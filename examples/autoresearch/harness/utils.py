from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEC_HASH_IGNORED_DIRS = {".git", "__pycache__"}
SPEC_HASH_IGNORED_FILES = {".DS_Store", "kustomization.yaml"}
SPEC_HASH_IGNORED_PREFIXES = ("..",)
LOG_FLUSH_BYTES = 256 * 1024
LOG_FLUSH_SECONDS = 1.0

# Single source of truth for the on-disk layout. The keys are the artifact names
# used in UI/HTTP requests; the values are the filenames the harness writes.
METADATA = "metadata.json"
AGENT_ARTIFACTS = {"agent": "agent.log", "notes": "notes.md"}
ATTEMPT_ARTIFACTS = {"logs": "attempt.log", "metrics": "metrics.jsonl", "diff": "diff.patch"}


@dataclass(frozen=True)
class AgentPaths:
  root: Path
  run_root: Path
  attempts: Path
  workspace: Path
  metadata: Path
  agent_log: Path
  notes: Path
  program: Path

  @classmethod
  def from_run(cls, log_root: Path, run_name: str, agent: str) -> AgentPaths:
    root = (log_root / run_name / "researchers" / agent).resolve()
    return cls(
      root,
      root.parent.parent,
      root / "attempts",
      root / "workspace",
      root / METADATA,
      root / AGENT_ARTIFACTS["agent"],
      root / AGENT_ARTIFACTS["notes"],
      root / "program.md",
    )

  def attempt_dir(self, name: str) -> Path:
    return self.attempts / name


def close_attempt_agent_logs(agent_dir: Path, end: int) -> None:
  for path in sorted((agent_dir / "attempts").glob(f"*/{METADATA}")):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("agent_log_end") is not None:
      continue
    start = int(data.get("agent_log_start", 0) or 0)
    data["agent_log_start"] = start
    data["agent_log_end"] = max(start, end)
    write_json_atomic(path, data)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with tmp.open("w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def agent_id(value: str) -> str:
  normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:40]
  if not normalized:
    raise ValueError("agent_id is required")
  return normalized


def run_id(value: str) -> str:
  normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80]
  if not normalized:
    raise ValueError("run_name is required")
  return normalized


def recipe_spec_hash(root: Path) -> str:
  root = root.resolve()
  digest = hashlib.sha256()
  for path in sorted(root.rglob("*")):
    rel = path.relative_to(root)
    if path.is_dir() or path.name in SPEC_HASH_IGNORED_FILES:
      continue
    if any(part in SPEC_HASH_IGNORED_DIRS for part in rel.parts):
      continue
    if any(part.startswith(SPEC_HASH_IGNORED_PREFIXES) for part in rel.parts):
      continue
    digest.update(rel.as_posix().encode())
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
  return "sha256:" + digest.hexdigest()


def git_text(*args: str, cwd: Path | None = None) -> str:
  result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
  return result.stdout if result.returncode == 0 else ""


def stop_process(process: subprocess.Popen, sig: signal.Signals = signal.SIGTERM) -> None:
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    return


def run_logged(
  command: list[str],
  log_path: Path,
  timeout_seconds: float | None = None,
  cwd: Path | None = None,
  env: dict[str, str] | None = None,
  idle_message: str = "",
  idle_seconds: float = 60.0,
  echo: bool = True,
) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("ab") as file:
    lock = threading.Lock()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    flush_state = {"pending": 0, "last_flush": time.monotonic()}
    last_output = [time.monotonic()]

    def flush(final: bool = False) -> None:
      if echo:
        tail = decoder.decode(b"", final=final)
        if tail:
          sys.stdout.write(tail)
      file.flush()
      if echo:
        sys.stdout.flush()
      flush_state["pending"] = 0
      flush_state["last_flush"] = time.monotonic()

    def maybe_flush(nbytes: int) -> None:
      flush_state["pending"] += nbytes
      now = time.monotonic()
      if flush_state["pending"] >= LOG_FLUSH_BYTES or now - flush_state["last_flush"] >= LOG_FLUSH_SECONDS:
        flush()

    def write_control(text: str) -> None:
      data = text.encode("utf-8")
      with lock:
        file.write(data)
        if echo:
          sys.stdout.write(text)
        maybe_flush(len(data))

    def write_child(data: bytes) -> None:
      with lock:
        file.write(data)
        if echo:
          sys.stdout.write(decoder.decode(data))
        maybe_flush(len(data))

    write_control(f"$ {shlex.join(command)}\n")
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    if process.stdout is None:
      raise RuntimeError("subprocess stdout pipe was not created")

    def copy_output() -> None:
      while chunk := process.stdout.read1(65536):
        last_output[0] = time.monotonic()
        write_child(chunk)
      with lock:
        flush(final=True)

    reader = threading.Thread(target=copy_output, daemon=True)
    reader.start()
    try:
      timeout = timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
      deadline = time.monotonic() + timeout if timeout else None
      while process.poll() is None:
        now = time.monotonic()
        if deadline and now >= deadline:
          break
        if idle_message and now - last_output[0] >= idle_seconds:
          write_control(f"{idle_message}\n")
          last_output[0] = now
        wait = 1.0
        if deadline:
          wait = min(wait, max(0, deadline - now))
        if idle_message:
          wait = min(wait, max(0, idle_seconds - (now - last_output[0])))
        try:
          process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
          pass
      if process.poll() is None:
        stop_process(process)
        try:
          process.wait(timeout=10)
        except subprocess.TimeoutExpired:
          stop_process(process, signal.SIGKILL)
          process.wait()
        reader.join(timeout=1)
        write_control(f"timed out after {timeout_seconds:.1f}s\n")
        with lock:
          flush()
        return 124
      reader.join(timeout=1)
      with lock:
        flush()
      return process.returncode
    finally:
      if process.poll() is None:
        stop_process(process)
      reader.join(timeout=1)
      with lock:
        flush()
