"""Run one autoresearch attempt and record UI artifacts."""

from __future__ import annotations

import json
import os
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chz
import tomllib

UI_EVENTS_FILE = "ui_events.jsonl"


@chz.chz
class AttemptConfig:
  recipe: Path = Path("autoresearch.toml")
  researcher: str = ""
  name: str = "attempt"
  log_root: Path = Path("artifacts/autoresearch/runs")
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))
  clean: bool = False


@dataclass(frozen=True)
class Recipe:
  task: str
  command: str
  editable: list[Path]
  metric: str
  metric_label: str
  metric_mode: str


def slug(text: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "attempt"


def researcher_name(researcher: str) -> str:
  researcher = re.sub(r"[^a-z0-9]+", "-", researcher.lower()).strip("-")[:40]
  if not researcher:
    raise ValueError("researcher is required")
  return researcher


def git_text(*args: str, cwd: Path | None = None) -> str:
  result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
  return result.stdout if result.returncode == 0 else ""


def git_snapshot() -> dict[str, str | None]:
  def value(*args: str) -> str | None:
    return git_text(*args).strip() or None

  return {
    "branch": value("branch", "--show-current"),
    "commit": value("rev-parse", "--short=7", "HEAD"),
    "parent": value("rev-parse", "--short=7", "HEAD^"),
  }


def git_commit_subject() -> str:
  return git_text("log", "-1", "--pretty=%s").strip()


def append_ui_events(event_dir: Path, events: list[dict[str, Any]]) -> None:
  path = event_dir / UI_EVENTS_FILE
  path.parent.mkdir(parents=True, exist_ok=True)
  now = time.time()
  with path.open("a", encoding="utf-8") as f:
    for event in events:
      f.write(json.dumps({"time": now, **event}, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def load_recipe(path: Path) -> Recipe:
  raw = tomllib.loads(path.read_text(encoding="utf-8"))
  mode = str(raw["metric_mode"])
  if mode not in {"max", "min"}:
    raise ValueError("metric_mode must be max or min")
  return Recipe(
    task=str(raw["task"]),
    command=str(raw["command"]),
    editable=[Path(value) for value in raw["editable"]],
    metric=str(raw["metric"]),
    metric_label=str(raw["metric_label"]),
    metric_mode=mode,
  )


def repo_root() -> Path:
  return Path(git_text("rev-parse", "--show-toplevel").strip() or Path.cwd()).resolve()


def relative_to_repo(path: Path, root: Path) -> str:
  absolute = path if path.is_absolute() else (Path.cwd() / path).resolve()
  return str(absolute.relative_to(root))


def git_paths_diff(root: Path, repo_paths: list[str], context: int) -> str:
  for args in [("HEAD^", "HEAD"), ()]:
    diff = git_text("diff", f"--unified={context}", *args, "--", *repo_paths, cwd=root)
    if diff:
      return diff
  chunks = []
  for repo_path in repo_paths:
    absolute = root / repo_path
    if absolute.exists() and not git_text("ls-files", "--error-unmatch", repo_path, cwd=root):
      result = subprocess.run(
        ["git", "diff", "--no-index", f"--unified={context}", "--", "/dev/null", str(absolute)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
      )
      chunks.append(result.stdout)
  return "\n".join(chunk for chunk in chunks if chunk)


def stop_process(process: subprocess.Popen, sig: signal.Signals = signal.SIGTERM) -> None:
  try:
    os.killpg(process.pid, sig)
  except ProcessLookupError:
    return


def run_logged(command: list[str], log_path: Path, timeout_seconds: float | None = None) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("a", encoding="utf-8") as file:

    def write(text: str) -> None:
      sys.stdout.write(text)
      sys.stdout.flush()
      file.write(text)
      file.flush()

    write(f"$ {shlex.join(command)}\n")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    assert process.stdout is not None
    timed_out = False
    kill_timer = None

    def expire() -> None:
      nonlocal kill_timer, timed_out
      timed_out = True
      stop_process(process)
      kill_timer = threading.Timer(10, lambda: stop_process(process, signal.SIGKILL) if process.poll() is None else None)
      kill_timer.start()

    timer = threading.Timer(timeout_seconds, expire) if timeout_seconds and timeout_seconds > 0 else None
    try:
      if timer:
        timer.start()
      while True:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready and (line := process.stdout.readline()):
          write(line)
        if process.poll() is not None:
          stop_process(process)
          while select.select([process.stdout], [], [], 0)[0]:
            if not (line := process.stdout.readline()):
              break
            write(line)
          break
      code = process.wait()
      if timed_out:
        write(f"timed out after {timeout_seconds:.1f}s\n")
        return 124
      return code
    finally:
      if timer:
        timer.cancel()
      if kill_timer:
        kill_timer.cancel()
      process.stdout.close()
      stop_process(process)


def extract_metrics(run_dir: Path, metric: str) -> list[tuple[float, int | float | None]]:
  found = []
  for row in read_jsonl(run_dir / "metrics.jsonl"):
    if metric in row:
      found.append((float(row[metric]), row.get("step")))
  return found


def replayed_attempts(log_root: Path, researcher: str) -> list[dict[str, Any]]:
  rows = {}
  for event_file in sorted(log_root.glob(f"*/{UI_EVENTS_FILE}")):
    for event in read_jsonl(event_file):
      if event.get("kind") != "attempt":
        continue
      if event["researcher"] != researcher:
        continue
      row = rows.setdefault(str(event["work_id"]), {"run_dir": event_file.parent, "git": {}, "experiment": {}, "status": "running"})
      for key in ("status", "order", "description"):
        if key in event:
          row[key] = event[key]
      row["git"].update(event.get("git") or {})
      row["experiment"].update(event.get("experiment") or {})
  return list(rows.values())


def ensure_attempt_is_new(args: AttemptConfig, recipe: Recipe, researcher: str, git: dict[str, str | None]) -> Path | None:
  attempts = replayed_attempts(args.log_root, researcher)
  if args.name == "baseline":
    existing = next((row for row in attempts if row.get("status") == "completed" and row.get("experiment", {}).get("name") == "baseline"), None)
    return Path(existing["run_dir"]) if existing else None
  if git_text("status", "--porcelain", "--", *[str(path) for path in recipe.editable]).strip():
    raise SystemExit("Non-baseline attempts must commit the declared editable files before running. Commit the edit, then rerun RUN_ATTEMPT_COMMAND.")
  commit = git.get("commit")
  existing = next(
    (row for row in attempts if commit and row.get("status") in {"completed", "timed_out"} and row.get("git", {}).get("commit") == commit),
    None,
  )
  if existing:
    raise SystemExit(f"Commit {commit} was already evaluated for {researcher}: {existing['run_dir']}. Commit a new change before rerunning.")
  return None


@dataclass
class AttemptRun:
  args: AttemptConfig
  recipe: Recipe
  researcher: str
  git: dict[str, str | None]
  attempt: int
  run_dir: Path

  @classmethod
  def create(cls, args: AttemptConfig, recipe: Recipe, researcher: str, git: dict[str, str | None]) -> AttemptRun:
    attempt = 1 + sum(1 for path in args.log_root.glob(f"{researcher}-attempt-*") if path.is_dir())
    run_dir = args.log_root / f"{researcher}-attempt-{attempt}-{slug(args.name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return cls(args, recipe, researcher, git, attempt, run_dir)

  @property
  def run_name(self) -> str:
    return self.run_dir.name

  @property
  def logs_path(self) -> Path:
    return self.run_dir / "logs.log"

  def events(self, *events: dict[str, Any]) -> None:
    append_ui_events(self.run_dir, [{"work_id": self.run_name, "researcher": self.researcher, **event} for event in events])

  def command(self) -> list[str]:
    values = {
      "attempt": self.attempt,
      "attempt_timeout_minutes": self.args.attempt_timeout_minutes,
      "log_root": self.args.log_root,
      "researcher": self.researcher,
      "run_dir": self.run_dir,
      "run_name": self.run_name,
    }
    parts = shlex.split(os.path.expandvars(self.recipe.command).format(**values))
    if parts and parts[0] in {"python", "python3"}:
      parts[0] = sys.executable
    return parts

  def start(self) -> None:
    self.events(
      {
        "kind": "attempt",
        "status": "running",
        "order": self.attempt,
        "description": git_commit_subject(),
        "attempt_timeout_minutes": self.args.attempt_timeout_minutes,
        "git": self.git,
        "experiment": {
          "name": self.args.name,
          "task": self.recipe.task,
          "attempt": self.attempt,
          "attempt_timeout_minutes": self.args.attempt_timeout_minutes,
        },
        "recipe": {"name": self.recipe.task, "editable": [str(path) for path in self.recipe.editable], "metric": self.recipe.metric},
        "tab": "logs",
        "path": "logs.log",
      }
    )
    if self.seed_agent_log():
      self.events({"tab": "agent", "path": "agent.log"})

  def capture_diffs(self) -> None:
    root = repo_root()
    paths = [relative_to_repo(path, root) for path in self.recipe.editable]
    diff = git_paths_diff(root, paths, 3)
    full = git_paths_diff(root, paths, 1000000)
    (self.run_dir / "code.diff").write_text(diff, encoding="utf-8")
    (self.run_dir / "code_full.diff").write_text(full or diff, encoding="utf-8")
    self.events({"tab": "diff", "path": "code.diff"}, {"tab": "diff_full", "path": "code_full.diff"})

  def agent_offset_path(self) -> Path:
    return Path(os.getenv("WORK_DIR") or self.args.log_root / f"{self.researcher}-activity") / "agent.offset"

  def seed_agent_log(self) -> bool:
    source = Path(agent_log) if (agent_log := os.getenv("RESEARCHER_LOG_PATH")) else None
    if not source or not source.exists():
      return False
    try:
      start = int(self.agent_offset_path().read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
      start = 0
    (self.run_dir / "agent.log").write_bytes(source.read_bytes()[min(start, source.stat().st_size) :])
    return True

  def mark_agent_boundary(self) -> None:
    source = Path(agent_log) if (agent_log := os.getenv("RESEARCHER_LOG_PATH")) else None
    if not source or not source.exists():
      return
    path = self.agent_offset_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(source.stat().st_size), encoding="utf-8")

  def finish(self, code: int) -> None:
    extracted = extract_metrics(self.run_dir, self.recipe.metric)
    status = "completed" if code == 0 and extracted else "timed_out" if code == 124 else "failed"
    with self.logs_path.open("a", encoding="utf-8") as f:
      if not extracted:
        print(f"missing metric={self.recipe.metric}", file=f)
      print(f"status={status}", file=f)
    if extracted:
      self.events(
        *[
          {
            "metric": {
              "name": self.recipe.metric,
              "label": self.recipe.metric_label,
              "value": value,
              "step": step,
              "mode": self.recipe.metric_mode,
            }
          }
          for value, step in extracted
        ]
      )
    self.events({"status": status})

  def run(self) -> Path:
    self.start()
    self.capture_diffs()
    code = run_logged(self.command(), self.logs_path, self.args.attempt_timeout_minutes * 60)
    self.mark_agent_boundary()
    self.capture_diffs()
    self.finish(code)
    return self.run_dir


def clean_artifacts(log_root: Path) -> None:
  shutil.rmtree(log_root, ignore_errors=True)


def run_attempt(args: AttemptConfig) -> Path:
  recipe = load_recipe(args.recipe)
  researcher = researcher_name(args.researcher)
  git = git_snapshot()
  if existing := ensure_attempt_is_new(args, recipe, researcher, git):
    print(f"baseline already exists for {researcher}: {existing}")
    return existing
  return AttemptRun.create(args, recipe, researcher, git).run()


def main() -> None:
  args = chz.entrypoint(AttemptConfig, allow_hyphens=True)
  if args.clean:
    clean_artifacts(args.log_root)
    print(f"cleaned {args.log_root}")
    return
  run_attempt(args)


if __name__ == "__main__":
  main()
