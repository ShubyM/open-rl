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
from pathlib import Path
from typing import Any

import chz
from harness.git import git_commit_subject, git_snapshot, git_text, repo_root
from harness.io import read_json, write_json_atomic
from harness.recipe import Recipe, load_recipe


@chz.chz
class AttemptConfig:
  recipe: Path = Path("autoresearch.toml")
  researcher: str = ""
  name: str = "attempt"
  log_root: Path = Path("artifacts/autoresearch/runs")
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))
  clean: bool = False


def slug(text: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "attempt"


def researcher_name(researcher: str) -> str:
  researcher = re.sub(r"[^a-z0-9]+", "-", researcher.lower()).strip("-")[:40]
  if not researcher:
    raise ValueError("researcher is required")
  return researcher


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def now() -> float:
  return time.time()


def repo_paths_for_editables(paths: list[Path], root: Path) -> list[str]:
  resolved = []
  for path in paths:
    if path.is_absolute():
      resolved.append(str(path.resolve().relative_to(root)))
    else:
      resolved.append(str(path))
  return resolved


def git_diff_names(root: Path, *revs: str) -> set[str]:
  return set(path for path in git_text("diff", "--name-only", *revs, cwd=root).splitlines() if path)


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


def diff_files(root: Path, repo_paths: list[str]) -> list[dict[str, str]]:
  def blob(rev: str, path: str) -> str:
    return git_text("show", f"{rev}:{path}", cwd=root)

  def changed(*revs: str) -> list[str]:
    return git_text("diff", "--name-only", *revs, "--", *repo_paths, cwd=root).splitlines()

  files = [{"name": path, "old_text": blob("HEAD^", path), "new_text": blob("HEAD", path)} for path in changed("HEAD^", "HEAD")]
  if files:
    return files

  paths = changed("HEAD")
  paths += git_text("ls-files", "--others", "--exclude-standard", "--", *repo_paths, cwd=root).splitlines()
  return [
    {
      "name": path,
      "old_text": blob("HEAD", path),
      "new_text": (root / path).read_text(encoding="utf-8", errors="replace") if (root / path).exists() else "",
    }
    for path in dict.fromkeys(path for path in paths if path)
  ]


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


def manifest_attempts(log_root: Path, researcher: str) -> list[dict[str, Any]]:
  root = log_root / "researchers" / researcher / "attempts"
  attempts = []
  for path in sorted(root.glob("*/attempt.json")):
    try:
      data = read_json(path)
    except (OSError, json.JSONDecodeError):
      continue
    attempts.append({**data, "run_dir": path.parent})
  return attempts


def ensure_attempt_is_new(args: AttemptConfig, recipe: Recipe, researcher: str, git: dict[str, str | None]) -> Path | None:
  attempts = manifest_attempts(args.log_root, researcher)
  if args.name == "default-config":
    existing = next((row for row in attempts if row.get("baseline") or row.get("experiment", {}).get("name") == "default-config"), None)
    return Path(existing["run_dir"]) if existing else None
  root = repo_root()
  dirty = git_text("status", "--porcelain", cwd=root).strip()
  if dirty:
    raise SystemExit("Working tree must be clean before running an attempt. Commit the edit, then rerun RUN_ATTEMPT_COMMAND.")
  commit = git.get("commit")
  existing = next(
    (row for row in attempts if commit and row.get("git", {}).get("commit") == commit),
    None,
  )
  if existing:
    raise SystemExit(f"Commit {commit} was already evaluated for {researcher}: {existing['run_dir']}. Commit a new change before rerunning.")
  if not git.get("parent"):
    raise SystemExit("Agent-edited attempts need a parent commit so the attempted diff can be evaluated.")
  allowed = set(repo_paths_for_editables(recipe.editable, root))
  changed = git_diff_names(root, "HEAD^", "HEAD")
  extra = changed - allowed
  if extra:
    raise SystemExit("Attempt commit changes files outside recipe.editable:\n" + "\n".join(sorted(extra)))
  if not changed:
    raise SystemExit("Attempt commit does not change any recipe.editable files.")
  return None


def new_run_dir(args: AttemptConfig, researcher: str) -> tuple[int, Path]:
  attempts_root = args.log_root / "researchers" / researcher / "attempts"
  attempts_root.mkdir(parents=True, exist_ok=True)
  if args.name == "default-config":
    run_dir = attempts_root / "000-baseline"
    attempt = 0
  else:
    attempt = 1 + sum(1 for path in attempts_root.iterdir() if path.is_dir() and path.name != "000-baseline")
    run_dir = attempts_root / f"{attempt:03d}-{slug(args.name)}"
  run_dir.mkdir(parents=True, exist_ok=False)
  return attempt, run_dir


def activity_dir(args: AttemptConfig, researcher: str) -> Path:
  return Path(os.getenv("WORK_DIR") or args.log_root / "researchers" / researcher)


def command_for_attempt(args: AttemptConfig, recipe: Recipe, researcher: str, attempt: int, run_dir: Path) -> list[str]:
  values = {
    "attempt": attempt,
    "attempt_timeout_minutes": args.attempt_timeout_minutes,
    "log_root": args.log_root,
    "researcher": researcher,
    "run_dir": run_dir,
    "run_name": run_dir.name,
  }
  parts = shlex.split(os.path.expandvars(recipe.command).format(**values))
  if parts and parts[0] in {"python", "python3"}:
    parts[0] = sys.executable
  return parts


def attempt_manifest(
  args: AttemptConfig,
  recipe: Recipe,
  researcher: str,
  attempt: int,
  run_dir: Path,
  status: str,
  git: dict[str, str | None],
  metric: dict[str, Any] | None = None,
) -> dict[str, Any]:
  name = "baseline" if args.name == "default-config" else slug(args.name)
  return {
    "schema_version": 1,
    "id": run_dir.name,
    "researcher": researcher,
    "name": name,
    "baseline": args.name == "default-config",
    "status": status,
    "started_at": run_dir.stat().st_ctime if run_dir.exists() else now(),
    "updated_at": now(),
    "attempt_timeout_minutes": args.attempt_timeout_minutes,
    "git": {**git, "description": git_commit_subject()},
    "recipe": {
      "task": recipe.task,
      "editable": [str(path) for path in recipe.editable],
      "metric": recipe.metric,
      "metric_label": recipe.metric_label,
      "metric_mode": recipe.metric_mode,
    },
    "metric": metric,
    "order": attempt,
    "artifacts": {
      "logs": "logs.log",
      "metrics": "metrics.jsonl",
      "diff": "code.diff",
      "diff_full": "code_full.diff",
      "diff_files": "diff_files.json",
      "agent": "agent.log" if (run_dir / "agent.log").exists() else "",
      "notes": os.path.relpath(notes_path(args, researcher), run_dir),
    },
  }


def write_attempt_manifest(
  args: AttemptConfig,
  recipe: Recipe,
  researcher: str,
  attempt: int,
  run_dir: Path,
  status: str,
  git: dict[str, str | None],
  metric: dict[str, Any] | None = None,
) -> None:
  write_json_atomic(run_dir / "attempt.json", attempt_manifest(args, recipe, researcher, attempt, run_dir, status, git, metric))


def notes_path(args: AttemptConfig, researcher: str) -> Path:
  path = activity_dir(args, researcher) / "notes.md"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.touch(exist_ok=True)
  return path


def seed_agent_log(args: AttemptConfig, researcher: str, run_dir: Path) -> bool:
  source = Path(agent_log) if (agent_log := os.getenv("RESEARCHER_LOG_PATH")) else None
  if not source or not source.exists():
    return False
  offset_path = activity_dir(args, researcher) / "agent.offset"
  try:
    start = int(offset_path.read_text(encoding="utf-8").strip())
  except (FileNotFoundError, ValueError):
    start = 0
  (run_dir / "agent.log").write_bytes(source.read_bytes()[min(start, source.stat().st_size) :])
  return True


def mark_agent_boundary(args: AttemptConfig, researcher: str) -> None:
  source = Path(agent_log) if (agent_log := os.getenv("RESEARCHER_LOG_PATH")) else None
  if not source or not source.exists():
    return
  path = activity_dir(args, researcher) / "agent.offset"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(str(source.stat().st_size), encoding="utf-8")


def capture_diffs(recipe: Recipe, run_dir: Path, researcher: str) -> None:
  root = repo_root()
  paths = repo_paths_for_editables(recipe.editable, root)
  diff = git_paths_diff(root, paths, 3)
  (run_dir / "code.diff").write_text(diff, encoding="utf-8")
  (run_dir / "code_full.diff").write_text(git_paths_diff(root, paths, 100_000), encoding="utf-8")
  (run_dir / "diff_files.json").write_text(json.dumps(diff_files(root, paths)), encoding="utf-8")


def finish_attempt(recipe: Recipe, run_dir: Path, researcher: str, code: int) -> tuple[str, dict[str, Any] | None]:
  extracted = extract_metrics(run_dir, recipe.metric)
  status = "completed" if code == 0 and extracted else "timed_out" if code == 124 else "failed"
  with (run_dir / "logs.log").open("a", encoding="utf-8") as f:
    if not extracted:
      print(f"missing metric={recipe.metric}", file=f)
    print(f"status={status}", file=f)
  if extracted:
    final_value, final_step = extracted[-1]
  return status, (
    {
      "name": recipe.metric,
      "label": recipe.metric_label,
      "value": final_value,
      "step": final_step,
      "mode": recipe.metric_mode,
    }
    if extracted
    else None
  )


def clean_artifacts(log_root: Path) -> None:
  shutil.rmtree(log_root, ignore_errors=True)


def run_attempt(args: AttemptConfig) -> Path:
  recipe = load_recipe(args.recipe)
  researcher = researcher_name(args.researcher)
  git = git_snapshot()
  if existing := ensure_attempt_is_new(args, recipe, researcher, git):
    print(f"default-config attempt already exists for {researcher}: {existing}")
    return existing

  attempt, run_dir = new_run_dir(args, researcher)
  logs_path = run_dir / "logs.log"
  seed_agent_log(args, researcher, run_dir)
  write_attempt_manifest(args, recipe, researcher, attempt, run_dir, "running", git)

  capture_diffs(recipe, run_dir, researcher)
  code = run_logged(command_for_attempt(args, recipe, researcher, attempt, run_dir), logs_path, args.attempt_timeout_minutes * 60)
  mark_agent_boundary(args, researcher)
  capture_diffs(recipe, run_dir, researcher)
  status, metric = finish_attempt(recipe, run_dir, researcher, code)
  write_attempt_manifest(args, recipe, researcher, attempt, run_dir, status, git, metric)
  return run_dir


def main() -> None:
  args = chz.entrypoint(AttemptConfig, allow_hyphens=True)
  if args.clean:
    clean_artifacts(args.log_root)
    print(f"cleaned {args.log_root}")
    return
  run_attempt(args)


if __name__ == "__main__":
  main()
