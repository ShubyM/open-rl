"""Launch one autoresearch agent and maintain its researcher manifest."""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chz

from harness.git import git_text
from harness.io import write_json_atomic


@chz.chz
class AgentConfig:
  researcher: str = os.getenv("RESEARCHER_ID", "")
  repo_dir: Path = Path(os.getenv("REPO_DIR", os.getcwd()))
  log_root: Path = Path(os.getenv("LOG_ROOT", "artifacts/autoresearch/runs"))
  recipe: Path = Path(os.getenv("RECIPE", "autoresearch.toml"))
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))
  agent_timeout_minutes: float = float(os.getenv("AGENT_TIMEOUT_MINUTES", "10"))
  agent_model: str = os.getenv("AGENT_MODEL", "gemini-3.1-pro-preview")
  agent_flags: str = os.getenv("AGENT_FLAGS", "--yolo --output-format stream-json")
  ready_urls: str = os.getenv("READY_URLS", "")
  ready_timeout_seconds: int = int(os.getenv("READY_TIMEOUT_SECONDS", "1200"))


@dataclass
class AgentPaths:
  work_dir: Path
  agent_log: Path
  launcher_log: Path
  notes: Path
  program: Path


SESSION_STATUS = "running"


def die(message: str) -> None:
  raise SystemExit(message)


def log(paths: AgentPaths, message: str) -> None:
  print(message)
  for path in (paths.launcher_log, paths.agent_log):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
      print(message, file=f)


def rel(path: Path, root: Path) -> str:
  try:
    return str(path.relative_to(root))
  except ValueError:
    return str(path)


def write_researcher(paths: AgentPaths, args: AgentConfig, status: str) -> None:
  root = paths.work_dir.resolve()
  root.mkdir(parents=True, exist_ok=True)
  for path in (paths.agent_log, paths.launcher_log, paths.notes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
  write_json_atomic(
    root / "researcher.json",
    {
      "schema_version": 1,
      "researcher": args.researcher,
      "status": status,
      "started_at": root.stat().st_ctime,
      "updated_at": time.time(),
      "agent_timeout_minutes": args.agent_timeout_minutes,
      "artifacts": {
        "agent": rel(paths.agent_log.resolve(), root),
        "launcher": rel(paths.launcher_log.resolve(), root),
        "notes": rel(paths.notes.resolve(), root),
      },
    },
  )


def append_exclude(repo_dir: Path, pattern: str) -> None:
  exclude = Path(git_text("rev-parse", "--git-path", "info/exclude", cwd=repo_dir).strip())
  if not exclude.is_absolute():
    exclude = repo_dir / exclude
  exclude.parent.mkdir(parents=True, exist_ok=True)
  existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
  if pattern not in existing.splitlines():
    with exclude.open("a", encoding="utf-8") as f:
      print(pattern, file=f)


def prepare_git(args: AgentConfig, paths: AgentPaths) -> None:
  subprocess.run(["git", "config", "--global", "--add", "safe.directory", str(args.repo_dir)], check=False)
  if not git_text("rev-parse", "--is-inside-work-tree", cwd=args.repo_dir).strip():
    die(f"REPO_DIR={args.repo_dir} is not a git repo.")
  log(paths, f"Using existing git repo at {git_text('rev-parse', '--show-toplevel', cwd=args.repo_dir).strip()}.")
  subprocess.run(["git", "config", "user.email", "agent@open-rl.local"], cwd=args.repo_dir, check=True)
  subprocess.run(["git", "config", "user.name", "Autoresearch Agent"], cwd=args.repo_dir, check=True)
  append_exclude(args.repo_dir, "__pycache__/")
  append_exclude(args.repo_dir, "*.pyc")
  repo_root = Path(git_text("rev-parse", "--show-toplevel", cwd=args.repo_dir).strip()).resolve()
  log_root = args.log_root.resolve()
  if log_root.is_relative_to(repo_root):
    append_exclude(args.repo_dir, f"/{log_root.relative_to(repo_root)}/")
  if git_text("status", "--porcelain", cwd=args.repo_dir).strip():
    die("Existing git repo has local changes. Run from an isolated clean copy.")


def wait_for_dependencies(args: AgentConfig, paths: AgentPaths) -> None:
  urls = [url for url in args.ready_urls.replace(",", " ").split() if url]
  if not urls:
    return
  log(paths, f"Waiting for dependencies before default-config attempt: {' '.join(urls)}")
  for url in urls:
    deadline = time.monotonic() + args.ready_timeout_seconds
    last_log = 0.0
    while time.monotonic() < deadline:
      if subprocess.run(["curl", "-fsS", "--max-time", "5", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0:
        log(paths, f"Dependency ready: {url}")
        break
      if time.monotonic() - last_log >= 30:
        log(paths, f"Waiting for dependency: {url}")
        last_log = time.monotonic()
      time.sleep(5)
    else:
      die(f"Timed out waiting for dependency after {args.ready_timeout_seconds}s: {url}")


def attempt_command(args: AgentConfig, name: str) -> str:
  return (
    "uv run --no-sync --package open-rl-client python -m harness.attempt "
    f"recipe={shlex.quote(str(args.recipe))} researcher={shlex.quote(args.researcher)} "
    f"attempt_timeout_minutes={args.attempt_timeout_minutes} name={shlex.quote(name)} "
    f"log_root={shlex.quote(str(args.log_root))}"
  )


def run_command(command: list[str], paths: AgentPaths, timeout_seconds: float | None = None) -> int:
  with paths.agent_log.open("a", encoding="utf-8") as log_file:
    log_file.write(f"$ {shlex.join(command)}\n")
    log_file.flush()
    process = subprocess.Popen(command, cwd=Path.cwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    assert process.stdout is not None
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    while True:
      line = process.stdout.readline()
      if line:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()
      if process.poll() is not None:
        break
      if deadline and time.monotonic() > deadline:
        os.killpg(process.pid, signal.SIGTERM)
        time.sleep(10)
        if process.poll() is None:
          os.killpg(process.pid, signal.SIGKILL)
        log_file.write(f"timed out after {timeout_seconds:.1f}s\n")
        return 124
    return process.wait()


def make_paths(args: AgentConfig) -> AgentPaths:
  work_dir = args.log_root / "researchers" / args.researcher
  return AgentPaths(
    work_dir=work_dir,
    agent_log=work_dir / "agent.log",
    launcher_log=work_dir / "launcher.log",
    notes=work_dir / "notes.md",
    program=work_dir / "program.md",
  )


def run_agent(args: AgentConfig) -> int:
  global SESSION_STATUS
  if not args.researcher:
    die("researcher is required; set RESEARCHER_ID or pass researcher=...")
  args.repo_dir = args.repo_dir.resolve()
  args.log_root = args.log_root.resolve()
  paths = make_paths(args)
  os.chdir(args.repo_dir)
  program_file = args.recipe.parent / "program.md"
  if not program_file.is_file():
    die(f"program file does not exist next to RECIPE: {program_file}")
  paths.work_dir.mkdir(parents=True, exist_ok=True)
  paths.program.write_text(program_file.read_text(encoding="utf-8"), encoding="utf-8")
  write_researcher(paths, args, "running")
  prepare_git(args, paths)
  wait_for_dependencies(args, paths)
  os.environ.update(
    {
      "RESEARCHER_ID": args.researcher,
      "ATTEMPT_TIMEOUT_MINUTES": str(args.attempt_timeout_minutes),
      "AGENT_TIMEOUT_MINUTES": str(args.agent_timeout_minutes),
      "LOG_ROOT": str(args.log_root),
      "WORK_DIR": str(paths.work_dir),
      "REPO_DIR": str(args.repo_dir),
      "RESEARCHER_LOG_PATH": str(paths.agent_log),
      "RECIPE": str(args.recipe),
      "RUN_ATTEMPT_COMMAND": attempt_command(args, "short-slug"),
      "DEFAULT_CONFIG_COMMAND": attempt_command(args, "default-config"),
      "GEMINI_CLI_TRUST_WORKSPACE": "true",
      "PYTHONDONTWRITEBYTECODE": "1",
    }
  )
  baseline = shlex.split(attempt_command(args, "default-config"))
  run_command(baseline, paths, args.attempt_timeout_minutes * 60)
  if not shutil.which("gemini"):
    die("gemini not found; install @google/gemini-cli in the image or local environment.")
  write_researcher(paths, args, "running")
  log(paths, f"Starting {args.agent_timeout_minutes} minute agent timeout.")
  code = run_command(
    ["gemini", "--model", args.agent_model, *shlex.split(args.agent_flags), "--prompt", paths.program.read_text(encoding="utf-8")],
    paths,
    args.agent_timeout_minutes * 60,
  )
  if code in {124, 137}:
    SESSION_STATUS = "timed_out"
    log(paths, f"Agent timed out after {args.agent_timeout_minutes} minutes.")
    return 0
  SESSION_STATUS = "completed" if code == 0 else "failed"
  return code


def main(argv: list[str] | None = None) -> None:
  args = chz.entrypoint(AgentConfig, argv=argv, allow_hyphens=True)
  paths = make_paths(args)
  try:
    code = run_agent(args)
  except KeyboardInterrupt:
    write_researcher(paths, args, "stopped")
    raise SystemExit(143)
  except SystemExit:
    write_researcher(paths, args, "failed")
    raise
  except Exception:
    write_researcher(paths, args, "failed")
    raise
  write_researcher(paths, args, SESSION_STATUS)
  log(paths, f"Autoresearch session exiting with code {code}.")
  raise SystemExit(code)


if __name__ == "__main__":
  main()
