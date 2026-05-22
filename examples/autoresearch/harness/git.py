"""Small git helpers for the autoresearch harness."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_text(*args: str, cwd: Path | None = None) -> str:
  result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
  return result.stdout if result.returncode == 0 else ""


def repo_root() -> Path:
  return Path(git_text("rev-parse", "--show-toplevel").strip() or Path.cwd()).resolve()


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
