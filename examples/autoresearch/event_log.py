"""Shared JSONL event writer for autoresearch UI state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

UI_EVENTS_FILE = "ui_events.jsonl"


def append_ui_events(event_dir: Path, events: list[dict[str, Any]]) -> None:
  path = event_dir / UI_EVENTS_FILE
  path.parent.mkdir(parents=True, exist_ok=True)
  now = time.time()
  with path.open("a", encoding="utf-8") as f:
    for event in events:
      f.write(json.dumps({"time": now, **event}, sort_keys=True) + "\n")


def activity_events(args: argparse.Namespace) -> list[dict[str, Any]]:
  base = {
    "attempt_timeout_minutes": args.attempt_timeout_minutes,
    "kind": "activity",
    "order": 0,
    "researcher": args.researcher,
    "status": args.status,
    "work_id": f"{args.researcher}-activity",
  }
  return [
    {**base, "tab": "agent", "path": args.agent_log},
    {**base, "tab": "notes", "path": args.notes},
  ]


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--event-dir", type=Path, required=True)
  parser.add_argument("--researcher", required=True)
  parser.add_argument("--status", required=True)
  parser.add_argument("--attempt-timeout-minutes", type=float, required=True)
  parser.add_argument("--agent-log", required=True)
  parser.add_argument("--notes", required=True)
  args = parser.parse_args(argv)
  append_ui_events(args.event_dir, activity_events(args))


if __name__ == "__main__":
  main()
