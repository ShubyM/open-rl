#!/usr/bin/env python3
"""Create a deterministic Harvey LAB train/eval task split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Create a deterministic task split.")
  parser.add_argument("--lab-root", default="./harvey-labs")
  parser.add_argument("--areas", action="append", default=None, help="Practice area directory name. Repeatable.")
  parser.add_argument("--eval-frac", type=float, default=0.2)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--out", default="split.json")
  return parser.parse_args()


def assignment_value(task_name: str, seed: int) -> float:
  digest = hashlib.sha256(f"{task_name}{seed}".encode()).hexdigest()
  return int(digest, 16) / float(1 << 256)


def main() -> int:
  args = parse_args()
  lab_root = Path(args.lab_root)
  tasks_root = lab_root / "tasks"
  if not tasks_root.exists():
    raise SystemExit(f"tasks directory not found: {tasks_root}")
  if not 0 <= args.eval_frac <= 1:
    raise SystemExit("--eval-frac must be between 0 and 1")

  areas = args.areas or sorted(p.name for p in tasks_root.iterdir() if p.is_dir())
  task_names: list[str] = []
  for area in areas:
    area_dir = tasks_root / area
    if not area_dir.is_dir():
      raise SystemExit(f"area not found: {area_dir}")
    for task_json in sorted(area_dir.glob("*/task.json")):
      task_names.append(str(task_json.parent.relative_to(tasks_root)))

  train = []
  eval_tasks = []
  for task_name in sorted(task_names):
    if assignment_value(task_name, args.seed) < args.eval_frac:
      eval_tasks.append(task_name)
    else:
      train.append(task_name)

  split = {
    "train": train,
    "eval": eval_tasks,
    "seed": args.seed,
    "areas": areas,
  }
  Path(args.out).write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
  print(f"train={len(train)} eval={len(eval_tasks)} total={len(task_names)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
