#!/usr/bin/env python3
"""Score one LAB run from the LAB virtual environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Score a Harvey LAB run for Open-RL.")
  parser.add_argument("--lab-root", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--task", required=True)
  parser.add_argument("--judge-model", required=True)
  parser.add_argument("--parallel", type=int, default=1)
  parser.add_argument("--max-criteria", type=int, default=None)
  parser.add_argument("--scores-out", required=True)
  return parser.parse_args()


def load_env(lab_root: Path) -> None:
  env_path = lab_root / ".env"
  if not env_path.exists():
    return
  for raw_line in env_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
      continue
    key, _, value = line.partition("=")
    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
  args = parse_args()
  lab_root = Path(args.lab_root).resolve()
  sys.path.insert(0, str(lab_root))
  load_env(lab_root)

  from evaluation.judge import Judge
  from evaluation.scoring import score_rubric

  task_dir = lab_root / "tasks" / Path(*args.task.split("/"))
  task_config = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
  criteria = list(task_config["criteria"])
  if args.max_criteria is not None:
    criteria = criteria[: args.max_criteria]

  run_dir = lab_root / "results" / args.run_id
  result = score_rubric(
    criteria=criteria,
    run_dir=run_dir,
    judge=Judge(model=args.judge_model),
    task_desc=task_config["title"],
    parallel=args.parallel,
  )
  n_criteria = len(result.criteria_results)
  n_passed = sum(1 for criterion in result.criteria_results if criterion["verdict"] == "pass")
  scores = {
    "score": result.score,
    "max_score": result.max_score,
    "summary": f"{n_passed}/{n_criteria} criteria passed.",
    "all_pass": n_criteria > 0 and n_passed == n_criteria,
    "n_criteria": n_criteria,
    "n_passed": n_passed,
    "criteria_results": result.criteria_results,
    "run_id": args.run_id,
    "task": args.task,
    "judge_model": args.judge_model,
    "scored_at": datetime.now(UTC).isoformat(),
  }

  scores_out = Path(args.scores_out)
  scores_out.parent.mkdir(parents=True, exist_ok=True)
  scores_out.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
  (run_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
