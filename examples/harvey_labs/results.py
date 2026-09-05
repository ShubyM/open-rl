"""Read Harvey run results for people, agents, and plots.

  uv --project examples run --no-sync python -m harvey_labs.results <run-dir> [--json]

The step coordinate is completed training batches: training batch 0 finishes
at step 1; its pre-update evaluation measures step 0. Stored indices stay intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ENV_NAMES = ("harvey-labs", "all")


def load_metrics(path: Path) -> list[dict]:
  lines = path.read_text().splitlines(keepends=True)
  rows = []
  for index, line in enumerate(lines):
    if not line.strip():
      continue
    try:
      rows.append(json.loads(line))
    except json.JSONDecodeError:
      # A running writer may still be appending its final record. Corruption
      # in a complete line must remain visible, rather than silently losing data.
      if index == len(lines) - 1 and not line.endswith("\n"):
        break
      raise
  return rows


def env_get(row: dict, prefix: str, suffix: str):
  for name in ENV_NAMES:
    value = row.get(f"{prefix}env/{name}/{suffix}")
    if value is not None:
      return value
  return None


def eval_result(row: dict) -> dict | None:
  # Keep numerator and denominator in the same namespace. Prefer LAB-specific
  # counts over aggregates when both are present.
  for name in ENV_NAMES:
    prefix = f"test/env/{name}/"
    passed = row.get(prefix + "lab/criteria_passed")
    total = row.get(prefix + "lab/criteria_total")
    episodes = row.get(prefix + "total_episodes")
    if passed is not None and total:
      return {
        "pass_rate": passed / total,
        "aggregation": "pooled_criteria",
        "criteria_passed": passed * episodes if episodes else None,
        "criteria_total": total * episodes if episodes else None,
        "episodes": episodes,
      }
  rate = env_get(row, "test/", "lab/criteria_pass_fraction")
  if rate is None:
    return None
  return {
    "pass_rate": rate,
    "aggregation": "mean_episode_fraction",
    "criteria_passed": None,
    "criteria_total": None,
    "episodes": env_get(row, "test/", "total_episodes"),
  }


def eval_pass_rate(row: dict) -> float | None:
  result = eval_result(row)
  return result["pass_rate"] if result is not None else None


def rollout_rewards(log_dir: Path) -> list[tuple[int, float]]:
  points = []
  for summary in sorted(log_dir.glob("**/iteration_*/train_rollout_summaries.jsonl")):
    step = int(summary.parent.name.split("_")[-1]) + 1
    points.extend((step, row["total_reward"]) for row in load_metrics(summary))
  return sorted(points)


def read_results(log_dir: Path, *, include_metrics: bool = False) -> dict:
  rows = load_metrics(log_dir / "metrics.jsonl")
  train, evaluations = {}, {}
  for row in rows:
    batch = row.get("progress/batch", row.get("step"))
    if batch is None:
      continue
    reward = env_get(row, "", "reward/total")
    if reward is not None:
      train[batch + 1] = {"step": batch + 1, "reward": reward}
    result = eval_result(row)
    if result is not None:
      evaluations[batch] = {"step": batch, "phase": row.get("eval_phase", "evaluation"), **result}
  train_rows = [train[key] for key in sorted(train)]
  eval_rows = [evaluations[key] for key in sorted(evaluations)]
  final = [row for row in eval_rows if row["phase"] == "final"]
  return {
    "schema_version": 1,
    "run_dir": str(log_dir),
    "step_unit": "completed_training_batches",
    "training_batches_with_metrics": len(train_rows),
    "last_train": train_rows[-1] if train_rows else None,
    "baseline_eval": evaluations.get(0),
    "latest_eval": eval_rows[-1] if eval_rows else None,
    "best_eval": max(eval_rows, key=lambda row: row["pass_rate"]) if eval_rows else None,
    "final_eval": final[-1] if final else None,
    "train": train_rows,
    "evaluations": eval_rows,
    **({"metrics": rows} if include_metrics else {}),
  }


def format_summary(results: dict) -> str:
  lines = [f"Run: {results['run_dir']}", f"Training batches with metrics: {results['training_batches_with_metrics']}"]
  if row := results["last_train"]:
    lines.append(f"Latest training reward: {row['reward']:.4f} (after batch {row['step']})")
  for key, label in (("baseline_eval", "Baseline"), ("best_eval", "Best"), ("latest_eval", "Latest"), ("final_eval", "Final")):
    row = results[key]
    if row is None:
      lines.append(f"{label} eval: unavailable")
      continue
    counts = ""
    if row["criteria_total"] is not None:
      counts = f", {row['criteria_passed']:.0f}/{row['criteria_total']:.0f} criteria"
    lines.append(f"{label} eval: {row['pass_rate']:.1%} ({row['aggregation']}{counts}; after batch {row['step']})")
  return "\n".join(lines)


def write_report(log_dir: Path) -> None:
  from .plot_run import plot_results

  results = read_results(log_dir)
  (log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
  plot_results(log_dir, results=results)
  print(format_summary(results))
  print(f"Saved {log_dir / 'results.json'} and {log_dir / 'run_plot.png'}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("log_dir", type=Path)
  parser.add_argument("--json", action="store_true", help="emit the versioned summary and unsmoothed series as JSON")
  parser.add_argument("--metrics", action="store_true", help="include all original metric rows in JSON for further analysis")
  args = parser.parse_args()
  if args.metrics and not args.json:
    parser.error("--metrics requires --json")
  try:
    results = read_results(args.log_dir, include_metrics=args.metrics)
  except (OSError, ValueError) as exc:
    parser.exit(1, f"Cannot read run results: {exc}\n")
  print(json.dumps(results, indent=2) if args.json else format_summary(results))


if __name__ == "__main__":
  main()
