#!/usr/bin/env python3
"""run49 in the shared plot_run.py shape: every per-rollout reward as a dot,
the per-step mean reward as a bold EMA, and the held-out criterion pass rate
as red diamonds at eval steps.

  python plot.py data [--out run49.png]

data/ holds metrics.jsonl and rollout_rewards.jsonl (step, total_reward per
rollout, extracted from iteration_*/train_rollout_summaries.jsonl).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RAW_COLOR = "#86b6ef"
SMOOTH_COLOR = "#2a78d6"
EVAL_COLOR = "#c8102e"
INK = "#3d3d3a"
MUTED = "#7f7e78"


def load_jsonl(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ema(values: list[float], alpha: float = 0.4) -> list[float]:
  smoothed: list[float] = []
  for value in values:
    smoothed.append(value if not smoothed else alpha * value + (1 - alpha) * smoothed[-1])
  return smoothed


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("log_dir", type=Path)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = load_jsonl(args.log_dir / "metrics.jsonl")
  raw = [(r["step"], r["total_reward"]) for r in load_jsonl(args.log_dir / "rollout_rewards.jsonl")]
  train = [(r["step"], r["env/all/reward/total"]) for r in rows if "env/all/reward/total" in r]
  evals = [(r["step"], r["test/env/all/lab/criteria_pass_fraction"]) for r in rows if "test/env/all/lab/criteria_pass_fraction" in r]

  fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

  jitter = random.Random(0)
  ax.scatter(
    [s + jitter.uniform(-0.28, 0.28) for s, _ in raw], [v for _, v in raw],
    s=14, color=RAW_COLOR, alpha=0.55, linewidths=0, label="rollout reward", zorder=2,
  )
  steps = [s for s, _ in train]
  smoothed = ema([v for _, v in train])
  ax.plot(steps, smoothed, color=SMOOTH_COLOR, linewidth=3, label="mean reward (EMA)", zorder=3)
  ax.annotate(f"{smoothed[-1]:.2f}", (steps[-1], smoothed[-1]), textcoords="offset points", xytext=(7, -3), color=SMOOTH_COLOR, fontsize=10, fontweight="bold")

  ex = [s for s, _ in evals]
  ey = [v for _, v in evals]
  ax.plot(ex, ey, color=EVAL_COLOR, linewidth=2, linestyle=(0, (2, 3)), zorder=3)
  ax.scatter(ex, ey, s=90, color=EVAL_COLOR, marker="D", label="held-out criterion pass rate (50 tasks)", zorder=4)
  for x, y in evals:
    ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", color=EVAL_COLOR, fontsize=9.5, fontweight="bold")

  ax.set_ylim(-0.15, 1.05)
  ax.set_xlim(-0.7, steps[-1] + 1.4)
  ax.set_xlabel("step", color=INK)
  ax.set_ylabel("reward / pass rate", color=INK)
  ax.set_title(f"run49 (Qwen3.8-27B, Automodel CP4, 262k ctx) through step {steps[-1]}", color=INK, fontsize=11, loc="left")
  ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
  for spine in ("left", "bottom"):
    ax.spines[spine].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)
  ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

  out = args.out or args.log_dir / "run49.png"
  fig.tight_layout()
  fig.savefig(out)
  print(out)


if __name__ == "__main__":
  main()
