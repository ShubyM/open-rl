#!/usr/bin/env python3
"""Plot a LAB run from its log directory: raw per-rollout rewards, the
smoothed per-step mean, and held-out criterion pass rate at eval steps.

  python plot_run.py artifacts/harvey-labs/<run> [--out run.png]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from harvey_labs.results import read_results, rollout_rewards

RAW_COLOR = "#86b6ef"
SMOOTH_COLOR = "#2a78d6"
EVAL_COLOR = "#008300"
INK = "#3d3d3a"
MUTED = "#7f7e78"


def ema(values: list[float], alpha: float = 0.4) -> list[float]:
  smoothed = []
  for value in values:
    smoothed.append(value if not smoothed else alpha * value + (1 - alpha) * smoothed[-1])
  return smoothed


def plot_results(log_dir: Path, out: Path | None = None, title: str | None = None, *, results: dict | None = None) -> Path:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  results = read_results(log_dir) if results is None else results
  raw = rollout_rewards(log_dir)
  train_rows = [(row["step"], row["reward"]) for row in results["train"]]
  evals = [(row["step"], row["pass_rate"]) for row in results["evaluations"]]

  fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
  rng = random.Random(0)
  if raw:
    xs = [step + rng.uniform(-0.18, 0.18) for step, _ in raw]
    ax.scatter(xs, [reward for _, reward in raw], s=14, color=RAW_COLOR, alpha=0.6, linewidths=0, label="rollout reward", zorder=2)
  if train_rows:
    steps = [step for step, _ in train_rows]
    smoothed = ema([reward for _, reward in train_rows])
    ax.plot(steps, smoothed, color=SMOOTH_COLOR, linewidth=2, label="mean reward (EMA)", zorder=3)
    ax.annotate(f"{smoothed[-1]:.2f}", (steps[-1], smoothed[-1]), textcoords="offset points", xytext=(6, -3), color=SMOOTH_COLOR, fontsize=9)
  if evals:
    ex = [step for step, _ in evals]
    ey = [rate for _, rate in evals]
    ax.plot(ex, ey, color=EVAL_COLOR, linewidth=2, linestyle=(0, (2, 3)), zorder=3)
    ax.scatter(ex, ey, s=64, color=EVAL_COLOR, marker="D", label="held-out criterion pass rate", zorder=4)
    for x, y in evals:
      ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", color=EVAL_COLOR, fontsize=9)

  ax.set_ylim(-0.15, 1.05)
  ax.set_xlabel("completed training batches", color=INK)
  ax.set_ylabel("reward / pass rate", color=INK)
  ax.set_title(title or log_dir.name, color=INK, fontsize=11, loc="left")
  ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
  for spine in ("left", "bottom"):
    ax.spines[spine].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)
  ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)

  out = out or log_dir / "run_plot.png"
  fig.tight_layout()
  fig.savefig(out)
  plt.close(fig)
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("log_dir", type=Path)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--title", default=None, help="plot title; defaults to the log directory name")
  args = parser.parse_args()

  print(plot_results(args.log_dir, args.out, args.title))


if __name__ == "__main__":
  main()
