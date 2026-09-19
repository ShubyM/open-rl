#!/usr/bin/env python3
"""Plot a LAB run from its log directory: raw per-rollout rewards, the
smoothed per-step mean, and held-out criterion pass rate at eval steps.

  uv run harvey-plot log_dir=<run-dir> out=run.png
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chz

from .results import read_results

RAW_COLOR = "#86b6ef"
SMOOTH_COLOR = "#2a78d6"
EVAL_COLOR = "#008300"
INK = "#3d3d3a"
MUTED = "#7f7e78"


@chz.chz
class PlotConfig:
  log_dir: Path
  out: Path | None = None
  title: str | None = chz.field(default=None, doc="Plot title; defaults to the log directory name")


def ema(values: list[float], alpha: float = 0.4) -> list[float]:
  smoothed = []
  for value in values:
    smoothed.append(value if not smoothed else alpha * value + (1 - alpha) * smoothed[-1])
  return smoothed


def plot_results(log_dir: Path, results: dict, out: Path | None = None, title: str | None = None) -> Path:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import MaxNLocator

  raw = [(row["step"], row["reward"]) for row in results["rollout_rewards"]]
  train_rows = [(row["step"], row["reward"]) for row in results["train"]]

  fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
  rng = random.Random(0)
  if raw:
    xs = [step + rng.uniform(-0.18, 0.18) for step, _ in raw]
    ax.scatter(xs, [reward for _, reward in raw], s=14, color=RAW_COLOR, alpha=0.6, linewidths=0, label="rollout reward", zorder=2)
  if train_rows:
    steps = [step for step, _ in train_rows]
    rewards = [reward for _, reward in train_rows]
    ax.plot(steps, rewards, color=SMOOTH_COLOR, linewidth=0.8, alpha=0.5, marker=".", label="batch mean reward", zorder=2)
    ax.plot(steps, ema(rewards), color=SMOOTH_COLOR, linewidth=2, label="batch mean reward (EMA, alpha=0.4)", zorder=3)
  for aggregation, label, color, marker in (
    ("pooled_criteria", "eval: pooled criterion pass rate", EVAL_COLOR, "D"),
    ("mean_episode_fraction", "eval: mean episode pass fraction (legacy)", MUTED, "s"),
  ):
    evals = [row for row in results["evaluations"] if row["aggregation"] == aggregation]
    if not evals:
      continue
    ex = [row["step"] for row in evals]
    ey = [row["pass_rate"] for row in evals]
    ax.plot(ex, ey, color=color, linewidth=2, linestyle=(0, (2, 3)), marker=marker, label=label, zorder=4)
    for row in evals:
      phase = "final\n" if row["phase"] == "final" else "baseline\n" if row["step"] == 0 else ""
      ax.annotate(
        f"{phase}{row['pass_rate']:.0%}",
        (row["step"], row["pass_rate"]),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        color=color,
        fontsize=9,
      )

  values = [reward for _, reward in raw + train_rows] + [row["pass_rate"] for row in results["evaluations"]]
  ax.set_ylim(min(-0.15, min(values, default=0) - 0.1), max(1.05, max(values, default=1) + 0.1))
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
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


def write_report(log_dir: Path, results: dict) -> dict:
  """Save results.json and run_plot.png next to the run's metrics."""
  (log_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
  plot_results(log_dir, results)
  return results


def main() -> None:
  args = chz.entrypoint(PlotConfig, allow_hyphens=True)
  print(plot_results(args.log_dir, read_results(args.log_dir), args.out, args.title))


if __name__ == "__main__":
  main()
