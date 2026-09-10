#!/usr/bin/env python3
"""run49 growth view: held-out criterion pass rate and train reward per step.

The shared plot_run.py shows every per-rollout reward across the full [-0.1, 1]
range, which flattens a 10-point move into a thin mid-plot ribbon. Here the
held-out criterion pass rate is the hero (it is the benchmark), the per-step
mean reward is kept as faint texture, and the axis is cropped to where the two
series actually live.

  python plot.py <log_dir> [--out run49.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TRAIN_COLOR = "#2a78d6"
TRAIN_DOT = "#9ec3ee"
EVAL_COLOR = "#008300"
INK = "#3d3d3a"
MUTED = "#7f7e78"

ENV_NAMES = ("all", "harvey-labs")


def load_metrics(path: Path) -> list[dict]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def env_get(row: dict, prefix: str, suffix: str):
  for name in ENV_NAMES:
    value = row.get(f"{prefix}env/{name}/{suffix}")
    if value is not None:
      return value
  return None


def eval_pass_rate(row: dict) -> float | None:
  return env_get(row, "test/", "lab/criteria_pass_fraction")


def ema(values: list[float], alpha: float = 0.4) -> list[float]:
  smoothed: list[float] = []
  for value in values:
    smoothed.append(value if not smoothed else alpha * value + (1 - alpha) * smoothed[-1])
  return smoothed


def linfit(xs: list[float], ys: list[float]) -> tuple[float, float]:
  n = len(xs)
  mx = sum(xs) / n
  my = sum(ys) / n
  denom = sum((x - mx) ** 2 for x in xs) or 1.0
  slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
  return slope, my - slope * mx


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("log_dir", type=Path)
  parser.add_argument("--out", type=Path, default=None)
  args = parser.parse_args()

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  rows = load_metrics(args.log_dir / "metrics.jsonl")
  train = [(r["step"], env_get(r, "", "reward/total")) for r in rows if "step" in r and env_get(r, "", "reward/total") is not None]
  evals = [(r["step"], eval_pass_rate(r)) for r in rows if "step" in r and eval_pass_rate(r) is not None]

  tsteps = [s for s, _ in train]
  tvals = [v for _, v in train]
  tsmooth = ema(tvals)
  esteps = [s for s, _ in evals]
  evals_y = [v for _, v in evals]

  fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)

  # Per-step mean reward as faint texture; the EMA is the read.
  ax.scatter(tsteps, tvals, s=22, color=TRAIN_DOT, alpha=0.7, linewidths=0, zorder=2)
  ax.plot(tsteps, tsmooth, color=TRAIN_COLOR, linewidth=2.4, label="mean train reward (EMA)", zorder=3)

  # Held-out benchmark: linear trend behind, bold markers in front.
  slope, intercept = linfit([float(s) for s in esteps], evals_y)
  fit_x = [esteps[0], esteps[-1]]
  ax.plot(fit_x, [slope * x + intercept for x in fit_x], color=EVAL_COLOR, linewidth=1.4, linestyle=(0, (1, 2)), alpha=0.6, zorder=3)
  ax.plot(esteps, evals_y, color=EVAL_COLOR, linewidth=2.6, zorder=4)
  ax.scatter(esteps, evals_y, s=95, color=EVAL_COLOR, marker="D", label="held-out criterion pass rate", zorder=5)
  for x, y in evals:
    ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points", xytext=(0, 11), ha="center", color=EVAL_COLOR, fontsize=9, fontweight="bold")

  # Crop to the band the two series occupy so the growth fills the frame.
  lo = min(min(tsmooth), min(evals_y), min(tvals))
  hi = max(max(tsmooth), max(evals_y), max(tvals))
  pad = (hi - lo) * 0.18
  ax.set_ylim(lo - pad, hi + pad)
  ax.set_xlim(-0.7, tsteps[-1] + 1.6)

  # Start -> end deltas for each series.
  d_eval = evals_y[-1] - evals_y[0]
  d_train = tsmooth[-1] - tsmooth[0]
  ax.annotate(f"{evals_y[-1]:.0%}", (esteps[-1], evals_y[-1]), textcoords="offset points", xytext=(10, -2), color=EVAL_COLOR, fontsize=10, fontweight="bold")
  ax.annotate(f"{tsmooth[-1]:.2f}", (tsteps[-1], tsmooth[-1]), textcoords="offset points", xytext=(10, -3), color=TRAIN_COLOR, fontsize=10)
  ax.text(
    0.015, 0.965,
    f"held-out  {evals_y[0]:.0%} → {evals_y[-1]:.0%}  ({d_eval:+.0%} over {esteps[-1]} steps)\n"
    f"train EMA  {tsmooth[0]:.2f} → {tsmooth[-1]:.2f}  ({d_train:+.2f})",
    transform=ax.transAxes, va="top", ha="left", fontsize=9.5, color=INK,
    bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f0", ec=MUTED, alpha=0.9),
  )

  ax.set_xlabel("step", color=INK)
  ax.set_ylabel("reward / pass rate", color=INK)
  ax.set_title("run49 (Qwen3.8-27B, Automodel CP4, 262k ctx) — held-out growth", color=INK, fontsize=12, loc="left")
  ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.5)
  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
  for spine in ("left", "bottom"):
    ax.spines[spine].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)
  ax.legend(loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK)

  out = args.out or args.log_dir / "run49.png"
  fig.tight_layout()
  fig.savefig(out)
  print(out)


if __name__ == "__main__":
  main()
