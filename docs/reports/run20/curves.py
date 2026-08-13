"""Reward and eval curves for run20 against the Qwen runs it has to beat.

    python docs/reports/run20/curves.py     # -> docs/reports/run20/run20-curves.png

criteria_pass_fraction is the metric on both panels because it is the honest
one: it averages over every task, scoring a failed episode 0 rather than
dropping it. rubric_reward looks far better and means far less, since it only
averages the episodes that survived to the judge.

Panel 1 puts run20 where it belongs -- next to run17 and run19/19r, the Qwen
9B runs on the identical 8x6 / seed-242 shape. Gemma starts roughly an order
of magnitude below where Qwen starts, so the interesting question is slope,
not level. Panel 2 is that slope: three steps in, the thing that was killing
the run is already coming apart.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _series import load, resume_steps

HERE = os.path.dirname(os.path.abspath(__file__))
R19 = os.path.join(os.path.dirname(HERE), "run19", "meta")
BLUE, LIGHT, RED, ORANGE = "#1f6fb2", "#9cc4e4", "#d1495b", "#e07b39"

TRAIN = "env/all/lab/criteria_pass_fraction"
EVAL = "test/env/all/lab/criteria_pass_fraction"

                                             # alpha on the thin train series:
                                             # run20 only has three points and
                                             # is the subject, so it is drawn
                                             # solid while the finished Qwen
                                             # runs stay backgrounded.
RUNS = [
    ("run 17 — Qwen3.5-9B", os.path.join(HERE, "meta", "run17.metrics.jsonl"), LIGHT, 0.45),
    ("run 19 — Qwen, collapsed", os.path.join(R19, "run19.metrics.jsonl"), RED, 0.35),
    ("run 19r — resumed from step 14", os.path.join(R19, "run19r.metrics.jsonl"), BLUE, 0.4),
    ("run 20 — Gemma-4-E4B (running)", os.path.join(HERE, "meta", "run20.metrics.jsonl"), ORANGE, 1.0),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 9), height_ratios=(1.25, 1))

# --- 1. cross-run ------------------------------------------------------------
for name, path, col, alpha in RUNS:
    rows = load(path)
    xs = [r["step"] for r in rows]
    ax1.plot(xs, [r[TRAIN] for r in rows], color=col, lw=1.1 if alpha < 1 else 2.0,
             alpha=alpha, zorder=2)
    ev = [(r["step"], r[EVAL]) for r in rows if EVAL in r]
    ax1.plot([s for s, _ in ev], [v for _, v in ev], color=col, lw=2.2,
             marker="o", ms=5.5, label=name, zorder=3)

# run19's last checkpoint before the collapse is where run19r picks up, so the
# two curves are one experiment forking at step 15, not two separate runs.
ax1.annotate("run19 collapses\n(frac_all_bad 0.88)", (18.6, 0.0), xytext=(26, 30),
             textcoords="offset points", ha="left", fontsize=8.2, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.9))
_p20 = os.path.join(HERE, "meta", "run20.metrics.jsonl")
_r20, _res = load(_p20), resume_steps(_p20)
_last = _r20[-1]
if _res:
    # Steps after the line were retrained from an older checkpoint rather than
    # continued, which no amount of curve-reading would reveal.
    ax1.axvline(_res[-1], color=ORANGE, ls=(0, (3, 3)), lw=1.1, zorder=1)
    ax1.annotate("FlexAttention OOM at step 14\nresumed from checkpoint %d" % _res[-1],
                 (_res[-1], _last[TRAIN]), xytext=(24, 34), textcoords="offset points",
                 ha="left", fontsize=8.2, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
else:
    ax1.annotate("run 20 is here\nstep %d of 40" % _last["step"],
                 (_last["step"], _last[TRAIN]), xytext=(26, 30),
                 textcoords="offset points", ha="left", fontsize=8.2, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.9))
ax1.set_xlim(-0.6, 40)
ax1.set_ylim(-0.03, 0.82)
ax1.set_xlabel("training step", fontsize=9.5, labelpad=6)
ax1.set_ylabel("criteria pass fraction", fontsize=9.5, labelpad=6)
ax1.set_title("run 20 against the Qwen runs, same 8x6 / seed-242 shape", fontsize=11, pad=10)
leg = ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.95, edgecolor="0.85")
ax1.add_artist(leg)
ax1.legend(handles=[Line2D([], [], color="0.4", lw=2.2, marker="o", ms=5.5, label="eval (held-out)"),
                    Line2D([], [], color="0.4", lw=1.1, alpha=0.45, label="train")],
           fontsize=8.2, loc="upper left", framealpha=0.95, edgecolor="0.85")

# --- 2. run20 close-up -------------------------------------------------------
# has_output is the gate: reward.py:66 short-circuits to 0.0 and never calls the
# judge when it is false, so every other series here is capped by this one.
rows = load(os.path.join(HERE, "meta", "run20.metrics.jsonl"))
xs = [r["step"] for r in rows]
SERIES = [
    ("env/all/lab/has_output", "wrote an output file", ORANGE, "-", 2.4),
    ("env/all/lab/graded", "reached the judge", BLUE, "-", 1.8),
    ("env/all/lab/criteria_pass_fraction", "criteria pass fraction", "#2e8b57", "-", 1.8),
    ("env/all/reward/total", "mean reward", "0.45", (0, (4, 3)), 1.6),
]
for key, name, col, ls, lw in SERIES:
    ys = [r.get(key, 0.0) for r in rows]
    ax2.plot(xs, ys, color=col, lw=lw, ls=ls, marker="o", ms=5, label=name, zorder=3)
    ax2.annotate("%.2f" % ys[-1], (xs[-1], ys[-1]), xytext=(7, -2),
                 textcoords="offset points", fontsize=8.2, color=col, va="center")
ax2.set_xticks(xs)
ax2.set_xlim(xs[0] - 0.3, xs[-1] + 0.9)
ax2.set_ylim(0, 1.0)
ax2.set_xlabel("training step", fontsize=9.5, labelpad=6)
ax2.set_title("run 20, steps %d–%d — the no-output failure unwinding" % (xs[0], xs[-1]),
              fontsize=11, pad=10)
ax2.legend(fontsize=8.5, loc="upper left", framealpha=0.95, edgecolor="0.85")

for ax in (ax1, ax2):
    ax.grid(color="0.9", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=4, color="0.7")
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("0.7")

fig.suptitle("harvey-labs — reward and eval curves", fontsize=12.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = os.path.join(HERE, "run20-curves.png")
fig.savefig(out, dpi=160)
print("wrote %s" % out)
for name, path, _, _ in RUNS:
    r = load(path)
    ev = [x[EVAL] for x in r if EVAL in x]
    print("  %-34s steps %2d-%-2d  train last %.3f  eval %s"
          % (name, r[0]["step"], r[-1]["step"], r[-1][TRAIN],
             " ".join("%.3f" % v for v in ev)))
