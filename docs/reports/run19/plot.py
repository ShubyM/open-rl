"""Render run19's reward curve from the committed data. No box access needed.

    python docs/reports/run19/plot.py         # -> docs/reports/run19/run19.png

Two metrics live in the logs. criteria_pass_fraction averages over every task,
scoring a failed episode 0; rubric_reward averages only over the episodes the
judge actually graded, so it hides the completion failures. Default to the
former -- it is the honest denominator.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METRIC = "env/all/lab/criteria_pass_fraction"
EVAL_EVERY = 5
ALPHA = 0.4

# run19's judge died partway through step 15 and its 15-19 are all-zero
# artefacts; run19r resumed from checkpoint 000015 and re-ran them.
RUNS = [("run19", lambda s: s < 15), ("run19r", lambda s: True)]

# Each resume re-ran the eval at its restart step over the same weights and
# overwrote the file. The overwritten readings are kept here: same weights,
# same 50 tasks, so each pair is a direct read of the eval's own noise.
#   step 20: 0.658 vs 0.609  (spread 0.049)
#   step 35: 0.710 vs 0.718  (spread 0.009, from backup-iteration_000035-finaleval)
# Two same-weights replicates disagreeing 6x is why the report says the floor
# is unknown-but-bounded rather than quoting a number.
SUPERSEDED = {20: [0.6584], 35: [0.7098]}

# Evals fired at `i_batch == end_batch - 1` rather than on the eval_every grid.
# This run stopped and resumed three times (max_steps 20, 35, 40), so each
# stopping point left one. 19 is real but omitted to match the report's table;
# 34 and 39 are the tail points worth showing. Pinning them explicitly matters
# while the run is live -- keying off "the last step we have" silently drops
# yesterday's tail point as soon as training moves past it.
KEEP_OFF_GRID = {34, 39}

# train.py's run_final_eval scores the saved `final` checkpoint after the loop
# has already exited, so it lands a step past the last in-loop eval and outside
# the run's own cadence. Dropped: at 0.645 against step 39's 0.725 it reads as a
# collapse, but 4 of its 50 episodes never completed (2 overflow, 1 no-output)
# while step 39 was the only eval all run to grade 50/50. On the episodes both
# graded they are 0.701 vs 0.725. That gap is the completion tail resampling
# itself, not one update at lr 2e-4 undoing the run. Step 39 is the last
# in-loop eval and the number the report should quote.
DROP_EVALS = {40}

steps, evals, resumed_from = {}, {}, None
for run, keep in RUNS:
    path = os.path.join(HERE, "meta", "%s.metrics.jsonl" % run)
    if not os.path.exists(path):
        continue
    for line in open(path):
        r = json.loads(line)
        if METRIC not in r or not keep(r["step"]):
            continue
        steps[r["step"]] = r[METRIC]
        if run != RUNS[0][0] and resumed_from is None:
            resumed_from = r["step"]
# one line per held-out episode, from build.py
by_iter = {}
for line in open(os.path.join(HERE, "eval_episodes.jsonl")):
    e = json.loads(line)
    keep = dict(RUNS)[e["run"]]
    if not keep(e["iter"]):
        continue
    by_iter.setdefault(e["iter"], []).append(e["metrics"].get("lab/criteria_pass_fraction", 0.0))
# Keep the eval_every grid plus the pinned tail evals. Everything else off-grid
# is an artefact of a crashed run.
for it, cpf in by_iter.items():
    if it in DROP_EVALS or (it % EVAL_EVERY and it not in KEEP_OFF_GRID):
        continue
    evals[it] = sum(cpf) / len(cpf)

x = sorted(steps)
y = [steps[i] for i in x]
ex, ey = sorted(evals), [evals[i] for i in sorted(evals)]
ex2 = [i for i, vs in SUPERSEDED.items() for _ in vs if i in evals]
ey2 = [v for i, vs in SUPERSEDED.items() if i in evals for v in vs]

sm = [y[0]]
for v in y[1:]:
    sm.append(ALPHA * v + (1 - ALPHA) * sm[-1])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, "-o", color="#9cc4e4", lw=1, ms=3.5, alpha=0.75, label="rollout reward")
ax.plot(x, sm, "-", color="#1f6fb2", lw=2.4, label="smoothed (EMA %s)" % ALPHA, zorder=4)
ax.plot(ex, ey, "D", color="#d1495b", ms=8, label="held-out eval (50 tasks)",
        mec="white", mew=1.1, zorder=6)
ax.plot(ex2, ey2, "D", color="#d1495b", ms=8, mfc="none", mew=1.4, zorder=6,
        label="same weights, resampled")
# the labels have to sit somewhere, and the reward curve runs through most of
# the plot -- a halo keeps them legible where they land on a line
HALO = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78)

for i, v in zip(ex, ey):
    # duck under the rollout line where it sits just above the diamond, and
    # under an eval that sits one step to the right (the 34/35 tail pair)
    near = [b for a, b in zip(x, y) if a == i]
    below = (bool(near) and 0 < near[0] - v < 0.03) or (i + 1) in evals
    ax.annotate("%.3f" % v, (i, v), textcoords="offset points", zorder=7,
                xytext=(0, -20 if below else 11), ha="center", color="#d1495b",
                fontsize=9, bbox=HALO)
for i, v in zip(ex2, ey2):
    # A resampled reading shares its step with the primary, and at step 35 the
    # two are 0.009 apart -- stacking the labels vertically puts them on top of
    # each other or into step 34's. Set them beside the diamond instead.
    ax.annotate("%.3f" % v, (i, v), textcoords="offset points", zorder=7,
                xytext=(12, -3), ha="left", color="#d1495b", fontsize=9, bbox=HALO)

last_x = max(max(x), max(ex))
# headroom so the legend clears the peaks; a little slack on the right so the
# resampled label at the final step is not clipped by the spine
ax.set_ylim(min(min(y), min(ey + ey2)) - 0.03, max(max(y), max(ey + ey2)) + 0.07)
ax.set_xlim(-0.8, last_x + 1.5)
if resumed_from:
    ax.axvline(resumed_from - 0.5, color="0.75", ls=(0, (4, 3)), lw=1, zorder=1)
    ax.text(resumed_from - 0.3, ax.get_ylim()[0] + 0.008,
            "judge died here; resumed\nfrom ckpt 000015 (run19r)",
            ha="left", va="bottom", color="0.45", fontsize=8, linespacing=1.4)

ax.set_xlabel("step", fontsize=10, labelpad=8)
ax.set_ylabel("criteria pass fraction", fontsize=10, labelpad=8)
ax.set_title("run 15 — Qwen3.5-9B LoRA r32, GRPO 8x6", fontsize=12, pad=12)
# label every 5th step and tick the rest; one label per step was a picket fence
ax.set_xticks(range(0, last_x + 1, 5))
ax.set_xticks(range(0, last_x + 1), minor=True)
ax.grid(axis="y", color="0.85", lw=0.7, zorder=0)
ax.grid(axis="x", color="0.92", lw=0.7, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(labelsize=9, length=4, color="0.7")
ax.tick_params(which="minor", length=2.5, color="0.8")
for edge in ("top", "right"):
    ax.spines[edge].set_visible(False)
for edge in ("left", "bottom"):
    ax.spines[edge].set_color("0.7")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95, edgecolor="0.85", borderpad=0.7)
fig.tight_layout()
out = os.path.join(HERE, "run19.png")
fig.savefig(out, dpi=160)
print("wrote %s  steps=%d..%d  evals=%s" % (
    out, min(x), max(x), [(i, round(v, 4)) for i, v in zip(ex, ey)]))
