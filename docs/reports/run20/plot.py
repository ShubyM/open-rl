"""Render run20's reward curve from the committed data. No box access needed.

    python docs/reports/run20/plot.py         # -> docs/reports/run20/run20.png

Same shape as docs/reports/run19/plot.py so the two runs can be read side by
side: rollout reward per step, an EMA through it, and the held-out eval as
diamonds. criteria_pass_fraction rather than rubric_reward for the same reason
as run19 -- it averages over every task and scores a failed episode 0, where
rubric_reward only averages the episodes that survived to the judge and so
hides exactly the failure mode this run has.

run19 read its evals from a per-episode eval_episodes.jsonl built by build.py.
run20 has no such build yet, so the eval points come straight out of
metrics.jsonl's test/ keys -- the same mean over the same 50 held-out tasks.

The run is live. Refresh with:
    scp h200:'~/open-rl/artifacts/harvey-labs/run20-gemma4-e4b/metrics.jsonl' \\
        docs/reports/run20/meta/run20.metrics.jsonl
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METRIC = "env/all/lab/criteria_pass_fraction"
EVAL_METRIC = "test/env/all/lab/criteria_pass_fraction"
ALPHA = 0.4

steps, evals = {}, {}
for line in open(os.path.join(HERE, "meta", "run20.metrics.jsonl")):
    r = json.loads(line)
    # The step-None rows are end-of-run evals with no training step attached.
    if r.get("step") is None:
        continue
    if METRIC in r:
        steps[r["step"]] = r[METRIC]
    if EVAL_METRIC in r:
        evals[r["step"]] = r[EVAL_METRIC]

x = sorted(steps)
y = [steps[i] for i in x]
ex = sorted(evals)
ey = [evals[i] for i in ex]

sm = [y[0]]
for v in y[1:]:
    sm.append(ALPHA * v + (1 - ALPHA) * sm[-1])

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, "-o", color="#9cc4e4", lw=1, ms=3.5, alpha=0.75, label="rollout reward")
ax.plot(x, sm, "-", color="#1f6fb2", lw=2.4, label="smoothed (EMA %s)" % ALPHA, zorder=4)
ax.plot(ex, ey, "D", color="#d1495b", ms=8, label="held-out eval (50 tasks)",
        mec="white", mew=1.1, zorder=6)
HALO = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78)
for i, v in zip(ex, ey):
    near = [b for a, b in zip(x, y) if a == i]
    below = bool(near) and 0 < near[0] - v < 0.03
    ax.annotate("%.3f" % v, (i, v), textcoords="offset points", zorder=7,
                xytext=(0, -20 if below else 11), ha="center", color="#d1495b",
                fontsize=9, bbox=HALO)

# The x-axis is pinned to the configured length rather than the last step we
# happen to have. A live run replotted every hour would otherwise rescale on
# every refresh and read as if it were nearly finished the whole way through.
TOTAL_STEPS = 40
lo = min(min(y), min(ey)) if ey else min(y)
hi = max(max(y), max(ey)) if ey else max(y)
ax.set_ylim(lo - 0.03, hi + 0.07)
ax.set_xlim(-0.8, TOTAL_STEPS + 0.8)
ax.axvspan(max(x) + 0.5, TOTAL_STEPS + 0.8, color="0.94", zorder=0)
ax.text((max(x) + TOTAL_STEPS) / 2, (lo + hi) / 2, "steps %d–%d not yet run"
        % (max(x) + 1, TOTAL_STEPS - 1), color="0.6", fontsize=9,
        ha="center", va="center", style="italic")

ax.set_xlabel("step", fontsize=10, labelpad=8)
ax.set_ylabel("criteria pass fraction", fontsize=10, labelpad=8)
ax.set_title("run 20 — Gemma-4-E4B-it LoRA r32, GRPO 8x6", fontsize=12, pad=12)
ax.set_xticks(range(0, TOTAL_STEPS + 1, 5))
ax.set_xticks(range(0, TOTAL_STEPS + 1), minor=True)
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
out = os.path.join(HERE, "run20.png")
fig.savefig(out, dpi=160)
print("wrote %s  steps=%d..%d of %d  evals=%s" % (
    out, min(x), max(x), TOTAL_STEPS, [(i, round(v, 4)) for i, v in zip(ex, ey)]))
