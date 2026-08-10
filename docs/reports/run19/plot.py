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

# Resuming past step 20 re-ran that eval over the same `final` weights and
# overwrote the file. Both readings are kept: same weights, same 50 tasks,
# 0.658 vs 0.609 -- the spread is the eval's own noise floor, and it is wider
# than most of the step-to-step movement in this series.
SUPERSEDED = {20: [0.6584]}

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
    # run_final_eval drops an extra summary at the last completed batch; the
    # real held-out points are the eval_every ones
    if not keep(e["iter"]) or e["iter"] % EVAL_EVERY:
        continue
    by_iter.setdefault(e["iter"], []).append(e["metrics"].get("lab/criteria_pass_fraction", 0.0))
for it, cpf in by_iter.items():
    evals[it] = sum(cpf) / len(cpf)

x = sorted(steps)
y = [steps[i] for i in x]
ex, ey = sorted(evals), [evals[i] for i in sorted(evals)]
ex2 = [i for i, vs in SUPERSEDED.items() for _ in vs if i in evals]
ey2 = [v for i, vs in SUPERSEDED.items() if i in evals for v in vs]

sm = [y[0]]
for v in y[1:]:
    sm.append(ALPHA * v + (1 - ALPHA) * sm[-1])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, "-o", color="#6aa9e0", lw=1, ms=4, alpha=0.5, label="rollout reward")
ax.plot(x, sm, "-", color="#1f6fb2", lw=2.2, label="smoothed (EMA %s)" % ALPHA)
ax.plot(ex, ey, "D", color="#d1495b", ms=9, label="held-out eval (50 tasks)", zorder=5)
ax.plot(ex2, ey2, "D", color="#d1495b", ms=9, mfc="none", zorder=5,
        label="same weights, resampled")
for i, v in zip(ex + ex2, ey + ey2):
    # duck under the rollout line where it sits just above the diamond
    near = [b for a, b in zip(x, y) if a == i]
    below = bool(near) and 0 < near[0] - v < 0.03
    ax.annotate("%.3f" % v, (i, v), textcoords="offset points",
                xytext=(0, -19 if below else 10), ha="center", color="#d1495b", fontsize=9)

# headroom so the legend clears the peaks
ax.set_ylim(min(min(y), min(ey + ey2)) - 0.03, max(max(y), max(ey + ey2)) + 0.07)
if resumed_from:
    ax.axvline(resumed_from - 0.5, color="0.6", ls="--", lw=1)
    ax.text(resumed_from - 0.4, ax.get_ylim()[0] + 0.006,
            "judge died here; resumed\nfrom ckpt 000015 (run19r)",
            ha="left", va="bottom", color="0.4", fontsize=8)

ax.set_xlabel("step")
ax.set_ylabel("criteria pass fraction")
ax.set_title("run19 — Qwen3.5-9B LoRA r32, GRPO 8x6")
ax.set_xticks(range(0, max(max(x), max(ex)) + 2))
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
fig.tight_layout()
out = os.path.join(HERE, "run19.png")
fig.savefig(out, dpi=160)
print("wrote %s  steps=%d..%d  evals=%s" % (
    out, min(x), max(x), [(i, round(v, 4)) for i, v in zip(ex, ey)]))
