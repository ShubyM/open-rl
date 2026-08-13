"""Optimization health for run20 -- the closest thing this loop has to a loss.

    python docs/reports/run20/optim.py       # -> docs/reports/run20/run20-optim.png

There is deliberately no loss curve here because there is no loss to plot. The
objective is an importance-sampling policy-gradient surrogate: it is built so
that its *gradient* is the policy gradient, and its value is an arbitrary
number that moves with advantage scale and ratio clipping. A falling surrogate
would not mean the policy improved and a rising one would not mean it got
worse. Nothing named loss appears anywhere in metrics.jsonl for run19 or run20.

The three quantities that do diagnose the optimizer:

  grad_norm  -- pre-clip gradient magnitude. Watch for a step-change, not a
                level; LoRA gradient norms are large by construction and the
                absolute number is not comparable across runs.
  entropy    -- per-token policy entropy. Collapse toward zero is the standard
                way an RL run dies: the policy stops exploring, every rollout
                in a group scores the same, advantages go to zero, learning
                stops. run19 died exactly this way (frac_all_bad 0.88).
  KL(sample||train) -- divergence between the weights that generated the
                rollouts and the weights the gradient is computed against.
                This should sit near zero; it is an infrastructure check, not
                a learning signal. If it climbs, the sampler is serving stale
                adapters and the run is silently off-policy.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _series import load, resume_steps

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, GREEN, ORANGE, RED = "#1f6fb2", "#2e8b57", "#e07b39", "#d1495b"

PATH = os.path.join(HERE, "meta", "run20.metrics.jsonl")
rows = load(PATH)
resumes = resume_steps(PATH)
xs = [r["step"] for r in rows]

PANELS = [
    ("grad_norm:mean", "gradient norm (pre-clip)", BLUE, None),
    ("optim/entropy", "policy entropy (nats/token)", GREEN, (0, None)),
    ("optim/kl_sample_train_v1", "KL(sampler ‖ trainer)", ORANGE, (0, None)),
]

fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
for ax, (key, title, col, ylim) in zip(axes, PANELS):
    ys = [r.get(key) for r in rows]
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=col, lw=2.0, ms=4.5, zorder=3)
    # A trend line through a 15-point series read by eye is guesswork, so the
    # mean is drawn explicitly: the question for all three panels is "is this
    # drifting or just noisy", and that is only answerable against a baseline.
    mean = sum(p[1] for p in pts) / len(pts)
    ax.axhline(mean, color="0.55", ls=(0, (4, 3)), lw=1.0, zorder=2)
    ax.annotate("mean %.4g" % mean, (xs[-1], mean), xytext=(6, 4),
                textcoords="offset points", fontsize=8, color="0.4")
    ax.set_title(title, fontsize=10.5, pad=8, loc="left")
    if ylim:
        ax.set_ylim(ylim[0], ax.get_ylim()[1] * 1.12)
    for rs in resumes:
        ax.axvline(rs, color=RED, ls=(0, (3, 3)), lw=1.1, zorder=1)

if resumes:
    axes[0].annotate("step-14 OOM,\nresumed from %d" % resumes[-1],
                     (resumes[-1], axes[0].get_ylim()[1]), xytext=(5, -4),
                     textcoords="offset points", fontsize=8, color=RED, va="top")

# KL is ~5e-4 across the whole run; without a reference the reader cannot tell
# whether that is small. 0.01 is the scale at which sampler staleness would
# start to bias the gradient.
axes[2].axhline(0.01, color="0.75", lw=1.0)
axes[2].annotate("0.01 — scale at which staleness would matter", (0, 0.01),
                 xytext=(4, 4), textcoords="offset points", fontsize=8, color="0.5")
axes[2].set_ylim(0, 0.012)
axes[2].set_xlabel("training step", fontsize=9.5, labelpad=6)
axes[2].set_xticks(xs)

for ax in axes:
    ax.grid(color="0.9", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=4, color="0.7")
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("0.7")

fig.suptitle("run 20 — optimization health (there is no loss curve; see the docstring)",
             fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.982))
out = os.path.join(HERE, "run20-optim.png")
fig.savefig(out, dpi=160)
print("wrote %s  steps %d-%d" % (out, xs[0], xs[-1]))
for key, title, _, _ in PANELS:
    v = [r[key] for r in rows if key in r]
    print("  %-28s first %.4g  last %.4g  min %.4g  max %.4g" % (title, v[0], v[-1], min(v), max(v)))
