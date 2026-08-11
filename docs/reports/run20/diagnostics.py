"""Render run20's iteration-0 diagnostics from the committed data. No box access.

    python docs/reports/run20/diagnostics.py  # -> docs/reports/run20/run20-iter0.png

run20 is Gemma-4-E4B-it on the same 8x6 / seed-242 shape as the 9B runs. At
iteration 0 it scores 0.076 criteria-pass against Qwen's 0.535, so the useful
question is not "how much reward" but "where do the episodes go". Three panels,
in descending order of how much they cost:

  1. Failure modes. no_output dominates -- the model plans, asks the user to
     upload files, and stops. Overflow is a rounding error next to it.
  2. Generation length. max_tokens was 32,768 against a measured max of 7,639.
  3. Observation length. Because message_env.py:94 reserves max_tokens from the
     trajectory budget every turn, that 32,768 fenced off the top quarter of
     Gemma's window: nothing ever ran past 98,304 of the nominal 131,072.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CONTEXT = 131072
GEN_WAS = 32768
GEN_NOW = 16384
BLUE, LIGHT, RED, GREY = "#1f6fb2", "#9cc4e4", "#d1495b", "0.55"

rows = [json.loads(l) for l in open(os.path.join(HERE, "iter0_episodes.jsonl"))]
train = [r for r in rows if r["split"] == "train"]
evalr = [r for r in rows if r["split"] == "eval"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9.5, 11.5))


def ecdf(vals):
    v = sorted(vals)
    return v, [(i + 1) / len(v) for i in range(len(v))]


# --- 1. failure modes -------------------------------------------------------
# has_output is the one bar where taller is better; it is the honest inverse of
# the whole panel, so it sits on top rather than mixed in with the failures.
LABELS = [
    ("lab/no_output", "no output written"),
    ("parse_error", "parse error"),
    ("context_overflow", "context overflow"),
    ("max_turns", "hit max_turns (40)"),
    ("has_output", "produced output"),
]
y = range(len(LABELS))
h = 0.38
for off, (grp, name, col) in enumerate(((train, "train (48 eps)", BLUE), (evalr, "eval (50 eps)", LIGHT))):
    fr = [100 * sum(r["flags"].get(k, 0) for r in grp) / len(grp) for k, _ in LABELS]
    bars = ax1.barh([i + (h / 2 if off == 0 else -h / 2) for i in y], fr, height=h,
                    color=col, label=name, zorder=3)
    for b, v in zip(bars, fr):
        ax1.text(v + 1, b.get_y() + b.get_height() / 2, "%.0f%%" % v,
                 va="center", fontsize=8.5, color="0.3")
ax1.set_yticks(list(y))
ax1.set_yticklabels([n for _, n in LABELS], fontsize=9.5)
ax1.invert_yaxis()
ax1.set_xlim(0, 78)
ax1.set_xlabel("% of episodes", fontsize=9.5, labelpad=6)
ax1.set_title("where iteration-0 episodes end up", fontsize=11, pad=10)
ax1.legend(fontsize=8.5, loc="lower right", framealpha=0.95, edgecolor="0.85")
ax1.axhline(3.5, color="0.85", lw=1)

# --- 2. generation length ---------------------------------------------------
ac = [a for r in rows for a in r["ac_lens"]]
v, p = ecdf(ac)
ax2.plot(v, [100 * x for x in p], color=BLUE, lw=2, zorder=4)
ax2.set_xscale("log")
ax2.set_xlim(100, 60000)
ax2.set_ylim(0, 104)
# the reserve we were paying for vs the longest turn anyone actually generated
ax2.axvspan(max(ac), GEN_WAS, color=RED, alpha=0.09, zorder=1)
# Three vertical lines inside one log decade collide if the labels all sit at
# the same height, so they are staggered and the rightmost reads leftward.
for x, lab, col, ls, ty, ha in (
    (max(ac), "longest turn\nobserved\n%s" % f"{max(ac):,}", "0.35", (0, (4, 3)), 60, "left"),
    (GEN_NOW, "max_tokens now\n%s" % f"{GEN_NOW:,}", BLUE, (0, (4, 3)), 36, "right"),
    (GEN_WAS, "max_tokens was\n%s" % f"{GEN_WAS:,}", RED, "-", 12, "left"),
):
    ax2.axvline(x, color=col, ls=ls, lw=1.4, zorder=2)
    ax2.text(x * (1.06 if ha == "left" else 0.94), ty, lab, fontsize=8.2,
             color=col, va="center", ha=ha)
# The band caption spans all three lines, so it needs an opaque backing or the
# 16,384 rule strikes straight through it.
ax2.text((max(ac) * GEN_WAS) ** 0.5, 92, "reserved, never used", ha="center",
         va="center", fontsize=8.5, color=RED, style="italic", zorder=5,
         bbox=dict(facecolor="white", edgecolor="none", pad=2.5))
ax2.set_xlabel("tokens generated in a turn (log scale)", fontsize=9.5, labelpad=6)
ax2.set_ylabel("% of turns below", fontsize=9.5, labelpad=6)
ax2.set_title("generation length, all %d turns  (p50 %s, p99 %s, max %s)"
              % (len(ac), f"{v[len(v) // 2]:,}", f"{v[int(len(v) * .99)]:,}", f"{max(ac):,}"),
              fontsize=11, pad=10)

# --- 3. observation length --------------------------------------------------
# The wall is not the context window. message_env.py:94 stops the episode when
# observation + max_tokens > max_trajectory_tokens, so the reachable ceiling is
# CONTEXT - GEN_TOKENS and the band above it is unreachable by construction.
ceil_was, ceil_now = CONTEXT - GEN_WAS, CONTEXT - GEN_NOW
ax3.hist([r["final_ob"] for r in rows], bins=26, range=(0, CONTEXT),
         color=LIGHT, edgecolor="white", lw=0.6, zorder=3)
# The two bands are adjacent, not stacked: blue is what lowering max_tokens
# hands back, red is what stays out of reach even after the fix. Overlapping
# them would blend into a third colour that means nothing.
ax3.axvspan(ceil_was, ceil_now, color=BLUE, alpha=0.11, zorder=1)
ax3.axvspan(ceil_now, CONTEXT, color=RED, alpha=0.09, zorder=1)
top = ax3.get_ylim()[1]
for x, lab, col, ty in ((ceil_was, "reachable ceiling\n%s" % f"{ceil_was:,}", RED, 0.96),
                        (ceil_now, "after the fix\n%s" % f"{ceil_now:,}", BLUE, 0.68),
                        (CONTEXT, "nominal window\n%s" % f"{CONTEXT:,}", "0.35", 0.96)):
    ax3.axvline(x, color=col, ls="-" if col != "0.35" else (0, (4, 3)), lw=1.4, zorder=4)
    ax3.text(x - 1400, top * ty, lab, fontsize=8.2, color=col, ha="right", va="top")
ax3.text((ceil_was + ceil_now) / 2, top * 0.30, "recovered\n+16.7%", ha="center",
         fontsize=8.2, color=BLUE, style="italic")
ax3.text((ceil_now + CONTEXT) / 2, top * 0.30, "still\nunreachable", ha="center",
         fontsize=8.2, color=RED, style="italic")
hi = max(r["final_ob"] for r in rows)
ax3.annotate("longest episode: %s" % f"{hi:,}", (hi, 1.2), xytext=(-16, 132),
             textcoords="offset points", ha="right", fontsize=8.5, color="0.3",
             arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))
ax3.set_xlim(0, CONTEXT * 1.005)
ax3.set_xlabel("final observation length (tokens)", fontsize=9.5, labelpad=6)
ax3.set_ylabel("episodes", fontsize=9.5, labelpad=6)
ax3.set_title("the top quarter of Gemma's window was never reachable", fontsize=11, pad=10)

for ax in (ax1, ax2, ax3):
    ax.grid(axis="x" if ax is ax1 else "y", color="0.88", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=4, color="0.7")
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("0.7")

fig.suptitle("run 20 — Gemma-4-E4B-it LoRA r32, GRPO 8x6 — iteration 0 diagnostics",
             fontsize=12.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = os.path.join(HERE, "run20-iter0.png")
fig.savefig(out, dpi=160)
print("wrote %s  episodes=%d turns=%d  max_ac=%s  max_ob=%s"
      % (out, len(rows), len(ac), f"{max(ac):,}", f"{hi:,}"))
