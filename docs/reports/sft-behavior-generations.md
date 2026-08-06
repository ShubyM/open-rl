# SFT Behavior Report: Three Generations of Teacher Distillation

**Scope:** How the Qwen3.5-9B student's behavior changed across three generations of
GLM-5.2 teacher distillation on Harvey LAB, what each generation's traces actually
taught, and the thinking-format defect that shaped the middle generation. All numbers
are recomputed from raw rollout summaries and trace files, not from run-time logs
(criteria scores use the /3,151 convention on the seed-0 random eval-50 unless noted).

**Setup common to all generations:** teacher episodes are collected under the
student's exact scaffold (same system prompt, tools, sandbox, output contract, tool
results truncated with the *student's* tokenizer), graded by the LAB rubric judge.
SFT re-renders the message-level traces through the student's own chat template
(rank-32 LoRA, lr 1e-4, 2 epochs, loss on assistant tokens only). Warm-started RL
runs then load the snapshot weights-only. Reference points: base model evals ~50%;
the RL-only record (run 11) reached 67.0% at step 15 and 73.9% at step 20.

---

## Generation 1 — no thinking traces (v1)

**Data.** 300 episodes, one per train task. The vLLM endpoint served GLM's reasoning
in the `reasoning_content` API field, which the collector never recorded. The traces
therefore contained only the teacher's *actions*: assistant turns of p50 334
characters — a one-line remark plus a tool call — with a single stray `</think>` on
each final turn (p50 0 characters of reasoning inside it). 149 episodes survived the
keep filter (reward ≥ 0.7, clean ending); pooled criteria of the kept set: 92.3%.
Teacher quality was never the problem: on graded episodes GLM scored 76.7% pooled
(86.2% excluding grading gaps), above the student's best-ever checkpoint.

**Training.** Two attempts:

- *B200:* a silent no-op. Per-datum losses were identical across epochs to within
  kernel nondeterminism (one datum bit-identical to 16 digits); the optimizer never
  updated weights. The plunging "loss curve" was a data-ordering artifact — epoch 2
  replayed epoch 1's shape exactly. Root cause hidden by silent-skip behavior:
  both the manual grad-clip and `AdamW.step()` skip params whose grads are `None`.
- *H200 (clean stack):* real learning. Epoch 2 beat epoch 1 by −0.044 mean loss,
  29/38 batches improved, adapter weights verifiably nonzero.

**Behavior after v1 SFT (run 12, warm-started RL).**

- Standalone eval of the snapshot: ~41% pooled — *below* base.
- Run 12 b0 eval: mean criteria-fraction 0.527, drifting to 0.466–0.504 by b5;
  train reward flat through 7 steps while run 11 had already climbed 45.9 → 60.6.
- **Correction recorded for honesty:** the initial diagnosis — "SFT suppressed
  thinking, turns collapsed to ~1.1k tokens" — conflated a *mean* tokens-per-turn
  (~1.1k, which the base model also exhibits) with run 11's *biggest-turn median*
  (7.5k). Re-analysis of run 12's own b0 rollouts showed normal-shaped episodes:
  biggest thinking turn p50 7,576 (base: 7,397), healthy rewards (mean 0.521).
  The defensible claims about v1 are narrower: the traces contained no reasoning,
  the warm start conferred no advantage, and the standalone 41% was never
  reconciled with run 12's healthier b0.

**Lesson.** Imitating actions without deliberation gave, at best, nothing. The
strongest v1 evidence is the absence of transfer, not the presence of damage.

---

## Generation 2 — thinking traces, no empty blocks (v2)

**Data.** Collection was rebuilt to capture reasoning: `reasoning_content` recorded
per assistant turn (kept off the wire and out of the trajectory budget), with a
fail-fast when the endpoint lacks a reasoning parser and normalization across the
three places servers put reasoning (`reasoning_content`, `reasoning`, inline
`<think>`). Final dataset: 385 episodes over all 100 family-disjoint SFT-pool tasks
(zero task or scenario-family overlap with RL train or eval pools), 266 keepers,
reasoning on **52.8% of assistant turns** — interior tool-calling turns, not just
finals (GLM's hybrid thinking genuinely skips trivial turns). Reasoning length p50
~500 chars with a heavy tail to ~49k.

**Behavior at step 0 (runs 13 and 14, same snapshot lineage).**

| metric | base (r11 b0) | v1 SFT (r12 b0) | v2 SFT (r13 b0) | teacher |
|---|---|---|---|---|
| turns/episode p50 | 10 | 10 | 12 | 15 |
| multi-tool-call turns | 16% | 18% | 20% | 27% |
| biggest thinking turn p50 | 7,397 | 7,576 | 7,040 | — |
| tool mix (bash/read) | 50/40 | 49/43 | 48/42 | 46/43 |
| pooled criteria | ~50% | — | 48.4% | — |

Thinking was preserved, and the student moved measurably *toward the teacher's
working style* — more turns, more parallel tool calls, slight `glob`/`edit`
adoption — while scores stayed at base level. A rank-32 LoRA over ~130–150 datums
is a prior nudge, not a personality transplant.

**Behavior over RL (run 14, batch 8×4, 32k tokens, same eval set as run 11).**

Live-graded trajectory: 50.6 → 49.6 → **55.9 (b10 peak)** → 54.4 → 54.4 → ~44–50
(final, degraded). Two behavioral stories inside that curve:

1. *RL washed out the teacher style it didn't reward.* The v2 fingerprint —
   parallel tool calls — decayed monotonically under RL (19% → 18% → 13% → 10% by
   b15) while run 11's RL-only policy independently discovered and amplified the
   same behavior (16% → 31% → 37% at its record checkpoint). The imitation prior
   was not sticky.
2. *The score driver was write-discipline, and SFT failed to pre-install it.*
   Run 14's climb tracks its wrote-a-deliverable count almost exactly
   (34/50 → 35 → 45 → 46), the same lesson run 11 learned from scratch at the
   same pace (38/50 → 50/50 at b20). Notably the SFT'd model *started* at
   34/50 — marginally below base — despite every kept teacher trace containing a
   written deliverable.

**The thinking-format defect.** Late in run 14 the policy degraded, and the
transcripts show why. The ungraded fraction of each eval tracked `has_output`
one-to-one (graded 0.76 vs has_output 0.83 at the final eval; genuine judge errors
were 2–4%) — these were not grading outages but episodes that produced *nothing to
grade*: turns announcing an action ("Now I'll generate the Word document:") followed
by no tool call, or empty final messages. Alongside them, stray `</think>` tokens
leaked into visible content on **8/50 final-eval episodes** (run 11 control at the
same step: 3), with mid-work stops rising from ~1–3 per eval at the healthy peak to
7 at the end:

| eval | think-tag leaks | mid-work stops | no-output episodes |
|---|---|---|---|
| b0 | 1 | 5 | 13 |
| b10 (peak) | 4 | 1 | 3 |
| b15 | 3 | 3 | 2 |
| b20 (final) | **8** | **7** | **10** |

Root cause: a convention mismatch baked into the v2 training targets. GLM marks a
non-thinking turn by *omitting* reasoning; Qwen marks it with an **empty**
`<think>\n\n</think>` block — every legal Qwen assistant turn carries think
structure (that is literally how `enable_thinking=False` is implemented). Our
converter mapped "no reasoning" to "no think tokens at all," so ~47% of trained
turns taught a turn shape the student never produces at inference and never sees in
history. The model learned that think structure is optional and irregular; twenty
steps of RL pressure amplified the tag errors into episode-ending failures, and the
run's ceiling (~56%) was set by format health, not by task ability.

**Lesson.** Same-scaffold imitation is necessary but not sufficient: the targets
must also be *format-legal under the student's own template*, including in the
cases where the teacher's convention differs. And RL is an amplifier — it magnified
a defect that was nearly invisible at step 0 (1 leaked tag) into the dominant
failure mode by step 20.

---

## Generation 3 — thinking traces with empty blocks (v3, in progress)

**The fix (`d05b15b`).** Every assistant turn now renders with think structure:
the teacher's reasoning when it deliberated, an empty block otherwise. The rendered
empty block is byte-identical to Qwen's own non-thinking prefill
(`<think>\n\n</think>\n\n`), verified through the real qwen3_5 renderer with both
turn types confirmed inside the loss mask. No re-collection required — the v2
traces were always correct (absent reasoning is GLM's honest signal); only the
translation was wrong. A side benefit: the student now learns *when* an empty
block is appropriate (the teacher skipped thinking on trivial turns), which is a
skill, not merely a syntax repair.

**Acceptance criteria for the v3 run**, independent of where the score lands:

1. `</think>` leakage at every eval ≤ the run-11 baseline (~1–3 episodes/50).
2. `has_output` ≥ 0.95 at late checkpoints (run 14 decayed to 0.83).
3. Mid-work stops not rising across checkpoints.

If those hold and the score still tracks the RL-only pace, the SFT-warm-start
question is answered cleanly: the format defect explained run 14's decay, and
~150-datum imitation simply does not accelerate this RL recipe.

---

## Cross-generation scoreboard (verified, /3,151 where pooled)

| checkpoint | run 11 (RL-only) | run 12 (v1 warm) | run 13 (v2 warm) | run 14 (v2 warm, full shape) |
|---|---|---|---|---|
| b0 | 45.9 (true ~50) | 0.527 mean-frac | 48.4 | 50.6 |
| b5 | 60.6 | 0.466–0.504 mean-frac | — | 49.6 |
| b10 | 66.8 | — | — | **55.9** |
| b15 | 67.0 | — | — | 54.4 (regrade claims 62.2, unverified) |
| b20 | **73.9** | — | — | ~44–50 (format decay; ~57 normalized to graded) |

Note: run 14's published "62.2% all-time record" claim is contradicted by run 11's
verified 67.0 at the same step on the identical eval set.

## Standing lessons

1. **Verify the gradient chain before trusting a loss curve.** Epoch-over-epoch
   per-datum comparison caught the B200 no-op; data-ordering artifacts can fake
   convergence.
2. **Distill thoughts, not just actions** — but capture them from the API channel
   the server actually uses, and fail fast when they're absent.
3. **Match the student's format conventions exactly**, including for the teacher's
   absences. "No reasoning" and "empty reasoning" are different token sequences.
4. **RL keeps what it rewards and discards the rest.** Style priors wash out;
   defects amplify; the behaviors that matter (write-discipline) get re-learned at
   full price either way.
5. **Distinguish measurement damage from behavior.** `graded` tracking `has_output`
   means the episodes failed; `reward_error` means the judge did. The two demand
   opposite responses.
