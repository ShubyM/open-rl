# run19 — Qwen3.5-9B on Harvey LAB

GRPO against the LAB rubric judge. Reward is the pure rubric pass fraction:
`process_reward_weight = 0.0`, so the shaping terms are logged but contribute
nothing. Held-out eval is 50 tasks, one rollout each, disjoint from the
training split via `train_split_seed=242`.

```
model_name=Qwen/Qwen3.5-9B  renderer_name=qwen3_5
learning_rate=2e-4  lora_rank=32           # constant LR, no schedule
batch_size=8  rollouts_per_example=6       # 48 rollouts/step
eval_every=5  task_set=random  judge_model=gpt-glm-5.2
stream_minibatches=True
max_tokens=32768  max_trajectory_tokens=163840  max_tool_result_tokens=16384
```

![run19](run19.png)

## Held-out eval

| step | criteria_pass_fraction | rubric (graded only) | graded | ctx overflow | no_output |
|-----:|-----:|-----:|-----:|-----:|-----:|
| 0 | 0.5351 | 0.6222 | 43 | 4 | 3 |
| 5 | 0.6254 | 0.6654 | 47 | 1 | 2 |
| 10 | 0.5826 | 0.6473 | 45 | 5 | 0 |
| 15 | 0.5796 | 0.6739 | 43 | 6 | 1 |
| 20 | 0.6090 | 0.6767 | 45 | 3 | 2 |
| 25 | 0.6774 | 0.7056 | 48 | 1 | 1 |
| 30 | 0.6903 | 0.7344 | 47 | 2 | 1 |
| 34 | 0.7088 | 0.7383 | 48 | 2 | 0 |
| 35 | 0.7098 | 0.7394 | 48 | 1 | 1 |

`criteria_pass_fraction` averages over all 50 tasks and scores a failed episode
0. `rubric_reward` averages only over the episodes the judge graded, so it
hides completion failures. Quote the former.

The last two rows are both real and both off the `eval_every` grid. The
cookbook fires an eval at the last batch (`i_batch == end_batch - 1`, so 34),
and `train.py`'s `run_final_eval` then scores the saved `final` checkpoint one
update later (logged as batch 35). **Headline number: 0.7098**, or 0.7093
pooled over all 100 tail episodes.

## Convergence

The run is done improving at this configuration.

| steps | rollout reward | grad_norm | entropy |
|---|---:|---:|---:|
| 0-6 | 0.554 | 1793 | 0.460 |
| 7-13 | 0.548 | 1728 | 0.473 |
| 14-20 | 0.646 | 1474 | 0.423 |
| 21-27 | 0.679 | 1144 | 0.422 |
| 28-34 | 0.675 | 1676 | 0.416 |

All of the gain lands in steps 14-21. Over the last 15 steps the reward slope
is **+0.0007/step against a per-step sd of 0.034** — 1/50th of the noise — and
over the last 10 it is negative. The final update moved the eval by +0.001.

`grad_norm` is not a convergence signal here and should not be read as one. In
GRPO the gradient scale is set by within-group advantage dispersion, so it
measures batch difficulty: `corr(grad_norm, reward) = -0.47`,
`corr(grad_norm, graded) = -0.60`. The smallest gradients in the run are its
easiest batches (step 27: 510 at reward 0.735). It is useful as an *alarm* —
the 119-277 values during the dead-judge steps are the signature of a constant
reward flattening every advantage, and that is worth alerting on.

Entropy is the better read: one compression around steps 14-20 (0.47 → 0.42),
flat since, with the step-to-step spread halving (0.043 → 0.020). Settled, not
collapsed.

Train and held-out track each other one-for-one — mean gap across the eight
checkpoints is **+0.003** (sd 0.068, entirely the two estimators' own noise),
and −0.008 / −0.026 / −0.008 at steps 25/30/34. Nothing is being memorised from
the training batches, and the train curve is a usable proxy for held-out at
1/5th the latency.

More steps will not help. The dataset caps it anyway: 300 train tasks at
batch_size 8 is `ceil(300/8) = 38` batches, and `end_batch = min(max_steps,
num_batches)`, so `max_steps=40` silently clamps to 38. Those 3 unused batches
(tasks 280-299) are unseen and safe to train on — `get_batch` is a plain slice
with no wraparound or reshuffle — but the last is partial (4 groups) and 3
steps is far inside the noise. Going past 38 needs `train_tasks=320`, and that
moves `eval_names = shuffled[num_train : num_train + num_eval]` to `[320:370]`,
i.e. **a different benchmark**. Don't.

## Two things that will mislead you in the raw data

**The judge died mid-step-15.** At 08:24:38 UTC on 2026-08-09 the GLM-5.2
server on the b200 box hit `TimeoutError: RPC call to sample_tokens timed out`
→ `EngineDeadError`, with 96 requests in flight against a `judge_parallel=16`
run — a 300-episode teacher-trace job was sharing the server. Nothing restarted
it. run19 kept going and logged steps 15-19 with `cpf = 0.0000` and grad_norm
collapsed to 119-277 (vs 1500-2200 normal); those steps still applied
gradients, so run19's `final`/`000020` checkpoint is contaminated. run19r
resumed from checkpoint `000015` (weights + optimizer state) against a healthy
judge and replayed them, with byte-identical hyperparameters. **Steps 0-14 come
from run19, 15+ from run19r.** The judge now runs under a respawn loop.

**The eval is noisy and we still don't know how noisy.** Resuming past step 20
re-ran that eval over the same `final` weights and overwrote the file: 0.6584
first time, 0.6090 second. Same weights, same 50 tasks, spread **0.049**. Both
readings are on the plot (filled vs open diamond); pooled over the 100 episodes
that checkpoint is 0.634.

That single pair was the basis for a "±0.05 noise floor" claim earlier in this
report's history, and the end of the run undercut it: steps 34 and 35 are one
update apart and landed **0.001** apart (0.7088, 0.7098). Two replicate pairs,
spreads differing by 50×. One draw cannot estimate a variance, and 0.049 may
simply have been an unlucky one. The defensible statement is that the spread is
*unknown and bounded somewhere under 0.05*, which is still wide enough to
swallow most step-to-step movement in the eval column — the 0.625 → 0.583
"regression" at step 10 that triggered a whole investigation was 0.04. One
rollout per task is too few to tell. Use 2-3 next time and measure this
properly instead of inferring it from accidents.

## What the eval column actually measured

Decomposing the apparent peak-to-trough decline (step 5 → 15, −0.1072 per task)
by what changed per task:

| | n | contribution |
|---|---:|---:|
| lost to context overflow | 8 | −0.1138 |
| quality on tasks scoring at both | 37 | +0.0381 |
| lost, graded zero | 1 | −0.0189 |
| lost, no output | 1 | −0.0158 |
| gained | 1 | +0.0131 |

The writing was improving the whole time; the entire regression was tasks
hitting the 163,840-token wall and scoring 0. Overflow is a tool-result
problem, not a verbosity problem — **assistant text is 0.8% of a trajectory and
tool results are 99.2%**, and with `max_tool_result_tokens=16384` a single call
can take 10% of the budget. The tasks that overflowed used ~1.5× the median
task's tokens at *every* checkpoint including step 0, and sat near 80% of
budget by step 5, while median per-task growth from 5 → 15 was +2%. A
stationary heavy tail drifting across a cliff, not a ballooning policy.

The `context_overflow_reward = -0.1` penalty is not the weak link: within-group
centering gives an overflowing rollout an advantage of −0.10 to −0.64 against
siblings scoring ~0.8. Constant-reward group filtering is not the weak link
either — all-overflow groups happened once in 15 steps. What the penalty cannot
do is teach *when* to stop reading, because the budget is never shown to the
model and the penalty has no slope.

Extending the run resolved it without any intervention. Overflow across evals:
4, 1, 5, 6, 3, 1, 2, 2, 1 — the tail stopped hitting the wall on its own by
step 25, and rubric quality and completion rate rose together instead of
trading off. By the final eval, 48 of 50 episodes complete and get graded, with
one overflow and one no-output left. No compaction, no prompt change, no budget
signal in the observation — RL alone closed it.

## Queued for the next run

* Expose remaining context budget in the observation. The model cannot learn a
  stopping policy conditioned on a variable it cannot see.
* 2-3 eval rollouts per task, so the signal clears the noise floor — and so the
  floor itself is measured rather than guessed from two accidental replicates.
* Abort a step on `lab/reward_error` or an all-zero batch. Five steps trained on
  a dead judge before anyone noticed. `grad_norm` collapsing an order of
  magnitude is the cheapest available detector.
* Optional, if the tail returns: `max_trajectory_tokens` 163840 → 262144, and
  `max_tool_result_tokens` well below 16384.
* Not more steps. The run converged by step 21 and the dataset caps at 38
  batches; see Convergence above.
* Capacity, if a bigger jump is wanted: LoRA rank above 32. Entropy at 0.42 says
  there is exploration left, so the ceiling looks like capacity or missing
  signal rather than optimisation.

## Contents

Committed:

| path | what |
|---|---|
| `eval_episodes.jsonl` | one line per held-out episode, every judge metric |
| `meta/<run>.metrics.jsonl` | per-step training metrics |
| `meta/<run>.config.json` | full resolved config |
| `meta/<run>.code.diff` | working tree diff at launch |
| `meta/<run>.checkpoints.jsonl` | checkpoint records (`tinker://` URIs) |
| `meta/<run>.logs.log.gz` | full training log |
| `plot.py` | regenerates `run19.png` from this directory alone |
| `build.py` | rebuilds this directory from a full artifact mirror |

Every table and figure above is recomputable from those alone.

Not committed — the repo keeps run data out of git (`.gitignore` line 60,
`runs/`), and these are 72 MB:

| path | what |
|---|---|
| `evals/<run>-<iter>.jsonl.gz` | held-out rollout transcripts, full fidelity |
| `train_episodes.jsonl.gz` | training rollouts, slim: every metric kept, each assistant message and tool result reduced to its length plus a 240-char head. Preserves the overflow, tool-volume, churn and error analyses at ~2% of raw size. |

Both rebuild from the mirror. The full 1.2 GB mirror — including
`train_rollout_summaries.jsonl`, `*_logtree.json` and the rendered `*.html`
viewers — lives in three places, none of them a spot instance:

| where | what |
|---|---|
| `gs://open-rl-runs/harvey-labs/` | project `open-rl`, us-east7 |
| `ShubyM/dump` @ `artifacts/harvey-labs/run19{,r}` | browsable, `logs.log` gzipped there (`*.log` is ignored) |
| `~/open-rl-runs/harvey-labs/` | laptop working copy |

The h200 and b200 boxes are spot instances and are not a copy. The h200's
service account has no storage scope, so the GCS push has to run from the
laptop.

Rebuild everything from the mirror:

```
ssh h200 'tar -C ~/open-rl/artifacts -czf - harvey-labs/run19 harvey-labs/run19r' \
  | tar -C ~/open-rl-runs -xzf -
python docs/reports/run19/build.py       # -> meta/, eval_episodes.jsonl, evals/
python docs/reports/run19/plot.py        # -> run19.png
cd ~/open-rl-runs && gcloud storage rsync -r --project=open-rl \
  harvey-labs gs://open-rl-runs/harvey-labs
```
