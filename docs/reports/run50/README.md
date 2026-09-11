# run50: Qwen3.8-27B, reference LAB episode limits, 0.8/0.2 reward

Same stack as run49 (`scripts/launch49.sh`: Automodel CP4 at 262144, four vLLM
samplers, GLM judge) with the episode limits matched to the reference LAB
harness and the reward changed. Driver: `scripts/run50-driver.sh`.

| knob | run49 | run50 |
|---|---|---|
| per-turn generation budget | 16,384 | 65,536 |
| turns | 40 | 200 |
| tool-result truncation | 4k | none |
| reasoning effort | medium | xhigh (Qwen3.8 has no "high") |
| budget-terminated episodes | −0.1, ungraded | graded on what was produced |
| reward | criterion pass fraction | 0.8 × pass fraction + 0.2 × all-pass |
| train tasks | 300 (last batch of 4 hung the run at step 37) | 304 |

## Step 0 baseline vs run49

| held-out, 50 tasks | run49 | run50 |
|---|---|---|
| criterion pass rate | 22.6% | 69.9% |
| all criteria passed | 0 | 7 |
| graded | 16 | 39 |
| killed by per-turn cap | 27 | 0 |
| context overflow | 0 | 16 (graded) |

run49's low baseline was the 16k per-turn cap, not the model.

## Training steps

| step | reward | pass rate | all-pass | graded | overflow (episodes) | no deliverable | step time |
|---|---|---|---|---|---|---|---|
| 0 | 0.436 | 0.530 | 0.062 | 62% | 34/48 | 38% | 5.0 h (incl. 2 h eval) |
| 1 | 0.610 | 0.716 | 0.188 | 77% | 17/48 | 23% | 2.3 h |

Reward equals 0.8 × pass fraction + 0.2 × all-pass on every episode (checked
from the rollout summaries). The 262k window, not the generation cap, is the
dominant terminal; xhigh triples generated tokens per episode (65k median vs
21k in run49) and keeps the samplers' KV cache at 85-99%, which is the pace.

Trainer on real trajectories: 220k-token backward peaks 100 GiB per GPU
(ladder: 92 GiB at 200k, 115 at 262k), no OOM through two steps.
