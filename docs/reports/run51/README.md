# run51: Gemma 4 31B on the Automodel worker

Measured on h200-vm (8x H200 141 GB), 2026-09-11, `scripts/automodel_probe.py`
in the Gemma venv (`~/automodel-gemma/.venv`: Automodel main b1f4031, transformers
5.15.1, ffpa-attn 0.2.4), LoRA rank 32, one sequence per pass.

## What is different from run49/run50 (Qwen3.8-27B)

Automodel's `Gemma4ForConditionalGeneration` owns its context parallelism: the
batch is sharded into contiguous slices, the forward embeds the full `input_ids`
and slices its own hidden states, and every attention layer carries a p2p ring
(FFPA for the 512-wide global heads, flex for the sliding layers). The worker
now goes through that contract (`prepare_model_inputs_for_cp`) for models that
declare `_owns_cp_attention`, and reads the final hidden states from a hook on
the last norm because that forward drops `output_hidden_states`. The round-robin
torch `context_parallel` path stays for Qwen.

The dense 31B keeps HF's plain layer list inside the Automodel class, so only HF
per-layer activation checkpointing applies (the worker's grouped checkpointing
needs Automodel's ModuleDict backbone).

## How to judge numerics on this model

Random tokens and raw text put the instruct model far off its distribution: on
a legal memo without chat formatting, vLLM (FlashAttention 3), HF eager, HF sdpa
and the worker all agree on a mean logprob of about -6.2 with 29% of tokens
below -10, and there any perturbation moves logprobs by ~0.3 nats on average:
a 0.1% LoRA perturbation, bf16 vs fp32, even fp32 TP2 vs fp32 CP2. Those numbers
say nothing about a layout. The comparisons below use the model's own vLLM sample
(chat format, 67-token prompt, 1542 sampled tokens, mean logprob -0.26).

| comparison (sample, 1600 tokens) | logprob mean abs diff | median | grad cosine median / min |
|---|---|---|---|
| sampler (vLLM FA3) vs worker, base model | 0.015 (p99 0.19, max 0.46) | | |
| worker B=0 vs plain transformers sdpa | 0.019 | 2e-4 | |
| TP2 vs single GPU, bf16 | 0.018 | 1.3e-4 | 0.994 / -0.09 |
| CP2 vs single GPU, bf16 | 0.016 | 1.2e-4 | 0.989 / -0.51 |
| CP4 vs single GPU, bf16 | 0.016 | 1.0e-4 | 0.992 / -0.29 |
| TP2 fp32 run twice (determinism) | 0 | 0 | 0.9999 / 0.997 |
| CP2 fp32 vs TP2 fp32 | 0.016 | 1.2e-4 | 0.994 / 0.66 |
| TP4 fp32 vs TP2 fp32 | 0.019 | 1.4e-4 | 0.993 / 0.64 |
| TP2 fp32 eager vs TP2 fp32 sdpa (same layout) | 0.020 | 1.6e-4 | 0.992 / 0.12 |

Logprobs agree across layouts at the kernel floor. A few attention adapters in
layers 14-24 have gradients that flip under any perturbation, including a kernel
swap inside one layout (cosine 0.12) and fp32 TP4 vs TP2 (0.64); CP2 at 0.66 is
within that band, so the ring is not adding error. The sampler-trainer mismatch
of 0.015 nats per sampled token is the quantity RL trains through.

The adapter save/load round trip is exact (820 tensors, optimizer state restored).

## Memory (per GPU, GiB, peak allocated / reserved)

| layout | 64k | 128k | 200k | 230k | 262144 |
|---|---|---|---|---|---|
| TP4 / DP1, per-layer AC | 77.6 | OOM | | | |
| CP4 / DP1, per-layer AC | 48.3 / 59.8 | 76.6 / 95.0 | 109.0 / 135.2 | 122.7 / 136.3 | OOM |

Where it goes (allocator snapshot, CP2 at 64k, live at peak): the per-layer stash
of 60 hidden states 17 GiB; fp32 RMSNorm copies of the layer being recomputed
5.8; the ring's collected K/V and its backward buffers ~13 (the 10 global layers
gather the full-sequence K/V on every rank and expand it to 32 heads; sliding
layers fetch one prior chunk); rotary, masks and repeat_kv ~5; the logprob head
3.7. The ring cost scales with the total context, not the local slice, so
grouped checkpointing (stash 42 to 10 GiB at 262k) would be largely eaten by the
extra live layers. 262144 is out of reach on four H200s with this implementation.

## Run configuration

`scripts/launch51.sh` (stack) and `scripts/run51-driver.sh` (client): trainer
on GPUs 0-3 at CP4, four single-GPU vLLM samplers (the 262144 window fits one
H200: 65.8 GiB KV, 1.98x concurrency), LoRA rank 32, lr 2e-4, batch 8 x 6, 304
train tasks, run50's episode limits (65536 generation tokens, 200 turns, tool
results uncapped) and 0.8/0.2 reward, GLM judge, renderer `gemma4`.
Context 180000 (readable observation window 180000 - 65536), chosen so that
reserved memory stays near run49's 123 GiB rather than the 135 GiB of the 200k
ladder point.

## Step-0 eval (held-out 50 tasks, base model, reference limits)

| | Gemma 4 31B (run51) | Qwen3.8-27B (run49, 16k cap) |
|---|---|---|
| criterion pass rate, graded episodes | 61.6% (44 graded) | 22.6% |
| all criteria passed | 0 | 0 |
| mean reward (0.8 pass fraction + 0.2 all pass, -0.1 ungraded) | 0.405 | |
| turns per episode | 9.9 mean, 15 max | |
| generated tokens per episode | 9.6k mean, 74k max | |
| longest observation | 52k mean, 114k max | |
| lost before grading | 4 parse errors (3 on the first turn), 2 no output, 1 context overflow | 27 killed by the 16k cap |

The cookbook's streaming loop awaits the step-0 eval before it launches any
train rollouts, so the first 35 minutes of the run are eval only; the 48 train
episodes started at 21:10Z.

## Step 0 (train, 8 groups x 6)

| metric | value |
|---|---|
| sampler-trainer KL (`optim/kl_sample_train_v1`) | 0.0004 (Qwen run49: 0.0007) |
| train reward | 0.505 |
| criterion pass fraction | 0.631 |
| all criteria passed | 0 |
| graded / no output / context overflow | 91.7% / 8.3% / 1.1% |
| turns per episode, generated tokens per turn | 11.7, 1154 |
| observation tokens per turn | 40.9k |
| every group mixed (frac_mixed) | 1.0 |
| trainer peak at 113k tokens, CP4 | 73.7 GiB allocated / 88.6 reserved per rank |

The KL confirms the FFPA-plus-flex trainer path scores Gemma's samples the way
vLLM produced them; the raw-text mismatch numbers earlier in this report do not
apply to the model's own outputs.
