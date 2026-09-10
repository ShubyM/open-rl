# run49: Qwen3.8-27B at full context on the Automodel worker

Measured on h200-vm (8x H200 141 GB), 2026-09-10, `scripts/automodel_probe.py`,
LoRA rank 32, one synthetic sequence per pass, peak memory per GPU in GiB.

## Correctness (Qwen3.5-9B, 4096 tokens, against a single-GPU reference)

| layout | logprob max abs diff | adapter grad cosine (min) | grad reldiff (median) |
|---|---|---|---|
| worker with B=0 vs plain transformers | 0.0 (bit-exact) | n/a | n/a |
| DP2 | 0.0 | n/a | n/a |
| TP2 | 0.157 | 0.99944 | 1.0e-2 |
| CP2 | 0.121 | 0.99950 | 9.1e-3 |
| CP4 | 0.227 | 0.99869 | 9.4e-3 |
| TP2 x CP2 | 0.186 | 0.99859 | 9.7e-3 |

Logprob differences are bf16 reduction-order noise on values near -13 and are
uniform across all sequence chunks. The save/load round trip (adapter plus
optimizer) reproduces logprobs exactly.

## Memory ladders (Qwen3.8-27B)

Automodel's own activation checkpointing wraps `self_attn`, `linear_attn` and
`mlp` separately on its native Qwen3.5 model, so each layer stashes about four
sequence-length tensors. The worker disables it and checkpoints groups of `k`
whole layers (`OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS`).

| layout | checkpointing | resident | 64k | 128k | 160k | 200k | 262144 |
|---|---|---|---|---|---|---|---|
| TP4 / DP1 | Automodel per-submodule | 21.4 | OOM (16k 68.6, 32k 108.1) | | | | |
| TP4 / DP1 | groups of 4 | 21.4 | 81.5 | OOM | | | |
| TP4 / DP1 | groups of 2 | 21.4 | 71.4 | 119.4 | OOM | | |
| CP4 / DP1 | Automodel per-submodule | 12.8 | 61.8 (32k 42.0) | | OOM | | |
| CP4 / DP1 | groups of 4 | 12.8 | | 66.4 | | 92.2 | 114.5 (123.0 reserved) |

TP4/DP1 cannot pass ~140k tokens on this backend: the gated-deltanet layers are
not tensor-parallel, so their recompute working set is replicated on every rank.
CP4 shards the sequence itself and reaches the model's full 262144 window with
~17 GiB of headroom. Forward+backward at 262144 takes 136 s at CP4.

## Run configuration

`scripts/launch49.sh` (stack) and `scripts/run49-driver.sh` (client): trainer on
GPUs 0-3 at CP4, four single-GPU vLLM samplers at max-model-len 262144, LoRA
rank 32, lr 2e-4, batch 8 x 6, 16k generation, 4k tool results, GLM judge
(`gpt-glm-5.2` on b200-vm), medium-reasoning Qwen3.8 renderer.
