#!/usr/bin/env bash
# Bring the stack up for run45: Qwen3.5-9B on the Megatron backend at TP=2/DP=2,
# 163,840-token context -- run19's recipe on a backend that has never run it.
#
# Separate from run45.sh because these are separate failure modes: this file
# decides the topology the servers come up with, and run45.sh refuses to start
# against a topology that came up wrong. Sourcing a script rather than typing
# this into tmux is deliberate -- a long send-keys line silently truncates, and
# a truncated env here is a run that trains with the wrong LoRA targets and
# looks perfectly healthy doing it.
#
# WHY TP=2 AND NOT TP=1. Measured, not estimated. A 9B at TP=1 costs 0.753 MiB
# of activations per token on top of 16.9 GiB of weights, so run19's 163,840
# tokens need 137.4 GiB of a 139.8 GiB card -- it does not fit, and the probe
# OOMs at 65,536 with the vocab projection unchunked. TP=2 halves the weights
# and brings activations to 0.565 MiB/token: ~102 GiB, with 38 GiB of margin.
#
# TP=2 buys context, not speed -- it adds two all-reduces per layer and full
# recompute pays them again in the backward, so TP=1 at 131,072 would be the
# faster run. Context parity with run19 was chosen over that, deliberately: the
# whole finding run19 left behind was that its eval decline came from context
# overflow on a heavy tail, and a shorter ceiling would reproduce the confound
# rather than test the backend.
#
# Only 0.72x, not 0.5x, because Qwen3.5's 24 gated-deltanet layers forbid
# sequence parallelism outright (core/ssm/gated_delta_net/gdn.py:91 is a bare
# assert), so LayerNorm and residual activations stay replicated at every TP
# degree. That same assert is why MEGATRON_SEQUENCE_PARALLEL is pinned off
# below: the worker turns SP on for any TP>1, TP=1 hid that behind an `and`,
# and leaving it alone would kill this run in model construction.
set -uo pipefail
cd "$HOME/open-rl" || exit 1

# 9b-128k for its shape, not its context: it carries run19's 8x6 batch and the
# seeded 300/50 split, where the plain 9b case sets neither. CONTEXT overrides
# its 131,072 default; SAMPLER_CONTEXT and TRAIN_TOKEN_BUDGET both follow it.
export MODEL=9b-128k
export CONTEXT=163840
export RUN_LABEL=lab-lora-qwen9b-160k

export TRAINER_BACKEND=megatron
export TRAIN_GPUS=4
# Spelled out because it is a decision, not an omission: MEGATRON_TP defaults
# to $TRAIN_GPUS, so leaving it unset would silently give TP=4.
export MEGATRON_TP=2
export MEGATRON_SEQUENCE_PARALLEL=0
# Still required by launch_work.sh under this backend, and four single-GPU
# samplers is the topology every context number above was measured on.
export AFFINITY=1
export MEGATRON_LORA_RANK=32
# The GDN mixers. The worker's default names only the four projections a dense
# transformer has, which on this checkpoint leaves 24 of 32 layers' mixers
# frozen. run19 trained them on the FSDP path, so this is parity rather than a
# second variable. Megatron fuses the checkpoint's four in_proj_* into one
# in_proj and save_hf_adapter splits them back out; all 496 exported tensors
# were name- and shape-checked against the base checkpoint before this run.
export MEGATRON_LORA_TARGETS=linear_qkv,linear_proj,linear_fc1,linear_fc2,in_proj,out_proj
export JUDGE_MODEL=gpt-glm-5.2

# One sampler change survives from run44: MTP speculative decoding. Qwen3.5
# ships a multi-token-prediction head and this vLLM registers it as a draft
# model; two drafted tokens are verified per target pass. Measured in run44
# step 0: 0.74 acceptance per draft token, 2.49 tokens per target step, but
# each step ~2x slower than plain decode (the gated-deltanet layers take the
# multi-token path), so the net was decode 50 -> 62 tok/s per stream.
#
# The fp8 KV cache run44 also tried is gone. It raised kl_sample_train from
# 3.5e-4 to 6.1e-3 at step 0 -- a 17x larger sampler/trainer mismatch -- for
# no separable speed gain. If kl stays high without it, MTP is the cause and
# goes next.
export SAMPLER_EXTRA="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}'"

echo "[launch45] $MODEL @ $CONTEXT, backend=$TRAINER_BACKEND TP=$MEGATRON_TP (DP=$((TRAIN_GPUS / MEGATRON_TP))) SP=$MEGATRON_SEQUENCE_PARALLEL"
echo "[launch45] LoRA rank=$MEGATRON_LORA_RANK targets=$MEGATRON_LORA_TARGETS"
echo "[launch45] sampler extra: $SAMPLER_EXTRA"
exec ./scripts/launch_work.sh
