#!/usr/bin/env bash
# run51: Gemma 4 31B on the Automodel worker, run49/run50's shape (4 trainer GPUs,
# 4 single-GPU samplers, LoRA r32, GLM judge). The trainer runs CP4 with the
# model-owned Gemma4 ring (FFPA on the 512-wide global heads, flex on the
# sliding layers); vLLM serves the samplers with FlashAttention 3 and fits the
# full 262144 window on one H200 (65.8 GiB KV). Trainer interpreter is the
# Gemma venv (Automodel main + ffpa-attn), built by ~/automodel-gemma/build.sh.
#
# Context is 180k, not the model's 262k. Automodel keeps HF's layer list for
# the dense 31B, so only per-layer activation checkpointing applies (60 stashed
# hidden states), and the ring gathers the full-sequence K/V of the 10 global
# layers on every rank. CP4 ladders (scripts/automodel_probe.py): 128k 76.6,
# 200k 109.0 (135 reserved), 230k 122.7 (136 reserved), 262144 OOM. TP4 is
# worse (the stash is replicated across TP; 128k OOM). 180k keeps reserved
# memory near run49's 123 GiB. Override AUTOMODEL_TP / AUTOMODEL_CP / CONTEXT.
set -uo pipefail
cd "$HOME/open-rl" || exit 1
export MODEL=12b
export MODEL_NAME_OVERRIDE=google/gemma-4-31B-it
export CONTEXT=${CONTEXT:-180000}
export SAMPLER_CONTEXT=${SAMPLER_CONTEXT:-$CONTEXT}
export RUN_LABEL=${RUN_LABEL:-run51-gemma4-31b-automodel}
export TRAINER_BACKEND=automodel
export TRAIN_GPUS=4
export AUTOMODEL_TP=${AUTOMODEL_TP:-1}
export AUTOMODEL_CP=${AUTOMODEL_CP:-4}
export AUTOMODEL_PYTHON=${AUTOMODEL_PYTHON:-$HOME/automodel-gemma/.venv/bin/python}
export AFFINITY=1
export AUTOMODEL_LORA_RANK=32
export JUDGE_MODEL=gpt-glm-5.2
export SAMPLER_EXTRA=""
echo "[launch51] $MODEL_NAME_OVERRIDE @ $CONTEXT (sampler $SAMPLER_CONTEXT), backend=$TRAINER_BACKEND TP=$AUTOMODEL_TP CP=$AUTOMODEL_CP DP=$((TRAIN_GPUS / (AUTOMODEL_TP * AUTOMODEL_CP)))"
echo "[launch51] LoRA rank=$AUTOMODEL_LORA_RANK, trainer python $AUTOMODEL_PYTHON"
exec ./scripts/launch_work.sh
