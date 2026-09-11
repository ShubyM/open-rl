#!/usr/bin/env bash
# run51: Gemma 4 31B on the Automodel worker, run49/run50's shape (4 trainer GPUs,
# 4 single-GPU samplers, LoRA r32, GLM judge). Gemma has no gated-deltanet, so
# TP4/DP1 shards every layer; the 512-wide global attention heads run through
# FFPA on the trainer (OPEN_RL_AUTOMODEL_ATTN=auto) and FlashAttention 3 on
# vLLM. Trainer interpreter is the Gemma venv (Automodel main + ffpa-attn),
# built by ~/automodel-gemma/build.sh on the box.
#
# Sampler context: one 262k sequence needs ~86 GB of KV for the 10 global
# layers next to 62 GB of weights, so a single H200 may not take the full
# window; vLLM refuses to start if max-model-len does not fit its KV cache.
# Lower SAMPLER_CONTEXT (and the driver's max_trajectory_tokens) if it does.
# Layout and context come from scripts/automodel_probe.py ladders; override
# AUTOMODEL_TP / AUTOMODEL_CP / CONTEXT to change them.
set -uo pipefail
cd "$HOME/open-rl" || exit 1
export MODEL=12b
export MODEL_NAME_OVERRIDE=google/gemma-4-31B-it
export CONTEXT=${CONTEXT:-262144}
export SAMPLER_CONTEXT=${SAMPLER_CONTEXT:-$CONTEXT}
export RUN_LABEL=${RUN_LABEL:-run51-gemma4-31b-automodel}
export TRAINER_BACKEND=automodel
export TRAIN_GPUS=4
export AUTOMODEL_TP=${AUTOMODEL_TP:-4}
export AUTOMODEL_CP=${AUTOMODEL_CP:-1}
export AUTOMODEL_PYTHON=${AUTOMODEL_PYTHON:-$HOME/automodel-gemma/.venv/bin/python}
export AFFINITY=1
export AUTOMODEL_LORA_RANK=32
export JUDGE_MODEL=gpt-glm-5.2
export SAMPLER_EXTRA=""
echo "[launch51] $MODEL_NAME_OVERRIDE @ $CONTEXT (sampler $SAMPLER_CONTEXT), backend=$TRAINER_BACKEND TP=$AUTOMODEL_TP CP=$AUTOMODEL_CP DP=$((TRAIN_GPUS / (AUTOMODEL_TP * AUTOMODEL_CP)))"
echo "[launch51] LoRA rank=$AUTOMODEL_LORA_RANK, trainer python $AUTOMODEL_PYTHON"
exec ./scripts/launch_work.sh
