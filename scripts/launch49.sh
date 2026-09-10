#!/usr/bin/env bash
# run49: Qwen3.8-27B on the Automodel worker, run48's shape (4 trainer GPUs, 4
# single-GPU samplers, LoRA r32, GLM judge). The trainer layout and context come
# from scripts/automodel_probe.py ladders on this box (see docs/reports/run49);
# override AUTOMODEL_TP / AUTOMODEL_CP / CONTEXT to change them.
set -uo pipefail
cd "$HOME/open-rl" || exit 1
export MODEL=9b-128k
export MODEL_NAME_OVERRIDE=Qwen/Qwen3.8-27B
export CONTEXT=${CONTEXT:-262144}
export RUN_LABEL=${RUN_LABEL:-run49-qwen38-27b-automodel}
export TRAINER_BACKEND=automodel
export TRAIN_GPUS=4
export AUTOMODEL_TP=${AUTOMODEL_TP:-1}
export AUTOMODEL_CP=${AUTOMODEL_CP:-4}
export AFFINITY=1
export AUTOMODEL_LORA_RANK=32
export JUDGE_MODEL=gpt-glm-5.2
export SAMPLER_EXTRA=""
echo "[launch49] $MODEL_NAME_OVERRIDE @ $CONTEXT, backend=$TRAINER_BACKEND TP=$AUTOMODEL_TP CP=$AUTOMODEL_CP DP=$((TRAIN_GPUS / (AUTOMODEL_TP * AUTOMODEL_CP)))"
echo "[launch49] LoRA rank=$AUTOMODEL_LORA_RANK"
exec ./scripts/launch_work.sh
