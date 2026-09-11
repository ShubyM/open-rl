#!/usr/bin/env bash
# run49 driver: run48c's client settings against the Automodel trainer stack
# brought up by scripts/launch49.sh. The trajectory cap must match the trainer's
# CONTEXT and the samplers' max-model-len from that launch.
# Run in the tmux "train" window after all four samplers report /v1/models.
set -uo pipefail
cd "$HOME/open-rl-k8s/examples" || exit 1
set -a; source "$HOME/open-rl/.env.judge"; set +a
export TINKER_API_KEY=tml-dummy
export FORCE_COLOR=1 COLUMNS=${COLUMNS:-160}
# Set unconditionally: shells in the launch_work tmux session inherit
# RUN_LABEL and CONTEXT from the launch script, and an inherited RUN_LABEL
# points the cookbook at the previous run's directory, where it auto-resumes
# from that run's checkpoints.jsonl.
CONTEXT=262144
RUN_LABEL=run49-qwen38-27b-automodel
echo "=== $RUN_LABEL driver start $(date -u +%FT%TZ) context=$CONTEXT"
curl -sf -m 10 "$OPENAI_BASE_URL/models" >/dev/null || { echo "judge at $OPENAI_BASE_URL is not answering"; exit 1; }
.venv/bin/harvey-train \
  model_name=Qwen/Qwen3.8-27B renderer_name=${RENDERER:-qwen3_8_medium_reasoning} base_url=http://127.0.0.1:9003 \
  lab_root=$HOME/open-rl/examples/harvey_labs/harvey-labs \
  learning_rate=2e-4 lora_rank=32 batch_size=8 rollouts_per_example=6 max_steps=40 \
  eval_every=${EVAL_EVERY:-10} eval_rollouts_per_task=1 save_every=${SAVE_EVERY:-10} judge_model=gpt-glm-5.2 stream_minibatches=True log_groups=0 \
  max_tokens=16384 max_trajectory_tokens=$CONTEXT max_tool_result_tokens=${TOOL_TOKENS:-4096} \
  log_path=$HOME/open-rl/artifacts/harvey-labs/$RUN_LABEL 2>&1 | tee -a $HOME/open-rl/artifacts/box-logs/$RUN_LABEL.log
echo "=== $RUN_LABEL driver exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
