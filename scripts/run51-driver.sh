#!/usr/bin/env bash
# run51 driver: run50's client settings (reference LAB episode limits, 0.8/0.2
# reward, 304 train tasks) against the Gemma 4 31B stack from scripts/launch51.sh.
# Gemma has one renderer (the recipe's tool-calling gemma4), no reasoning
# effort levels. max_trajectory_tokens must match the trainer's CONTEXT from
# that launch (180k by default, see launch51.sh for why not 262k); the
# samplers accept up to that as well.
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
CONTEXT=${RUN51_CONTEXT:-180000}
RUN_LABEL=run51-gemma4-31b-automodel-reference-limits
echo "=== $RUN_LABEL driver start $(date -u +%FT%TZ) context=$CONTEXT"
curl -sf -m 10 "$OPENAI_BASE_URL/models" >/dev/null || { echo "judge at $OPENAI_BASE_URL is not answering"; exit 1; }
.venv/bin/harvey-train \
  model_name=google/gemma-4-31B-it renderer_name=gemma4 base_url=http://127.0.0.1:9003 \
  lab_root=$HOME/open-rl/examples/harvey_labs/harvey-labs \
  train_tasks=304 learning_rate=2e-4 lora_rank=32 batch_size=8 rollouts_per_example=6 max_steps=40 \
  eval_every=${EVAL_EVERY:-10} eval_rollouts_per_task=1 save_every=${SAVE_EVERY:-10} judge_model=gpt-glm-5.2 stream_minibatches=True log_groups=0 \
  max_turns=${MAX_TURNS:-200} max_tokens=${GEN_TOKENS:-65536} max_trajectory_tokens=$CONTEXT max_tool_result_tokens=${TOOL_TOKENS:-$CONTEXT} \
  log_path=$HOME/open-rl/artifacts/harvey-labs/$RUN_LABEL 2>&1 | tee -a $HOME/open-rl/artifacts/box-logs/$RUN_LABEL.log
echo "=== $RUN_LABEL driver exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
