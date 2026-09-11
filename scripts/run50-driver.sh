#!/usr/bin/env bash
# run50 driver: run49's stack (scripts/launch49.sh, Qwen3.8-27B, Automodel CP4,
# 262144 context) with the episode limits matched to the reference LAB harness
# and the vals.ai leaderboard runs instead of the caps run48c chose for memory:
#
#   reference: model's full output budget per turn, 200 turns, tool results
#              never truncated, maximum reasoning effort
#   run49:     16k per turn, 40 turns, 4k tool results, medium reasoning
#
# run49's step-0 eval lost 27 of 50 episodes to the 16k per-turn cap alone.
#
# One deliberate deviation. The env reserves max_tokens out of the trajectory
# budget on every turn, so the usable observation window is 262144 - max_tokens.
# The reference runs models with 500k-1M windows and 128k+ outputs; on this
# model's 262144 window a 128k reserve would halve the readable context, so the
# per-turn budget is 65536 (four times run49, observation ceiling 196k).
# Tool results are capped at the window itself, which is no truncation in
# practice. Sampling temperature is not configurable in the recipe (the
# reference evaluates at 0); RL sampling stays as is.
#
# train_tasks=304: with 300 tasks and batch 8 the last batch holds 4 groups
# while the streaming trainer still waits for 8 minibatches, so runs 47 and 49
# both hung at "step 37, minibatch 3/8" and never ran their final eval. 304
# gives 38 full batches; the held-out slice is fixed by seed and unchanged.
#
# Expect slower steps than run49: longer thinking, more turns, longer contexts.
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
RUN_LABEL=run50-qwen38-27b-automodel-reference-limits
echo "=== $RUN_LABEL driver start $(date -u +%FT%TZ) context=$CONTEXT"
curl -sf -m 10 "$OPENAI_BASE_URL/models" >/dev/null || { echo "judge at $OPENAI_BASE_URL is not answering"; exit 1; }
.venv/bin/harvey-train \
  model_name=Qwen/Qwen3.8-27B renderer_name=${RENDERER:-qwen3_8_xhigh_reasoning} base_url=http://127.0.0.1:9003 \
  lab_root=$HOME/open-rl/examples/harvey_labs/harvey-labs \
  train_tasks=304 learning_rate=2e-4 lora_rank=32 batch_size=8 rollouts_per_example=6 max_steps=40 \
  eval_every=${EVAL_EVERY:-10} eval_rollouts_per_task=1 save_every=${SAVE_EVERY:-10} judge_model=gpt-glm-5.2 stream_minibatches=True log_groups=0 \
  max_turns=${MAX_TURNS:-200} max_tokens=${GEN_TOKENS:-65536} max_trajectory_tokens=$CONTEXT max_tool_result_tokens=${TOOL_TOKENS:-$CONTEXT} \
  log_path=$HOME/open-rl/artifacts/harvey-labs/$RUN_LABEL 2>&1 | tee -a $HOME/open-rl/artifacts/box-logs/$RUN_LABEL.log
echo "=== $RUN_LABEL driver exit=${PIPESTATUS[0]} $(date -u +%FT%TZ)"
