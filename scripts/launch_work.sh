#!/usr/bin/env bash
# Bring up the whole Harvey-LAB RL stack in one tmux session called "work":
# a stock vLLM sampler on GPUs 1-N, the gateway + LoRA trainer on GPU 0, and
# typed (not yet run) train/eval commands.
#
#   MODEL=9b  ./scripts/launch_work.sh    # Qwen3.5-9B, full 262K window (default)
#   MODEL=27b ./scripts/launch_work.sh    # Qwen3.5-27B, 98K (measured H200 ceiling)
#
# Windows:
#   0 sampler   vllm serve, GPUs 1-7, data-parallel            auto-starts
#   1 gateway   gateway + LoRA trainer, GPU 0                  auto-starts (waits for sampler)
#   2 train     training command — TYPED, press Enter to launch
#   3 eval      eval_checkpoint command — TYPED, edit checkpoint= then Enter
#   4 gpu       nvidia-smi watch
#
# Re-running never kills anything: an existing session is attached as-is.
# The LAB checkout is bootstrapped via setup_lab.sh if missing. Logs tee to
# artifacts/box-logs/. Overridable: MODEL, RUN_LABEL, GEN_TOKENS.
set -euo pipefail

SESSION=work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB_ROOT="$REPO/examples/harvey_labs/harvey-labs"
LOGS="$REPO/artifacts/box-logs"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running — attaching (tmux kill-session -t $SESSION to reset)"
  if [ -t 1 ]; then
    exec tmux attach -t "$SESSION"
  else
    exit 0
  fi
fi

if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "WARNING: GEMINI_API_KEY is not set — rubric grading will fail without it." >&2
fi
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is not set — deliverable-name matching errors on mismatched filenames without it." >&2
fi

mkdir -p "$LOGS"
cd "$REPO"

if [ ! -d "$LAB_ROOT" ]; then
  echo "[work] LAB checkout missing — running setup_lab.sh (clones the fork, installs pandoc/podman)..."
  ./examples/harvey_labs/setup_lab.sh
fi
echo "[work] LAB judge at: $(git -C "$LAB_ROOT" log --oneline -1 -- evaluation/judge.py 2>/dev/null || echo 'unknown')"

# The 27B ceiling on a 141GB H200 is 98K tokens (measured, activation offload
# on); 9B fits its full 262K window with room to spare.
MODEL=${MODEL:-9b}
case "$MODEL" in
  9b)
    MODEL_NAME=Qwen/Qwen3.5-9B
    CONTEXT=262144
    GEN_TOKENS=${GEN_TOKENS:-32768}
    TASK_SET=${TASK_SET:-random}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen9b}
    ;;
  27b)
    MODEL_NAME=Qwen/Qwen3.5-27B
    CONTEXT=98304
    # 32K tool results in a 98K window would let a few parallel document
    # reads overflow the whole trajectory budget; 16K is the proven value.
    GEN_TOKENS=${GEN_TOKENS:-16384}
    # Curated run-3/4 task lists so 27B numbers stay comparable across runs.
    TASK_SET=${TASK_SET:-bootstrap}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen27b}
    ;;
  *)
    echo "Unknown MODEL=$MODEL (use 9b or 27b)" >&2
    exit 1
    ;;
esac
echo "[work] MODEL=$MODEL -> $MODEL_NAME, context $CONTEXT, log $RUN_LABEL"

SAMPLER_CMD="CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
uv run --extra gpu --extra vllm vllm serve $MODEL_NAME \
--port 8000 --enable-lora --max-lora-rank 64 --max-loras 2 \
--data-parallel-size 7 --api-server-count 1 \
--max-model-len $CONTEXT --gpu-memory-utilization 0.92 \
--language-model-only |& tee -a $LOGS/sampler.log"

GATEWAY_CMD="until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do echo 'waiting for sampler...'; sleep 10; done; \
CUDA_VISIBLE_DEVICES=0 BASE_MODEL=$MODEL_NAME SAMPLER_BASE_URL=http://127.0.0.1:8000 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPEN_RL_TRAIN_TOKEN_BUDGET=$CONTEXT OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 \
OPEN_RL_LOG_CUDA_MEMORY=1 \
uv run --extra gpu --extra vllm python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003 |& tee -a $LOGS/gateway.log"

TRAIN_CMD="TINKER_API_KEY=tml-dummy uv --project examples run python examples/harvey_labs/train.py \
model_name=$MODEL_NAME renderer_name=qwen3_5 base_url=http://127.0.0.1:9003 \
learning_rate=2e-4 lora_rank=32 \
batch_size=5 rollouts_per_example=2 max_steps=20 eval_every=5 \
task_set=$TASK_SET \
max_tokens=$GEN_TOKENS max_trajectory_tokens=$CONTEXT max_tool_result_tokens=$GEN_TOKENS \
log_path=artifacts/harvey-labs/$RUN_LABEL"

EVAL_CMD="TINKER_API_KEY=tml-dummy uv --project examples run python examples/harvey_labs/eval_checkpoint.py \
checkpoint=/tmp/open-rl/peft/CHANGE-ME/final model_name=$MODEL_NAME renderer_name=qwen3_5 \
base_url=http://127.0.0.1:9003 task_set=$TASK_SET \
max_tokens=$GEN_TOKENS max_trajectory_tokens=$CONTEXT max_tool_result_tokens=$GEN_TOKENS"

# set-option needs a running tmux server, so the session must exist first.
tmux new-session -d -s "$SESSION" -n sampler -c "$REPO"
tmux set-option -t "$SESSION" history-limit 100000
tmux send-keys -t "$SESSION:sampler" "$SAMPLER_CMD" C-m

tmux new-window -t "$SESSION" -n gateway -c "$REPO"
tmux send-keys -t "$SESSION:gateway" "$GATEWAY_CMD" C-m

tmux new-window -t "$SESSION" -n train -c "$REPO"
tmux send-keys -t "$SESSION:train" "$TRAIN_CMD"          # typed, NOT run

tmux new-window -t "$SESSION" -n eval -c "$REPO"
tmux send-keys -t "$SESSION:eval" "$EVAL_CMD"            # typed, NOT run

tmux new-window -t "$SESSION" -n gpu -c "$REPO"
tmux send-keys -t "$SESSION:gpu" "watch -n 5 nvidia-smi" C-m

tmux select-window -t "$SESSION:train"
echo "[work] up. sampler+gateway starting; train/eval commands are typed and waiting."
if [ -t 1 ]; then
  exec tmux attach -t "$SESSION"
fi
