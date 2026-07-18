#!/usr/bin/env bash
# Start a persistent OpenRL development stack on a single GPU VM.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session="${OPEN_RL_TMUX_SESSION:-openrl}"
base_model="${BASE_MODEL:-Qwen/Qwen2.5-0.5B}"
trainer_gpus="${CUDA_VISIBLE_DEVICES:-0}"
sampler_gpus="${SAMPLER_CUDA_VISIBLE_DEVICES:-$trainer_gpus}"
port="${PORT:-9003}"
redis_port="${REDIS_PORT:-6379}"
tmp_dir="${OPEN_RL_TMP_DIR:-/tmp/open-rl}"
log_dir="${OPEN_RL_LOG_DIR:-$tmp_dir/logs}"
log_max_bytes="${OPEN_RL_LOG_MAX_BYTES:-26214400}"
socket_path="${OPEN_RL_ACCEL_TIMESLICER_SOCKET:-$tmp_dir/accel-timeslicer.sock}"
runtime_library_path="$repo_root/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
attach="${ATTACH:-auto}"
dry_run="${OPEN_RL_SETUP_DRY_RUN:-0}"
bootstrap="${OPEN_RL_BOOTSTRAP:-auto}"

run() {
  if [[ "$dry_run" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing %s. %s\n' "$1" "$2" >&2
    exit 1
  fi
}

require_tool nvidia-smi "Use a GCP GPU image with the NVIDIA driver installed."

if [[ "$dry_run" != "1" && "$bootstrap" != "0" ]]; then
  missing_packages=()
  command -v tmux >/dev/null 2>&1 || missing_packages+=(tmux)
  command -v redis-server >/dev/null 2>&1 || missing_packages+=(redis-server)
  command -v redis-cli >/dev/null 2>&1 || missing_packages+=(redis-tools)
  command -v curl >/dev/null 2>&1 || missing_packages+=(curl)
  command -v git >/dev/null 2>&1 || missing_packages+=(git)
  command -v python3 >/dev/null 2>&1 || missing_packages+=(python3)
  compgen -G '/usr/include/python3.*/Python.h' >/dev/null || missing_packages+=(python3-dev)
  if (( ${#missing_packages[@]} )); then
    command -v sudo >/dev/null 2>&1 || { printf 'sudo is required to install: %s\n' "${missing_packages[*]}" >&2; exit 1; }
    sudo apt-get update
    sudo apt-get install -y build-essential "${missing_packages[@]}"
  fi
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

require_tool tmux "Install it with: sudo apt-get install -y tmux"
require_tool redis-server "Install it with: sudo apt-get install -y redis-server"
require_tool redis-cli "Install it with: sudo apt-get install -y redis-tools"
require_tool curl "Install it with: sudo apt-get install -y curl"
require_tool uv "Install it from https://docs.astral.sh/uv/."
require_tool python3 "Install it with: sudo apt-get install -y python3"

if ! [[ "$log_max_bytes" =~ ^[1-9][0-9]*$ ]]; then
  printf 'OPEN_RL_LOG_MAX_BYTES must be a positive integer\n' >&2
  exit 1
fi

read -r -d '' log_sink <<'PY' || true
import os
import sys

path = sys.argv[1]
limit = int(sys.argv[2])
log = open(path, "a", encoding="utf-8")
for line in sys.stdin:
  sys.stdout.write(line)
  sys.stdout.flush()
  log.write(line)
  log.flush()
  if log.tell() >= limit:
    log.close()
    os.replace(path, path + ".1")
    log = open(path, "a", encoding="utf-8")
log.close()
PY

timeslicer_backend="${OPEN_RL_ACCEL_TIMESLICER_BACKEND:-}"
if [[ -z "$timeslicer_backend" ]]; then
  if command -v cuda-checkpoint >/dev/null 2>&1; then
    timeslicer_backend="cuda"
  else
    timeslicer_backend="noop"
    printf 'warning: cuda-checkpoint is unavailable; using the no-op time-slicer\n' >&2
  fi
fi

printf 'OpenRL VM setup\n'
printf '  repository:   %s\n' "$repo_root"
printf '  model:        %s\n' "$base_model"
printf '  trainer GPUs: %s\n' "$trainer_gpus"
printf '  sampler GPUs: %s\n' "$sampler_gpus"
printf '  time-slicer:  %s\n' "$timeslicer_backend"
printf '  logs:         %s\n' "$log_dir"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h "$repo_root" "$(dirname "$tmp_dir")" 2>/dev/null | awk 'NR == 1 || !seen[$1]++'

run mkdir -p "$tmp_dir" "$log_dir"
run rm -f "$socket_path"

if [[ "$dry_run" != "1" ]]; then
  printf 'Installing the locked CUDA and vLLM environment...\n'
  uv sync --frozen --extra gpu --extra vllm
  env LD_LIBRARY_PATH="$runtime_library_path" uv run --frozen --no-sync python -c \
    'import torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA"; from server.vllm_sampler import VLLM_AVAILABLE; assert VLLM_AVAILABLE, "vLLM failed to import"'
fi

if tmux has-session -t "$session" 2>/dev/null; then
  if [[ "${OPEN_RL_RESTART:-0}" != "1" ]]; then
    printf "tmux session '%s' already exists; attaching without restarting it\n" "$session"
    if [[ "$attach" == "1" || ( "$attach" == "auto" && -t 1 ) ]]; then
      exec tmux attach-session -t "$session"
    fi
    exit 0
  fi
  run tmux kill-session -t "$session"
fi

redis_url="redis://127.0.0.1:$redis_port/0"
if redis-cli -p "$redis_port" ping >/dev/null 2>&1; then
  redis_command="printf 'Using existing Redis on port %s\\n' '$redis_port'; exec bash"
else
  printf -v redis_command 'set -o pipefail; redis-server --save "" --appendonly no --bind 127.0.0.1 --port %q 2>&1 | exec python3 -u -c %q %q %q' \
    "$redis_port" "$log_sink" "$log_dir/redis.log" "$log_max_bytes"
fi

printf -v timeslicer_command \
  'cd %q && set -o pipefail; uv run --frozen --no-sync python -m accel_timeslicer.serve socket=%q backend=%q 2>&1 | exec python3 -u -c %q %q %q' \
  "$repo_root" "$socket_path" "$timeslicer_backend" "$log_sink" "$log_dir/timeslicer.log" "$log_max_bytes"

printf -v gateway_command \
  'cd %q && set -o pipefail; env PYTHONUNBUFFERED=1 BASE_MODEL=%q CUDA_VISIBLE_DEVICES=%q SAMPLER_CUDA_VISIBLE_DEVICES=%q LD_LIBRARY_PATH=%q REDIS_URL=%q OPEN_RL_ENABLE_FFT=true SAMPLING_BACKEND=vllm OPEN_RL_TMP_DIR=%q OPEN_RL_LOG_DIR=%q OPEN_RL_ACCEL_TIMESLICER_SOCKET=%q uv run --frozen --no-sync python -m uvicorn server.gateway:app --host 127.0.0.1 --port %q 2>&1 | exec python3 -u -c %q %q %q' \
  "$repo_root" "$base_model" "$trainer_gpus" "$sampler_gpus" "$runtime_library_path" "$redis_url" "$tmp_dir" "$log_dir" "$socket_path" "$port" "$log_sink" "$log_dir/gateway.log" "$log_max_bytes"

printf -v gpu_command 'exec watch -n 1 nvidia-smi'

run tmux new-session -d -s "$session" -n redis "$redis_command"
run tmux new-window -t "$session" -n timeslicer "$timeslicer_command"
run tmux new-window -t "$session" -n gateway "$gateway_command"
run tmux new-window -t "$session" -n gpu "$gpu_command"
run tmux select-window -t "$session:gateway"

if [[ "$dry_run" != "1" ]]; then
  deadline=$((SECONDS + ${OPEN_RL_STARTUP_TIMEOUT_SECONDS:-300}))
  until curl --silent --fail "http://127.0.0.1:$port/api/v1/healthz" >/dev/null; do
    if (( SECONDS >= deadline )); then
      printf 'Gateway did not become healthy. Last gateway output:\n' >&2
      tmux capture-pane -p -t "$session:gateway" -S -80 >&2 || true
      exit 1
    fi
    sleep 1
  done
fi

printf '\nOpenRL is ready at http://127.0.0.1:%s\n' "$port"
printf 'Attach: tmux attach -t %s\n' "$session"
printf 'Stop:   tmux kill-session -t %s\n' "$session"
printf 'CLI:    uv run --no-sync openrl doctor --json\n'

if [[ "$dry_run" != "1" && ( "$attach" == "1" || ( "$attach" == "auto" && -t 1 ) ) ]]; then
  exec tmux attach-session -t "$session"
fi
