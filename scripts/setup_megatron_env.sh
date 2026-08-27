#!/usr/bin/env bash
# Build the interpreter the Megatron trainer runs on.
#
# Why this is not a uv extra in pyproject.toml. megatron-core, megatron-bridge
# and transformer-engine each pin torch and transformers ranges that disagree
# with the ones vLLM and the FSDP path need, so a real resolution either fails
# or drags every other backend's environment somewhere it should not go. They
# are installed --no-deps here instead: their actual runtime imports are the
# short list below, all of which we already satisfy, and none of the pins bite
# at import time. That is a deliberate trade -- the resolver's guarantee for a
# hand-checked one -- and it is why this lives in its own venv rather than in
# .venv, where a stray `uv sync` would undo it.
#
# transformer_engine_torch ships only as an sdist and compiles against the
# box's nvcc, which takes tens of minutes on a cold uv cache. Everything else
# is a wheel. Verified on the 8xH200 box (CUDA 12.9, driver cu129) with
# megatron-core 0.19.0 / megatron-bridge 0.6.1 / transformer-engine 2.9.0.
#
#   ./scripts/setup_megatron_env.sh              # builds ~/megatron-probe/.venv
#   MEGATRON_VENV=~/other ./scripts/setup_megatron_env.sh
#
# Point launch_work.sh at the result with MEGATRON_PYTHON=<venv>/bin/python.
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
VENV="${MEGATRON_VENV:-$HOME/megatron-probe/.venv}"
V="$VENV/bin/python"

if [ ! -x "$V" ]; then
  uv venv --python 3.12 "$VENV"
fi

# cu129 torch, matching the box toolkit so nvcc can build TE against it. This
# is the same build vLLM pulls, which is what lets one venv hold both.
uv pip install --python "$V" torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129

# What megatron-bridge and megatron-core actually import at runtime, plus the
# open-rl trainer's own dependencies (it runs server.training_requests_processor
# off PYTHONPATH, not an install).
uv pip install --python "$V" \
  "numpy<2.0.0" packaging ninja pybind11 absl-py einops peft accelerate \
  safetensors rich typing-extensions regex pyyaml tqdm omegaconf \
  "transformers==5.12.1" datasets \
  "chz>=0.4.0" fastapi pydantic "redis>=5.0.0" uvicorn httpx \
  opentelemetry-api opentelemetry-sdk opentelemetry-instrumentation-fastapi \
  opentelemetry-exporter-gcp-trace

uv pip install --python "$V" --no-deps megatron-core==0.19.0 megatron-bridge==0.6.1
uv pip install --python "$V" --no-deps transformer_engine_cu12==2.9.0 transformer-engine==2.9.0

# transformer_engine_torch compiles. Its setup.py finds cuDNN and the CUDA libs
# through the environment, and on this box those come from pip wheels under
# site-packages/nvidia rather than from the system CUDA install, so point the
# compiler at them explicitly. Without this it fails on a missing cudnn.h.
SP="$($V -c 'import site; print(site.getsitepackages()[0])')"
INC=$(find "$SP/nvidia" -maxdepth 2 -type d -name include | tr '\n' ':')
LIB=$(find "$SP/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')
export CUDNN_PATH="$SP/nvidia/cudnn"
export CPATH="$INC${CPATH:-}"
export LIBRARY_PATH="$LIB${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:-}"
MAX_JOBS=$(nproc) NVTE_FRAMEWORK=pytorch \
  uv pip install --python "$V" --no-deps --no-build-isolation transformer_engine_torch==2.9.0

# vLLM, for weight_transfer.py only: the trainer imports its NCCL group helper
# and the send side of the transfer engine. It never starts an engine.
uv pip install --python "$V" \
  "vllm==0.25.1.dev29+gf378f79b7.cu129" \
  --extra-index-url https://wheels.vllm.ai/f378f79b7c34d7ca94d53db34837929da2db03ed/cu129 \
  --index-strategy unsafe-best-match

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$REPO/src" "$V" - <<'EOF'
import server.training_requests_processor  # noqa: F401
import training.megatron_worker  # noqa: F401
import training.weight_transfer  # noqa: F401
from megatron.bridge import AutoBridge  # noqa: F401
from vllm.distributed.weight_transfer.nccl_common import (  # noqa: F401
  stateless_init_process_group,
)

print("megatron env OK")
EOF
echo "[setup] done: $V"
