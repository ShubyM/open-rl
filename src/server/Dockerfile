FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS gpu-cr-builder

ARG GPU_CR_REF=e9bbb52e1f52986587fc631217c0f2b50b46245a
ARG GPU_CR_SHM_SIZE_GB=40

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/gpu-os/GPU-CR.git /tmp/GPU-CR \
    && cd /tmp/GPU-CR \
    && git checkout "$GPU_CR_REF" \
    && cmake -S . -B build -DGPU_VENDOR=NVIDIA -DSHM_SIZE_GB="$GPU_CR_SHM_SIZE_GB" \
    && cmake --build build -j"$(nproc)"

# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_FROZEN=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

# Install Python 3.12 via uv
RUN uv python install 3.12

# Install the NVIDIA CUDA checkpoint/restore utility used by the in-repo
# snapshot agent. The utility is driver-backed; keep the node driver compatible.
RUN curl -fsSL https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint \
    -o /usr/local/bin/cuda-checkpoint \
    && chmod +x /usr/local/bin/cuda-checkpoint

RUN mkdir -p /usr/local/lib/open-rl/gpu-cr
COPY --from=gpu-cr-builder /tmp/GPU-CR/build/vGPU-NVIDIA.so /usr/local/lib/open-rl/gpu-cr/vGPU-NVIDIA.so
COPY --from=gpu-cr-builder /tmp/GPU-CR/build/cr_client /usr/local/bin/cr_client
COPY --from=gpu-cr-builder /tmp/GPU-CR/build/multi_cr_client /usr/local/bin/multi_cr_client
ENV GPUCR_PRELOAD=/usr/local/lib/open-rl/gpu-cr/vGPU-NVIDIA.so

WORKDIR /app

# Copy dependency files
COPY pyproject.toml .
COPY uv.lock .

# Configure uv for Docker and caching
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies using uv with BuildKit cache
# --frozen ensures it uses the uv.lock exactly
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra gpu --extra vllm --extra cluster

# Copy source code
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra gpu --extra vllm --extra cluster

# TODO: this is a hack until vllm is fixed upstream
# Apply Gemma 4 LoRA deduplication patch
RUN uv run python /app/src/server/scripts/patch_vllm_lora_dedup.py

# Expose ports (Gateway: 8000, vLLM Worker: 8001)
EXPOSE 8000
EXPOSE 8001

# Start the Gateway using uv run to ensure the venv is active
CMD ["uv", "run", "uvicorn", "server.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
