# Configuration

Open-RL is configured with environment variables. The examples below use plain
shell commands so they work even if `make` is not installed. The root
`Makefile` wraps the same commands for convenience.

## Run outside Kubernetes

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Start the API server and trainer with the default torch sampling backend:

```bash
cd src/server
BASE_MODEL=google/gemma-4-e2b \
SAMPLING_BACKEND=torch \
uv run --extra cpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

Because `REDIS_URL` is unset, this starts the API server and trainer loop in one
process on the same workstation or VM.

For a separate vLLM sampler, use two terminals:

```bash
# Terminal 1: vLLM sampler
cd src/server
BASE_MODEL=google/gemma-4-e2b \
VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM \
CUDA_VISIBLE_DEVICES=0 \
uv run --extra vllm python -m vllm_sampler
```

```bash
# Terminal 2: API server and trainer
cd src/server
BASE_MODEL=google/gemma-4-e2b \
SAMPLING_BACKEND=vllm \
CUDA_VISIBLE_DEVICES=1 \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

The equivalent Makefile shortcuts are:

```bash
make server BASE_MODEL=google/gemma-4-e2b
VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM make vllm BASE_MODEL=google/gemma-4-e2b
make server BASE_MODEL=google/gemma-4-e2b SAMPLING_BACKEND=vllm
```

## Core variables

| Env var | Default | What it does |
| --- | --- | --- |
| `BASE_MODEL` | unset | Hugging Face model id loaded by the trainer and, when using vLLM, by the sampler. |
| `SAMPLING_BACKEND` | `torch` locally, `vllm` when distributed | Sampling backend selector. `torch` samples in the training process. `vllm` forwards sampling requests to a vLLM worker. |
| `REDIS_URL` | unset | Enables distributed mode by switching the request store to Redis. Leave unset for a single-machine run. |
| `VLLM_URL` | `http://127.0.0.1:8001` | API server URL for the vLLM worker when `SAMPLING_BACKEND=vllm`. |

## Server paths

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Root directory for adapter snapshots under `peft/` and saved states under `checkpoints/`. |
| `CUDA_VISIBLE_DEVICES` | unset | Standard PyTorch GPU selector. Use different devices when the vLLM worker and trainer run on separate GPUs. |

## Weight sync

Most runs do not need weight-sync-specific configuration. When `SAMPLING_BACKEND=vllm`
or `VLLM_URL` is set, the trainer publishes a durable PEFT adapter snapshot and
notifies the Open-RL vLLM worker with a versioned adapter reload request. This
works with a shared `OPEN_RL_TMP_DIR`, including mounted storage in Kubernetes.

The practical required variables are:

| Env var | Default | What it does |
| --- | --- | --- |
| `VLLM_URL` | `http://127.0.0.1:8001` | Address of the Open-RL vLLM worker. |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Shared local or mounted root for checkpoints, adapter snapshots, and vLLM materialized LoRA adapters. |

Advanced/debug-only knobs:

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_WEIGHT_SYNC_TRANSPORT` | `vllm_lora_adapter_reload` | Override the sync transport. `vllm_lora_tensors_http` sends a safetensors payload in the control body for small/debug payloads; `vllm_lora_tensors_shm` uses POSIX shared memory and requires colocated processes with a shared IPC namespace. |
| `OPEN_RL_WEIGHT_SYNC_TIMEOUT` | `30.0` | Timeout, in seconds, for the trainer to notify the vLLM worker. |
| `OPEN_RL_WEIGHT_SYNC_STRICT` | `0` | `1` makes a failed vLLM sync fail the sampler save request instead of falling back to adapter checkpoint state. |
| `OPEN_RL_WEIGHT_SYNC_TENSOR_DTYPE` | unset | Cast tensors before sending, for example `float16` or `bfloat16`, if the inference runtime requires it. |
| `OPEN_RL_WEIGHT_SYNC_CHECKSUM` | `0` | Computes and verifies hot-path tensor checksums. Durable checkpoints keep their own manifest and payload metadata. |

## vLLM variables

| Env var | Default | What it does |
| --- | --- | --- |
| `MOCK_VLLM` | `0` | `1` starts the vLLM worker without initializing the engine, useful for local API debugging. |
| `VLLM_ARCHITECTURE_OVERRIDE` | unset | Optional architecture override passed to the in-repo vLLM worker. Gemma 4 examples use `Gemma4ForCausalLM`. |

## Client variables

| Env var | Default | What it does |
| --- | --- | --- |
| `TINKER_BASE_URL` | `http://127.0.0.1:9003` | Base URL used by example clients and scripts. |
| `TINKER_API_KEY` | `tml-dummy-key` | Passed through to the Tinker SDK. Local Open-RL does not enforce auth. |
| `HF_TOKEN` | unset | Required for gated Hugging Face models. `uv run hf auth login` is the easiest setup path. |
| `ENABLE_GCP_TRACE` | `0` | `1` exports OpenTelemetry traces to Google Cloud Trace. |
| `ENABLE_CONSOLE_TRACE` | `0` | `1` prints trace spans to stdout for debugging. |

## Distributed deployment

Kubernetes deployment manifests set these variables in pod specs. The important split is:

```bash
# API server pod
REDIS_URL=redis://redis-service:6379 \
VLLM_URL=http://vllm-service:8001 \
BASE_MODEL=google/gemma-4-e2b \
uv run uvicorn src.gateway:app --host 0.0.0.0 --port 8000
```

```bash
# Trainer worker pod
REDIS_URL=redis://redis-service:6379 \
VLLM_URL=http://vllm-service:8001 \
BASE_MODEL=google/gemma-4-e2b \
uv run python -m src.clock_cycle
```

```bash
# vLLM worker pod
BASE_MODEL=google/gemma-4-e2b \
uv run uvicorn src.vllm_sampler:app --host 0.0.0.0 --port 8001
```
