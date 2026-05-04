# Tinker Cookbook Recipes on Open-RL

The `tinker_cookbook` package ships ready-to-run recipes from Thinking
Machines. Open-RL implements enough of the Tinker API to run the cookbook's
training and sampling loops against a local server. This guide documents the
commands to use and the current checkpoint limitation.

## Setup

Install the example dependencies from this repository:

```bash
cd examples
uv sync
```

This environment includes the cookbook `math-rl` extra because the RL command
below uses GSM8K. Other cookbook recipes may require different extras; add
those to `examples/pyproject.toml` before syncing if you want to try them.

## Start the Server

From the repository root, start one vLLM sampler and one Open-RL gateway, then
reuse that server for the cookbook commands below. For a single-GPU local run,
use a small model:

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 BASE_MODEL="Qwen/Qwen3-0.6B" uv run --extra vllm python -m vllm_sampler
```

In another shell:

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 \
BASE_MODEL="Qwen/Qwen3-0.6B" \
SAMPLING_BACKEND=vllm \
VLLM_URL=http://127.0.0.1:8001 \
TINKER_API_KEY=tml-dummy-key \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

CPU mode is useful for tiny model fixtures, but Qwen-sized cookbook runs should
use GPU/vLLM.

## Supervised Learning Loop

`sl_loop` fine-tunes on the No Robots chat dataset with cross-entropy loss and
writes JSONL metrics under `log_path`.

```bash
cd examples
TINKER_API_KEY=tml-dummy-key uv run --frozen python -m tinker_cookbook.recipes.sl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-0.6B" \
  log_path=artifacts/tinker-cookbook/sl_loop \
  batch_size=8 \
  max_length=2048 \
  lora_rank=16 \
  save_every=0
```

For a very small local check, lower `batch_size`, use a small dataset patch or
fixture, and keep `max_length` short.

## RL Training Loop

`rl_loop` runs a GRPO-style loop with reward-centered advantages and importance
sampling loss. The default `rl_loop` dataset is GSM8K, so this command uses the
cookbook `math-rl` extra.

```bash
cd examples
TINKER_API_KEY=tml-dummy-key uv run --frozen python -m tinker_cookbook.recipes.rl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-0.6B" \
  log_path=artifacts/tinker-cookbook/rl_loop \
  batch_size=2 \
  group_size=2 \
  max_tokens=64 \
  lora_rank=16 \
  save_every=0
```

For faster local runs, use the cookbook `ArithmeticDatasetBuilder` with a small
`n_batches` value.

## Qwen3-4B-Instruct Validation

We validated the cookbook RL path against `Qwen/Qwen3-4B-Instruct-2507` on a
two-GPU RunPod box with one GPU running vLLM and one GPU running the Open-RL
training gateway. Use at least two L4-class GPUs for this run; the sampler and
trainer should be on separate devices.

Sampler:

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 \
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
VLLM_MAX_MODEL_LEN=2048 \
VLLM_MAX_NUM_SEQS=16 \
VLLM_GPU_MEMORY_UTILIZATION=0.88 \
uv run --extra vllm python -m vllm_sampler
```

Gateway:

```bash
cd src/server
OPEN_RL_TMP_DIR=/tmp/open-rl-qwen4b-gsm8k \
CUDA_VISIBLE_DEVICES=1 \
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
SAMPLING_BACKEND=vllm \
VLLM_URL=http://127.0.0.1:8001 \
TINKER_API_KEY=tml-dummy-key \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

Recipe:

```bash
cd examples
TINKER_API_KEY=tml-dummy-key uv run --frozen python -m tinker_cookbook.recipes.rl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-4B-Instruct-2507" \
  log_path=/tmp/open-rl-logs-qwen4b-gsm8k \
  batch_size=128 \
  group_size=16 \
  max_tokens=256 \
  lora_rank=32 \
  save_every=0
```

## Checkpoint Limitation

Open-RL can refresh sampler weights during training, which is what the cookbook
RL loop needs for rollouts. It does not yet implement Tinker's durable
checkpoint system for periodic `save_state` outputs, `weights.download`, TTL,
publish/unpublish, or full resume parity.

Set `save_every=0` for cookbook runs so the recipe does not ask the server for
periodic durable checkpoints. Sampler refreshes still happen between training
steps through `save_weights_and_get_sampling_client`; this setting only
disables the cookbook's extra checkpoint saves.
