# Tinker Cookbook Recipes

The `tinker_cookbook` package ships ready-to-run recipes from Thinking Machines.
Open-RL exposes the Tinker-compatible API those recipes call, so you can point
the cookbook at a local Open-RL server.

## Setup

Install the example dependencies from this repository:

```bash
cd examples
uv sync
```

The examples environment already depends on `tinker==0.18.1`,
`tinker_cookbook`, `datasets`, `torch`, and plotting utilities.

## Start the Server

From the repository root, start one vLLM sampler and one Open-RL gateway, then
reuse that server for the cookbook commands below.

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 BASE_MODEL="Qwen/Qwen3-0.6B" uv run --extra vllm python -m vllm_sampler
```

In another shell:

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 \
BASE_MODEL="Qwen/Qwen3-0.6B" \
SINGLE_PROCESS=1 \
SAMPLER=vllm \
VLLM_URL=http://127.0.0.1:8001 \
TINKER_API_KEY=tml-dummy-key \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

CPU mode is useful for tiny model fixtures, but Qwen-sized cookbook runs should
use GPU/vLLM. The cookbook recipes use pure LoRA adapters; saving and loading
`train_unembed` modules is still a server-side TODO.

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
  save_every=20
```

For a very small local check, lower `batch_size`, use a small dataset patch or
fixture, and keep `max_length` short.

## RL Training Loop

`rl_loop` runs a GRPO-style loop with reward-centered advantages and importance
sampling loss.

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
  save_every=20 \
  eval_every=0 \
  num_groups_to_log=0
```

The default `rl_loop` dataset is GSM8K. For faster API smoke tests, use the
cookbook `ArithmeticDatasetBuilder` with a small `n_batches` value.
