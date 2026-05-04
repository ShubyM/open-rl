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

If you want to try other recipes, you may need to install other extras or dependencies.

## Start the Server

From the repository root, start one vLLM sampler and one Open-RL gateway, then
reuse that server for the cookbook commands below.

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" uv run --extra vllm python -m vllm_sampler
```

In another shell:

```bash
cd src/server
CUDA_VISIBLE_DEVICES=0 \
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
SINGLE_PROCESS=1 \
SAMPLING_BACKEND=vllm \
VLLM_URL=http://127.0.0.1:8001 \
TINKER_API_KEY=tml-dummy-key \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

CPU mode is useful for tiny model fixtures, but Qwen-sized cookbook runs should
use GPU/vLLM. The cookbook recipes use pure LoRA adapters; saving and loading
`train_unembed` modules is still a server-side TODO.

## Checkpointing Limitation
Open-RL currently saves local PEFT adapter state, but it does not implement
Tinker's checkpoint service model: there is no checkpoint registry, no archive
URL flow, no publish/unpublish lifecycle, and no durable training-run/session
metadata. Cookbook paths that only need `save_state` and `load_state` can work,
but workflows that depend on true Tinker checkpoint management are out of scope
until Open-RL has real checkpointing semantics.

## Supervised Learning Loop

`sl_loop` fine-tunes on the No Robots chat dataset with cross-entropy loss. You can run it directly from the repository root:

```bash
TINKER_API_KEY=tml-dummy-key uv run \
  --with tinker \
  --with datasets \
  --with torch \
  python -m tinker_cookbook.recipes.sl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-4B-Instruct-2507" \
  log_path=artifacts/tinker-cookbook/sl_loop
```


## RL Training Loop

`rl_loop` runs a GRPO-style loop with reward-centered advantages and importance sampling loss. You can run it directly from the repository root:

```bash
TINKER_API_KEY=tml-dummy-key uv run \
  --with tinker \
  --with datasets \
  --with torch \
  python -m tinker_cookbook.recipes.rl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-4B-Instruct-2507" \
  log_path=artifacts/tinker-cookbook/rl_loop
```


