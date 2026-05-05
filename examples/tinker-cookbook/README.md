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
SAMPLING_BACKEND=vllm \
VLLM_URL=http://127.0.0.1:8001 \
TINKER_API_KEY=tml-dummy-key \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

CPU mode is useful for tiny model fixtures, but Qwen-sized cookbook runs should
use GPU/vLLM. The cookbook recipes use pure LoRA adapters; saving and loading
`train_unembed` modules is still a server-side TODO.

## Checkpointing Limitation

Set `save_every=0` for cookbook runs so the recipe does not ask the server for periodic durable checkpoints. Sampler refreshes still happen between training steps through `save_weights_and_get_sampling_client`; this setting only disables the cookbook's extra checkpoint saves.

## Supervised Learning Loop

`sl_loop` fine-tunes on the No Robots chat dataset with cross-entropy loss. You can run it by moving into the `examples` directory:

```bash
cd examples
TINKER_API_KEY=tml-dummy-key uv run python -m tinker_cookbook.recipes.sl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-1.7B" \
  log_path=artifacts/tinker-cookbook/sl_loop \
  save_every=0
```

![SFT Loss Curve](plots/sl_loss_plot.png)

## Shorter Response Preference RL Loop

`train` runs an ultra-fast GRPO-style reinforcement learning loop that optimizes the policy to generate highly compliant, short responses. You can run it by moving into the `examples` directory:

```bash
cd examples
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 TINKER_TELEMETRY=0 uv run python -m tinker_cookbook.recipes.preference.shorter.train \
  model_name="Qwen/Qwen3-1.7B" \
  renderer_name="qwen3_disable_thinking" \
  batch_size=4 \
  group_size=8 \
  max_steps=20 \
  log_path=artifacts/tinker-cookbook/shorter_rl \
  behavior_if_log_dir_exists=delete \
  save_every=0
```

![RL Length Curve](plots/rl_length_plot.png)
![RL Format Curve](plots/rl_format_plot.png)


