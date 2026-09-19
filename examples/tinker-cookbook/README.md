# Tinker Cookbook Recipes

Since OpenRL implements Tinker-compatible APIs, you can use
[`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook)
recipes with OpenRL endpoints.

## Setup

Assuming you have cloned this repository, install the example dependencies:

```bash
cd examples
uv sync
```

If you want to try other recipes, you may need to install other extras or dependencies.
The GSM8K recipe (`tinker_cookbook.recipes.rl_loop`) needs the cookbook's `math-rl`
extra, which the pin above already includes.

### Client SDK versions

Tinker SDK 0.25.0 and later send training requests as protobuf and only read
training and sampling results as protobuf. The API server accepts both encodings, so any
SDK from 0.23 onward works, including the one a fresh `uv sync` of the upstream
`tinker-cookbook` repository resolves. `docs/tinker-client-compatibility.md` lists
the supported client methods for the SDK version it was generated from.

### Models

The upstream cookbook resolves a renderer from its own model registry and raises
`KeyError` for models that are not in it. Small Qwen models that OpenRL runs on a
single L4 and that the registry knows are `Qwen/Qwen3-4B-Instruct-2507` (renderer
`qwen3_instruct`) and, on larger GPUs, `Qwen/Qwen3-8B` (renderer `qwen3`). Recipes
whose default renderer is hardcoded for a different family, such as
`preference.shorter.train`, need `renderer_name=qwen3_instruct` on the command line.

## Start the Server

From the repository root, start one vLLM sampler and one OpenRL API server on
separate GPUs. These examples are written for two L4 GPUs or better.

```bash
CUDA_VISIBLE_DEVICES=0 BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" uv run --extra vllm python -m server.vllm_sampler
```

In another shell:

```bash
CUDA_VISIBLE_DEVICES=1 \
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
SAMPLING_BACKEND=vllm \
VLLM_URL=http://127.0.0.1:8001 \
TINKER_API_KEY=tml-dummy-key \
uv run --extra gpu python -m uvicorn server.api_server:app --host 127.0.0.1 --port 9003
```

CPU mode is useful for tiny model fixtures, but Qwen-sized cookbook runs should
use GPU/vLLM.

## Checkpointing Limitation

OpenRL does not yet implement full Tinker-compatible durable checkpoint management. For recipes that expose periodic checkpoint saves, set `save_every=0`; See [gke-labs/open-rl#83](https://github.com/gke-labs/open-rl/issues/83) for more details.

## Supervised Learning Loop

`sl_loop` fine-tunes on the No Robots chat dataset with cross-entropy loss. You can run it by moving into the `examples` directory:

```bash
cd examples
TINKER_API_KEY=tml-dummy-key uv run python -m tinker_cookbook.recipes.sl_loop \
  base_url=http://127.0.0.1:9003 \
  model_name="Qwen/Qwen3-4B-Instruct-2507" \
  log_path=artifacts/tinker-cookbook/sl_loop \
  save_every=0
```

![SFT Loss Curve](plots/sl_loss_plot.png)

## Shorter Response Preference RL Loop

`train` runs an ultra-fast GRPO-style reinforcement learning loop that optimizes the policy to generate highly compliant, short responses. You can run it by moving into the `examples` directory:

```bash
cd examples
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 TINKER_TELEMETRY=0 uv run python -m tinker_cookbook.recipes.preference.shorter.train \
  model_name="Qwen/Qwen3-4B-Instruct-2507" \
  batch_size=4 \
  group_size=4 \
  max_tokens=64 \
  max_steps=40 \
  log_path=artifacts/tinker-cookbook/shorter_rl \
  behavior_if_log_dir_exists=delete
```

![RL Length Curve](plots/rl_length_plot.png)
![RL Format Curve](plots/rl_format_plot.png)
