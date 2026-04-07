# Open-RL Client

This directory contains the client-side scripts and SDK for interacting with the Open-RL API.

## Getting Started with `uv`

This repo uses `uv` for both the client and server. From a fresh machine:

### 1. Install `uv`

On macOS:

```bash
brew install uv
```

Or on macOS/Linux with the upstream installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify:

```bash
uv --version
```

### 2. Clone the repo

```bash
git clone <your-open-rl-repo-url>
cd open-rl
```

### 3. Sync the two Python projects

The repo is split into:

- `server/` for the gateway, trainer worker, and sampler worker
- `client/` for demos and training scripts

Sync the client:

```bash
cd client
uv sync
cd ..
```

Then choose the server environment you need:

Gateway/core only:

```bash
cd server
uv sync
cd ..
```

Local single-process training flows such as Pig Latin SFT or FunctionGemma (CPU PyTorch):

```bash
cd server
uv sync --extra cpu
cd ..
```

Linux GPU/vLLM worker flows (CUDA PyTorch on Linux/WSL):

```bash
cd server
uv sync --extra gpu --extra vllm
cd ..
```

### 4. Run common workflows with `uv`

Start the gateway directly:

```bash
cd server
uv run uvicorn src.gateway:app --host 127.0.0.1 --port 8000
```

Start the local single-process Pig Latin server:

```bash
cd server
OPEN_RL_SINGLE_PROCESS=1 \
OPEN_RL_BASE_MODEL="Qwen/Qwen3-0.6B" \
SAMPLER_BACKEND=torch \
uv run --extra cpu uvicorn src.gateway:app --host 127.0.0.1 --port 9001
```

Start the Linux GPU/vLLM worker:

```bash
cd server
CUDA_VISIBLE_DEVICES=0 \
VLLM_MODEL="Qwen/Qwen3-4B-Instruct-2507" \
uv run --extra gpu --extra vllm python -m src.vllm_sampler
```

Run the Pig Latin SFT example:

```bash
cd client
uv run python -u piglatin_sft.py qwen base_url="http://127.0.0.1:9001"
```

Run the RLVR demo:

```bash
cd client
TINKER_BASE_URL="http://127.0.0.1:8000" \
uv run python rlvr.py --jobs 1 --steps 5 --base-model "Qwen/Qwen3-4B-Instruct-2507"
```

Run the Gemma 4 Text-to-SQL SFT + GRPO-style recipe on the local engine backend:

```bash
make run-text-to-sql-gemma4-server
make run-text-to-sql-sft-grpo
```

Run the same recipe with vLLM sampling on GPU:

```bash
# terminal 1
make run-text-to-sql-gemma4-vllm

# terminal 2
make run-text-to-sql-gemma4-server-gpu

# terminal 3
make run-text-to-sql-sft-grpo
```

This recipe reuses the existing SQLite execution checks for reward shaping:

- compile reward for any valid SQLite query
- execution-match reward when the generated query returns the same rows as the target SQL

If you are running against a hosted Tinker backend with checkpoint restore enabled, you can also save and resume state:

```bash
make run-text-to-sql-sft-grpo ARGS='save_sft_state_name=texttosql-gemma4-sft'
make run-text-to-sql-sft-grpo ARGS='phase=rl_only resume_state_path=tinker://...'
```

You can also use the repo Make targets if you prefer:

```bash
make run-server
make run-pig-latin-server
make run-pig-latin-sft
make run-rlvr
make run-text-to-sql-sft-grpo
```

Notes:

- `server/uv sync --extra cpu` installs the local training stack with CPU PyTorch for the single-process engine flows.
- `server/uv sync --extra gpu --extra vllm` adds the Linux-only vLLM worker dependencies and resolves CUDA PyTorch wheels.
- `vllm` is Linux-only here. On a Mac, use the gateway-only or single-process `cpu` flows unless you are running the Linux container story.
- `tinker-cookbook` is not required for the standard client demos in this repo.
- FunctionGemma examples require Hugging Face auth and model access.
- The local Open-RL backend in this repo supports saving sampler/state snapshots, but full checkpoint restore may depend on the backend you point the Tinker SDK at.

## Available Guides & Examples

Detailed walkthroughs for building with the Open-RL framework have been moved to the centralized docs folder:

- **Supervised Fine-Tuning**
  - [FunctionGemma Demo](../docs/guides/supervised/function-gemma.md)
  - [Pig Latin SFT](../docs/guides/supervised/pig-latin.md)
- **Reinforcement Learning**
  - [RLVR Demo](../docs/guides/reinforcement-learning/rlvr.md)
