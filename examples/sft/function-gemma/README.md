# FunctionGemma Supervised Fine-Tuning Guide

This guide shows how to run the local FunctionGemma SFT demo.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   uv sync --extra cpu
   uv --project examples sync
   ```
2. **Accept the model terms**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
3. **Authenticate with Hugging Face** (required for gated models):
   ```bash
   uv run --no-sync hf auth login
   ```

## Running the Training Server

Start the local server preloaded with FunctionGemma:
```bash
BASE_MODEL=google/functiongemma-270m-it SAMPLING_BACKEND=torch \
  uv run --no-sync python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003
```

## Running the SFT Script

Execute the training script:
```bash
cd examples/sft/function-gemma
uv run --no-sync python functiongemma_sft.py
```

## Contents

* `functiongemma_sft.py`: The main training script.
* `README.md`: This documentation file.
