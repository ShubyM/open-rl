# FunctionGemma Supervised Fine-Tuning Guide

This guide shows how to run the local FunctionGemma SFT demo.

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   uv sync --project src/server --extra cpu
   uv sync --package open-rl-client
   ```
2. **Accept the model terms**: [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it)
3. **Authenticate with Hugging Face** (required for gated models):
   ```bash
   uv run --package open-rl-client hf auth login
   ```

## Running the Training Server

Start the local server preloaded with FunctionGemma:
```bash
make server BASE_MODEL=google/functiongemma-270m-it
```

## Running the SFT Script

Execute the training script:
```bash
uv run --package open-rl-client python examples/sft/function-gemma/functiongemma_sft.py
```

## Contents

* `functiongemma_sft.py`: The main training script.
* `README.md`: This documentation file.
