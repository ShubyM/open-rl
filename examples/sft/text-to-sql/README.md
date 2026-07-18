# Gemma 3 Text-to-SQL SFT

This directory contains the Gemma 3 supervised fine-tuning example for
Text-to-SQL tasks using OpenRL. The Gemma 4 SFT+RL recipe lives under
[`../../rl/text-to-sql`](../../rl/text-to-sql).

## Prerequisites

1. **Install dependencies**:
   Set up the server and client environments:
   ```bash
   uv sync --extra cpu
   uv --project examples sync
   ```

## Running the Training Server

Start the local single-process OpenRL server:
```bash
BASE_MODEL=google/gemma-3-1b-pt SAMPLING_BACKEND=torch \
  uv run --no-sync python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003
```

## Running the SFT Script

Execute the training script:
```bash
cd examples/sft/text-to-sql
uv run --no-sync python texttosql_sft.py gemma
```

## Contents

* `texttosql_sft.py`: Gemma 3 SFT script and shared Text-to-SQL data/eval helpers.
* `texttosql_sft_notebook.ipynb`: Jupyter notebook demonstrating the Gemma 3 SFT process.
* `README.md`: This documentation file.
