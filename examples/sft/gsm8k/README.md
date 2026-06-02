# GSM8K full fine-tuning (via the tinker server)

Full-parameter SFT of a small model on [GSM8K](https://github.com/openai/grade-school-math)
(7,473 grade-school math problems with chain-of-thought), driven through the open-rl gateway
with the tinker SDK.

## Why full-FT goes through an env var

The tinker SDK's `create_lora_training_client()` is LoRA-only (it rejects `training_mode`). To do
full fine-tuning without changing the client, start the gateway with **`OPEN_RL_TRAINING_MODE=full`**,
which makes every created model a full-FT model. (The eventual plan is a
`create_fft_training_client()` client helper; until then this env var keeps the client unchanged.)

## Run (4-Pane Setup)

This setup demonstrates multi-tenant full fine-tuning (FFT) SFT jobs sharing a single GPU through explicit scheduler handoff. 

Assumes you start at the project root in all terminal panes:

### Pane 1: Scheduler
```bash
uv --project src/server run python src/server/trainer_scheduler.py
```

### Pane 2: Gateway
```bash
OPEN_RL_STORE_DIR=/tmp/open-rl/store \
OPEN_RL_SCHEDULER_SOCKET=/tmp/open-rl/trainer-scheduler.sock \
OPEN_RL_TRAINING_MODE=full \
BASE_MODEL=Qwen/Qwen2.5-0.5B \
uv --project src/server run uvicorn gateway:app --app-dir src/server --host 127.0.0.1 --port 9003
```

### Pane 3: SFT Job A
```bash
uv --project examples run python examples/sft/gsm8k/gsm8k_sft.py \
  --log-path=examples/sft/gsm8k/artifacts/job_a \
  --max-steps=20 \
  --base-model=Qwen/Qwen2.5-0.5B \
  --behavior-if-log-dir-exists=delete
```

### Pane 4: SFT Job B
```bash
uv --project examples run python examples/sft/gsm8k/gsm8k_sft.py \
  --log-path=examples/sft/gsm8k/artifacts/job_b \
  --max-steps=20 \
  --base-model=Qwen/Qwen2.5-0.5B \
  --behavior-if-log-dir-exists=delete
```

---

Training uses `tinker_cookbook.supervised.train`, so batching, LR scheduling, metric logging, and final checkpoint export are handled by the cookbook loop. The training script will print the directory where the final checkpoint is saved.

## Eval

Eval is decoupled (point a server at the saved dir). With vLLM (run from the project root):

```bash
python examples/sft/gsm8k/vllm_eval.py \
  --path <saved_dir> --data gsm8k_test.json
```

## Results (Qwen2.5-0.5B, 1 epoch, lr 2e-5, 0-shot exact-match on 250 test problems)

| | GSM8K |
|---|---|
| base (our 0-shot format) | ~1.5% |
| **after full-FT SFT** | **~36%** |


Files: `gsm8k_sft.py` (training via tinker server), `vllm_eval.py` (fast eval of the saved dir).
