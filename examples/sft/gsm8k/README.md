# GSM8K full fine-tuning

Full-parameter SFT of a small model on GSM8K, driven through the OpenRL gateway
with the Tinker SDK.

## Why full fine-tuning goes through dedicated workers

The public Tinker SDK entrypoint is still `create_lora_training_client()`. For
now, OpenRL routes that same client flow to a full fine-tuning worker when the
gateway is started with `OPEN_RL_ENABLE_FFT=true`.

## Run

This branch launches one full fine-tuning worker process per created model. That
worker shares requests and futures with the gateway through Redis.

Start from the repository root in separate terminals.

### Terminal 1: Redis

```bash
redis-server --port 6379 --save "" --appendonly no
```

### Terminal 2: Gateway

```bash
cd src/server
REDIS_URL=redis://127.0.0.1:6379 \
OPEN_RL_ENABLE_FFT=true \
BASE_MODEL=Qwen/Qwen2.5-0.5B \
SAMPLING_BACKEND=torch \
uv run --extra gpu python -m uvicorn gateway:app --host 127.0.0.1 --port 9003
```

### Terminal 3: SFT Job

```bash
uv --project examples run python examples/sft/gsm8k/gsm8k_sft.py \
  --log-path=examples/sft/gsm8k/artifacts/job_a \
  --max-steps=20 \
  --base-model=Qwen/Qwen2.5-0.5B
```

Training uses `tinker_cookbook.supervised.train`, so batching, LR scheduling,
metric logging, and final checkpoint export are handled by the cookbook loop.
The example deletes an existing log directory by default so stale checkpoint
metadata does not trigger resume. The training script prints
`eval_model_path=...` when it can resolve a final checkpoint path.

## Eval

Eval is decoupled from OpenRL. Point vLLM at the saved Hugging Face checkpoint:

```bash
python examples/sft/gsm8k/vllm_eval.py \
  --path <eval_model_path> \
  --data gsm8k_test.json
```

## Result

Single-job result from the original FFT prototype run:

| Setup | GSM8K |
| --- | --- |
| Qwen2.5-0.5B base, 0-shot exact match on 250 examples | ~1.5% |
| Qwen2.5-0.5B after full-FT SFT, 1 epoch, lr 2e-5 | ~36% |

Files:

- `gsm8k_sft.py`: training via the OpenRL/Tinker server.
- `vllm_eval.py`: fast eval of the saved checkpoint directory.
