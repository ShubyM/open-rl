# GSM8K full fine-tuning (via the tinker server)

Full-parameter SFT of a small model on [GSM8K](https://github.com/openai/grade-school-math)
(7,473 grade-school math problems with chain-of-thought), driven through the open-rl gateway
with the tinker SDK — the same client path as `../pig-latin`, but full fine-tuning instead of LoRA.

## Why full-FT goes through an env var

The tinker SDK's `create_lora_training_client()` is LoRA-only (it rejects `training_mode`). To do
full fine-tuning without changing the client, start the gateway with **`OPEN_RL_TRAINING_MODE=full`**,
which makes every created model a full-FT model. (The eventual plan is a
`create_fft_training_client()` client helper; until then this env var keeps the client unchanged.)

## Run

```bash
# Terminal 1 — gateway (single-process, in-memory store, torch sampling)
cd src/server
OPEN_RL_TRAINING_MODE=full BASE_MODEL=Qwen/Qwen2.5-0.5B \
  python -m uvicorn gateway:app --host 127.0.0.1 --port 9003

# Terminal 2 — train (chz config)
python examples/sft/gsm8k/gsm8k_sft.py base_model=Qwen/Qwen2.5-0.5B epochs=1 lr=2e-5
```

Training uses `tinker_cookbook.supervised.train`, so batching, LR scheduling, metric logging, and
final checkpoint export are handled by the cookbook loop. The final checkpoint is recorded under
`artifacts/gsm8k_sft/checkpoints.jsonl`; for full-FT the sampler checkpoint path points at a
standard HF model directory.

## Eval

Eval is decoupled (point a server at the saved dir). With vLLM:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python examples/sft/gsm8k/vllm_eval.py \
  --path <saved_dir> --data gsm8k_test.json   # ~5s for 250 problems
```

## Results (Qwen2.5-0.5B, 1 epoch, lr 2e-5, 0-shot exact-match on 250 test problems)

| | GSM8K |
|---|---|
| base (our 0-shot format) | ~1.5% |
| **after full-FT SFT** | **~36%** |
| Qwen2.5-0.5B base *(published, 4-shot)* | 41.6% |
| Qwen2.5-0.5B-Instruct *(published)* | 49.6% |

In the published ~0.5B band (and our number is the harder 0-shot vs their 4-shot). Qwen3-0.6B-Base
under the same recipe reaches ~48%.

Files: `gsm8k_sft.py` (training via tinker server), `vllm_eval.py` (fast eval of the saved dir).
