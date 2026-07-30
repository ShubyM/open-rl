# Quickstart

Shortest path to a working training run. LoRA first — it is the stock path,
fits in memory without any of the FFT machinery, and is the right way to
validate the stack end to end. Full fine-tuning and multi-GPU come after and
have their own docs ([journey.md](journey.md) for the why,
[configuration.md](configuration.md) for every knob).

## Prerequisites

- Linux box with one CUDA GPU. gemma-4-e2b LoRA fits on 24GB; gemma-4-E4B-it
  is comfortable on 48GB+ (the sampler holds the base model in bf16).
- [`uv`](https://astral.sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- A Hugging Face token with the Gemma license accepted
  (`uv run hf auth login`), for gated model downloads.

## Step 1 — start the backend

**Path A — LoRA, single process (start here).** One command, one GPU: gateway,
LoRA trainer, and a torch sampler in one process. No Redis, no extra
terminals.

```bash
make server BASE_MODEL=google/gemma-4-E4B-it
```

The torch sampler is slow (plain HF `generate`), so keep demo runs small —
`max_tokens=128`-ish generations are snappy; multi-thousand-token LAB rollouts
will crawl. It is 100% correct, just not fast.

**Path B — LoRA with vLLM sampling (two GPUs).** Two variants; adapters are
applied per request either way, no reload cycle needed.

*B1 — bring your own `vllm serve` (simplest, no Redis).* Launch a stock
OpenAI-compatible vLLM server yourself, on whatever GPU/env you want:

```bash
make vllm-serve BASE_MODEL=google/gemma-4-E4B-it \
  VLLM_SERVE_ARGS="--max-model-len 32768 --language-model-only"
# equivalently: VLLM_ALLOW_RUNTIME_LORA_UPDATING=true vllm serve <model> \
#   --port 8001 --enable-lora --max-lora-rank 64 ...
```

then point the gateway at it:

```bash
SAMPLER_BASE_URL=http://127.0.0.1:8001 \
BASE_MODEL=google/gemma-4-E4B-it \
uv run --extra gpu python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003
```

The gateway registers each sampler snapshot as a fresh LoRA adapter via
`/v1/load_lora_adapter` (retiring the previous one) and samples through
`/v1/completions`, so a long-lived server never serves stale weights. The
vLLM process must share the trainer's filesystem (adapters live under
`$OPEN_RL_TMP_DIR/peft`). LoRA only — FFT checkpoint reloads need the
managed workers below.

*B2 — managed queue workers.* The gateway hosts the LoRA trainer in-process
and launches a per-model vLLM sampler worker on demand:

```bash
# Redis (queue + futures):
sudo service redis-server start

REDIS_URL=redis://127.0.0.1:6379 \
OPEN_RL_TIME_SLICING=off \
SAMPLER_CUDA_VISIBLE_DEVICES=1 \
VLLM_MAX_MODEL_LEN=32768 \
BASE_MODEL=google/gemma-4-E4B-it \
uv run --extra gpu --extra vllm python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003
```

Sampler backend defaults to vLLM once `REDIS_URL` is set. The trainer uses
the gateway's GPU; `SAMPLER_CUDA_VISIBLE_DEVICES` pins the sampler worker to
its own. Multimodal base models are sampled text-only by default
(`OPEN_RL_VLLM_LANGUAGE_MODEL_ONLY=1` skips the vision tower's memory
budgets; set `0` to re-enable multimodal inputs). LoRA rank must be ≤ 64
(the sampler's `max_lora_rank`).

**Path C — full fine-tuning.** Same command as Path B plus
`OPEN_RL_ENABLE_FFT=true` and `VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM`
(mandatory for Gemma FFT — text-only checkpoints do not match the multimodal
graph without it). The worker manager launches a dedicated trainer process
and the sampler hot-reloads full checkpoints per revision
(`OPEN_RL_TIME_SLICING=off` assumes trainer and sampler get their own GPUs;
leave it on to share one GPU via cuda-checkpoint).

Sanity check before training:

```bash
curl -s http://127.0.0.1:9003/api/v1/healthz          # {"status":"ok"}
```

In Paths B and C, also check the sampler worker's log after the first model is
created: it must print `Engine initialized successfully`, **not**
`vllm not installed, bypassing real engine init` (mock mode returns dummy
tokens — never train against it).

## Step 2 — smoke test (math RL, no LAB dependencies)

```bash
export PATH=$PATH:$HOME/.local/bin
TINKER_API_KEY=tml-dummy-key TINKER_BASE_URL=http://127.0.0.1:9003 \
uv --project examples run python examples/autoresearch/recipes/math_rl/train_gemma.py \
  model_name=google/gemma-4-E4B-it renderer_name=gemma4 env=gsm8k \
  group_size=2 groups_per_batch=1 max_steps=1 max_tokens=128 \
  base_url=http://127.0.0.1:9003 save_every=0 eval_every=0 \
  behavior_if_log_dir_exists=delete log_path=artifacts/smoke
```

One step completing end to end (sample → reward → forward_backward →
optim_step) proves the whole stack.

## Step 3 — the Harvey LAB recipe

```bash
examples/harvey_labs/setup_lab.sh        # clones the fixed LAB fork + sandbox deps
export GEMINI_API_KEY=...                # rubric judge; training needs no key
```

Then follow the LoRA quickstart in
[examples/harvey_labs/README.md](../examples/harvey_labs/README.md) — it has a
one-task tiny run and the full config surface. One rule to remember: the
sampler's `VLLM_MAX_MODEL_LEN` and the client's `max_trajectory_tokens` must
agree (default recipe assumes Gemma's full 131,072; shrink both together on
smaller GPUs, e.g. 32768 for a first run).

## Step 4 — LoRA has headroom: use it

LoRA is not memory-bound the way FFT is (the base model is frozen; only
adapters train), so on any reasonable GPU you can raise throughput
immediately:

| Knob | Where | Effect |
| --- | --- | --- |
| `group_size` / `groups_per_batch` | recipe config | more rollouts per step; gradients accumulate over all of them in one optimizer step |
| `OPEN_RL_TRAIN_TOKEN_BUDGET` | trainer env | packs multiple examples into one padded forward/backward instead of sequential passes. Set to `max_seq_len × desired_batch`, e.g. `131072` packs 4×32K examples per pass |
| `VLLM_MAX_NUM_SEQS` | sampler env (Path B) | concurrent sequences in the sampler (default 64); raise if rollouts queue |
| `max_tokens` | recipe config | generation length per turn — the main lever on rollout wall-clock, especially on Path A's torch sampler |

Suggested first LoRA run (Path A, one 80GB GPU): `group_size=4
groups_per_batch=2`, `OPEN_RL_TRAIN_TOKEN_BUDGET=32768`, `max_tokens=256`.
Watch step time, then scale group sizes up — LoRA gradients are cheap; the
sampler is the bottleneck. (For FFT, none of this applies — batch size is a
memory negotiation; see [configuration.md](configuration.md) "Long-context
training mechanics".)

## Troubleshooting, 30 seconds each

- **Sampler log says `bypassing real engine init` (mock mode)** — vLLM failed
  to import in the sampler worker; check its first log lines (a wrong wheel
  variant shows `libcudart` ImportErrors). Never train against a mock sampler.
- **HF 403 on model download** — Gemma license not accepted for this token.
- **Port 9003 busy** — `make server` kills the old listener itself; anything
  else on the port, stop manually.
- **Garbled / nonsense generations** — see the garble table in
  [journey.md](journey.md); first checks are the mock flag above and (FFT only)
  a missing `VLLM_ARCHITECTURE_OVERRIDE`.
- **Judge errors / 429s in LAB runs** — rubric judge rate limits; lower
  parallelism or wait; training requests themselves are unaffected.
- **Unit tests** — `make test` (no GPU needed) should always be green.
