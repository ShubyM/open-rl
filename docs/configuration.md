# Configuration

OpenRL is configured with environment variables. The examples below use plain
shell commands so they work even if `make` is not installed. The root
`Makefile` wraps the same commands for convenience.

## Run outside Kubernetes

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Start the API server and trainer with the default torch sampling backend:

```bash
BASE_MODEL=google/gemma-4-e2b \
SAMPLING_BACKEND=torch \
uv run --extra cpu python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003
```

Because `REDIS_URL` is unset, this starts the API server and trainer loop in one
process on the same workstation or VM.

For vLLM-backed sampling there are two shapes (see
[quickstart.md](quickstart.md) Path B). With `SAMPLER_BASE_URL` set, the
gateway samples through an externally launched stock `vllm serve`
(OpenAI-compatible API, dynamic LoRA adapter registration; LoRA only, works
without Redis, and you fully control the vLLM process's environment — the
escape hatch when the managed workers inherit something unfortunate, e.g.
NCCL init trouble). Without it, the sampler runs in queue mode: the gateway
launches a worker per training model, for LoRA and FFT alike. LoRA workers
serve the base model and apply adapters per request; FFT workers hot-reload
full checkpoints per revision (FFT always uses queue mode). `make vllm`
remains only as a hand-launch debugging tool for queue workers
(`make vllm BASE_MODEL=... MODEL_ID=...`); `make vllm-serve` launches the
stock server for the `SAMPLER_BASE_URL` mode.

## Core variables

| Env var | Default | What it does |
| --- | --- | --- |
| `BASE_MODEL` | unset | Hugging Face model id loaded by the trainer and, when using vLLM, by the sampler. |
| `SAMPLING_BACKEND` | `torch` locally, `vllm` when distributed | Sampling backend selector. `torch` samples in the training process. `vllm` forwards sampling requests to a vLLM worker. |
| `REDIS_URL` | unset | Enables distributed mode by switching the request store to Redis. Leave unset for a single-machine run. |
| `OPEN_RL_FUTURE_TTL_S` | `300` | How long resolved request results stay readable by `retrieve_future` after a worker resolves them. |
| `VLLM_URL` | `http://127.0.0.1:8001` | Legacy HTTP sampler address, only used by the gateway's single-process preflight. Queue-mode sampler workers (the current path) do not listen on HTTP. |
| `OPEN_RL_FSDP_WORLD_SIZE` | `1` | Number of FSDP ranks for FFT trainer workers. Set it to the number of trainer GPUs (with matching `CUDA_VISIBLE_DEVICES`) and the worker manager launches the trainer under `torchrun` with FULL_SHARD FSDP; params, grads, and AdamW state are sharded across ranks, each `forward_backward` splits its datums round-robin across ranks for data-parallel compute, and checkpoints are gathered to rank 0 in HF format. `1` keeps the single-process trainer. |
| `OPEN_RL_TIME_SLICING` | `on` | `off` disables the checkpoint/time-slice lifecycle entirely: `time_slicer_client_from_env()` returns a no-op client, so FFT trainer workers skip snapshot/restore coordination and the vLLM sampler makes no time-slicer calls (and never sleeps its engine between batches). Use when trainer and sampler have their own GPUs, or on driver stacks where `cuda-checkpoint` fails (WSL, RunPod r550). Weight hot-reload stays active. |

## Server paths

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_TMP_DIR` | `/tmp/open-rl` | Root directory for adapter snapshots under `peft/` and saved states under `checkpoints/`. Both of the two settings below default to a subdirectory of this. |
| `OPEN_RL_SNAPSHOT_DIR` | `$OPEN_RL_TMP_DIR/peft` | Where the trainer writes sampler adapter snapshots. Rewritten every optimizer step and pruned to the last four, so nothing here needs to survive a restart — `/dev/shm/open-rl/peft` puts the handoff on tmpfs. **Node-local when set to tmpfs:** the sampler must share a kernel with the trainer. Samplers in their own pods need this on a shared filesystem. |
| `OPEN_RL_CHECKPOINT_DIR` | `$OPEN_RL_TMP_DIR/checkpoints` | Where training state (adapter plus optimizer) is written. This is what a resume reads, so on a preemptible machine point it at persistent storage — under the default, a reboot that clears `/tmp` leaves every `state_path` in `checkpoints.jsonl` dangling and the run restarts from scratch. |
| `OPEN_RL_TRAIN_TOKEN_BUDGET` | `0` | Maximum `batch_size * max_sequence_length` for padded trainer chunks inside one `forward_backward` request. `0` keeps the previous one-datum-at-a-time execution path. |
| `OPEN_RL_FUSED_LOGPROB` | `1` | Compute target logprobs by running the backbone and projecting through the vocabulary in chunks, so the full `[batch, seq, vocab]` logits tensor is never materialized. `0` falls back to full logits. |
| `OPEN_RL_LOGPROB_CHUNK` | `128` | Token rows per vocabulary projection chunk in the fused logprob path. Lower values trade speed for less peak memory on large-vocab models. |
| `OPEN_RL_ACTIVATION_CPU_OFFLOAD` | `0` | Store backbone tensors saved for backward in pinned CPU memory. Set `1` to enable, trading PCIe traffic and host RAM for lower VRAM at long sequence lengths. |
| `OPEN_RL_OPTIM_CPU_STEP` | `0` | Run the FFT optimizer step on the host: params and grads move to CPU for the step and the AdamW moments stay in host RAM. Frees ~2x model size of VRAM for activations (with activation offload this reaches 131K-token Gemma-4-E4B and 147K-token Qwen3.5-9B steps on one 80GB H100) for ~30s/step of PCIe traffic. |
| `OPEN_RL_LOG_CUDA_MEMORY` | `0` | Log allocated, reserved, free, and peak CUDA memory around forward, backward, and optimizer steps. OOMs always print a memory summary. |
| `OPEN_RL_ATTN_IMPLEMENTATION` | unset | Optional Transformers attention override. Leave unset to select tuned FlexAttention for Gemma 3n/4 and SDPA for other models such as Qwen 3.5. |
| `OPEN_RL_SDPA_NO_MATH` | `1` | Prevent quadratic SDPA math fallback. A `No available kernel` error then means all fused backends rejected the input; inspect the warnings immediately above it. |
| `CUDA_VISIBLE_DEVICES` | unset | Standard PyTorch GPU selector. Use different devices when the vLLM worker and trainer run on separate GPUs. |

## Long-context training mechanics

The four `OPEN_RL_*` memory knobs above exist because a full fine-tuning step
has four tenants competing for one GPU: bf16 params, bf16 grads, fp32 AdamW
moments (8 bytes/param — the largest), and activations that grow linearly with
sequence length. For Qwen3.5-9B the first three alone are ~108GB; an 80GB H100
is over budget before the first token. Each knob evicts or shrinks one tenant.
Measured ceilings and step timings are in
[docs/fft/single-h100-long-context.md](fft/single-h100-long-context.md).

### Fused logprob path (`OPEN_RL_FUSED_LOGPROB`, `OPEN_RL_LOGPROB_CHUNK`)

Training only needs the logprob of each *target* token, but the naive path
materializes logits for the whole vocabulary at every position — for Gemma 4's
262K vocabulary at 131K tokens that is a ~64GB bf16 tensor, larger than the
GPU. The fused path runs the transformer backbone once, then projects hidden
states through the LM head in chunks of `OPEN_RL_LOGPROB_CHUNK` token rows,
computing target logprobs per chunk so full logits never exist. Values and
gradients are exactly equal to the full-logits computation
(`tests/test_compute_target_logprobs.py` asserts both). Leave it on; set
`OPEN_RL_FUSED_LOGPROB=0` only to bisect a suspected bug in it, at short
context. Lower the chunk size if the projection phase is the OOM site.

### Activation CPU offload (`OPEN_RL_ACTIVATION_CPU_OFFLOAD`)

Activations saved for backward scale linearly with sequence length and are
what actually exhausts VRAM at long context. With this knob, each layer's
saved tensors stream to pinned host RAM during the forward pass and stream
back during backward, so the GPU holds roughly one layer's working set. Cost:
~10% on forward+backward — the PCIe copies mostly overlap compute — plus tens
of GB of host RAM. Enable it only above the baseline ceilings (Gemma-4-E4B
49K, Qwen3.5-9B 32K); below them it is pure overhead.

### CPU optimizer step (`OPEN_RL_OPTIM_CPU_STEP`)

AdamW's two fp32 moments are the single biggest tenant (~72GB for a 9B
model) and are only touched once per optimizer step. With this knob they live
permanently in host RAM: at step time grads move to the host, the update runs
on CPU, and updated params move back. Cost: a flat ~23-29s per step for a 9B
model regardless of sequence length — and GRPO takes one optimizer step per
rollout group, so this amortizes to a few percent of step time. Requires
roughly 150GB of host RAM for a 9B model. Same rule as above: off unless the
run actually needs the ceiling.

### Attention backend (`OPEN_RL_ATTN_IMPLEMENTATION`, `OPEN_RL_SDPA_NO_MATH`)

The trainer picks the attention implementation per architecture: FlexAttention
with low-resource tiles for Gemma 3n/4, SDPA (Flash) for models like Qwen 3.5.
This is not cosmetic — Gemma's global-attention layers and wide heads
(`head_dim` 512) fall off SDPA's fused kernels, and the math fallback
materializes a quadratic attention matrix that OOMs long before the ceilings
above. **Leave `OPEN_RL_ATTN_IMPLEMENTATION` unset for Gemma** so the tuned
FlexAttention path is selected; forcing `sdpa` on Gemma at long context is a
guaranteed OOM. `OPEN_RL_SDPA_NO_MATH=1` (default) makes that failure loud
instead of silent: rather than quietly falling back to the quadratic kernel, a
rejected input raises `No available kernel`, and the warnings directly above
the error say which fused backend refused and why.

### Sampler weight hot-reload and time slicing (`OPEN_RL_TIME_SLICING`)

Full fine-tuning rollouts are only on-policy if the vLLM sampler serves the
weights the trainer just wrote. Under `OPEN_RL_ENABLE_FFT=true` the gateway
stamps every sampling request with the full-model checkpoint path saved by
`save_weights_for_sampler` (under `OPEN_RL_TMP_DIR/sampler_full/`), and the
sampler reloads it in place — sleep, `reload_weights`, wake — only when the
revision differs from what is already loaded. Readiness is advertised through
the `open_rl:sampler_ready:<model_id>` Redis key.

When trainer and sampler share one GPU, the accelerator time-slicer (see the
Worker manager section below) serializes them: each side acquires the
accelerator before touching CUDA and is checkpointed on release, and the
sampler sleeps its engine between batches so the trainer gets the memory
back. Set `OPEN_RL_TIME_SLICING=off` to drop this whole lifecycle — no
snapshot/restore, no acquire/release, no engine sleeps — when each side has a
dedicated GPU, or when the driver stack cannot run `cuda-checkpoint` at all
(WSL, RunPod r550-era drivers). Hot-reload keeps working either way; `off`
only removes GPU time sharing.

## Sensible setups

**Gemma-4-E4B, long context (the Harvey LAB default, 131K):**

```bash
OPEN_RL_ENABLE_FFT=true REDIS_URL=redis://127.0.0.1:6379 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 OPEN_RL_OPTIM_CPU_STEP=1 \
VLLM_MAX_MODEL_LEN=131072 <launch gateway>
```

Do not set `OPEN_RL_ATTN_IMPLEMENTATION` (Gemma auto-selects FlexAttention).
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids allocator
fragmentation when sequence lengths vary between steps. Host needs ~150GB+
RAM. Set `OPEN_RL_LOG_CUDA_MEMORY=1` while tuning.

**Qwen3.5-9B:** same recipe for long context; additionally its gated-deltanet
(linear-attention) layers need the `flash-linear-attention` package with
TileLang JIT kernels — if kernel builds fail, align `nvcc` with the CUDA
toolkit fla was built against (12.9 verified).

**Short context (trajectories ≤32K):** leave both offload knobs off. At Qwen
32K the identical step goes from ~7s to ~36s with them on, for zero memory
benefit. The fused logprob path and attention selection stay on — they are
free.
## Worker manager

| Env var | Default | What it does |
| --- | --- | --- |
| `OPEN_RL_WORKER_MANAGER` | `local` | Trainer worker manager mode. Use `local` for subprocess workers or `kubernetes` for the DRA worker-manager deployment. |
| `OPEN_RL_ACCEL_TIMESLICER_SOCKET` | `/tmp/open-rl/accel-timeslicer.sock` | Unix socket path for a local accelerator time-slicer. Used when `OPEN_RL_ACCEL_TIMESLICER_HOST` is unset. |
| `OPEN_RL_ACCEL_TIMESLICER_HOST` | unset | Node-local accelerator time-slicer host for Kubernetes workers. When set, the worker uses TCP instead of the Unix socket; Kubernetes sets this from `status.hostIP`. |
| `OPEN_RL_ACCEL_TIMESLICER_PORT` | `9753` | Node-local accelerator time-slicer TCP port for Kubernetes workers. |

For local FFT subprocess mode, start `python -m accel_timeslicer.serve` before the
workers run. The local launcher tags each worker with a time-slice job id and
starts it in its own process group so the CUDA checkpoint backend can discover
the active GPU PIDs. Kubernetes deploys the equivalent process with the
`open-rl-accel-timeslicer` DaemonSet, which layers on top of the llm-d snapshot
backend by default for physical checkpoint/restore.

## vLLM variables

| Env var | Default | What it does |
| --- | --- | --- |
| `MOCK_VLLM` | `0` | `1` starts the vLLM worker without a real vLLM engine, useful for local API debugging. |
| `VLLM_ARCHITECTURE_OVERRIDE` | unset | Optional architecture override passed to the in-repo vLLM worker. Gemma 4 examples use `Gemma4ForCausalLM`. |
| `VLLM_ENABLE_MULTIMODAL` | `0` | By default the samplers pass `limit_mm_per_prompt={"image": 0, "video": 0}`. Text checkpoints published as `*ForConditionalGeneration` otherwise make vLLM reserve a multi-GiB encoder cache during startup that no OpenRL code path can use, which can OOM engine init. Set to `1` to restore stock vLLM behaviour. Note this does not change the constructed graph or its weight names: multimodal base models like Gemma additionally need `VLLM_ARCHITECTURE_OVERRIDE` (e.g. `Gemma4ForCausalLM`) so text-only FFT checkpoint keys match at reload; without it, reloads are silently skipped (`Following weights were not loaded from checkpoint` in the vLLM log) and the sampler keeps serving stale weights. |
| `OPEN_RL_VLLM_SLEEP_LEVEL` | `1` | Sleep level used when the FFT sampler yields the GPU. `1` keeps weights in host RAM; `2` discards them, forcing a checkpoint reload on every wake. |
| `OPEN_RL_VLLM_ATTENTION_BACKEND` | unset | Optional vLLM attention backend override passed through as the `attention_backend` engine arg. |

## Client variables

| Env var | Default | What it does |
| --- | --- | --- |
| `TINKER_BASE_URL` | `http://127.0.0.1:9003` | Base URL used by example clients and scripts. |
| `TINKER_API_KEY` | `tml-dummy-key` | Passed through to the Tinker SDK. Local OpenRL does not enforce auth. |
| `HF_TOKEN` | unset | Required for gated Hugging Face models. `uv run hf auth login` is the easiest setup path. |
| `ENABLE_GCP_TRACE` | `0` | `1` exports OpenTelemetry traces to Google Cloud Trace. |
| `ENABLE_CONSOLE_TRACE` | `0` | `1` prints trace spans to stdout for debugging. |

## Distributed deployment

Kubernetes deployment manifests set these variables in pod specs. The important split is:

```bash
# API server pod
REDIS_URL=redis://redis-service:6379 \
VLLM_URL=http://vllm-service:8001 \
BASE_MODEL=google/gemma-4-e2b \
uv run uvicorn server.gateway:app --host 0.0.0.0 --port 8000
```

```bash
# Trainer worker pod
REDIS_URL=redis://redis-service:6379 \
BASE_MODEL=google/gemma-4-e2b \
uv run python -m server.training_requests_processor
```

```bash
# vLLM worker pod (queue mode: drains one model's sampling queue from Redis)
REDIS_URL=redis://redis-service:6379 \
BASE_MODEL=google/gemma-4-e2b \
uv run --extra vllm python -m server.vllm_sampler --model-id <model-id>
```

In FFT mode the worker manager launches these per model automatically; the
manual invocation above is for static deployments.
