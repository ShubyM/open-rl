# Long-context FFT on a single H100: Qwen3.5-9B and Gemma-4-E4B-it

What sequence length can one full-fine-tuning (FFT) training step actually reach on a
single 80 GB H100, for the two Harvey-labs candidate models? Measured 2026-07-12 on
an H100 (80GB HBM3, 196 GB host RAM, driver 550 / torch 2.11 cu129 /
transformers 5.13), driving the real `FFTTrainingWorker` path (`create_model` →
`forward_backward(cross_entropy)` → `optim_step`) with one synthetic datum per run and a
fresh process per point. "Steady state" = a second forward/backward/optim step with AdamW
state resident, which is the number that matters for a real GRPO loop.

## Headline results (max tokens in one datum, batch size 1)

| Configuration | Qwen3.5-9B (8.95 B) | Gemma-4-E4B-it (7.46 B text-only) |
|---|---|---|
| Repo default, single step | **98,304** ✓ / 114,688 ✗ | **98,304** ✓ / 114,688 ✗ |
| Repo default, steady state | **32,768** ✓ / 49,152 ✗ | **49,152** ✓ / 65,536 ✗ |
| + activation CPU offload + CPU AdamW, steady state | **147,456** ✓ / 163,840 ✗ | **131,072** ✓ = model's full context window |

"Repo default" = bf16, HF gradient checkpointing, fused chunked-logprob head
(`OPEN_RL_FUSED_LOGPROB=1`, chunk 128), sdpa-flash (Qwen) / flex_attention (Gemma),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, AdamW(`foreach=False`) on GPU.

The offload configuration is `OPEN_RL_ACTIVATION_CPU_OFFLOAD=1` (the repo's
`save_on_cpu(pin_memory=True)` wrap around the backbone forward) plus
`OPEN_RL_OPTIM_CPU_STEP=1`, which steps AdamW on the CPU (params+grads move to host for
the step, moments live in host RAM — ZeRO-Offload style).

**Gemma covers its entire 131k context window on one H100.** Qwen3.5 reaches 147k of its
262k architectural window; going further needs multi-GPU (see below).

Step cost at the ceiling (single H100, offload config):

| | forward+backward | CPU AdamW step |
|---|---|---|
| Qwen3.5-9B @ 147k | 45–85 s | ~31 s |
| Gemma-4-E4B @ 131k | 102–127 s | ~26 s |

For GRPO: each rollout in the group runs as its own micro-batch
(`OPEN_RL_TRAIN_TOKEN_BUDGET=0`), so the ceiling applies to the *longest single rollout*;
group size multiplies step time, not memory. A group of 8 at the ceiling ≈ 7–15 min per
optimizer step. The `importance_sampling`/`ppo` losses add only per-token fp32 vectors
(a few MB); ceilings are identical to the `cross_entropy` numbers above.

## Where the memory goes

Static (resident regardless of sequence length), measured:

| Component | Qwen3.5-9B | Gemma-4-E4B | Notes |
|---|---|---|---|
| bf16 weights | 16.7 GiB | 13.9 GiB | Gemma loaded text-only via `load_text_causal_lm` |
| bf16 gradients | 16.7 GiB | 13.9 GiB | allocated during backward, freed only by `zero_grad` |
| AdamW exp_avg + exp_avg_sq | 33.4 GiB | 27.8 GiB | bf16, created lazily at first `optim_step` |
| **Total steady static** | **66.8 GiB** | **55.7 GiB** | leaves 12 / 23 GiB for activations on GPU |

Per-token activation cost with gradient checkpointing ON, measured from
forward+backward peak deltas:

- Qwen: ~410–560 KB/token. Roughly 256 KB/token is the checkpoint-stored layer-boundary
  hidden states (4096 hidden × 2 B × 32 layers); the rest is *transient*: per-layer
  recompute graphs and the fla gated-deltanet fwd/bwd workspaces (fp32 recurrent/chunk
  states; individual allocations reach 3–3.75 GiB at 131k+).
- Gemma: ~580 KB/token. Boundary states are only ~210 KB/token (2560 hidden × 42 layers),
  but the per-layer-input embeddings (256/layer), AltUp-style extras, and
  flex-attention backward push the total higher.

Why each config tops out where it does:

1. **Steady-state default (33k / 49k):** optimizer states + grads + weights = 66.8 / 55.7
   GiB static, so activations get only ~12 / 23 GiB. This is the binding constraint for a
   real training loop and the first thing to fix.
2. **Single-step default (98k):** without optimizer states the activation budget is
   ~44 / 49 GiB; at ~0.4–0.6 MB/token that runs out just above 98k (both models OOM at
   114,688 in backward; Gemma missed 131k by 2.5 GiB).
3. **Offload config (147k / 131k):** stored activations move to pinned host RAM and
   optimizer state leaves the GPU entirely. What remains on GPU is weights + grads
   (33.4 / 27.8 GiB) plus the *non-offloadable transients* (~290–340 KB/token of
   recompute + kernel workspaces). Qwen OOMs at 163,840 when those transients
   (~47 GiB) no longer fit — the fla backward workspace is the residual hotspot.
   Host RAM peak is roughly 100–130 GB at the Qwen ceiling; the 196 GB box is fine, a
   smaller node would host-OOM.

Not a hotspot anymore, but worth remembering: the lm_head logits. The fused chunked
logprob head (`0c8860f` lineage) is what makes any of this possible — full
`[1, 131072, 248320]` fp32 logits would be ~130 GB on their own.

## "What FSDP did we need?"

None — and on one GPU, none is possible: FSDP sharding is a no-op at world size 1. The
single-GPU levers, in the order they bought us headroom:

1. Gradient checkpointing (repo default) — mandatory at these lengths.
2. Fused chunked logprob head (repo default) — mandatory for 248k/262k vocabs.
3. Flash-class attention everywhere: sdpa flash/efficient with the math backend hard-
   disabled (Qwen), flex_attention with the wide-head (head_dim 512) low-resource tiles
   from the uncommitted worktree changes (Gemma).
4. Activation CPU offload (`OPEN_RL_ACTIVATION_CPU_OFFLOAD=1`, already in the repo) —
   turns the stored ~210–256 KB/token into host traffic; costs ~1.5–2× on backward time.
5. CPU-resident AdamW (`OPEN_RL_OPTIM_CPU_STEP=1`) — deletes the entire 33 / 28 GiB
   optimizer-state term and makes steady state equal to single step. This is exactly
   what FSDP1 `CPUOffload(offload_params=True)` / FSDP2 `CPUOffloadPolicy` /
   ZeRO-Offload do; at world size 1 the trainer implements it as
   `model.to("cpu") → step → model.to("cuda")` (~30 s/step). A paged/8-bit optimizer
   would be the zero-latency alternative at a lower ceiling.

What a multi-GPU FSDP FULL_SHARD trainer (the `feat/lab-rl` branch carries one) buys
beyond this: it shards the *static* terms 1/N — at 4 ranks Qwen's 66.8 GiB drops to
~17 GiB/GPU — but it does **not** shard per-token activations, which after offload are
transient workspaces pinned to whichever GPU runs the layer. So FSDP alone raises the
Qwen ceiling only modestly past 164k. To train Qwen3.5 at its full 262k window you need
sequence/context parallelism (ring attention or equivalent splitting the *sequence*
across GPUs), which no branch implements today.

## Reproduction

On the box: harness and logs in `~/open-rl-bench/bench/` (`fft_step_bench.py`,
`sweep.sh`, `results.jsonl`, per-run logs under `logs/`). Code is this branch
(`feat/harvey-lab`) rsynced to `~/open-rl-bench`, reusing the `~/open-rl/.venv`
interpreter.

```bash
export CUDA_HOME=$HOME/cuda129 PATH=$HOME/cuda129/bin:$PATH   # see gotcha below
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OPEN_RL_LOG_CUDA_MEMORY=1
OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 OPEN_RL_OPTIM_CPU_STEP=1 ~/open-rl/.venv/bin/python -u \
  ~/open-rl-bench/bench/fft_step_bench.py --model Qwen/Qwen3.5-9B \
  --seq-len 147456 --steps 2
```

Environment gotchas hit on the way (relevant for any Hopper box running Qwen3.5):

- Qwen3.5's gated-deltanet needs `flash-linear-attention==0.5.1`. On Hopper with
  Triton ≥ 3.4 its backward *requires* the TileLang backend (upstream correctness issue
  fla#640) which JIT-compiles CUDA at runtime — both are declared in the `gpu` extra,
  but TileLang additionally needs a working nvcc ≥ ~12.6 on the machine.
- Every toolchain on the box failed that compile: system nvcc 12.4 is too old for the
  TMA instructions, and the venv's pip cu13 toolkit is both internally inconsistent and
  unloadable on the r550 driver. Fix: standalone CUDA 12.9 nvcc via
  `micromamba create -p ~/cuda129 -c nvidia -c conda-forge cuda-nvcc_linux-64=12.9
  cuda-cudart-dev=12.9 cuda-cccl=12.9` and `CUDA_HOME=$HOME/cuda129`.
- Kernel selection can also be forced with `FLA_TILELANG=0` (Triton fallback), but fla
  refuses it for the gated backward on Hopper — the nvcc fix is the real one.
