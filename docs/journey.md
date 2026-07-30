# From Teacher Traces to On-Policy GRPO: How We Got Here

This is the condensed record of the Harvey LAB training effort: each approach we
took, where it broke, and what replaced it. The deep dives live in
[fft/fft.md](fft/fft.md) (architecture),
[fft/single-h100-long-context.md](fft/single-h100-long-context.md) (memory
measurements), and [configuration.md](configuration.md) (every knob). This page
is the map.

## Phase 1: teacher traces + SFT — and why it collapsed

The first plan was distillation: collect traces from a strong teacher (Gemini
Flash) solving LAB tasks, keep the good ones, and SFT gemma-4-E4B on them
(`experiments/lab-traces` on the lab-rl branch). The collector only kept
attempts that passed every LAB scoring check and stored full patched
transcripts.

Three things made this far more complicated than it sounds:

1. **Yield and judge fragility.** Only passing attempts count, so throughput is
   gated on teacher pass rate — and on the judge itself, which needed its own
   hardening series (JSON parsing failures, 429 retries, thinking-mode
   artifacts, deliverable naming). A meaningful share of the engineering went
   into the measurement apparatus, not the model.
2. **Rate limits.** The teacher tier allowed 2M input tokens/min; multi-turn
   agentic tasks over document workspaces burn that fast, capping parallelism
   at 1–2 workers.
3. **The killer: context alignment.** Full agentic transcripts routinely
   exceed the trainable context, so traces must be *compacted* — tool outputs
   dropped or summarized — before they fit an SFT sequence. But then three
   context budgets have to line up: what the teacher saw at collection, what
   compaction preserves, and what the student trains on. Every compaction
   choice changes the training distribution away from what the student will
   actually see at inference. We never got a compaction scheme we trusted; each
   fix reshuffled the data and invalidated comparisons.

Verdict: a brittle pipeline where every improvement required re-collecting, and
a distribution-mismatch problem baked into the design. We pivoted.

## Phase 2: live-rollout GRPO

Instead of teaching from static traces, sample the policy itself on LAB tasks,
score terminal states with the rubric judge, and run GRPO
(`examples/harvey_labs`, on tinker-cookbook's loop). This dissolves the Phase 1
problems by construction: there is no compaction step because the model trains
on exactly the tokens it sampled (token-in/token-out), and context budgets
can't misalign because there is only one — whatever the policy produces.

The price is that you now need a *system*: a trainer and a sampler that stay
on-policy with each other, at 131K context, on hardware that doesn't naturally
fit the model. That bought us two wars.

## Phase 3: the memory war

A full fine-tuning step has four tenants competing for GPU memory: bf16 params,
bf16 grads, fp32 AdamW moments (8 bytes/param — the largest), and activations
that grow linearly with sequence length. For a 9B model the first three alone
are ~108GB; an 80GB H100 is over budget before the first token. Every fix below
evicts or shrinks exactly one tenant; measured ceilings are in
[fft/single-h100-long-context.md](fft/single-h100-long-context.md).

| OOM / limit we hit | Cause | Fix |
| --- | --- | --- |
| Logits OOM at long context | naive loss materializes `[batch, seq, 262K-vocab]` logits — ~64GB bf16 at 131K | fused chunked logprob head: project hidden states through the LM head in checkpointed chunks, reducing to `logit[target] − logsumexp`; full logits never exist (`OPEN_RL_FUSED_LOGPROB`) |
| Attention OOM on Gemma | Gemma's 512-dim global heads fall off SDPA's fused kernels; the math fallback materializes a quadratic attention matrix | FlexAttention with low-resource 16×16 kernel tiles (default tiles need 256KB shared memory/block; sm_90 tops out at 227KB), plus `OPEN_RL_SDPA_NO_MATH=1` to make fallback loud |
| Activations OOM above ~49K | saved-for-backward tensors scale linearly with tokens | activation CPU offload — saved tensors stream to pinned host RAM and back (`OPEN_RL_ACTIVATION_CPU_OFFLOAD`, ~10% step cost) |
| AdamW moments don't fit at all | fp32 moments are ~2× model size and touched once per step | ZeRO-Offload-style CPU optimizer step: moments live in host RAM permanently (`OPEN_RL_OPTIM_CPU_STEP`, flat ~25–30s/step for 9B) |
| Fragmentation OOMs across variable-length steps | allocator can't reuse segments | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

Result: 131K-token Gemma-4-E4B and 147K-token Qwen3.5-9B steps on **one** 80GB
H100.

Multi-GPU came next (FSDP FULL_SHARD, `OPEN_RL_FSDP_WORLD_SIZE`): params,
grads, and moments shard across ranks, `forward_backward` stripes its datums
round-robin for real data parallelism, and checkpoints gather back to rank 0 in
HF format so the sampler pipeline is untouched. Two lessons paid for in H200
OOMs: **activations don't shard** (each rank holds its own datum's full
activation footprint, so long context still needs the offload knob), and every
single-GPU memory fix has to be *explicitly wired into the FSDP forward path* —
both the FlexAttention tiles and activation offload were silently bypassed
there at first. With sharding, the CPU optimizer step becomes unnecessary at
≥2 GPUs, and the wall-clock win at moderate context comes as much from turning
the offload knobs off as from parallel compute.

## Phase 4: the harness and serving war

### The LAB harness itself

Rewards are only as good as the harness. Stock upstream LAB had bugs that
corrupted the reward signal: `write` accepted fake binary deliverables, output
paths didn't normalize, nested deliverables never scored, and tool metrics were
wrong. Our fork (default clone in `setup_lab.sh`) carries the fixes, upstreamed
as LAB PRs #85–#90. Rewards from unfixed harnesses are not comparable — this
alone can look like "RL isn't working."

### Keeping the sampler on-policy

GRPO is only on-policy if vLLM serves the weights the trainer just wrote. The
mechanism: every `save_weights_for_sampler` writes a full HF-format checkpoint;
the gateway stamps each sampling request with that path plus a revision id; the
sampler lazily hot-reloads (sleep → `reload_weights` → wake) only when the
revision changes. GPU time-slicing (cuda-checkpoint via the accel-timeslicer)
lets trainer and sampler share one GPU, and `OPEN_RL_TIME_SLICING=off` disables
that whole lifecycle for dedicated-GPU or broken-driver (WSL, RunPod r550)
deployments.

Getting this *correct* was a war of its own, because every failure mode below
produced fluent-looking garbage rather than an error:

| Garbled-output bug | Root cause | Fix |
| --- | --- | --- |
| Gibberish after sleep/wake | sleep level 2 discards weights; wake without reload samples uninitialized memory | weight-preserving level 1 by default; waking from level 2 without a checkpoint now raises |
| Reloads silently no-op; model stuck at base weights | `language_model_only` only disables multimodal *inputs* — it doesn't change the graph, so text-only checkpoint keys don't match the multimodal graph | `VLLM_ARCHITECTURE_OVERRIDE=Gemma4ForCausalLM` is required for multimodal base models; the sampler now warns loudly when it's missing |
| Garble after OOM + resume | interrupted in-place saves leave dirs mixing old and new shards; vLLM 0.25.0 loads the mix silently | atomic checkpoint writes (stage + rename swap) and a vLLM build with the shard-integrity check, so corruption fails loudly |
| Intermittent garble at save boundaries | a new-revision request could sleep/hot-swap the engine underneath another request's live decode | drain gate: reloads wait for in-flight generations to hit zero |
| Garble with perfectly good checkpoints | CUDA-13 wheel variant fails to import on cu129 stacks; sampler silently fell back to mock mode returning zero tokens | pin the `.cu129` wheel variant; check `/healthz` for `"mock": true` |

The through-line: each fix converts silent wrongness into loud failure. The
remaining open item is a possible residue of vLLM-side parameters on Gemma's
KV-shared layers that HF checkpoints legitimately omit — benign if present
(those tensors are untrainable, so boot values are correct), tracked via the
reload log's "weights were not loaded" list.

## Where we are

One branch (`feat/harvey-lab`, based on upstream `fft`) now carries: the LAB
GRPO recipe, the long-context single-GPU memory work, sampler hot-reload with
optional time-slicing, multi-GPU FSDP with data parallelism, and the full
garble-hardening series — with the unit suite green and every knob documented
in [configuration.md](configuration.md). To run it, start with
[quickstart.md](quickstart.md).
