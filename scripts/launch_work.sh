#!/usr/bin/env bash
# Bring up the whole Harvey-LAB RL stack in one tmux session called "work":
# a stock vLLM sampler on GPUs 1-N, the gateway + LoRA trainer on GPU 0, and
# typed (not yet run) train/eval commands.
#
#   MODEL=9b  ./scripts/launch_work.sh                # Qwen3.5-9B, full 262K window (default)
#   MODEL=27b ./scripts/launch_work.sh                # Qwen3.5-27B, 98K (measured H200 ceiling)
#   MODEL=e4b ./scripts/launch_work.sh                # Gemma-4-E4B-it, 131K (its full window)
#   TRAIN_GPUS=4 MODEL=27b ./scripts/launch_work.sh   # data-parallel LoRA trainer on 4 GPUs
#   WORKLOAD=sft ./scripts/launch_work.sh             # train window types sft.py (RL is the default)
#
# Windows:
#   sampler   vllm serve, data-parallel on the non-trainer GPUs   auto-starts
#   gateway   API gateway (in-gateway trainer when TRAIN_GPUS=1)  auto-starts (waits for sampler)
#   trainer   torchrun data-parallel LoRA trainer                 auto-starts (TRAIN_GPUS>1 only)
#   train     training command — TYPED, press Enter to launch
#   eval      eval_checkpoint command — TYPED, edit checkpoint= then Enter
#   gpu       nvidia-smi watch
#
# Re-running never kills anything: an existing session is attached as-is.
# The LAB checkout is bootstrapped via setup_lab.sh if missing. Logs tee to
# artifacts/box-logs/. Overridable: MODEL, TRAIN_GPUS, RUN_LABEL, GEN_TOKENS,
# JUDGE_MODEL (gemini-* via GEMINI_API_KEY, or glm-* via a self-deployed
# Vertex SGLang endpoint — needs VERTEX_JUDGE_ENDPOINT and ADC).
set -euo pipefail

SESSION=work
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LAB_ROOT="$REPO/examples/harvey_labs/harvey-labs"
LOGS="$REPO/artifacts/box-logs"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"

# Storage roots. These default together to /tmp/open-rl, which on a spot VM
# means a preemption clears the optimizer checkpoints and every state_path in
# checkpoints.jsonl dangles -- run20 came within one backup of losing five
# steps that way. Split them: snapshots are a trainer->sampler handoff worth a
# few hundred MB that is regenerated every optim step, so tmpfs is ideal and
# losing them costs nothing; checkpoints are the only thing a resume can read,
# so they belong on the boot disk. See src/training/paths.py.
#
# /dev/shm is node-local. This is correct only because the trainer and the
# samplers are processes on one box; if samplers ever move to their own nodes,
# OPEN_RL_SNAPSHOT_DIR has to go back to a shared filesystem.
export OPEN_RL_SNAPSHOT_DIR="${OPEN_RL_SNAPSHOT_DIR:-/dev/shm/open-rl/peft}"
export OPEN_RL_CHECKPOINT_DIR="${OPEN_RL_CHECKPOINT_DIR:-$HOME/open-rl-checkpoints}"
mkdir -p "$OPEN_RL_SNAPSHOT_DIR" "$OPEN_RL_CHECKPOINT_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already running — attaching (tmux kill-session -t $SESSION to reset)"
  if [ -t 1 ]; then
    exec tmux attach -t "$SESSION"
  else
    exit 0
  fi
fi

# Judge selection: gemini-* grades through the API key; glm-* grades through
# a self-deployed SGLang endpoint on Vertex, reached via ADC (no API key).
JUDGE_MODEL=${JUDGE_MODEL:-gemini-3.5-flash}
JUDGE_ENV=""
case "$JUDGE_MODEL" in
  glm*)
    if [ -z "${VERTEX_JUDGE_ENDPOINT:-}" ]; then
      echo "ERROR: JUDGE_MODEL=$JUDGE_MODEL needs the Vertex endpoint:" >&2
      echo "  export VERTEX_JUDGE_ENDPOINT=projects/<project>/locations/<region>/endpoints/<id>" >&2
      exit 1
    fi
    VERTEX_JUDGE_TOKENIZER=${VERTEX_JUDGE_TOKENIZER:-zai-org/GLM-5.2-FP8}
    JUDGE_ENV="VERTEX_JUDGE_ENDPOINT=$VERTEX_JUDGE_ENDPOINT VERTEX_JUDGE_TOKENIZER=$VERTEX_JUDGE_TOKENIZER"
    ;;
  *)
    if [ -z "${GEMINI_API_KEY:-}" ]; then
      echo "WARNING: GEMINI_API_KEY is not set — rubric grading will fail without it." >&2
    fi
    ;;
esac
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "WARNING: ANTHROPIC_API_KEY is not set — deliverable-name matching errors on mismatched filenames without it." >&2
fi

mkdir -p "$LOGS"
cd "$REPO"

# Disk preflight: podman layers + per-episode results exhaust small disks
# mid-run, which kills episodes in confusing ways. Refuse to start low.
GRAPHROOT=$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo "$HOME/.local/share/containers/storage")
for path in "$REPO" "$GRAPHROOT"; do
  [ -e "$path" ] || path=$(dirname "$path")
  AVAIL_GB=$(df -BG --output=avail "$path" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${AVAIL_GB:-0}" -lt 20 ]; then
    echo "ERROR: only ${AVAIL_GB:-?}G free on $path (< 20G). Free space before training:" >&2
    echo "  old runs:      rm -rf $LAB_ROOT/results/<old-run-id>" >&2
    echo "  podman layers: move graphroot in ~/.config/containers/storage.conf to a big disk" >&2
    exit 1
  fi
done

# Reap sandbox containers leaked by crashed episodes (normal teardown is
# handled by env cleanup; these are only crash leftovers).
LEAKED=$(podman ps -a --filter name=lab-sandbox --format '{{.Names}}' 2>/dev/null)
if [ -n "$LEAKED" ]; then
  echo "[work] removing leaked sandbox containers: $LEAKED"
  echo "$LEAKED" | xargs -r podman rm -f >/dev/null
fi

if [ ! -d "$LAB_ROOT" ]; then
  echo "[work] LAB checkout missing — running setup_lab.sh (clones the fork, installs pandoc/podman)..."
  ./examples/harvey_labs/setup_lab.sh
fi
echo "[work] LAB judge at: $(git -C "$LAB_ROOT" log --oneline -1 -- evaluation/judge.py 2>/dev/null || echo 'unknown')"

if [[ "$JUDGE_MODEL" == glm* ]]; then
  if ! "$LAB_ROOT/.venv/bin/python" -c "import google.cloud.aiplatform, transformers" 2>/dev/null; then
    echo "[work] installing GLM judge deps into the LAB venv..."
    "$LAB_ROOT/.venv/bin/pip" install -q google-cloud-aiplatform transformers
  fi
  if ! "$LAB_ROOT/.venv/bin/python" -c "import google.auth; google.auth.default()" 2>/dev/null; then
    echo "ERROR: no Application Default Credentials — the GLM judge cannot authenticate to Vertex." >&2
    echo "  Fix one of:" >&2
    echo "    gcloud auth application-default login --no-launch-browser" >&2
    echo "    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json" >&2
    echo "    (on a GCP VM: attach a service account with the Vertex AI User role)" >&2
    exit 1
  fi
  echo "[work] JUDGE_MODEL=$JUDGE_MODEL via $VERTEX_JUDGE_ENDPOINT (ADC ok)"
fi

# The 27B ceiling on a 141GB H200 is 98K tokens (measured, activation offload
# on); 9B fits its full 262K window with room to spare.
MODEL=${MODEL:-9b}
case "$MODEL" in
  9b)
    MODEL_NAME=Qwen/Qwen3.5-9B
    CONTEXT=${CONTEXT:-262144}
    GEN_TOKENS=${GEN_TOKENS:-32768}
    TASK_SET=${TASK_SET:-random}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen9b}
    ;;
  9b-128k)
    # Signal-hunting shape: big groups for GRPO contrast, the seeded random
    # 300/50 split for task diversity — and the 50-task eval's ~3,150
    # criteria cut eval noise to ~±1%, so small gains are detectable.
    MODEL_NAME=Qwen/Qwen3.5-9B
    CONTEXT=${CONTEXT:-131072}
    GEN_TOKENS=${GEN_TOKENS:-32768}
    TASK_SET=${TASK_SET:-random}
    BATCH_SIZE=${BATCH_SIZE:-8}
    ROLLOUTS=${ROLLOUTS:-6}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen9b-128k}
    ;;
  e4b)
    # Gemma-4-E4B against the same 8x6 / seed-242 shape as the 9B runs, so the
    # two curves are directly comparable. 131,072 is Gemma's full window, so
    # unlike the Qwen runs max_trajectory_tokens lands at the architectural
    # ceiling rather than a chosen budget -- the only headroom left to trade is
    # the generation reserve, which is what GEN_TOKENS below is about.
    MODEL_NAME=google/gemma-4-E4B-it
    CONTEXT=${CONTEXT:-131072}
    # message_env.py:94 reserves max_tokens from the trajectory budget on every
    # turn (observation + generation_reserve > max_trajectory_tokens), so the
    # real observation ceiling is CONTEXT - GEN_TOKENS. At 32K that fenced off a
    # quarter of Gemma's window before turn one: run20's longest observation was
    # 98,301 against a computed ceiling of 98,304, and no episode ever saw the
    # last 32K. The reserve was ~50x oversized -- over 902 run20 turns the
    # generation length was p50 336, p99 3,874, max 7,639, and nothing crossed
    # 8,192. 16,384 is 2.1x the observed max and still lifts the ceiling to
    # 114,688 (+16.7%). Not 8,192 (+25%): it also truncates nothing measured,
    # but leaves only 7% over the observed max, and a cap that clips a turn
    # mid-thought is the failure that cost runs 8-10 (see the 9b warning below).
    GEN_TOKENS=${GEN_TOKENS:-16384}
    TASK_SET=${TASK_SET:-random}
    BATCH_SIZE=${BATCH_SIZE:-8}
    ROLLOUTS=${ROLLOUTS:-6}
    RENDERER=gemma4
    # NB: do NOT set --hf-overrides Gemma4ForCausalLM here. That is the right
    # move for FFT, where text-only checkpoint keys must match the graph, but
    # LoRA is the opposite: _remap_adapter_to_hub_layout deliberately emits
    # hub-layout adapter keys (model.language_model.*) because vLLM resolves
    # adapters through the *multimodal* model's hf_to_vllm_mapper. Forcing the
    # text graph makes those keys match nothing, and vLLM applies no adapter at
    # all — sampling silently serves the base model for the whole run.
    RUN_LABEL=${RUN_LABEL:-lab-lora-gemma4-e4b}
    ;;
  12b)
    # Gemma-4-12B, the dense sibling of the E-series. Same 8x6 / seed-242 shape
    # as run20 so the curve is directly comparable to the E4B one.
    MODEL_NAME=google/gemma-4-12B-it
    # Trainer backward is the binding constraint, not the sampler and not the
    # model -- Gemma-4-12B itself does 262,144 (max_position_embeddings), so
    # everything below is a memory budget, not an architectural limit.
    #
    # Refit on 178 clean backward samples from run22 (the earlier 244-sample fit
    # off run21 is superseded; it read the ceiling ~26k tokens too low):
    #   peak_GiB ~= 0.7437 * ktokens + 22.46,  worst-case residual +8.63
    # against 139.80 GiB of H200, putting the OOM line near 146k tokens. The
    # worst backward actually observed was 109.84 GiB at 114,352 tokens -- 78.6%
    # of the card, ~30 GiB free -- so 114,688 was leaving real headroom unused.
    #
    # Why the two fits disagree so much: peak tracks the longest sequence in the
    # microbatch, not just its token total, because gradient checkpointing
    # recomputes a whole sequence at a time. run21 packed 4x32296 and peaked at
    # 136.64 GiB; run22 packs 1x79154 at a similar total and peaks at 109.84.
    # Same tokens, very different peaks. Treat any ceiling here as shape-
    # dependent and keep the margin.
    #
    # Raised 131,072 -> 143,360 after run27-29. A third fit, this one over 14,097
    # `[CUDA_MEMORY] phase=backward[NxM]:end` samples binned by N*M, reads
    #   p99 peak_allocated ~= 0.691 * ktokens + 27.4
    # which is the run22 line within noise (0.7437 / 22.46) and independently
    # puts the ceiling in the same place. At 143,360 the two fits give 126.5 and
    # 129.1 GiB of 139.80; adding run22's worst-case residual of 8.63 still lands
    # under the card. 147,456 sits *on* the run22 OOM line, so it is not taken.
    #
    # Read `peak_allocated` and nothing else. `allocated` at :end is post-free
    # (~25 GiB in every bin) and `peak_reserved` is the caching allocator's pool
    # high-water mark under expandable_segments (138.5-138.8 GiB in every bin,
    # including 0-16k). Both are flat in sequence length and both are wrong;
    # each one has already produced a confidently wrong ceiling here.
    #
    # Where the 0.69 GiB/ktoken goes: gradient checkpointing saves one input per
    # layer, 48 x 3,840 x 2 bytes = 0.37 GiB/ktoken, and the per-layer recompute
    # working set (intermediate_size 15,360) is most of the rest. The 262,144
    # vocab is not implicated -- the fused chunked logprob head is live
    # (OPEN_RL_FUSED_LOGPROB=1, chunk 128) with no fallback warnings. To go
    # meaningfully past this, activation offload (trainer_worker.py:423, today
    # mutually exclusive with the fused head) or sequence parallelism across the
    # six trainer ranks, not another tuning pass.
    #
    # Separately, 9 of the 14,097 backwards OOM'd (0.06%) inside
    # torch/utils/checkpoint.py recompute_fn, and they are length-INDEPENDENT --
    # one at 6,464 tokens while holding 123 GiB. Something transient reaches
    # ~110 GiB regardless of shape; cause unknown. That spike, not the fit, is
    # what the remaining ~10 GiB of margin is for.
    #
    # This raise moves SAMPLER_CONTEXT too, so it needs a vLLM restart. The KV
    # cache is 1,844,349 tokens, so the window costs nothing: 14.07x concurrency
    # at 131,072 becomes 12.86x at 143,360.
    #
    # The intercept is 22.46 GiB of *frozen base weights* replicated on all four
    # ranks (12B x 2 bytes bf16 = 22.35 GiB) -- the LoRA path has no FSDP wrap.
    # That, not optimizer state, is the only large thing left to attack; Adam's
    # moments for 131M rank-32 params are under 1 GiB (see OPEN_RL_OPTIM_CPU_STEP).
    #
    # A single trajectory cannot be split across microbatches -- the packer only
    # refuses to *add* to a non-empty batch -- so max_trajectory_tokens is bounded
    # by one backward, which is why CONTEXT and SAMPLER_CONTEXT must differ.
    CONTEXT=${CONTEXT:-143360}
    # The sampler window deliberately matches CONTEXT rather than reaching for
    # 12B's 256K. A bigger window would buy nothing: the KV cache is sized from
    # --gpu-memory-utilization, not --max-model-len (run21: 0.92 -> 128.62 GiB,
    # less 23.83 weights / 1.99 activation / 1.25 CUDA-graph / 0.22 non-torch =
    # 102.57 GiB = 1,837,317 tokens, the same number at any window), and the env
    # already caps a prompt at CONTEXT - GEN_TOKENS = 110,592, so nothing above
    # CONTEXT is reachable.
    #
    # Matching also keeps a guardrail. The trainer cannot split one trajectory
    # across microbatches, so a prompt longer than CONTEXT is an OOM later; with
    # the windows equal vLLM rejects it at the sampler with a 400 instead. That
    # is the failure that killed run21, and it arrived as an unrelated-looking
    # "did not produce one loss_fn_output per input datum" three hours in.
    SAMPLER_CONTEXT=${SAMPLER_CONTEXT:-$CONTEXT}
    # 16,384 was sized at ~2x E4B's observed max (p99 3,874, max 7,639). 12B is
    # a different animal: it writes whole documents inline as `write` arguments
    # and ran a single turn straight into the cap. In run21's step-0 eval that
    # killed 15 of 50 episodes -- stop_reason=="length" ends the episode at
    # message_env.py:119 with -0.1 and no grading at all, so 30% of the batch
    # was pure negative noise the judge never saw. Doubling to 32,768 leaves a
    # 98,304-token observation ceiling; context_overflow was only 1/50, so
    # spending trajectory budget on generation is the right side of the trade.
    #
    # !! run22 says this trade did not pay off and should probably be reverted.
    # Its step-0 eval, same weights and same 50 tasks as run21, moved the
    # failures rather than removing them: gen-cap kills 15 -> 8 but overflow
    # 1 -> 5 and parse 5 -> 7, so episodes reaching the judge went 29 -> 30 of
    # 50 and mean reward 0.1246 -> 0.1192 (noise). See docs/reports/run22/.
    # Worse, on the *training* split 17 of 48 rollouts still ran the full 32,768
    # and 29 of 48 returned -0.100, and each one costs twice the sampling and
    # backward of a 16,384 cap -- step 0 took ~2.5h. The verbosity is a tool-use
    # problem (12B writes whole documents inline as `write` arguments), and no
    # value of GEN_TOKENS fixes it. Consider 16384 plus a write-tool fix.
    #
    # run27 settles it: reverting to 16384. Over 21 training steps and 5 evals
    # the generation cap went almost unused -- max_tokens_reached was 0% of
    # episodes on both of the last two evals and under 6% on most training steps
    # -- while context_overflow cost a steady 10-19% of episodes. The 32,768
    # reserve was being held for a failure mode that had stopped happening, and
    # charged against the one that had not.
    #
    # run28 ran that experiment for 40 steps and refuted it, so this is back at
    # 32768. Halving the cap did kill context overflow (18% -> 2% of episodes)
    # but max_tokens_reached rose to fill the gap and overshot, 4% -> 32%. Total
    # truncation went from run27's ~14-22% to 34% and the fraction of episodes
    # reaching the judge never moved (0.64-0.74 in both runs). The generation
    # length distribution is adaptive, not fixed: p99 is only 9,260 tokens, so a
    # 16,384 cap should have bound on ~1% of turns and instead bound on a third
    # of episodes. Context overflow is the cheaper of the two failures.
    #
    # That leaves the observation ceiling at CONTEXT - 32,768, which the CONTEXT
    # raise above takes from 98,304 to 110,592 (+12.5%). The claim that used to
    # sit here -- that CONTEXT could not move, from a 6,419-sample fit of
    # 0.657 * ktokens + 5.4 on a 31.3 GiB baseline -- was measured off the wrong
    # allocator field and is retracted; see the CONTEXT block.
    GEN_TOKENS=${GEN_TOKENS:-32768}
    TASK_SET=${TASK_SET:-random}
    BATCH_SIZE=${BATCH_SIZE:-8}
    ROLLOUTS=${ROLLOUTS:-6}
    RENDERER=gemma4
    # !! VERIFY THE ADAPTER IS ACTUALLY APPLIED BEFORE TRUSTING A 12B CURVE.
    # E4B routes LoRA through the multimodal model's hf_to_vllm_mapper, which is
    # why _remap_adapter_to_hub_layout emits model.language_model.* keys. 12B is
    # encoder-free -- image and audio project straight into the decoder -- so it
    # is a different graph and there is no reason to assume the same key layout.
    # If the keys match nothing, vLLM applies no adapter and silently samples the
    # base model for the entire run, which is a curve that looks like "no
    # learning" rather than an error. Check step 0 sampling against the base
    # model before letting it run overnight.
    RUN_LABEL=${RUN_LABEL:-lab-lora-gemma4-12b}
    ;;
  27b)
    MODEL_NAME=Qwen/Qwen3.5-27B
    CONTEXT=${CONTEXT:-98304}
    # 32K tool results in a 98K window would let a few parallel document
    # reads overflow the whole trajectory budget; 16K is the proven value.
    GEN_TOKENS=${GEN_TOKENS:-16384}
    # Curated run-3/4 task lists so 27B numbers stay comparable across runs.
    TASK_SET=${TASK_SET:-bootstrap}
    RUN_LABEL=${RUN_LABEL:-lab-lora-qwen27b}
    ;;
  *)
    echo "Unknown MODEL=$MODEL (use 9b, 9b-128k, e4b, 12b, or 27b)" >&2
    exit 1
    ;;
esac
TOOL_TOKENS=${TOOL_TOKENS:-16384}
BATCH_SIZE=${BATCH_SIZE:-5}
ROLLOUTS=${ROLLOUTS:-2}
RENDERER=${RENDERER:-qwen3_5}
SAMPLER_EXTRA=${SAMPLER_EXTRA:-}

# Speculative decoding. Rollout generation is the wall-clock bottleneck on the
# 12B LAB runs: run22's step 0 took ~2.5h, and 17 of its 48 rollouts emitted the
# full 32,768-token cap one token per target forward pass. A draft model
# proposes SPEC_TOKENS at a time and the target verifies them in a single pass,
# so the win scales with exactly the long, low-entropy, document-shaped output
# that is making these runs slow.
#
#   SPEC_MODEL=google/gemma-4-12B-it-assistant ./scripts/launch_work.sh
#
# Off unless SPEC_MODEL is set, because the failure mode is a hard one: vLLM
# resolves the draft model at startup and exits if the name does not exist, and
# the sampler is the slowest part of the stack to bring back (~3-5 min to reload
# 12B across 4 ranks). Two things to confirm the first time this is used on a
# given box, neither of which can be checked from here:
#   - the draft repo actually exists and is pullable (it is not a HF-canonical
#     name pattern, so it may be an internal or gated artifact)
#   - the installed vLLM supports speculative decoding *together with*
#     --enable-lora and --data-parallel-size; several releases reject that
#     combination outright, and this stack needs both
# Verify against a throwaway serve before committing a run to it.
SPEC_MODEL=${SPEC_MODEL:-}
SPEC_TOKENS=${SPEC_TOKENS:-4}
if [ -n "$SPEC_MODEL" ]; then
  SAMPLER_EXTRA="$SAMPLER_EXTRA --speculative-config '{\"model\":\"$SPEC_MODEL\",\"num_speculative_tokens\":$SPEC_TOKENS}'"
  echo "[work] speculative decoding: $SPEC_MODEL, $SPEC_TOKENS draft tokens"
fi

# Deltanet is a Qwen3.5 hybrid-attention path; Gemma has no equivalent, and the
# import fails there for the boring reason that the module does not exist.
case "$MODEL_NAME" in
  Qwen/*)
    FP=$(uv run --no-sync python -c "from transformers.models.qwen3_5 import modeling_qwen3_5 as m; print(m.is_fast_path_available)" 2>/dev/null)
    if [ "$FP" != "True" ]; then
      echo "WARNING: Qwen deltanet fast path is NOT available — training runs the eager" >&2
      echo "         fallback (2-5x slower). Run ./scripts/setup_vm.sh to build causal-conv1d." >&2
    fi
    ;;
esac

WORKLOAD=${WORKLOAD:-rl}
if [ "$WORKLOAD" = "sft" ]; then
  # The trainer packs padded [max_len x count] forwards up to
  # OPEN_RL_TRAIN_TOKEN_BUDGET (= CONTEXT here). At 262K it packs two ~131K
  # SFT datums into one forward and OOMs; 163840 matches sft.py max_length,
  # so the worst packed forward stays at one max-size datum's scale.
  CONTEXT=${SFT_CONTEXT:-163840}
fi

# CONTEXT used to drive three unrelated limits at once: the sampler's window,
# the trainer's microbatch packing budget, and the episode's trajectory cap.
# They are bounded by different hardware -- KV cache on the sampler GPUs versus
# backward activations on the trainer GPUs -- and on 12B those bounds are an
# order of magnitude apart (1.8M tokens of KV against ~120K of backward). Tying
# them together meant every run had to pick the minimum and waste the rest.
#
# Both still default to CONTEXT, so every existing model case and every
# CONTEXT=... invocation behaves exactly as before; only a case or caller that
# sets them explicitly sees a difference.
SAMPLER_CONTEXT=${SAMPLER_CONTEXT:-$CONTEXT}
TRAIN_TOKEN_BUDGET=${TRAIN_TOKEN_BUDGET:-$CONTEXT}

# A trajectory the trainer cannot fwd_bwd in one microbatch is an OOM waiting
# for the one long episode that reaches it, and it fails mid-run rather than at
# launch. Catch it here instead.
if [ "$CONTEXT" -gt "$TRAIN_TOKEN_BUDGET" ]; then
  echo "WARNING: CONTEXT=$CONTEXT exceeds TRAIN_TOKEN_BUDGET=$TRAIN_TOKEN_BUDGET — a single" >&2
  echo "         max-length trajectory cannot be split across microbatches and will OOM" >&2
  echo "         the trainer. Lower CONTEXT or raise TRAIN_TOKEN_BUDGET." >&2
fi
echo "[work] MODEL=$MODEL -> $MODEL_NAME, context $CONTEXT (sampler $SAMPLER_CONTEXT, train budget $TRAIN_TOKEN_BUDGET), gen $GEN_TOKENS, tool $TOOL_TOKENS, batch ${BATCH_SIZE}x${ROLLOUTS}, log $RUN_LABEL"
if [ "$GEN_TOKENS" -lt 32768 ] && [[ "$MODEL" == 9b* ]]; then
  echo "WARNING: GEN_TOKENS=$GEN_TOKENS < 32768 — the 16K cap killed episodes mid-thought in runs 8-10" >&2
  echo "         (run 11's record needed 32K). Unset GEN_TOKENS or export GEN_TOKENS=32768." >&2
fi

# fla kernel backend by architecture: Hopper requires TileLang (Triton>=3.4
# dropped the deltanet path there); Blackwell runs the proven Triton backend
# (TileLang on sm_100 has produced misaligned-address kernel faults).
GPU0=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
case "$GPU0" in
  *H100*|*H200*) FLA_TILELANG=${FLA_TILELANG:-1} ;;
  *)             FLA_TILELANG=${FLA_TILELANG:-0} ;;
esac
echo "[work] GPU: $GPU0 -> FLA_TILELANG=$FLA_TILELANG"

# TRAIN_GPUS=1 (default): trainer runs inside the gateway on GPU 0.
# TRAIN_GPUS=N>1: dedicated torchrun trainer (data-parallel LoRA) on GPUs
# 0..N-1 via the Redis queue; the sampler shrinks to the remaining GPUs.
TRAIN_GPUS=${TRAIN_GPUS:-1}
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l); NUM_GPUS=${NUM_GPUS:-8}
SAMPLER_DP=$((NUM_GPUS - TRAIN_GPUS))
SAMPLER_DEV=$(seq -s, "$TRAIN_GPUS" $((NUM_GPUS - 1)))
TRAIN_DEV=$(seq -s, 0 $((TRAIN_GPUS - 1)))
QUEUE_ENV=""
if [ "$TRAIN_GPUS" -gt 1 ]; then
  # Ephemeral queue: no RDB snapshots or AOF — background persistence of
  # multi-MB training payloads is pure stall risk for zero value. Raise the
  # fd limit before daemonizing: redis derives maxclients from it, and every
  # pending gateway request holds one BLPOP connection.
  ulimit -n 65535 2>/dev/null || true
  pgrep -x redis-server >/dev/null || redis-server --daemonize yes --save '' --appendonly no --maxclients 8192
  QUEUE_ENV="REDIS_URL=redis://127.0.0.1:6379 OPEN_RL_EXTERNAL_TRAINER=1"
  echo "[work] TRAIN_GPUS=$TRAIN_GPUS -> torchrun trainer on GPUs $TRAIN_DEV, sampler DP$SAMPLER_DP on $SAMPLER_DEV"
fi

# AFFINITY=1: one single-GPU vllm serve per sampler GPU (ports 8000+i) and
# prefix-hash routing in the gateway, so every turn of an episode hits the
# instance that already holds its KV/prefix cache. Default: one DP server
# (per-request round-robin, no cross-turn cache reuse).
AFFINITY=${AFFINITY:-0}
if [ "$AFFINITY" = "1" ]; then
  SAMPLER_CMD=""
  SAMPLER_URLS=""
  for i in $(seq 0 $((SAMPLER_DP - 1))); do
    GPU_ID=$((TRAIN_GPUS + i))
    PORT=$((8000 + i))
    SAMPLER_URLS="$SAMPLER_URLS,http://127.0.0.1:$PORT"
    SAMPLER_CMD="$SAMPLER_CMD CUDA_VISIBLE_DEVICES=$GPU_ID VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
uv run --extra gpu --extra vllm --extra fastpath vllm serve $MODEL_NAME \
--port $PORT --enable-lora --max-lora-rank 64 --max-loras 2 --enable-prefix-caching \
--max-model-len $SAMPLER_CONTEXT --gpu-memory-utilization 0.92 \
--language-model-only $SAMPLER_EXTRA |& tee -a $LOGS/sampler-$i.log &"
  done
  SAMPLER_CMD="${SAMPLER_CMD} wait"
  SAMPLER_URLS="${SAMPLER_URLS#,}"
  SAMPLER_ENV="SAMPLER_BASE_URLS=$SAMPLER_URLS"
  LAST_PORT=$((8000 + SAMPLER_DP - 1))
  SAMPLER_WAIT="until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && curl -sf http://127.0.0.1:$LAST_PORT/v1/models >/dev/null 2>&1; do echo 'waiting for samplers...'; sleep 10; done"
  echo "[work] AFFINITY=1 -> $SAMPLER_DP single-GPU samplers on ports 8000-$LAST_PORT"
else
  SAMPLER_ENV="SAMPLER_BASE_URL=http://127.0.0.1:8000"
  SAMPLER_WAIT="until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do echo 'waiting for sampler...'; sleep 10; done"
  SAMPLER_CMD="CUDA_VISIBLE_DEVICES=$SAMPLER_DEV VLLM_ALLOW_RUNTIME_LORA_UPDATING=true \
uv run --extra gpu --extra vllm --extra fastpath vllm serve $MODEL_NAME \
--port 8000 --enable-lora --max-lora-rank 64 --max-loras 2 --enable-prefix-caching \
--data-parallel-size $SAMPLER_DP --api-server-count 1 \
--max-model-len $SAMPLER_CONTEXT --gpu-memory-utilization 0.92 \
--language-model-only $SAMPLER_EXTRA |& tee -a $LOGS/sampler.log"
fi

GATEWAY_DEV=0
[ "$TRAIN_GPUS" -gt 1 ] && GATEWAY_DEV=""
GATEWAY_CMD="$SAMPLER_WAIT; \
CUDA_VISIBLE_DEVICES=$GATEWAY_DEV $QUEUE_ENV FLA_TILELANG=$FLA_TILELANG BASE_MODEL=$MODEL_NAME $SAMPLER_ENV \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPEN_RL_TRAIN_TOKEN_BUDGET=$TRAIN_TOKEN_BUDGET OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 \
OPEN_RL_OPTIM_CPU_STEP=${OPTIM_CPU_STEP:-1} \
OPEN_RL_LOG_CUDA_MEMORY=1 \
uv run --extra gpu --extra vllm --extra fastpath python -m uvicorn server.gateway:app --host 127.0.0.1 --port 9003 |& tee -a $LOGS/gateway.log"

# Overlapping fwd_bwd with sampling cuts ~40% off the median step at 8x6
# (run10 4778s vs run8/run9 6378-8903s at identical rollout cost) and the
# gradient is unchanged at num_substeps=1. On by default; STREAM_MINIBATCHES=0
# to compare against the un-overlapped path.
STREAM_MINIBATCHES=${STREAM_MINIBATCHES:-1}
case "$STREAM_MINIBATCHES" in
  1|true|True) STREAM_ARG="stream_minibatches=True" ;;
  0|false|False) STREAM_ARG="stream_minibatches=False" ;;
  *) echo "STREAM_MINIBATCHES must be 0 or 1 (got '$STREAM_MINIBATCHES')" >&2; exit 1 ;;
esac

# The eval split is the benchmark and stays on task_split_seed=0; 242 redraws
# only the train pool, excluding scenario siblings of eval tasks. See
# examples/harvey_labs/ARCHITECTURE.md "Task split".
TRAIN_SPLIT_SEED=${TRAIN_SPLIT_SEED:-242}

# max_steps clamps to ceil(train_tasks / batch_size), so past 300/8 = 38 batches
# more steps need more train tasks. TRAIN_EXTRA appends them *after* the eval
# slice, which leaves eval and every earlier batch byte-identical — raising
# TRAIN_TASKS instead would slide the eval window and change the benchmark.
MAX_STEPS=${MAX_STEPS:-20}
TRAIN_EXTRA=${TRAIN_EXTRA:-0}

# 2e-4 diverges. It held for four steps and then ran away: across run26 steps
# 3-6 entropy went 0.222 -> 2.127, KL 0.0012 -> 0.0516 and reward 0.324 -> 0.073,
# i.e. below where step 0 started. The mechanism is a feedback loop -- a looser
# policy writes longer turns (ac_tokens_per_turn 780 -> 2955), those hit the
# max_tokens cap (6% -> 54% of episodes), capped episodes end at the -0.1 floor
# without ever reaching the judge (failed_before_grading 0.125 -> 0.604), and the
# thinner reward signal loosens the policy further. No earlier run at 2e-4 ever
# got past step 4 to show this: run22 died at 1 step, run24 at 2, run25 at 4, all
# on infrastructure faults, so the LR looked survivable when it never had been.
# 2e-5 is a 10x cut and still ~7x the recipe default in train.py.
LEARNING_RATE=${LEARNING_RATE:-2e-5}

TRAIN_CMD="TINKER_API_KEY=tml-dummy $JUDGE_ENV uv --project examples run python -m harvey_labs.train \
model_name=$MODEL_NAME renderer_name=$RENDERER base_url=http://127.0.0.1:9003 \
learning_rate=$LEARNING_RATE lora_rank=32 \
batch_size=$BATCH_SIZE rollouts_per_example=$ROLLOUTS max_steps=$MAX_STEPS eval_every=5 \
task_set=$TASK_SET judge_model=$JUDGE_MODEL $STREAM_ARG \
max_tokens=$GEN_TOKENS max_trajectory_tokens=$CONTEXT max_tool_result_tokens=$TOOL_TOKENS \
log_path=artifacts/harvey-labs/$RUN_LABEL"

# train_split_seed / train_tasks_extra only apply to task_set=random; train.py
# rejects them otherwise.
if [ "$TASK_SET" = "random" ]; then
  TRAIN_CMD="$TRAIN_CMD train_split_seed=$TRAIN_SPLIT_SEED"
  if [ "$TRAIN_EXTRA" -gt 0 ]; then
    TRAIN_CMD="$TRAIN_CMD train_tasks_extra=$TRAIN_EXTRA"
  fi
fi

# WORKLOAD=sft: same stack (the sampler still serves the post-SFT eval), but
# the train window types the SFT warm-start script instead of RL. Traces
# default to the public HF dataset inside sft.py.
if [ "$WORKLOAD" = "sft" ]; then
  TRAIN_CMD="TINKER_API_KEY=tml-dummy uv --project examples run python -m harvey_labs.sft \
model_name=$MODEL_NAME"
elif [ "$WORKLOAD" != "rl" ]; then
  echo "Unknown WORKLOAD=$WORKLOAD (use 'rl' or 'sft')" >&2
  exit 1
fi

# LOAD_CHECKPOINT=tinker://<id>/sampler_weights/<name> warm-starts RL from an
# SFT snapshot (weights only, fresh optimizer).
if [ "$WORKLOAD" = "rl" ] && [ -n "${LOAD_CHECKPOINT:-}" ]; then
  TRAIN_CMD="$TRAIN_CMD load_checkpoint_path=$LOAD_CHECKPOINT"
  echo "[work] RL warm start from $LOAD_CHECKPOINT"
fi

if [ "$WORKLOAD" = "rl" ]; then
  if [ "$TASK_SET" = "random" ]; then
    echo "[work] $STREAM_ARG, train_split_seed=$TRAIN_SPLIT_SEED (eval split unchanged)"
  else
    echo "[work] $STREAM_ARG"
  fi
fi

EVAL_CMD="TINKER_API_KEY=tml-dummy $JUDGE_ENV uv --project examples run python -m harvey_labs.eval_checkpoint \
checkpoint=/tmp/open-rl/peft/CHANGE-ME/final model_name=$MODEL_NAME renderer_name=$RENDERER \
base_url=http://127.0.0.1:9003 task_set=$TASK_SET judge_model=$JUDGE_MODEL \
max_tokens=$GEN_TOKENS max_trajectory_tokens=$CONTEXT max_tool_result_tokens=$TOOL_TOKENS"

# set-option needs a running tmux server, so the session must exist first.
tmux new-session -d -s "$SESSION" -n sampler -c "$REPO"
tmux set-option -t "$SESSION" history-limit 100000
# Panes get a fresh interactive shell, so they inherit this script's exports
# only when new-session also started the tmux server. If a server was already
# up for some unrelated session, the panes get that server's much older
# environment instead and the storage roots silently revert to /tmp. Setting
# them on the session covers every window created below. The sampler pane is
# already running and does not need them: it is handed absolute lora_paths in
# each request and never derives a root itself.
tmux set-environment -t "$SESSION" OPEN_RL_SNAPSHOT_DIR "$OPEN_RL_SNAPSHOT_DIR"
tmux set-environment -t "$SESSION" OPEN_RL_CHECKPOINT_DIR "$OPEN_RL_CHECKPOINT_DIR"
tmux send-keys -t "$SESSION:sampler" "$SAMPLER_CMD" C-m

tmux new-window -t "$SESSION" -n gateway -c "$REPO"
tmux send-keys -t "$SESSION:gateway" "$GATEWAY_CMD" C-m

if [ "$TRAIN_GPUS" -gt 1 ]; then
  TRAINER_CMD="CUDA_VISIBLE_DEVICES=$TRAIN_DEV FLA_TILELANG=$FLA_TILELANG REDIS_URL=redis://127.0.0.1:6379 \
OPEN_RL_FSDP_WORLD_SIZE=$TRAIN_GPUS OPEN_RL_WORKER_PROBE_PORT=8090 \
BASE_MODEL=$MODEL_NAME PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OPEN_RL_TRAIN_TOKEN_BUDGET=$TRAIN_TOKEN_BUDGET OPEN_RL_ACTIVATION_CPU_OFFLOAD=1 \
OPEN_RL_OPTIM_CPU_STEP=${OPTIM_CPU_STEP:-1} \
OPEN_RL_LOG_CUDA_MEMORY=1 \
uv run --extra gpu --extra fastpath torchrun --standalone --nproc-per-node=$TRAIN_GPUS -m server.training_requests_processor |& tee -a $LOGS/trainer.log"
  tmux new-window -t "$SESSION" -n trainer -c "$REPO"
  tmux send-keys -t "$SESSION:trainer" "$TRAINER_CMD" C-m
fi

tmux new-window -t "$SESSION" -n train -c "$REPO"
tmux send-keys -t "$SESSION:train" "$TRAIN_CMD"          # typed, NOT run

tmux new-window -t "$SESSION" -n eval -c "$REPO"
tmux send-keys -t "$SESSION:eval" "$EVAL_CMD"            # typed, NOT run

tmux new-window -t "$SESSION" -n gpu -c "$REPO"
tmux send-keys -t "$SESSION:gpu" "watch -n 5 nvidia-smi" C-m

tmux select-window -t "$SESSION:train"
echo "[work] up. sampler+gateway starting; train/eval commands are typed and waiting."
if [ -t 1 ]; then
  exec tmux attach -t "$SESSION"
fi
