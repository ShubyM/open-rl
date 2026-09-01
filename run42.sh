#!/usr/bin/env bash
# run42: run19's recipe, unchanged, on the Megatron backend.
#
# run19 is the best result this project has: Qwen3.5-9B, FSDP, eval 0.5351 ->
# 0.7098 over 35 steps. +17.5pp against a 9.7pp minimum detectable interval, so
# it is signal rather than noise. Every Megatron run since (31-41) has been
# Gemma-4-12B, which starts ~17pp lower and has never cleared 0.39. run42 asks
# one question: does the Megatron backend reproduce run19?
#
# Every hyperparameter below is run19's, deliberately, including the ones that
# look wrong. In particular learning_rate=2e-4 is the number that diverged
# twice on this backend -- run37 and run40 both went entropy 0.15 -> 8ish with
# eval collapsing to 0.0000 at lr=1e-4, and run38 diverged at 3e-5. Every one
# of those was Gemma-4-12B; no Qwen has ever run on Megatron, and Qwen took 2e-4
# on FSDP happily while Gemma diverged at the same LR on the same backend. So
# the LR bracket may be a property of the model rather than the backend. That
# is the bet, it was made knowingly, and the watchdog is armed on the thing that
# would falsify it.
#
# WHAT HAD TO BE BUILT TO GET HERE, all of it verified before this run existed:
#   - Qwen3.5's checkpoint is multimodal and megatron-bridge dispatches it to a
#     vision-language model. models/qwen35.py re-points the architecture at a
#     text-only GPTModel subclass of the bridge that already exists. Loads
#     8.95B params, 307/307 tensors, perplexity 5.547.
#   - The gated-deltanet kernels need fla, which splits across two
#     distributions: flash-linear-attention ships fla/layers and fla/models,
#     while fla.modules and fla.ops -- the ones the kernel path imports -- are
#     in fla-core. Installing only the first leaves HAVE_FLA quietly False.
#   - The GDN *backward* additionally needs tilelang. fla refuses to run it on
#     Hopper under Triton>=3.4 because the Triton kernel is silently wrong
#     there. The trainer runs out of ~/megatron-probe/.venv, which did not have
#     it. This is checked below rather than trusted: it is the failure that
#     would land at the first optim step, after a full rollout is already paid
#     for, and it is invisible in a forward pass.
#   - LoRA covers the GDN mixers (in_proj/out_proj), matching run19. Megatron
#     fuses the four in_proj_* into one; save_hf_adapter splits them back out
#     and all 496 exported tensors were name- and shape-checked against the
#     base checkpoint.
#
# Stated in advance so it can be falsified:
#   - eval criteria_pass_fraction should start near run19's 0.5351. Materially
#     below that means the Megatron load is subtly wrong in a way perplexity
#     5.547 did not catch, and nothing after step 0 is worth reading.
#   - it should end near 0.71. Between 0.55 and 0.65 is a real but smaller
#     effect; flat is a backend that trains nothing.
#   - optim/entropy should sit in 0.15-0.30. Above 0.6 is the divergence
#     signature from runs 37/38/40 and means 2e-4 did not transfer.
#   - grad_norm must never be exactly 0. Zero is the zero-advantage lock, and
#     it is what run37 and run40 printed after they had already died.
#   - env/all/max_tokens_reached is per-TURN, roughly 15x understated per
#     episode. Read it as a trend, not a rate.
set -uo pipefail
cd "$HOME/open-rl" || exit 1

LORA_RANK=32
CONTEXT=163840
JUDGE_MODEL=gpt-glm-5.2
MEGATRON_PY="$HOME/megatron-probe/.venv/bin/python"

# Grading runs as a subprocess that inherits os.environ. With the env absent the
# client constructor raises before a single rubric is scored, reward.py catches
# it, and every episode returns lab/reward_error=1.0 with lab/graded=0.0 -- the
# run then trains on the -0.1 floor at full GPU cost.
if [ ! -f .env.judge ]; then
  echo "ERROR: .env.judge missing -- grading would fail silently on every episode." >&2
  exit 1
fi
set -a
. ./.env.judge
set +a

if ! curl -sf -m 20 "${OPENAI_BASE_URL:?OPENAI_BASE_URL not set by .env.judge}/models" >/dev/null; then
  echo "ERROR: judge endpoint at \$OPENAI_BASE_URL is not answering /models." >&2
  exit 1
fi
# Reachable is not enough: train.py's judge_model defaults to gemini-3.5-flash,
# so dropping the flag below grades against a different model than this endpoint
# serves -- quietly, at full GPU cost.
if ! curl -sf -m 20 "$OPENAI_BASE_URL/models" | grep -q "\"$JUDGE_MODEL\""; then
  echo "ERROR: judge endpoint does not serve $JUDGE_MODEL. It lists:" >&2
  curl -sf -m 20 "$OPENAI_BASE_URL/models" >&2
  exit 1
fi
echo "[run42] judge endpoint reachable, serving $JUDGE_MODEL"

# THE CHECK THIS RUN EXISTS BECAUSE OF. Twenty-four of Qwen3.5's thirty-two
# layers are gated-deltanet, and a gradient reaching layer 0's LoRA has to pass
# through all of them -- so this binds regardless of which modules LoRA targets.
# fla raises rather than returning wrong numbers, which is the good outcome, but
# it raises in the backward: the model loads, the forward is correct, the first
# forty minutes of rollout succeed, and then the first optim step dies. Checked
# in the trainer's own interpreter, which is not the project venv and did not
# have tilelang until it was installed for this run.
if ! "$MEGATRON_PY" -c 'import tilelang' >/dev/null 2>&1; then
  echo "ERROR: no tilelang in $MEGATRON_PY." >&2
  echo "  fla refuses the gated-deltanet backward on Hopper under Triton>=3.4 without it," >&2
  echo "  so this run would die at its first optim step. Install tilelang==0.1.9 and" >&2
  echo "  apache-tvm-ffi==0.1.9 (0.1.13 breaks 'import megatron.bridge' outright)." >&2
  exit 1
fi
if ! "$MEGATRON_PY" -c 'import megatron.core.ssm.gated_delta_net as g; raise SystemExit(0 if g.HAVE_FLA else 1)' >/dev/null 2>&1; then
  echo "ERROR: HAVE_FLA is False in $MEGATRON_PY -- fla-core is missing or not importable." >&2
  echo "  flash-linear-attention alone is not enough; fla.modules and fla.ops live in fla-core." >&2
  exit 1
fi
echo "[run42] trainer interpreter: tilelang present, HAVE_FLA true"

# Four single-GPU samplers, each of which must be at the right window AND able
# to take an adapter at runtime. A sampler that came up without --enable-lora,
# or with VLLM_ALLOW_RUNTIME_LORA_UPDATING unset, serves perfectly well and
# ignores every adapter push -- the run then trains for 40 steps while every
# rollout comes from the base model, and the only symptom is a flat curve.
for PORT in 8000 8001 8002 8003; do
  LIVE_LEN=$(curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])' 2>/dev/null)
  if [ "$LIVE_LEN" != "$CONTEXT" ]; then
    echo "ERROR: sampler :$PORT max_model_len is '$LIVE_LEN', want $CONTEXT. Stale vLLM -- restart the stack." >&2
    exit 1
  fi
  # [v]llm so this never matches the line running it.
  SAMPLER_PID=$(pgrep -f "[v]llm serve .*--port $PORT( |$)" | head -1)
  if [ -z "$SAMPLER_PID" ]; then
    echo "ERROR: sampler :$PORT answers HTTP but no 'vllm serve --port $PORT' process was found." >&2
    exit 1
  fi
  SAMPLER_CMD=$(tr '\0' ' ' < "/proc/$SAMPLER_PID/cmdline")
  case "$SAMPLER_CMD" in
    *--enable-lora*) ;;
    *) echo "ERROR: sampler :$PORT started without --enable-lora." >&2; exit 1 ;;
  esac
  MAX_RANK=$(sed -n 's/.*--max-lora-rank \([0-9]\{1,\}\).*/\1/p' <<<"$SAMPLER_CMD")
  if [ -z "$MAX_RANK" ] || [ "$MAX_RANK" -lt "$LORA_RANK" ]; then
    echo "ERROR: sampler :$PORT has --max-lora-rank '$MAX_RANK', below the trained rank $LORA_RANK." >&2
    exit 1
  fi
  if ! grep -qx 'VLLM_ALLOW_RUNTIME_LORA_UPDATING=true' < <(tr '\0' '\n' < "/proc/$SAMPLER_PID/environ"); then
    echo "ERROR: sampler :$PORT has no VLLM_ALLOW_RUNTIME_LORA_UPDATING=true." >&2
    echo "  /v1/load_lora_adapter is not mounted without it, and every rollout would be base weights." >&2
    exit 1
  fi
  echo "[run42] sampler :$PORT window $LIVE_LEN, LoRA to rank $MAX_RANK, runtime updating on"
done

TRAINER_PID=$(pgrep -f 'server.training_requests_processor' | head -1)
if [ -z "$TRAINER_PID" ]; then
  echo "ERROR: no trainer process. Bring the stack up with scripts/launch42.sh first." >&2
  exit 1
fi
TRAINER_ENV=$(tr '\0' '\n' < "/proc/$TRAINER_PID/environ")
# OPEN_RL_MEGATRON_SEQUENCE_PARALLEL=0 is not a tuning choice. The worker turns
# SP on for any TP>1, and megatron's gated-deltanet asserts against it
# (core/ssm/gated_delta_net/gdn.py:91), so at TP=2 the default value kills the
# run in model construction. FLA_TILELANG=1 selects the kernel backend the
# tilelang check above proved is installed.
for VAR in OPEN_RL_TRAINER_BACKEND=megatron OPEN_RL_MEGATRON_TP=2 \
           OPEN_RL_MEGATRON_SEQUENCE_PARALLEL=0 "OPEN_RL_MEGATRON_LORA_RANK=$LORA_RANK" \
           FLA_TILELANG=1; do
  if ! grep -qx "$VAR" <<<"$TRAINER_ENV"; then
    echo "ERROR: trainer pid $TRAINER_PID is missing $VAR." >&2
    echo "  Its env has: $(grep -E '^(OPEN_RL_(TRAINER_BACKEND|MEGATRON)|FLA_)' <<<"$TRAINER_ENV" | tr '\n' ' ')" >&2
    exit 1
  fi
done
# The GDN mixers specifically. A targets string that silently lost in_proj
# trains 8 of 32 mixers and still produces a valid adapter and a plausible
# curve, so it cannot be caught after the fact.
TARGETS=$(grep '^OPEN_RL_MEGATRON_LORA_TARGETS=' <<<"$TRAINER_ENV" | head -1)
case "$TARGETS" in
  *in_proj*out_proj*|*out_proj*in_proj*) ;;
  *) echo "ERROR: trainer LoRA targets do not cover the GDN mixers: '${TARGETS:-unset}'." >&2
     echo "  24 of 32 layers would train nothing but their MLP. run19 trained them." >&2
     exit 1 ;;
esac
if ! grep -qx 'SAMPLER_BASE_URLS=http://127.0.0.1:8000,http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003' <<<"$TRAINER_ENV"; then
  echo "ERROR: trainer has no SAMPLER_BASE_URLS for all four samplers." >&2
  echo "  Got: $(grep -E '^SAMPLER_BASE_URLS?=' <<<"$TRAINER_ENV")" >&2
  exit 1
fi
# The trainer writes the adapter and the gateway hands the sampler its path, so
# the two processes have to mean the same directory. Disagreeing is a
# FileNotFoundError at the first rollout, or worse, a stale adapter that loads.
GATEWAY_PID=$(pgrep -f '[s]erver.gateway|[u]vicorn.*gateway' | head -1)
TRAINER_SNAPSHOT=$(grep '^OPEN_RL_SNAPSHOT_DIR=' <<<"$TRAINER_ENV" | head -1)
if [ -n "$GATEWAY_PID" ]; then
  GATEWAY_SNAPSHOT=$(tr '\0' '\n' < "/proc/$GATEWAY_PID/environ" | grep '^OPEN_RL_SNAPSHOT_DIR=' | head -1)
  if [ "$TRAINER_SNAPSHOT" != "$GATEWAY_SNAPSHOT" ]; then
    echo "ERROR: trainer and gateway disagree about the adapter directory." >&2
    echo "  trainer: '${TRAINER_SNAPSHOT:-unset}'  gateway: '${GATEWAY_SNAPSHOT:-unset}'" >&2
    exit 1
  fi
fi
echo "[run42] trainer pid $TRAINER_PID: megatron TP=2 SP=off, rank $LORA_RANK, GDN mixers targeted"

# Sandbox concurrency against the kernel keyring quota. podman allocates one
# keyring per container and the per-user default is 200 keys; the eval set is 50
# tasks and run41 died at launch -- before a single GPU cycle -- asking for
# exactly 200 at once, with the error reported as "Disk quota exceeded". The
# raise is runtime-only, so a spot reboot silently restores the failure.
MAXKEYS=$(cat /proc/sys/kernel/keys/maxkeys)
if [ "$MAXKEYS" -lt 1000 ]; then
  echo "ERROR: kernel.keys.maxkeys=$MAXKEYS is too low; run41 died at launch on this." >&2
  echo "  sudo sysctl -w kernel.keys.maxkeys=5000 kernel.keys.maxbytes=500000" >&2
  exit 1
fi
LEAKED=$(podman ps -aq 2>/dev/null | wc -l)
if [ "$LEAKED" -gt 20 ]; then
  echo "ERROR: $LEAKED sandbox containers already exist; each holds a keyring." >&2
  exit 1
fi

# Megatron checkpoints do not rotate. At ~19 GiB for a 9B and save_every=10 that
# is four saves over 40 steps; the failure mode is a *healthy* run filling the
# disk mid-save, which corrupts the checkpoint and takes the run with it.
FREE_G=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
if [ "$FREE_G" -lt 100 ]; then
  echo "ERROR: only ${FREE_G}G free on /home; four ~19G checkpoints need ~80G plus room." >&2
  echo "  609G sits in ~/open-rl-checkpoints from dead runs -- reclaim some first." >&2
  exit 1
fi
echo "[run42] ${FREE_G}G free, maxkeys=$MAXKEYS, $LEAKED containers -- preflight clean"

# lora_rank below is cosmetic on this backend: the gateway sends full_config
# under FFT and the worker reads OPEN_RL_MEGATRON_LORA_RANK. It is left in so
# run19's and run42's command lines diff cleanly; the env check above is what
# actually enforces the rank.
TINKER_API_KEY=tml-dummy uv --project examples run python examples/harvey_labs/train.py \
  model_name=Qwen/Qwen3.5-9B \
  renderer_name=qwen3_5 \
  base_url=http://127.0.0.1:9003 \
  learning_rate=2e-4 \
  lora_rank=32 \
  batch_size=8 \
  rollouts_per_example=6 \
  max_steps=40 \
  eval_every=5 \
  save_every=10 \
  task_set=random \
  judge_model=$JUDGE_MODEL \
  stream_minibatches=True \
  max_tokens=32768 \
  max_trajectory_tokens=163840 \
  max_tool_result_tokens=16384 \
  train_split_seed=242 \
  log_path=artifacts/harvey-labs/run42-qwen35-9b-megatron-160k \
  |& tee -a "$HOME/open-rl/artifacts/box-logs/run42.log"
