#!/usr/bin/env bash
# Report-only watchdog for run42. It never kills anything -- it writes a line
# per step to watch42.log and shouts when a metric crosses a threshold, and the
# decision to stop the run stays with a human.
#
# The three thresholds are the three ways this specific run is expected to fail:
#
#   entropy > 0.6   The divergence signature. Runs 37, 38 and 40 all went this
#                   way on Megatron -- entropy climbs, the policy stops being a
#                   policy, reward follows it down a few steps later. run42
#                   deliberately took lr=2e-4 for run19 parity even though
#                   every Megatron divergence on record sat at a lower LR, so
#                   this is the accepted risk of the run and the thing most
#                   worth catching early. Entropy leads reward, which is why it
#                   is the trip wire rather than reward itself.
#
#   grad_norm == 0  Zero-advantage lock. If every rollout in a group scores the
#                   same, advantages are all zero, and the run keeps paying
#                   full GPU cost while learning exactly nothing. It looks
#                   completely healthy from the outside -- steps advance, loss
#                   prints -- so nothing else catches it.
#
#   max_tokens_reached > 0.03
#                   Context overflow. This is per-TURN, and an episode runs
#                   ~15 turns, so 0.03 here is closer to a third of episodes
#                   truncated. run19's eval decline turned out to be exactly
#                   this, and holding 163,840 tokens is the reason run42 pays
#                   for TP=2 at all -- if it trips anyway, the parity argument
#                   for this configuration is what is failing.
set -uo pipefail

M="$HOME/open-rl/artifacts/harvey-labs/run42-qwen35-9b-megatron-160k/metrics.jsonl"
OUT="$HOME/open-rl/artifacts/box-logs/watch42.log"
seen=0

echo "WATCH42_START $(date -u +%Y-%m-%dT%H:%M:%SZ) -> $M" >> "$OUT"

while true; do
  if [ -f "$M" ]; then
    n=$(wc -l < "$M")
    if [ "$n" -gt "$seen" ]; then
      # Re-read from the last seen offset so a burst of steps all get reported
      # rather than only the newest one.
      tail -n +$((seen + 1)) "$M" | python3 -c '
import json, sys

def get(d, *names):
    # Exact keys only. A suffix fallback was tried and picked up
    # time/do_group_rollout_and_filter_constant_reward:mean as the reward,
    # reporting 728.49 -- a plausible-looking number from the wrong metric is
    # worse than a missing one, because it reads as a healthy signal.
    for n in names:
        if n in d:
            return d[n]
    return None

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except json.JSONDecodeError:
        continue

    step = get(d, "step")
    ent = get(d, "optim/entropy")
    gn = get(d, "grad_norm:mean")
    rew = get(d, "env/all/reward/total")
    mt = get(d, "env/all/max_tokens_reached")
    # The turn-level counter above read 0.0 at step 0 while this read 1%, so
    # they are not measuring the same thing and truncation would show up here
    # first. Reported, not tripped on.
    ovf = get(d, "env/all/context_overflow")
    kl = get(d, "optim/kl_sample_train_v1")
    skip = get(d, "optim_step_skipped:mean")

    def s(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    print(f"step={s(step)} entropy={s(ent)} reward={s(rew)} grad_norm={s(gn)} "
          f"kl={s(kl)} ovf={s(ovf)} max_tok={s(mt)} skipped={s(skip)}")

    if ent is not None and ent > 0.6:
        print(f"  !! TRIP entropy={ent:.4f} > 0.6 -- runs 37/38/40 divergence signature")
    if gn is not None and gn == 0:
        print("  !! TRIP grad_norm == 0 -- zero-advantage lock, run is learning nothing")
    if mt is not None and mt > 0.03:
        print(f"  !! TRIP max_tokens_reached={mt:.4f} > 0.03 (per-turn; ~15x that per episode)")
    # Step 0 measured 3e-4. This is the one number that proves the Megatron ->
    # save_hf_adapter -> vLLM export stayed numerically correct, so a jump here
    # means the samplers are running a different policy than the trainer thinks.
    if kl is not None and kl > 0.05:
        print(f"  !! TRIP kl_sample_train={kl:.5f} > 0.05 -- sampler/trainer policy mismatch")
    if skip is not None and skip > 0.5:
        print(f"  !! TRIP optim_step_skipped={skip:.4f} -- optimizer is refusing steps")
' >> "$OUT" 2>&1
      seen=$n
    fi
  fi

  # The driver is the liveness signal, not the metrics file: a step at this
  # context takes long enough that a quiet metrics.jsonl proves nothing.
  if ! pgrep -f '[t]rain.py' > /dev/null; then
    echo "WATCH42_END $(date -u +%Y-%m-%dT%H:%M:%SZ) -- train.py gone after $seen step(s)" >> "$OUT"
    exit 0
  fi
  sleep 60
done
