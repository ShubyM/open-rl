#!/usr/bin/env bash
# Churn workers against a live scheduler and check the books every round.
#
# Run it after kind-smoke.sh with KEEP=1 (or against any cluster where the
# scheduler is deployed and nodes are labeled). Each round creates workers up
# to WORKERS, waits for the fleet to settle, asserts the seat arithmetic --
# running == min(live, GPUS x MAXW), everyone else Pending with a reason,
# never more claims than GPUs -- then deletes a random third and goes again.
# The last round deletes everything and waits for the reclaim sweep to return
# the namespace to empty: no workers, no managed claims, no worker pods.
#
#   WORKERS=24 ROUNDS=5 GPUS=8 MAXW=2 MEMORY=50Gi ./hack/stress.sh
set -euo pipefail

WORKERS=${WORKERS:-24}
ROUNDS=${ROUNDS:-5}
GPUS=${GPUS:-8}
MAXW=${MAXW:-2}
MEMORY=${MEMORY:-50Gi}
SEED=${SEED:-42}
RANDOM=$SEED

say() { printf '\n== %s\n' "$*"; }
count() { kubectl -n default get "$1" --no-headers 2>/dev/null | grep -c . || true; }

phase_counts() {
  kubectl -n default get openrlworkers -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}'
}

settle() {
  local live=$1 seats=$((GPUS * MAXW)) want_running want_pending
  want_running=$((live < seats ? live : seats))
  want_pending=$((live - want_running))
  for _ in $(seq 120); do
    local running pending failed
    running=$(phase_counts | grep -c '^Running$' || true)
    pending=$(phase_counts | grep -c '^Pending$' || true)
    failed=$(phase_counts | grep -c '^Failed$' || true)
    if [ "$failed" != 0 ]; then
      echo "FAIL: $failed workers Failed"
      kubectl -n default get openrlworkers
      exit 1
    fi
    if [ "$running" = "$want_running" ] && [ "$pending" = "$want_pending" ]; then
      local claims
      claims=$(count resourceclaims)
      if [ "$claims" -gt "$GPUS" ]; then
        echo "FAIL: $claims claims for $GPUS GPUs"
        kubectl -n default get resourceclaims
        exit 1
      fi
      echo "settled: $running running, $pending pending, $claims claims"
      # Every pending worker must say why.
      kubectl -n default get openrlworkers -o jsonpath='{range .items[?(@.status.phase=="Pending")]}{.metadata.name}: {.status.reason}{"\n"}{end}'
      return 0
    fi
    sleep 2
  done
  echo "FAIL: never settled at $want_running running / $want_pending pending"
  kubectl -n default get openrlworkers
  exit 1
}

for round in $(seq "$ROUNDS"); do
  say "round $round: topping up to $WORKERS workers"
  for i in $(seq "$WORKERS"); do
    kubectl -n default get openrlworker "stress-$i" >/dev/null 2>&1 && continue
    kubectl apply -f - >/dev/null <<EOF
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorker
metadata: {name: stress-$i, namespace: default}
spec: {role: trainer, modelId: stress-$i, ownerId: stress-$((i % 3)), memory: $MEMORY}
EOF
  done
  settle "$WORKERS"

  say "round $round: deleting a random third"
  deleted=0
  for i in $(seq "$WORKERS"); do
    if [ $((RANDOM % 3)) = 0 ]; then
      kubectl -n default delete openrlworker "stress-$i" --wait=false >/dev/null 2>&1 || true
      deleted=$((deleted + 1))
    fi
  done
  # Deletion is asynchronous; wait for the census to match before asserting.
  for _ in $(seq 60); do
    [ "$(count openrlworkers)" = "$((WORKERS - deleted))" ] && break
    sleep 2
  done
  settle "$((WORKERS - deleted))"
done

say "final: deleting everything and waiting for the reclaim sweep"
kubectl -n default delete openrlworkers --all --wait=true >/dev/null
# Claims outlive their workers by design: a 2m grace plus the sweep interval.
for _ in $(seq 100); do
  claims=$(count resourceclaims)
  pods=$(kubectl -n default get pods --no-headers 2>/dev/null | grep -c '^orw-' || true)
  if [ "$claims" = 0 ] && [ "$pods" = 0 ]; then
    say "PASS: namespace is clean -- 0 workers, 0 claims, 0 worker pods"
    exit 0
  fi
  sleep 5
done
echo "FAIL: leftovers after deletion: $claims claims, $pods worker pods"
kubectl -n default get pods,resourceclaims
exit 1
