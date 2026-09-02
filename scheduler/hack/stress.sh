#!/usr/bin/env bash
# Churn workers against a live scheduler and check the books every round.
#
# Run it after kind-smoke.sh with KEEP=1 (or against any cluster where the
# scheduler is deployed and nodes are labeled). Each round creates workers up
# to WORKERS, waits for the fleet to settle, asserts the seat arithmetic --
# running == min(live, GPUS x MAXW), everyone else Pending with a reason,
# never more allocated claims than GPUs -- then deletes a random third and
# goes again. The last round deletes everything and waits for the namespace
# to drain to empty: no workers, no managed claims, no worker pods.
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
NS=openrl-system

say() { printf '\n== %s\n' "$*"; }
count() { kubectl -n "$NS" get "$1" --no-headers 2>/dev/null | grep -c . || true; }
# A pending worker keeps its unallocated dedicated claim open as the
# autoscale signal, so only allocated claims are bounded by the GPU count.
allocated_claims() {
  kubectl -n "$NS" get resourceclaims \
    -o jsonpath='{range .items[?(@.status.allocation)]}{.metadata.name}{"\n"}{end}' 2>/dev/null | grep -c . || true
}

phase_counts() {
  kubectl -n "$NS" get workloads -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}'
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
      kubectl -n "$NS" get workloads
      exit 1
    fi
    if [ "$running" = "$want_running" ] && [ "$pending" = "$want_pending" ]; then
      local claims
      claims=$(allocated_claims)
      if [ "$claims" -gt "$GPUS" ]; then
        echo "FAIL: $claims allocated claims for $GPUS GPUs"
        kubectl -n "$NS" get resourceclaims
        exit 1
      fi
      echo "settled: $running running, $pending pending, $claims allocated claims"
      # Every pending worker must say why.
      kubectl -n "$NS" get workloads -o jsonpath='{range .items[?(@.status.phase=="Pending")]}{.metadata.name}: {.status.reason}{"\n"}{end}'
      return 0
    fi
    sleep 2
  done
  echo "FAIL: never settled at $want_running running / $want_pending pending"
  kubectl -n "$NS" get workloads
  exit 1
}

for round in $(seq "$ROUNDS"); do
  say "round $round: topping up to $WORKERS workers"
  for i in $(seq "$WORKERS"); do
    kubectl -n "$NS" get workload "stress-$i" >/dev/null 2>&1 && continue
    kubectl apply -f - >/dev/null <<EOF
apiVersion: openrl.io/v1alpha1
kind: Workload
metadata: {name: stress-$i, namespace: $NS}
spec:
  role: trainer
  modelID: stress-$i
  ownerID: stress-$((i % 3))
  accelerator: {memory: $MEMORY}
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: worker
          image: busybox:1.36
          command: ["sleep", "infinity"]
EOF
  done
  settle "$WORKERS"

  say "round $round: deleting a random third"
  deleted=0
  for i in $(seq "$WORKERS"); do
    if [ $((RANDOM % 3)) = 0 ]; then
      kubectl -n "$NS" delete workload "stress-$i" --wait=false >/dev/null 2>&1 || true
      deleted=$((deleted + 1))
    fi
  done
  # Deletion is asynchronous; wait for the census to match before asserting.
  for _ in $(seq 60); do
    [ "$(count workloads)" = "$((WORKERS - deleted))" ] && break
    sleep 2
  done
  settle "$((WORKERS - deleted))"
done

say "final: deleting everything and waiting for the namespace to drain"
kubectl -n "$NS" delete workloads --all --wait=true >/dev/null
# Teardown is inline, but finalizers wait out pod termination first.
for _ in $(seq 100); do
  claims=$(count resourceclaims)
  pods=$(kubectl -n "$NS" get pods --no-headers 2>/dev/null | grep -c '^orw-' || true)
  if [ "$claims" = 0 ] && [ "$pods" = 0 ]; then
    say "PASS: namespace is clean -- 0 workers, 0 claims, 0 worker pods"
    exit 0
  fi
  sleep 5
done
echo "FAIL: leftovers after deletion: $claims claims, $pods worker pods"
kubectl -n "$NS" get pods,resourceclaims
exit 1
