#!/usr/bin/env bash
# The pipeline smoke test: OpenRLWorker -> ResourceClaim -> allocation -> Pod
# Running, against a real API server, real kube-scheduler, and real DRA.
#
# Fake GPUs by default: a kind cluster plus the upstream DRA example driver,
# which publishes synthetic ResourceSlices (8 GPUs x 80Gi per node). The
# controller cannot tell the difference -- everything it reads comes from
# ResourceSlices and node labels -- so the same script against a cluster with
# real hardware exercises the identical path:
#
#   ./hack/kind-smoke.sh                      # kind + fake GPUs
#   KEEP=1 ./hack/kind-smoke.sh               # leave the cluster up afterwards
#   USE_EXISTING_CLUSTER=1 LOAD_INTO=<kind-cluster> \
#     DEVICE_CLASS=gpu.nvidia.com GPUS=2 MEMORY=10Gi ./hack/kind-smoke.sh
#                                             # current kubectl context, real DRA
#
# GPUS is how many devices the target node exposes (the fake driver's 8 by
# default); MEMORY must fit one of its devices. The run asks for GPUS+1
# workers and asserts spread-then-share. LOAD_INTO names a kind cluster to
# load the locally built image into; a non-kind cluster needs the image
# pushed somewhere it can pull.
#
# What the behavior tests cannot verify and this does: claim allocation and
# immutability, pod binding, and watch-driven status flow.
set -euo pipefail

CLUSTER=${CLUSTER:-openrl-smoke}
DEVICE_CLASS=${DEVICE_CLASS:-gpu.example.com}
DEVICE_DRIVER=${DEVICE_DRIVER:-$DEVICE_CLASS}
USE_EXISTING_CLUSTER=${USE_EXISTING_CLUSTER:-0}
LOAD_INTO=${LOAD_INTO:-}
KEEP=${KEEP:-0}
GPUS=${GPUS:-8}
MEMORY=${MEMORY:-50Gi}
# The smoke overlay pins this tag; it is not overridable for that reason.
IMG=open-rl/placement-controller:smoke

controller=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$controller/.." && pwd)

say() { printf '\n== %s\n' "$*"; }

if [ "$USE_EXISTING_CLUSTER" != 1 ]; then
  say "creating kind cluster $CLUSTER"
  kind create cluster --name "$CLUSTER" --wait 120s
  if [ "$KEEP" != 1 ]; then
    trap 'kind delete cluster --name "$CLUSTER"' EXIT
  fi
fi

# Fake GPUs. The example driver is the reference DRA implementation: it
# publishes ResourceSlices whose devices carry a memory capacity, exactly like
# the NVIDIA driver, without needing hardware.
if [ "$DEVICE_CLASS" = gpu.example.com ]; then
  say "installing the DRA example driver (fake GPUs)"
  tmp=$(mktemp -d)
  git clone --quiet --depth 1 https://github.com/kubernetes-sigs/dra-example-driver "$tmp/driver"
  helm upgrade --install dra-example-driver "$tmp/driver/deployments/helm/dra-example-driver" \
    --create-namespace --namespace dra-example-driver --wait --timeout 180s
fi

say "waiting for ResourceSlices from $DEVICE_DRIVER"
for _ in $(seq 60); do
  kubectl get resourceslices -o jsonpath='{.items[*].spec.driver}' 2>/dev/null | grep -q "$DEVICE_DRIVER" && break
  sleep 2
done
kubectl get resourceslices

say "building and loading the controller image"
docker build -q -t "$IMG" "$controller"
if [ "$USE_EXISTING_CLUSTER" != 1 ]; then
  kind load docker-image "$IMG" --name "$CLUSTER"
elif [ -n "$LOAD_INTO" ]; then
  kind load docker-image "$IMG" --name "$LOAD_INTO"
fi

say "deploying the scheduler (smoke overlay)"
kubectl apply -k "$repo/k8s/deploy/scheduler-smoke"
if [ "$DEVICE_CLASS" != gpu.example.com ]; then
  # The overlay defaults to the fake driver; real-hardware runs override it.
  kubectl -n default set env deployment/open-rl-scheduler \
    OPEN_RL_DEVICE_CLASS="$DEVICE_CLASS" OPEN_RL_DEVICE_DRIVER="$DEVICE_DRIVER"
fi

say "opting nodes in"
for node in $(kubectl get nodes -o name); do
  kubectl label --overwrite "$node" \
    openrl.io/enabled=true openrl.io/trainer=true openrl.io/sampler=true \
    openrl.io/max-workers-per-claim=2
done

if ! kubectl -n default rollout status deployment/open-rl-scheduler --timeout=240s; then
  echo "FAIL: the scheduler never became ready"
  kubectl -n default describe pods -l app=open-rl-scheduler | tail -30
  kubectl -n default logs deployment/open-rl-scheduler --tail=40 || true
  exit 1
fi

# One more worker than the node has GPUs, so the pipeline shows both halves
# of the policy: spread while devices are free, then share under contention.
WORKERS=$((GPUS + 1))

say "requesting $WORKERS workers of $MEMORY against $GPUS GPUs"
for i in $(seq "$WORKERS"); do
  kubectl apply -f - <<EOF
apiVersion: openrl.io/v1alpha1
kind: OpenRLWorker
metadata: {name: smoke-$i, namespace: default}
spec: {role: trainer, modelId: smoke-$i, ownerId: smoke, memory: $MEMORY}
EOF
done

say "waiting for every worker to run"
for i in $(seq "$WORKERS"); do
  ok=0
  for _ in $(seq 90); do
    phase=$(kubectl -n default get openrlworker "smoke-$i" -o jsonpath='{.status.phase}' 2>/dev/null || true)
    [ "$phase" = Running ] && ok=1 && break
    sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo "FAIL: smoke-$i never reached Running (phase: ${phase:-none})"
    kubectl -n default get openrlworkers
    kubectl -n default get pods,resourceclaims
    exit 1
  fi
done

say "asserting spread, sharing, and real allocations"
claims=$(kubectl -n default get openrlworkers -o jsonpath='{range .items[*]}{.status.claimName}{"\n"}{end}')
distinct=$(echo "$claims" | sort -u | grep -c .)
if [ "$distinct" != "$GPUS" ]; then
  echo "FAIL: $WORKERS workers hold $distinct claims, want $GPUS: one per GPU, with the extra worker sharing"
  kubectl -n default get openrlworkers
  exit 1
fi
for claim in $(echo "$claims" | sort -u); do
  driver=$(kubectl -n default get resourceclaim "$claim" -o jsonpath='{.status.allocation.devices.results[0].driver}')
  if [ "$driver" != "$DEVICE_DRIVER" ]; then
    echo "FAIL: claim $claim was not allocated by $DEVICE_DRIVER (got: ${driver:-nothing})"
    exit 1
  fi
done
echo "$WORKERS workers on $distinct claims, every claim allocated by $DEVICE_DRIVER"

say "the run, as an operator would see it"
kubectl -n default get openrlworkers
kubectl -n default get resourceclaims
kubectl -n default get pods -o wide

say "PASS"
