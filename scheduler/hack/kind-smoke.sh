#!/usr/bin/env bash
# The pipeline smoke test: Workload -> tiered ResourceClaim -> allocation ->
# Pod Running, against a real API server, real kube-scheduler, and real DRA.
#
# Fake GPUs by default: a kind cluster plus the upstream DRA example driver,
# which publishes synthetic ResourceSlices (8 GPUs x 80Gi per node). The
# controller cannot tell the difference -- everything it reads comes from
# ResourceSlices and node labels -- so the same script against a cluster with
# real hardware exercises the identical path:
#
#   ./hack/kind-smoke.sh                      # kind + fake GPUs
#   SCHEDULER_DIR=/path/to/other/checkout/scheduler ./hack/kind-smoke.sh
#                                             # test another checkout
#   KEEP=1 ./hack/kind-smoke.sh               # leave the cluster up afterwards
#   USE_EXISTING_CLUSTER=1 LOAD_INTO=<kind-cluster> \
#     DEVICE_CLASS=gpu.nvidia.com GPUS=2 MEMORY=10Gi ./hack/kind-smoke.sh
#                                             # current kubectl context, real DRA
#
# Hosts with cgroup v1 can use KIND_NODE_IMAGE=kindest/node:v1.34.0.
# KIND_NODE_IMAGE pins Kubernetes; DRA_DRIVER_COMMIT pins the fake driver.
# The spread policy is explicit: the standalone controller defaults to binpack.
# A private kubeconfig is used; existing-cluster runs require an empty namespace.
#
# GPUS is how many devices the target node exposes (the fake driver's 8 by
# default); MEMORY must fit one of its devices. The run asks for GPUS+1
# workers: GPUS of them win dedicated claims, and the extra one falls back to
# a seat on an existing ClaimLedger once kube-scheduler declines its pod. LOAD_INTO
# names a kind cluster to load the locally built image into; a non-kind
# cluster needs the image pushed somewhere it can pull.
#
# What the behavior tests cannot verify and this does: claim allocation and
# immutability, pod binding, and watch-driven status flow.
set -euo pipefail

CLUSTER=${CLUSTER:-openrl-smoke-$$}
KIND_NODE_IMAGE=${KIND_NODE_IMAGE:-kindest/node:v1.35.5}
DRA_DRIVER_COMMIT=bab2dce4b045e729565c9a3b85554738247cda30 # v0.4.0
DEVICE_CLASS=${DEVICE_CLASS:-gpu.example.com}
DEVICE_DRIVER=${DEVICE_DRIVER:-$DEVICE_CLASS}
USE_EXISTING_CLUSTER=${USE_EXISTING_CLUSTER:-0}
LOAD_INTO=${LOAD_INTO:-}
KEEP=${KEEP:-0}
GPUS=${GPUS:-8}
MEMORY=${MEMORY:-50Gi}
NS=openrl-system
# The smoke overlay pins this tag; it is not overridable for that reason.
IMG=open-rl/scheduler:smoke

scheduler=${SCHEDULER_DIR:-$(cd "$(dirname "$0")/.." && pwd)}

say() { printf '\n== %s\n' "$*"; }

# A private kubeconfig keeps both kind and kubectl off the caller's context.
umask 077
tmp=$(mktemp -d)
created=0
cleanup() {
  result=$?
  if [ "$result" != 0 ] && [ "$created" = 1 ]; then
    kind export logs "$tmp/logs" --name "$CLUSTER" || true
  fi
  if [ "$result" != 0 ] && [ -s "$KUBECONFIG" ]; then
    kubectl -n "$NS" get workloads,pods,resourceclaims,claimledgers -o wide || true
    kubectl -n "$NS" get events --sort-by=.lastTimestamp | tail -40 || true
    kubectl -n "$NS" logs deployment/open-rl-scheduler --tail=80 || true
  fi
  if [ "$KEEP" = 1 ]; then
    echo "Kept kubeconfig: $tmp/kubeconfig"
    echo "Inspect with: KUBECONFIG=$tmp/kubeconfig kubectl -n $NS get workloads,pods,resourceclaims"
    [ "$created" != 1 ] || echo "Clean up with: kind delete cluster --name $CLUSTER"
  else
    if [ "$created" = 1 ]; then kind delete cluster --name "$CLUSTER"; fi
    rm -rf "$tmp"
  fi
  exit "$result"
}
if [ "$USE_EXISTING_CLUSTER" = 1 ]; then
  context=${CONTEXT:-$(kubectl config current-context)}
  kubectl --context "$context" config view --minify --flatten > "$tmp/kubeconfig"
fi
export KUBECONFIG="$tmp/kubeconfig"
trap cleanup EXIT
if [ "$USE_EXISTING_CLUSTER" != 1 ]; then
  say "creating kind cluster $CLUSTER ($KIND_NODE_IMAGE)"
  # Reject name collisions before recording ownership for cleanup.
  if kind get clusters | grep -Fxq "$CLUSTER"; then
    echo "FAIL: cluster $CLUSTER already exists; choose a fresh CLUSTER name"
    exit 1
  fi
  created=1
  kind create cluster --name "$CLUSTER" --image "$KIND_NODE_IMAGE" --kubeconfig "$KUBECONFIG" --retain --wait 120s
fi
if kubectl get namespace "$NS" >/dev/null 2>&1; then
  echo "FAIL: namespace $NS already exists; use an empty test cluster"
  exit 1
fi

# Fake GPUs. The example driver is the reference DRA implementation: it
# publishes ResourceSlices whose devices carry a memory capacity, exactly like
# the NVIDIA driver, without needing hardware.
if [ "$DEVICE_CLASS" = gpu.example.com ]; then
  say "installing the DRA example driver (fake GPUs)"
  git init --quiet "$tmp/driver"
  git -C "$tmp/driver" fetch --quiet --depth 1 https://github.com/kubernetes-sigs/dra-example-driver "$DRA_DRIVER_COMMIT"
  git -C "$tmp/driver" checkout --quiet --detach FETCH_HEAD
  helm upgrade --install dra-example-driver "$tmp/driver/deployments/helm/dra-example-driver" \
    --create-namespace --namespace dra-example-driver --wait --timeout 180s \
    --set kubeletPlugin.numDevices="$GPUS"
fi

say "waiting for ResourceSlices from $DEVICE_DRIVER"
ready=0
for _ in $(seq 60); do
  if kubectl get resourceslices -o jsonpath='{range .items[*]}{.spec.driver}{"\n"}{end}' | grep -Fxq "$DEVICE_DRIVER"; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" != 1 ]; then
  echo "FAIL: no ResourceSlices from $DEVICE_DRIVER"
  exit 1
fi
kubectl get resourceslices

say "building and loading the controller image"
docker build -q -t "$IMG" "$scheduler/controller"
if [ "$USE_EXISTING_CLUSTER" != 1 ]; then
  kind load docker-image "$IMG" --name "$CLUSTER"
elif [ -n "$LOAD_INTO" ]; then
  kind load docker-image "$IMG" --name "$LOAD_INTO"
fi

say "deploying the scheduler (smoke overlay)"
kubectl apply --server-side -k "$scheduler/deploy/overlays/smoke"
kubectl -n "$NS" set env deployment/open-rl-scheduler OPEN_RL_PLACEMENT_STRATEGY=spread
if [ "$DEVICE_CLASS" != gpu.example.com ]; then
  # The overlay defaults to the fake driver; real-hardware runs override it.
  kubectl -n "$NS" set env deployment/open-rl-scheduler \
    OPEN_RL_DEVICE_CLASS="$DEVICE_CLASS" OPEN_RL_DEVICE_DRIVER="$DEVICE_DRIVER"
fi

say "opting nodes in"
for node in $(kubectl get nodes -o name); do
  kubectl label --overwrite "$node" \
    openrl.io/enabled=true openrl.io/trainer=true openrl.io/sampler=true
done

if ! kubectl -n "$NS" rollout status deployment/open-rl-scheduler --timeout=240s; then
  echo "FAIL: the scheduler never became ready"
  kubectl -n "$NS" describe pods -l app=open-rl-scheduler | tail -30
  kubectl -n "$NS" logs deployment/open-rl-scheduler --tail=40 || true
  exit 1
fi

# One more worker than the node has GPUs, so the pipeline shows both halves
# of the policy: dedicated claims while devices are free, then a ClaimLedger
# seat on the unschedulable verdict under contention.
WORKERS=$((GPUS + 1))

say "requesting $WORKERS workers of $MEMORY against $GPUS GPUs"
for i in $(seq "$WORKERS"); do
  kubectl apply -f - <<EOF
apiVersion: openrl.io/v1alpha1
kind: Workload
metadata: {name: smoke-$i, namespace: $NS}
spec:
  role: trainer
  modelID: smoke-$i
  ownerID: smoke
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

say "waiting for every worker to run"
kubectl -n "$NS" wait workloads --all --for=jsonpath='{.status.phase}'=Running --timeout=240s

say "asserting dedicated claims, the sharing fallback, and real allocations"
claims=$(kubectl -n "$NS" get workloads -o jsonpath='{range .items[*]}{.status.claimName}{"\n"}{end}')
distinct=$(echo "$claims" | sort -u | grep -c .)
if [ "$distinct" != "$GPUS" ]; then
  echo "FAIL: $WORKERS workers hold $distinct claims, want $GPUS: one per GPU, with the extra worker seated on an existing ledger"
  kubectl -n "$NS" get workloads
  exit 1
fi
for claim in $(echo "$claims" | sort -u); do
  driver=$(kubectl -n "$NS" get resourceclaim "$claim" -o jsonpath='{.status.allocation.devices.results[0].driver}')
  if [ "$driver" != "$DEVICE_DRIVER" ]; then
    echo "FAIL: claim $claim was not allocated by $DEVICE_DRIVER (got: ${driver:-nothing})"
    exit 1
  fi
done
echo "$WORKERS workers on $distinct claims, every claim allocated by $DEVICE_DRIVER"

say "the run, as an operator would see it"
kubectl -n "$NS" get workloads
kubectl -n "$NS" get claimledgers
kubectl -n "$NS" get resourceclaims
kubectl -n "$NS" get pods -o wide

say "PASS"
