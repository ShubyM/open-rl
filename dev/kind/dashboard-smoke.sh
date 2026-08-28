#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${KIND_CLUSTER_NAME:-open-rl-dashboard}
IMAGE=${DASHBOARD_IMAGE:-open-rl-dashboard:dev}
PORT=${DASHBOARD_PORT:-9013}
CONTEXT="kind-${CLUSTER_NAME}"
if [[ -n ${DASHBOARD_BUILD_VERSION:-} ]]; then
  BUILD_VERSION=$DASHBOARD_BUILD_VERSION
else
  BUILD_VERSION=$(git rev-parse --short=12 HEAD 2>/dev/null || echo dev)
  if ! git diff --quiet --ignore-submodules -- || ! git diff --cached --quiet --ignore-submodules --; then
    BUILD_VERSION="${BUILD_VERSION}-dirty"
  fi
fi

for command in kind kubectl docker curl python3; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 1; }
done

if ! kind get clusters | grep -Fxq "$CLUSTER_NAME"; then
  kind create cluster --name "$CLUSTER_NAME" --wait 90s
fi

docker build --build-arg VERSION="$BUILD_VERSION" -f src/server/Dockerfile.gateway -t "$IMAGE" .
kind load docker-image "$IMAGE" --name "$CLUSTER_NAME"
kubectl --context "$CONTEXT" apply -k dev/kind/dashboard
kubectl --context "$CONTEXT" rollout restart deployment/open-rl-dashboard
kubectl --context "$CONTEXT" rollout status deployment/open-rl-dashboard --timeout=120s

forward_log=$(mktemp)
snapshot_file=$(mktemp)
detail_file=$(mktemp)
kubectl --context "$CONTEXT" port-forward service/open-rl-dashboard "${PORT}:8000" >"$forward_log" 2>&1 &
forward_pid=$!
trap 'kill "$forward_pid" 2>/dev/null || true; rm -f "$forward_log" "$snapshot_file" "$detail_file"' EXIT

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/v1/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

curl -fsS "http://127.0.0.1:${PORT}/api/v1/dashboard/snapshot" >"$snapshot_file"
curl -fsS "http://127.0.0.1:${PORT}/dashboard" | grep -Fq "open-rl operations"

dashboard_pod=$(kubectl --context "$CONTEXT" get pods -l app=open-rl-dashboard -o jsonpath='{.items[0].metadata.name}')
curl -fsS "http://127.0.0.1:${PORT}/api/v1/dashboard/pods/${dashboard_pod}/logs?tail=5" |
  python3 -c 'import json,sys; assert "text" in json.load(sys.stdin)'
kubectl --context "$CONTEXT" auth can-i delete pods --as system:serviceaccount:default:open-rl-dashboard | grep -Fxq yes
kubectl --context "$CONTEXT" auth can-i list events --as system:serviceaccount:default:open-rl-dashboard | grep -Fxq yes

# `kubectl auth can-i` requires API discovery, but vanilla Kind deliberately
# has no Metrics Server. A SubjectAccessReview verifies the RBAC independently.
assert_metrics_access() {
  local resource=$1
  local namespace=${2:-}
  python3 -c 'import json,sys; print(json.dumps({"apiVersion":"authorization.k8s.io/v1","kind":"SubjectAccessReview","spec":{"user":"system:serviceaccount:default:open-rl-dashboard","resourceAttributes":{"namespace":sys.argv[2],"verb":"list","group":"metrics.k8s.io","resource":sys.argv[1]}}}))' "$resource" "$namespace" |
    kubectl --context "$CONTEXT" create --raw /apis/authorization.k8s.io/v1/subjectaccessreviews -f - |
    python3 -c 'import json,sys; assert json.load(sys.stdin)["status"]["allowed"]'
}
assert_metrics_access pods default
assert_metrics_access nodes
if kubectl --context "$CONTEXT" logs deployment/open-rl-dashboard | grep -Fq "Building open-rl"; then
  echo "gateway performed an unexpected runtime package rebuild" >&2
  exit 1
fi

python3 - "$snapshot_file" "$BUILD_VERSION" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
  snapshot = json.load(f)

assert snapshot["schema_version"] == 1, snapshot.get("schema_version")
cluster = snapshot["cluster"]
assert cluster["kubernetes"]["available"], cluster["kubernetes"]
assert cluster["kubernetes"]["namespace"] == "default", cluster["kubernetes"]
assert cluster["gateway"]["build"]["revision"] == sys.argv[2], cluster["gateway"]["build"]
assert cluster["kubernetes"]["metrics"] == {
  "installed": False,
  "available": False,
  "error": None,
  "pods_available": False,
  "nodes_available": False,
  "pods_observed": 0,
  "nodes_observed": 0,
}, cluster["kubernetes"]["metrics"]
assert cluster["pools"], "expected the Kind node in a cluster pool"
assert any(pod["name"].startswith("open-rl-dashboard-") for pod in cluster["pods"]), cluster["pods"]
dashboard = next(pod for pod in cluster["pods"] if pod["name"].startswith("open-rl-dashboard-"))
assert dashboard["containers"][0]["image_id"], dashboard["containers"]
assert cluster["scheduler"]["installed"] is False, cluster["scheduler"]
checks = {check["id"]: check for check in snapshot["health"]["checks"]}
assert checks["kubernetes"]["status"] == "ok", checks["kubernetes"]
assert checks["scheduler"]["status"] == "off", checks["scheduler"]
assert checks["visibility.events"]["status"] == "ok", checks["visibility.events"]
assert checks["visibility.metrics"]["status"] == "off", checks["visibility.metrics"]
for stat in snapshot["health"]["stats"]:
  assert {"value_number", "unit", "context", "status"} <= stat.keys(), stat
assert not snapshot["problems"]["problems"], snapshot["problems"]
print("Kind dashboard smoke passed:", len(cluster["pods"]), "pod(s),", len(cluster["pools"]), "pool(s)")
PY

run_id=$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/dashboard/runs" \
  -H 'Content-Type: application/json' \
  -d '{"base_model":"Qwen/Qwen3-0.6B"}' |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["request_id"])')
curl -fsS "http://127.0.0.1:${PORT}/api/v1/dashboard/snapshot" >"$snapshot_file"
curl -fsS "http://127.0.0.1:${PORT}/api/v1/dashboard/runs/${run_id}" >"$detail_file"
python3 - "$snapshot_file" "$detail_file" "$run_id" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
  snapshot = json.load(f)
with open(sys.argv[2]) as f:
  detail = json.load(f)
run_id = sys.argv[3]
run = next(run for run in snapshot["runs"]["runs"] if run["run_id"] == run_id)
assert run["state"]["phase"] in {"queued", "starting", "ready", "failed"}, run
assert isinstance(run["queue_depth"], int), run
assert isinstance(run["telemetry"], dict), run
assert detail["run_id"] == run_id, detail
assert detail["state"]["phase"] == run["state"]["phase"], (run, detail)
assert isinstance(detail["gpu_devices"], int), detail
assert isinstance(detail["diagnostics"], list), detail
assert detail["telemetry"] == run["telemetry"], (run, detail)
PY

echo "Dashboard UI, build/image identity, persistent launch inspection, runtime telemetry, pod logs, and stop permission verified in Kind"
