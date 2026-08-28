#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${KIND_CLUSTER_NAME:-open-rl-dashboard}
PORT=${DASHBOARD_UI_PORT:-9014}
CONTEXT="kind-${CLUSTER_NAME}"

for command in kubectl curl uvx; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 1; }
done

forward_log=$(mktemp)
kubectl --context "$CONTEXT" port-forward service/open-rl-dashboard "${PORT}:8000" >"$forward_log" 2>&1 &
forward_pid=$!
trap 'kill "$forward_pid" 2>/dev/null || true; rm -f "$forward_log"' EXIT

for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/api/v1/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "http://127.0.0.1:${PORT}/api/v1/healthz" >/dev/null

uvx --from playwright playwright install chromium >/dev/null
BASE_URL="http://127.0.0.1:${PORT}" uvx --from playwright python dev/kind/dashboard-ui-smoke.py
