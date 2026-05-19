#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${OVERLAY:-examples/rl/autoresearch/text_sql}"
NAMESPACE="${NAMESPACE:-default}"
DELETE_ARTIFACTS="${DELETE_ARTIFACTS:-0}"
LOG_ROOT="${LOG_ROOT:-}"

kubectl -n "${NAMESPACE}" delete -k "${OVERLAY}" --ignore-not-found=true

if [ "${DELETE_ARTIFACTS}" = "1" ]; then
  if [ -z "${LOG_ROOT}" ]; then
    echo "DELETE_ARTIFACTS=1 requires LOG_ROOT" >&2
    exit 2
  fi
  rm -rf "${LOG_ROOT}"
fi
