#!/usr/bin/env bash
set -euo pipefail

OVERLAY="${OVERLAY:-examples/rl/autoresearch/text_sql}"
NAMESPACE="${NAMESPACE:-default}"
DELETE_ARTIFACTS="${DELETE_ARTIFACTS:-0}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"

kubectl -n "${NAMESPACE}" delete -k "${OVERLAY}" --ignore-not-found=true

if [ "${DELETE_ARTIFACTS}" = "1" ]; then
  if [ -z "${ARTIFACT_ROOT}" ]; then
    echo "DELETE_ARTIFACTS=1 requires ARTIFACT_ROOT" >&2
    exit 2
  fi
  rm -rf "${ARTIFACT_ROOT}"
fi
