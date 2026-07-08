#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PINNED_SHA="f46ef86e4788545622db25dcffa3aebb7a139929"
LAB_ROOT="harvey-labs"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    if [[ "$1" == "podman" ]]; then
      echo "missing required command: podman"
      echo "install hint: sudo apt-get install -y podman"
    else
      echo "missing required command: $1"
    fi
    exit 1
  fi
}

require_cmd git
require_cmd uv
require_cmd podman

if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "warning: GOOGLE_API_KEY is not set"
fi

if [[ ! -d "$LAB_ROOT" ]]; then
  git clone https://github.com/harveyai/harvey-labs "$LAB_ROOT"
fi

git -C "$LAB_ROOT" checkout "$PINNED_SHA"

if git -C "$LAB_ROOT" apply --reverse --check ../patches/full-transcript.patch >/dev/null 2>&1; then
  echo "full transcript patch already applied"
else
  git -C "$LAB_ROOT" apply --check ../patches/full-transcript.patch
  git -C "$LAB_ROOT" apply ../patches/full-transcript.patch
fi

(
  cd "$LAB_ROOT"
  uv sync
)

echo "ready: harvey-labs is pinned, patched, and synced"
echo "next: python3 make_split.py --areas banking-finance --out split.json"
echo "next: python3 collect.py --split split.json --subset train --model gemini-3.5-flash --judge-model gemini-3.5-flash"
