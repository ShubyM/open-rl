#!/usr/bin/env bash
# Vendor the Tinker public protobuf schema from a released tinker SDK wheel.
#
# The .proto source is not published; the SDK ships only the generated module
# and its type stubs. This copies both into src/server/proto/ with a header
# recording the SDK version they came from.
#
#   scripts/sync_tinker_proto.sh 0.29.0
set -euo pipefail
version="${1:?usage: $0 <tinker-version>}"
dest="$(cd "$(dirname "$0")/.." && pwd)/src/server/proto"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
uv pip install --quiet --target "$tmp" --no-deps "tinker==${version}"
for f in tinker_public_pb2.py tinker_public_pb2.pyi; do
  {
    echo "# Vendored from the tinker SDK ${version} (tinker/proto/${f}), Apache-2.0."
    echo "# Do not edit; regenerate with scripts/sync_tinker_proto.sh ${version}."
    cat "$tmp/tinker/proto/$f"
  } > "$dest/$f"
done
echo "vendored tinker ${version} proto into $dest"
