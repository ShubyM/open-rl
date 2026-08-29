"""Publish one LoRA adapter snapshot into the directory layout samplers read.

Two backends write adapters for sampling and they produce the files very
differently -- peft's save_pretrained on the FSDP path, the bridge's collective
save_hf_adapter on the Megatron one -- but the *directory* contract they have to
honour is identical, because gateway.sampler_adapter_path resolves every
sampling ref through it:

    peft/<adapter_id>/<session_label>/    the immutable snapshot vLLM loads
    peft/<adapter_id>/<alias>            symlink, for tinker://.../final refs
    peft/<adapter_id>/metadata.json

The contract is not obvious and getting it wrong fails in ways that look like
something else. Writing in place while a sampler reads gave "<dir> doesn't
contain tensors"; a missing alias symlink returns a ref to a directory that was
never written. This lives in one place so a second backend cannot rediscover
either bug.
"""

import json
import os
import shutil
import time
from collections.abc import Callable
from datetime import datetime

from training import paths
from training.distributed import is_primary

SNAPSHOT_KEEP = 4


def publish(
  adapter_id: str,
  write_files: Callable[[str], str],
  alias: str | None = None,
  session_label: str | None = None,
  keep: int = SNAPSHOT_KEEP,
) -> str:
  """Stage an adapter, then move it into place with a rename.

  write_files is handed a staging directory and returns the path it wrote the
  adapter into (peft nests it under the adapter id; the bridge does not). It is
  called on EVERY rank, because a tensor-parallel export is collective even
  though only rank 0 has anything to write -- everything after it is rank 0's
  alone. Returns the snapshot directory.
  """
  adapter_root = os.path.join(paths.snapshot_root(), adapter_id)
  final_dir = os.path.join(adapter_root, session_label or adapter_id)
  staging_root = os.path.join(adapter_root, f".staging-{os.getpid()}-{time.time_ns()}")
  if is_primary():
    os.makedirs(staging_root, exist_ok=True)

  try:
    staged_adapter = write_files(staging_root)
    if not is_primary():
      return final_dir

    if os.path.exists(final_dir):
      # Only the legacy label (adapter_id itself) can collide; snapshot
      # labels are unique per save. Move the old dir aside, never delete
      # under a reader.
      os.replace(final_dir, os.path.join(staging_root, "replaced"))
    os.rename(staged_adapter, final_dir)

    if alias and alias != os.path.basename(final_dir):
      # Alias-named refs (e.g. tinker://<id>/sampler_weights/final) resolve
      # to peft/<id>/<alias>, but the adapter itself lives in the snapshot
      # dir — without this link the returned ref points at a directory that
      # was never written and every sample against it fails.
      alias_path = os.path.join(adapter_root, alias)
      staged_link = os.path.join(staging_root, "alias-link")
      os.symlink(os.path.basename(final_dir), staged_link)
      if os.path.isdir(alias_path) and not os.path.islink(alias_path):
        os.replace(alias_path, os.path.join(staging_root, "replaced-alias"))
      os.replace(staged_link, alias_path)
  finally:
    shutil.rmtree(staging_root, ignore_errors=True)

  metadata = {"model_id": adapter_id, "created_at": datetime.now().isoformat(), "timestamp": time.time()}
  if alias is not None:
    metadata["alias"] = alias
  with open(os.path.join(adapter_root, "metadata.json"), "w") as f:
    json.dump(metadata, f)

  prune(adapter_root, keep=keep, current=final_dir)
  return final_dir


def prune(adapter_root: str, keep: int, current: str) -> None:
  """Delete all but the newest `keep` snapshot dirs (in-flight rollouts may
  still sample from a recent previous snapshot). Snapshots an alias symlink
  (e.g. "final") points at are kept regardless of age."""
  try:
    entries = os.listdir(adapter_root)
    alias_targets = {os.path.realpath(os.path.join(adapter_root, name)) for name in entries if os.path.islink(os.path.join(adapter_root, name))}
    snapshots = sorted(
      (
        os.path.join(adapter_root, name)
        for name in entries
        if name.startswith("sampler-") and os.path.isdir(os.path.join(adapter_root, name)) and not os.path.islink(os.path.join(adapter_root, name))
      ),
      key=os.path.getmtime,
      reverse=True,
    )
    for stale in snapshots[keep:]:
      if stale != current and os.path.realpath(stale) not in alias_targets:
        shutil.rmtree(stale, ignore_errors=True)
  except OSError:
    pass
