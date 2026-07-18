"""Storage helpers for active model state and durable run identity."""

import json
from typing import Any

MODEL_META_PREFIX = "open_rl:model_meta:"
RUN_META_PREFIX = "open_rl:run_meta:"

DURABLE_METADATA_FIELDS = frozenset(
  {
    "base_model",
    "created_at",
    "name",
    "stopped_at",
    "tracker_url",
    "training_kind",
  }
)


def decode_metadata(raw: object) -> dict[str, Any]:
  if raw is None:
    return {}
  try:
    parsed = json.loads(raw) if isinstance(raw, str | bytes | bytearray) else raw
  except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
    return {"base_model": str(raw)}
  return parsed if isinstance(parsed, dict) else {"base_model": str(parsed)}


def durable_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
  """Keep the small identity record useful after active model cleanup."""
  return {field: metadata[field] for field in DURABLE_METADATA_FIELDS if field in metadata}


async def create_run_metadata(store: Any, run_id: str, metadata: dict[str, Any]) -> None:
  """Create both the active-model record and its durable run identity."""
  await store.set_value(f"{MODEL_META_PREFIX}{run_id}", json.dumps(metadata))
  await store.set_value(f"{RUN_META_PREFIX}{run_id}", json.dumps(durable_metadata(metadata)))


async def update_run_metadata(
  store: Any,
  run_id: str,
  updates: dict[str, Any],
  *,
  update_active: bool = False,
) -> None:
  """Merge durable metadata, updating an existing active record when requested.

  The active key is never created here. This prevents a late lifecycle update
  from resurrecting a model after normal cleanup deleted it.
  """
  durable_raw = await store.get_value(f"{RUN_META_PREFIX}{run_id}")
  active_raw = await store.get_value(f"{MODEL_META_PREFIX}{run_id}")
  if durable_raw is None and active_raw is None:
    return
  durable = decode_metadata(durable_raw) or decode_metadata(active_raw)
  durable.update(durable_metadata(updates))
  await store.set_value(f"{RUN_META_PREFIX}{run_id}", json.dumps(durable_metadata(durable)))

  if update_active and active_raw is not None:
    active = decode_metadata(active_raw)
    active.update(updates)
    await store.set_value(f"{MODEL_META_PREFIX}{run_id}", json.dumps(active))
