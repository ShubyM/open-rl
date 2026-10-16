"""Durable placement history: which worker held which GPUs on which node, and when.

The standalone dashboard records it in the run store every few seconds while a
placement is live, so the Nodes page can draw allocation bars for the whole
selected window and the record outlives the worker.
"""

import asyncio
import json
import time
from typing import Any

KEY_PREFIX = "open_rl:placement:"
RETENTION_SECONDS = 7 * 86400
RECORD_INTERVAL_SECONDS = 10


def changed(previous: dict, placement: dict) -> bool:
  """Compare placement identity, allowing missing device or pod-UID observations."""
  old_uuids, uuids = previous.get("device_uuids") or {}, placement.get("device_uuids") or {}
  return (
    any(previous.get(field) != placement.get(field) for field in ("node", "pod"))
    or bool(previous.get("pod_uid") and placement.get("pod_uid") and previous["pod_uid"] != placement["pod_uid"])
    or bool(previous.get("devices") and placement.get("devices") and set(previous["devices"]) != set(placement["devices"]))
    or any(old_uuids[device] != uuids[device] for device in old_uuids.keys() & uuids.keys())
  )


def reconcile(previous: dict, placement: dict):
  """Extend one placement observation, preserving its history across teardown and replacement."""
  if previous and changed(previous, placement):
    yield {**previous, "id": f"{placement['id']}:{previous['first_seen']}", "worker_id": placement["id"]}
    previous = {}
  yield {
    **previous,
    **placement,
    "worker_id": placement["id"],
    "run_ids": list(dict.fromkeys([*previous.get("run_ids", []), *(placement.get("run_ids") or [])])),
    "label": previous["label"] if previous.get("run_ids") and not placement.get("run_ids") else placement["label"],
    "pod_uid": placement.get("pod_uid") or previous.get("pod_uid"),
    "devices": placement.get("devices") or previous.get("devices") or [],
    "device_uuids": {**(previous.get("device_uuids") or {}), **(placement.get("device_uuids") or {})},
    "shared_runtime": placement.get("shared_runtime") if placement.get("shared_runtime") is not None else previous.get("shared_runtime"),
  }


async def record(store, placements: list[dict], now: float | None = None) -> None:
  """Extend the observed interval, archiving it when the worker changes placement."""
  now = time.time() if now is None else now
  for placement in placements:
    if not placement.get("node"):
      continue
    raw = await store.get_value(KEY_PREFIX + placement["id"])
    previous = json.loads(raw) if raw else {}
    for entry in reconcile(previous, placement):
      if entry["id"] == placement["id"]:
        entry.update(first_seen=entry.get("first_seen", now), last_seen=now)
      await store.set_value(KEY_PREFIX + entry["id"], json.dumps(entry), ttl_seconds=RETENTION_SECONDS)


async def read(store, since: float) -> list[dict[str, Any]]:
  """Placements seen at or after `since`, newest first."""
  entries = [json.loads(raw) for raw in (await store.scan_values(KEY_PREFIX)).values()]
  return sorted((entry for entry in entries if entry["last_seen"] >= since), key=lambda entry: entry["first_seen"], reverse=True)


async def recorder(store, snapshot) -> None:
  """Background loop: ask for the current placements and record them."""
  while True:
    try:
      await record(store, (await snapshot.current(store))["placements"])
    except Exception:
      pass  # the next pass records again; a missed tick costs one interval of history
    await asyncio.sleep(RECORD_INTERVAL_SECONDS)
