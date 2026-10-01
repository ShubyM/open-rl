"""GPU utilization and memory, and the worker's CPU and memory, for one allocation.

Devices are matched by UUID, which the inventory (DRA ResourceSlices on
Kubernetes, nvidia-smi locally) and the hardware source both publish, so
nothing is inferred from a GPU index. The worker is matched by pod name.
"""

from datetime import datetime

import httpx

from server.dashboard import gke, sources
from server.dashboard.snapshot import snapshot
from server.store import get_store

UNAVAILABLE = "GPU metrics query unavailable"


def empty(identity: str, uuid: str | None, reason: str) -> dict:
  return {"id": identity, "uuid": uuid, "utilization": [], "memory_mib": [], "reason": reason}


def device_uuids(state: dict) -> dict[str, str | None]:
  return {device["id"]: device.get("uuid") for node in state["cluster"]["nodes"] for device in node["devices"]}


async def gpu_history(placement_id: str, since=None, until=None) -> dict:
  start, end = gke.time_range(since, until)
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  state = await snapshot.current(get_store())
  placement = next((p for p in state["placements"] if p["id"] == placement_id), None) or next(
    (p for p in state["history"] if p["id"] == placement_id), None
  )
  if placement is None:
    return {"available": False, "reason": "Allocation not found", "devices": [], "worker": None}
  backend, reason = sources.hardware()
  if backend is None:
    return {"available": False, "reason": reason, "devices": [], "worker": None}
  known = device_uuids(state)
  uuids = {identity: known.get(identity) for identity in placement["devices"]}
  result = {"available": False, "reason": None, "source": backend.name, "since": start, "until": end}
  try:
    series = await backend.devices([u for u in uuids.values() if u], low, high)
    result["devices"] = [
      {"id": identity, "uuid": uuid, **series[uuid]} if uuid else empty(identity, None, "GPU UUID unavailable") for identity, uuid in uuids.items()
    ]
  except (httpx.HTTPError, ValueError, KeyError, TypeError):
    result["devices"] = [empty(identity, uuid, UNAVAILABLE) for identity, uuid in uuids.items()]
  pod = placement.get("pod")
  result["worker"] = None
  if pod:
    try:
      result["worker"] = {"pod": pod, **(await backend.workers_series([pod], low, high))[pod]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
      result["worker"] = {"pod": pod, "cpu_cores": [], "memory_bytes": [], "reason": "Worker CPU and memory unavailable"}
  result["available"] = any(d.get("utilization") for d in result["devices"])
  return result
