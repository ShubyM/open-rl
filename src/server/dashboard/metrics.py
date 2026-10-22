"""Physical GPU, worker and vLLM sampler series, independent of allocation lookup."""

import asyncio
import os
from datetime import datetime

from server.dashboard import inspection, sources
from server.dashboard.sources import PromQL


def hardware() -> tuple[PromQL | None, str | None]:
  """The series source: an explicit Prometheus, else GKE's managed one, else (None, why)."""
  config = sources.configuration()
  url = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/")
  if url:
    return PromQL("prometheus", url, None, config["cluster"] or None, sources.namespace()), None
  if config["configured"]:
    endpoint = f"https://monitoring.googleapis.com/v1/projects/{config['project']}/location/global/prometheus"
    return PromQL("gke", endpoint, sources.access_token, config["cluster"], sources.namespace(), sources.quota_headers), None
  return None, "No GPU metrics source: not on GKE and OPEN_RL_PROMETHEUS_URL is unset"


async def series(state: dict, node_id=None, since=None, until=None, run_id=None, kind="hardware", samples=()) -> dict:
  """A node's devices and workers, idle GPUs included; or a run's. kind=devices is the GPUs alone. Sampler charts are named queries."""
  start, end = sources.log_window(since, until)
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  placements = [p for p in inspection.sources(state, run_id=run_id, node_id=node_id, samples=samples) if p["start"] <= high and p["end"] >= low]
  nodes = [n for n in state["cluster"]["nodes"] if not node_id or n.get("name") == node_id]
  current = {d["id"]: d.get("uuid") for n in nodes for d in n["devices"]}
  devices, workers = {}, {}
  for placement in placements:
    for identity in placement["devices"]:
      uuid = (placement.get("device_uuids") or {}).get(identity) or (None if placement["ended"] else current.get(identity))
      devices[identity, uuid] = None
    identity, pod = placement.get("allocation_id") or placement["id"], placement.get("pod")
    uid = placement.get("pod_uid")
    if pod:
      worker = workers.setdefault((identity, uid, pod), {"id": identity, "pod": pod, "pod_uid": uid, "first_seen": high, "last_seen": low})
      worker["first_seen"] = min(worker["first_seen"], placement["start"])
      worker["last_seen"] = max(worker["last_seen"], placement["end"])
  known = {identity for identity, _ in devices}
  # The browser serializes snapshot timestamps to milliseconds.
  live_window = high >= datetime.fromisoformat(state["observed_at"]).timestamp() - 0.001
  if not run_id:
    devices.update({(identity, uuid): None for identity, uuid in current.items() if live_window or identity not in known})
  backend, reason = hardware()

  async def query(method, rows, key, fields):
    identities = list(dict.fromkeys(row[key] for row in rows if row[key]))
    found, errors, succeeded = {}, {}, 0
    if backend is not None and identities:
      found, errors = await getattr(backend, method)(identities, low, high)
      succeeded = len(fields) - len(errors)
    elif reason:
      errors = dict.fromkeys(fields, reason)
    for row in rows:
      row["errors"] = errors if row[key] else dict.fromkeys(fields, "GPU UUID unavailable")
      row["reason"] = next(iter(row["errors"].values()), None)
      for field in fields:
        row[field] = [p for p in found.get(field, {}).get(row[key], []) if row.get("first_seen", low) <= p[0] <= row.get("last_seen", high)]
    return rows, errors, succeeded

  (gpu, gpu_errors, gpu_ok), (cpu, cpu_errors, cpu_ok) = await asyncio.gather(
    query("devices", [{"id": identity, "uuid": uuid} for identity, uuid in devices], "uuid", ("utilization", "memory_mib")),
    query("workers_series", list(workers.values()) if kind != "devices" else [], "pod", ("cpu_cores", "memory_bytes")),
  )
  error = next(iter([*gpu_errors.values(), *cpu_errors.values()]), None)
  result = {
    "source": backend.name if backend else None,
    "reason": reason or error,
    "error": reason or (error if not gpu_ok + cpu_ok else None),
    "since": start,
    "until": end,
    "devices": [{**row, "slot": backend.slots.get(row["uuid"])} if backend and row["uuid"] in backend.slots else row for row in gpu],
    "workers": cpu,
  }
  result["available"] = any(d["utilization"] or d["memory_mib"] for d in gpu) or any(w["cpu_cores"] or w["memory_bytes"] for w in cpu)
  return result
