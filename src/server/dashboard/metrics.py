"""Physical GPU and worker series, and each node's GPUs as DCGM reports them."""

import asyncio
import os
import time

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


async def node_gpus() -> dict[str, list[dict]] | None:
  """Each node's GPUs now, by UUID and DCGM slot; None without a metrics source. DCGM names the slot DRA does."""
  await sources.discover()
  backend, _ = hardware()
  if backend is None:
    return None
  now = time.time()
  found: dict[str, list[dict]] = {}
  for row in await backend.range_rows(f'max by (Hostname, UUID, gpu) (DCGM_FI_DEV_GPU_UTIL{{UUID!=""{backend.scope("cluster")}}})', now, now):
    labels = row["metric"]
    found.setdefault(labels.get("Hostname"), []).append({"id": labels["UUID"], "uuid": labels["UUID"], "name": f"gpu-{labels.get('gpu')}"})
  return {node: sorted(gpus, key=lambda d: (len(d["name"]), d["name"])) for node, gpus in found.items()}


async def series(state: dict, span: sources.Window, node_id=None, run_id=None, kind="hardware") -> dict:
  """A node's GPUs and workers, idle GPUs included; or a run's. kind=devices is the GPUs alone. Sampler charts are named queries."""
  start, end, low, high = span
  placements = [p for p in inspection.lanes(state, run_id=run_id, node_id=node_id) if p["start"] <= high and p["end"] >= low]
  devices = {uuid: None for p in placements for uuid in p["devices"]}
  if not run_id:
    devices.update({d["uuid"]: None for n in state["cluster"]["nodes"] if not node_id or n.get("name") == node_id for d in n["devices"]})
  workers = {}
  for placement in placements:
    if pod := placement.get("pod"):
      identity = placement.get("allocation_id") or placement["id"]
      worker = workers.setdefault(identity, {"id": identity, "pod": pod, "first_seen": high, "last_seen": low})
      worker["first_seen"] = min(worker["first_seen"], placement["start"])
      worker["last_seen"] = max(worker["last_seen"], placement["end"])
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
    query("devices", [{"id": uuid, "uuid": uuid} for uuid in devices], "uuid", ("utilization", "memory_mib")),
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
