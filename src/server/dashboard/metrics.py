"""Physical GPU, worker and vLLM sampler series, independent of allocation lookup."""

import asyncio
import os
from datetime import datetime

from server.dashboard.inspection import segments
from server.telemetry import gke, kubernetes
from server.telemetry.prometheus import PromQL


def hardware() -> tuple[PromQL | None, str | None]:
  """The series source: an explicit Prometheus, else GKE's managed one, else (None, why)."""
  config = gke.configuration()
  url = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/")
  if url:
    return PromQL("prometheus", url, None, config["cluster"] or None, kubernetes.namespace()), None
  if config["configured"]:
    endpoint = f"https://monitoring.googleapis.com/v1/projects/{config['project']}/location/global/prometheus"
    return PromQL("gke", endpoint, gke.access_token, config["cluster"], kubernetes.namespace()), None
  return None, "No GPU metrics source: not on GKE and OPEN_RL_PROMETHEUS_URL is unset"


async def series(state: dict, node_id=None, since=None, until=None, run_id=None) -> dict:
  """A node's devices and workers, idle GPUs included; or a run's, with its samplers' vLLM series."""
  start, end = gke.time_range(since, until)
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  placements = [p for p in segments(state, run_id=run_id, node_id=node_id) if p["start"] <= high and p["end"] >= low]
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
    found, error = {}, reason
    if backend is not None and identities:
      try:
        found = await getattr(backend, method)(identities, low, high)
      except Exception:
        error = "GPU metrics query unavailable" if method == "devices" else "Worker CPU and memory unavailable"
    for row in rows:
      row["reason"] = error if row[key] else "GPU UUID unavailable"
      row.update(
        {
          field: [p for p in found.get(row[key], {}).get(field, []) if row.get("first_seen", low) <= p[0] <= row.get("last_seen", high)]
          for field in fields
        }
      )
    return rows, error

  async def samplers():
    pods = sorted({p["pod"] for p in placements if p.get("role") == "sampler" and p.get("pod")})
    if not run_id or backend is None or not pods:
      return {}, None if not run_id or pods else "No sampler pods for this run in this window"
    try:
      return await backend.samplers(pods, low, high), None
    except Exception:
      return {}, "vLLM sampler metrics unavailable"

  (gpu, gpu_error), (cpu, cpu_error), (vllm, vllm_error) = await asyncio.gather(
    query("devices", [{"id": identity, "uuid": uuid} for identity, uuid in devices], "uuid", ("utilization", "memory_mib")),
    query("workers_series", list(workers.values()), "pod", ("cpu_cores", "memory_bytes")),
    samplers(),
  )
  result = {
    "source": backend.name if backend else None,
    "reason": reason or gpu_error or cpu_error,
    "since": start,
    "until": end,
    "devices": gpu,
    "workers": cpu,
    **({"samplers": vllm, "samplers_reason": vllm_error or reason} if run_id else {}),
  }
  result["available"] = any(d["utilization"] or d["memory_mib"] for d in result["devices"]) or any(
    w["cpu_cores"] or w["memory_bytes"] for w in result["workers"]
  )
  return result
