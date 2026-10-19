"""Physical GPU, worker and vLLM sampler series, independent of allocation lookup."""

import asyncio
import os
from datetime import datetime

from server.dashboard.inspection import sources
from server.telemetry import gke, kubernetes
from server.telemetry.prometheus import SAMPLER_QUERIES, PromQL


def hardware() -> tuple[PromQL | None, str | None]:
  """The series source: an explicit Prometheus, else GKE's managed one, else (None, why)."""
  config = gke.configuration()
  url = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/")
  if url:
    return PromQL("prometheus", url, None, config["cluster"] or None, kubernetes.namespace()), None
  if config["configured"]:
    endpoint = f"https://monitoring.googleapis.com/v1/projects/{config['project']}/location/global/prometheus"
    return PromQL("gke", endpoint, gke.access_token, config["cluster"], kubernetes.namespace(), gke.quota_headers), None
  return None, "No GPU metrics source: not on GKE and OPEN_RL_PROMETHEUS_URL is unset"


async def series(state: dict, node_id=None, since=None, until=None, run_id=None, kind="all", samples=()) -> dict:
  """A node's devices and workers, idle GPUs included; or a run's, with its samplers' vLLM series."""
  start, end = gke.time_range(since, until)
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  placements = [p for p in sources(state, run_id=run_id, node_id=node_id, samples=samples) if p["start"] <= high and p["end"] >= low]
  nodes = [n for n in state["cluster"]["nodes"] if not node_id or n.get("name") == node_id]
  current = {d["id"]: d.get("uuid") for n in nodes for d in n["devices"]}
  devices, workers = {}, {}
  for placement in placements if kind != "samplers" else []:
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
  if not run_id and kind != "samplers":
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

  async def samplers():
    pods = sorted({p["pod"] for p in placements if p.get("role") == "sampler" and p.get("pod")})
    if kind == "hardware" or not run_id or not pods:
      return {}, {}, 0
    if backend is None:
      return {}, dict.fromkeys(SAMPLER_QUERIES, reason), 0
    found, errors = await backend.samplers(pods, low, high)
    return found, errors, len(SAMPLER_QUERIES) - len(errors)

  (gpu, gpu_errors, gpu_ok), (cpu, cpu_errors, cpu_ok), (vllm, vllm_errors, vllm_ok) = await asyncio.gather(
    query("devices", [{"id": identity, "uuid": uuid} for identity, uuid in devices], "uuid", ("utilization", "memory_mib")),
    query("workers_series", list(workers.values()), "pod", ("cpu_cores", "memory_bytes")),
    samplers(),
  )
  error = next(iter([*gpu_errors.values(), *cpu_errors.values(), *vllm_errors.values()]), None)
  result = {
    "source": backend.name if backend else None,
    "reason": reason or error,
    "error": reason or (error if not gpu_ok + cpu_ok + vllm_ok else None),
    "since": start,
    "until": end,
    "devices": gpu,
    "workers": cpu,
    **(
      {
        "samplers": vllm,
        "samplers_errors": vllm_errors,
        "samplers_reason": next(iter(vllm_errors.values()), reason) or (None if vllm else "No sampler pods for this run in this window"),
      }
      if run_id and kind != "hardware"
      else {}
    ),
  }
  result["available"] = (
    any(d["utilization"] or d["memory_mib"] for d in gpu) or any(w["cpu_cores"] or w["memory_bytes"] for w in cpu) or any(vllm.values())
  )
  return result
