"""Named PromQL queries: one copy of each expression for the dashboard and agents.

A template names its subject with {gpus} (a run's or node's GPU UUIDs) or {pods} (a run's sampler pods), and may use
{window}, the span in seconds. A query asked for one point is evaluated once at the end of the window.
"""

import asyncio

from server.dashboard import inspection
from server.dashboard.metrics import hardware
from server.dashboard.sources import SAMPLER_QUERIES, Window, any_of

GPU = "{{UUID=~{gpus}{scope}}}"
QUERIES = {
  # By UUID and gpu: DCGM's slot index names a GPU after its node is gone.
  "gpu_util": f"max by (UUID, gpu) (DCGM_FI_DEV_GPU_UTIL{GPU})",
  "gpu_memory_mib": f"max by (UUID, gpu) (DCGM_FI_DEV_FB_USED{GPU})",
  # Duty cycle over the window: the share of time a kernel ran, one number per GPU.
  "gpu_busy": f"max by (UUID, gpu) (avg_over_time(DCGM_FI_DEV_GPU_UTIL{GPU}[{{window}}s]))",
  **{f"vllm_{name}": expression.replace("{M}", "{{pod=~{pods}{scope}}}") for name, expression in SAMPLER_QUERIES.items()},
}


def subject(state: dict, run_id: str | None, node_id: str | None, low: float, high: float) -> tuple[list[str], list[str]]:
  """The GPUs and sampler pods a run's placements held during the window, or a node's GPUs."""
  if node_id and not run_id:
    node = next((n for n in state["cluster"]["nodes"] if n.get("name") == node_id), None)
    return [d["uuid"] for d in (node or {}).get("devices", [])], []
  held = [p for p in inspection.lanes(state, run_id=run_id) if p["start"] <= high and p["end"] >= low]
  return sorted({uuid for p in held for uuid in p["devices"]}), sorted({p["pod"] for p in held if p["role"] == "sampler" and p["pod"]})


async def run(name: str, state: dict, run_id: str | None, node_id: str | None, span: Window, points: int) -> dict:
  start, end, low, high = span
  result = {"name": name, "since": start, "until": end, "series": []}
  backend, reason = hardware()
  if backend is None:
    return {**result, "error": reason}
  gpus, pods = subject(state, run_id, node_id, low, high)
  template = QUERIES[name]
  if ("{gpus}" in template and not gpus) or ("{pods}" in template and not pods):
    return {**result, "error": "No GPUs or sampler pods recorded for this subject in this window"}
  query = template.format(gpus=any_of(gpus), pods=any_of(pods), scope=backend.scope("cluster"), window=max(1, int(high - low)))
  first = high if points == 1 else low
  try:
    rows = await backend.range_rows(query, first, high, points=points)
  except Exception as exc:
    return {**result, "query": query, "error": f"Query failed: {type(exc).__name__}"}
  series = [
    {"labels": r["metric"], "points": [[float(t), float(v)] for t, v in r.get("values", []) if v not in ("NaN", "+Inf", "-Inf")]} for r in rows
  ]
  return {**result, "query": query, "source": backend.name, "series": series}


async def accelerators(state: dict, run_id: str | None, node_id: str | None, span: Window) -> dict:
  """One row per GPU: where it is, what it is, who held it last in the window, and how busy and full it is, from DCGM."""
  start, end, low, high = span
  result = {"since": start, "until": end, "gpus": []}
  backend, reason = hardware()
  if backend is None:
    return {**result, "error": reason}
  gpus = subject(state, run_id, node_id, low, high)[0] if run_id or node_id else None
  if gpus == []:
    return {**result, "error": "No GPUs recorded for this subject in this window"}
  where = f"{{UUID=~{any_of(gpus)}{backend.scope('cluster')}}}" if gpus else f"{{{backend.scope('cluster')[1:]}}}"
  expressions = {
    "util": f"max by (Hostname, UUID, gpu, modelName) (DCGM_FI_DEV_GPU_UTIL{where})",
    "busy": f"max by (UUID) (avg_over_time(DCGM_FI_DEV_GPU_UTIL{where}[{max(1, int(high - low))}s]))",
    "memory_used_mib": f"max by (UUID) (DCGM_FI_DEV_FB_USED{where})",
    "memory_total_mib": f"max by (UUID) (DCGM_FI_DEV_FB_TOTAL{where})",
  }
  try:
    found = dict(zip(expressions, await asyncio.gather(*(backend.range_rows(q, high, high, points=1) for q in expressions.values()))))
  except Exception as exc:
    return {**result, "error": f"Query failed: {type(exc).__name__}"}
  value = {name: {r["metric"].get("UUID"): float(r["values"][-1][1]) for r in rows if r.get("values")} for name, rows in found.items()}
  held = sorted((lane for lane in inspection.lanes(state, run_id=run_id) if lane["start"] <= high and lane["end"] >= low), key=lambda lane: lane["start"])
  holders = {uuid: lane for lane in held for uuid in lane["devices"]}  # the latest placement in the window wins
  for row in found["util"]:
    uuid, labels = row["metric"].get("UUID"), row["metric"]
    lane = holders.get(uuid)
    result["gpus"].append({
      "uuid": uuid,
      "node": labels.get("Hostname"),
      "slot": labels.get("gpu"),
      "model": labels.get("modelName"),
      **{name: round(value[name][uuid], 1) for name in expressions if uuid in value[name]},
      **({"holder": {k: lane[k] for k in ("run_ids", "role", "pod", "start", "end", "ended")}} if lane else {}),
    })
  result["gpus"].sort(key=lambda g: (g["node"] or "", int(g["slot"] or 0)))
  return {**result, "source": backend.name}
