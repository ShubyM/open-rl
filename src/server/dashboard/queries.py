"""Named PromQL queries: one copy of each expression for the dashboard, agents and checks.

A template names its subject with {gpus} (a run's or node's GPU UUIDs) or {pods} (a run's sampler pods), and may use
{window}, the span in seconds. A query asked for one point is evaluated once at the end of the window.
"""

from datetime import datetime

from server.dashboard import inspection
from server.dashboard.metrics import hardware
from server.dashboard.sources import SAMPLER_QUERIES, any_of, log_window

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


async def run(name: str, state: dict, run_id: str | None, node_id: str | None, since, until, points: int) -> dict:
  start, end = log_window(since, until)
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
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
