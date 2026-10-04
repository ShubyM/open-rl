"""The operator UI and the identical interface used by agents. Everything is
read-only except deleting the records of finished or abandoned runs."""

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from server.dashboard import experiments, metrics
from server.dashboard.snapshot import TERMINAL_STATUSES, snapshot
from server.session_registry import SessionRegistry
from server.store import get_store
from server.telemetry import backends, gke, kubernetes, ops

router = APIRouter()
STATIC = Path(__file__).parent / "static"


@router.get("/dashboard", include_in_schema=False)
@router.get("/dashboard/", include_in_schema=False)
async def dashboard_page():
  return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/dashboard/assets/{name}", include_in_schema=False)
async def dashboard_asset(name: str):
  path = (STATIC / name).resolve()
  if path.parent != STATIC.resolve() or not path.is_file():
    raise HTTPException(404, "Not found")
  return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.get("/api/v1/dashboard")
async def inspection_index():
  """Entry point for read-only agent inspection; no browser automation required."""
  return {
    "schema_version": 2,
    "scope": {"namespace": kubernetes.namespace(), "identity": "shared_operator"},
    "links": {
      "snapshot": "/api/v1/dashboard/snapshot",
      "run": "/api/v1/dashboard/runs/{run_id}",
      "run_logs": "/api/v1/dashboard/runs/{run_id}/logs",
      "run_metrics": "/api/v1/dashboard/runs/{run_id}/metrics",
      "pod_logs": "/api/v1/dashboard/pods/{pod}/logs",
      "gpu_metrics": "/api/v1/dashboard/allocations/{placement_id}/metrics",
      "gpu_turns": "/api/v1/dashboard/allocations/{placement_id}/turns",
      "experiments": "/api/v1/dashboard/experiments",
      "delete_runs": "POST /api/v1/dashboard/runs/delete",
      "openapi": "/openapi.json",
    },
    "workflow": [
      "Read snapshot for run IDs, pods, placements and their history, and source errors.",
      "Read the run to resolve shared-runtime membership before attributing logs or GPU activity.",
      "Read logs and metrics with explicit since/until timestamps; keep filters fixed when following next_cursor.",
    ],
    "capabilities": {"read_only": False, "delete_runs": True, "pod_exec": False, "filesystem": False, "secrets": False},
    "limits": {"placement_history_days": 7, "operation_samples_per_run": ops.SAMPLE_LIMIT},
  }


@router.get("/api/v1/dashboard/snapshot")
async def snapshot_view():
  await gke.discover()
  state = await snapshot.current(get_store())
  return {**state, "telemetry_sources": await asyncio.to_thread(backends.current().describe)}


@router.get("/api/v1/dashboard/experiments")
async def experiment_metrics():
  """Reward, correctness and optimizer curves from each run's metrics.jsonl on the shared volume."""
  return await experiments.experiments()


async def run_of(run_id: str) -> tuple[dict, dict]:
  state = await snapshot.current(get_store())
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
  if run is None:
    raise HTTPException(404, "Run not found")
  return state, run


@router.get("/api/v1/dashboard/runs/{run_id}")
async def run_detail(run_id: str):
  state, run = await run_of(run_id)
  return {"observed_at": state["observed_at"], **run, "placements": [p for p in state["placements"] if run_id in p["run_ids"]]}


class DeleteRuns(BaseModel):
  run_ids: list[str] = Field(min_length=1, max_length=1000)


@router.post("/api/v1/dashboard/runs/delete")
async def delete_runs(body: DeleteRuns):
  """Forget finished or abandoned runs: their record and their recorded ops.
  Workers, checkpoints and metrics files are not touched. A run that still has
  workers or a live client session is kept and reported with the reason."""
  store = get_store()
  runs = {r["run_id"]: r for r in (await snapshot.current(store))["runs"]}
  sessions = SessionRegistry(store)
  deleted, kept = [], []
  for run_id in dict.fromkeys(body.run_ids):
    run = runs.get(run_id)
    reason = "Run not found" if run is None else run["delete_blocker"]
    if reason is None and str(run["status"]).lower() not in TERMINAL_STATUSES:
      session_id = ((await store.get_model_metadata(run_id)) or {}).get("session_id")
      if session_id and await sessions.live(session_id):
        reason = "The run's client session is still live"
    if reason:
      kept.append({"run_id": run_id, "reason": reason})
      continue
    await store.delete_values(f"open_rl:model_meta:{run_id}", ops.key(run_id))
    deleted.append(run_id)
  if deleted:
    snapshot.invalidate()
  return {"deleted": deleted, "kept": kept}


@router.get("/api/v1/dashboard/runs/{run_id}/metrics")
async def run_metrics(run_id: str, since: str | None = None, until: str | None = None):
  """The worker's own operation records: outcome, timing and queue delay per request."""
  try:
    start, end = gke.time_range(since, until)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  result = await ops.read(get_store(), run_id)
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  result["samples"] = [sample for sample in result["samples"] if low <= sample.get("at", 0) <= high]
  return {**result, "available": bool(result["samples"]), "since": start, "until": end}


@router.get("/api/v1/dashboard/runs/{run_id}/logs")
async def run_logs(
  run_id: str,
  q: str = Query("", max_length=1024),
  pod: str | None = None,
  node: str | None = None,
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: str | None = None,
  until: str | None = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: str | None = Query(None, max_length=4096),
):
  """Cloud Logging for every pod the run has owned, live or gone."""
  await gke.discover()
  state, run = await run_of(run_id)
  sources = gke.pod_sources(run, state["history"], state["observed_at"])
  try:
    result = await gke.run_logs(run_id, sources, since=since, until=until, cursor=cursor, limit=limit, q=q, pod=pod, node=node, severity=severity)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  return {**result, "shared_runtime": run["shared_runtime"], "runtime_run_ids": run["runtime_run_ids"]}


@router.get("/api/v1/dashboard/pods/{pod}/logs")
async def pod_logs(pod: str, container: str | None = None, previous: bool = False, tail: int = Query(200, ge=1, le=1000)):
  """Current or previous container output straight from the kubelet, for any pod in the namespace; locally, the worker's log file."""
  try:
    return await asyncio.to_thread(backends.current().pod_logs, pod, container, tail, previous)
  except Exception as exc:
    raise HTTPException(503, "Pod logs unavailable in the configured namespace") from exc


@router.get("/api/v1/dashboard/allocations/{placement_id}/turns")
async def allocation_turns(placement_id: str, since: str | None = None, until: str | None = None):
  """The exclusive GPU turns the time-slicer granted this allocation's worker, recorded by the worker itself."""
  try:
    start, end = gke.time_range(since, until)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  state = await snapshot.current(get_store())
  placement = next((p for p in state["placements"] if p["id"] == placement_id), None) or next(
    (p for p in state["history"] if p["id"] == placement_id), None
  )
  if placement is None:
    raise HTTPException(404, "Allocation not found")
  low, high = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  turns = [t for t in await ops.read_turns(get_store(), placement["name"]) if t.get("at", 0) >= low and t.get("started_at", 0) <= high]
  return {"placement_id": placement_id, "workload": placement["name"], "samples": turns, "available": bool(turns), "since": start, "until": end}


@router.get("/api/v1/dashboard/allocations/{placement_id}/metrics")
async def allocation_metrics(placement_id: str, since: str | None = None, until: str | None = None):
  try:
    return await metrics.gpu_history(placement_id, since, until)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
