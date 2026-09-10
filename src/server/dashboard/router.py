"""The operator UI and the identical read-only interface used by agents."""

import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from server import observability as telemetry
from server.dashboard import experiments, gke, kubernetes, logs, metrics
from server.dashboard.data import observations
from server.store import get_store

router = APIRouter()
STATIC = Path(__file__).parent / "static"


@router.get("/dashboard", include_in_schema=False)
@router.get("/dashboard/", include_in_schema=False)
async def dashboard():
  return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/dashboard/assets/{name}", include_in_schema=False)
async def asset(name: str):
  if name not in {"app.js", "ui.js", "views.js", "timeline.js", "style.css", "time-range.js", "time-range.css", "charts.js", "charts.css"}:
    raise HTTPException(404)
  return FileResponse(STATIC / name, headers={"Cache-Control": "no-cache"})


@router.get("/api/v1/dashboard")
async def inspection_index():
  """Entry point for read-only agent inspection; no browser automation required."""
  return {
    "schema_version": 1,
    "scope": {"namespace": kubernetes.k8s_namespace(), "identity": "shared_operator"},
    "links": {
      "snapshot": "/api/v1/dashboard/snapshot",
      "run": "/api/v1/dashboard/runs/{run_id}",
      "run_logs": "/api/v1/dashboard/runs/{run_id}/logs",
      "run_metrics": "/api/v1/dashboard/runs/{run_id}/metrics",
      "pod_logs": "/api/v1/dashboard/pods/{pod}/logs",
      "gpu_metrics": "/api/v1/dashboard/allocations/{placement_id}/metrics",
      "experiments": "/api/v1/dashboard/experiments",
      "openapi": "/openapi.json",
    },
    "workflow": [
      "Read snapshot for run IDs, pod UIDs, scheduler placements, and source errors.",
      "Read the run to resolve shared-runtime membership before attributing logs or GPU activity.",
      "Read logs and metrics using explicit since/until timestamps; keep filters fixed when following next_cursor.",
      "Report missing sources and coverage gaps alongside findings.",
    ],
    "capabilities": {"read_only": True, "pod_exec": False, "filesystem": False, "secrets": False, "inflight_operation_traces": False},
    "limits": {
      "allocation_history_minutes": 30,
      "operation_samples_per_run": 2000,
      "local_log_records_total": 20000,
      "local_log_retention_days": 7,
      "historical_queries_require_observed_pod_identity": True,
    },
  }


@router.get("/api/v1/dashboard/experiments")
async def experiment_metrics():
  """Reward, correctness and optimizer curves from each run's metrics.jsonl on the shared volume."""
  return await experiments.experiments()


@router.get("/api/v1/dashboard/snapshot")
async def snapshot():
  await gke.discover()
  state = await observations.snapshot(get_store())
  return {**state, "telemetry_sources": {"gke": gke.configuration()}}


@router.get("/api/v1/dashboard/runs/{run_id}")
async def run_detail(run_id: str):
  state = await snapshot()
  run = next((run for run in state["runs"] if run["run_id"] == run_id), None)
  if run is None:
    raise HTTPException(404, "Run not found")
  return {"schema_version": 1, "observed_at": state["observed_at"], "coverage": state["coverage"], **run}


LOCAL_FALLBACK_NOTE = "Cloud Logging is not readable from this gateway; showing the local archive"


async def run_context(run_id: str) -> tuple[dict | None, dict]:
  """The run's current record and the pod lifetimes (live and archived) its logs can come from."""
  state = await snapshot()
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
  archive = await asyncio.to_thread(logs.query, run_id, limit=1)
  return run, {"run": run, "sources": gke.pod_sources(run, archive["sources"], state["observed_at"])}


def with_run_identity(result: dict, run: dict | None, sources: list[dict]) -> dict:
  result["shared_runtime"] = bool(run and run["shared_runtime"]) or any(s.get("shared_runtime") for s in sources)
  result["runtime_run_ids"] = run["runtime_run_ids"] if run else []
  return result


@router.get("/api/v1/dashboard/runs/{run_id}/logs")
async def run_logs(
  run_id: str,
  source: str = Query("auto", pattern="^(auto|local|gke)$"),
  q: str = Query("", max_length=1024),
  pod: str | None = None,
  container: str | None = None,
  node: str | None = None,
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  attempt: int | None = Query(None, ge=0),
  since: str | None = None,
  until: str | None = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: str | None = Query(None, max_length=4096),
):
  filters = dict(
    q=q, pod=pod, container=container, node=node, severity=severity, attempt=attempt, since=since, until=until, limit=limit, cursor=cursor
  )
  note = None
  if source != "local":
    await gke.discover()
  if source == "gke" or (source == "auto" and gke.configuration()["enabled"]):
    run, context = await run_context(run_id)
    try:
      result = await gke.run_logs(run_id, context["sources"], **filters)
    except ValueError as exc:
      raise HTTPException(400, str(exc)) from exc
    # Auto means "the best source that works". When Cloud Logging says no on
    # a first page, the local archive answers and says why the switch happened.
    if not (source == "auto" and result.get("error") and not cursor):
      return with_run_identity(result, run, context["sources"])
    note = LOCAL_FALLBACK_NOTE
  # Archive queries remain valid after a run or pod has gone away.
  try:
    result = await asyncio.to_thread(logs.query, run_id, **filters)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  if note is None:
    run, _ = await run_context(run_id)
  result["source"] = "local"
  if note:
    result["source_note"] = note
  return with_run_identity(result, run, result["sources"])


@router.get("/api/v1/dashboard/allocations/{placement_id}/metrics")
async def allocation_metrics(placement_id: str, since: str | None = None, until: str | None = None):
  try:
    return await metrics.gpu_history(placement_id, since, until)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc


@router.get("/api/v1/dashboard/pods/{pod}/logs")
async def pod_logs(
  pod: str,
  container: str | None = None,
  previous: bool = False,
  tail: int = Query(200, ge=1, le=1000),
):
  from server.dashboard import kubernetes

  try:
    return await asyncio.to_thread(kubernetes.k8s_pod_logs, pod, container, tail, previous)
  except Exception as exc:
    raise HTTPException(503, "Pod logs unavailable in the configured namespace") from exc


@router.get("/api/v1/dashboard/runs/{run_id}/metrics")
async def run_metrics(run_id: str, since: str | None = None, until: str | None = None):
  try:
    start, end = gke.time_range(since, until)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  await gke.discover()
  result = await telemetry.read(get_store(), run_id)
  start_seconds, end_seconds = datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  result["samples"] = [sample for sample in result["samples"] if start_seconds <= sample.get("at", 0) <= end_seconds]
  result["available"] = bool(result["samples"])
  result.update(since=start, until=end)
  if gke.configuration()["enabled"]:
    state = await snapshot()
    run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
    archive = await asyncio.to_thread(logs.query, run_id, limit=1)
    sources = gke.pod_sources(run, archive["sources"], state["observed_at"])
    result["gke"] = await gke.resource_metrics(sources, since, until)
  return result
