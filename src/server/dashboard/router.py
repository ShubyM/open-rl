# HTTP surface of the operational dashboard. The JSON endpoints are the same primitives the
# ops CLI uses (health, problems, inspect, logs, stop); the static files are the human UI.

import asyncio
from pathlib import Path

from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from server.dashboard import data, demo
from server.dashboard import logs as log_store
from server.store import get_store

STATIC_DIR = Path(__file__).parent / "static"

router = APIRouter(prefix="/api/v1/dashboard")


@router.get("/snapshot")
async def dashboard_snapshot(request: Request):
  if data.demo_mode_enabled():
    return {
      "schema_version": 1,
      "demo": True,
      "notice": demo.DEMO_NOTICE,
      "cluster": demo.demo_cluster(),
      "runs": demo.demo_runs(),
      "health": demo.demo_health(),
      "problems": demo.demo_problems(),
    }
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  snapshot = await data.diagnostic_snapshot(get_store(), request.app.state.fft_worker_manager, k8s)
  return {**snapshot, "schema_version": 1}


@router.get("/cluster")
async def dashboard_cluster():
  if data.demo_mode_enabled():
    return demo.demo_cluster()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  return await data.cluster_snapshot(get_store(), k8s)


@router.get("/runs")
async def dashboard_runs(request: Request):
  if data.demo_mode_enabled():
    return demo.demo_runs()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  return await data.runs_snapshot(get_store(), request.app.state.fft_worker_manager, k8s["pods"], k8s.get("scheduler"))


@router.post("/runs")
async def dashboard_launch_run(payload: dict):
  if data.demo_mode_enabled():
    return {
      "demo": True,
      "notice": demo.DEMO_NOTICE,
      "request_id": "demo-run-not-created",
      "launched": False,
    }
  from server import gateway

  return await gateway.create_model(payload)


@router.get("/runs/{run_id}")
async def dashboard_run_detail(run_id: str, request: Request, logs: int = 0):
  log_tail = min(max(logs, 0), 2000)
  if data.demo_mode_enabled():
    detail = demo.demo_run_detail(run_id, log_tail)
  else:
    k8s = await asyncio.to_thread(data.k8s_snapshot)
    detail = await data.run_detail(get_store(), request.app.state.fft_worker_manager, run_id, k8s, log_tail)
  if detail is None:
    return JSONResponse(status_code=404, content={"error": f"unknown run: {run_id}"})
  return detail


@router.get("/runs/{run_id}/logs")
async def dashboard_run_logs(
  run_id: str,
  request: Request,
  since: str | None = None,
  until: str | None = None,
  pod: str | None = None,
  container: str | None = None,
  node: str | None = None,
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  q: str = Query("", max_length=512),
  attempt: int | None = Query(None, ge=0),
  limit: int = Query(200, ge=1, le=1000),
  cursor: str | None = Query(None, max_length=2048),
  refresh: bool = True,
):
  options = dict(
    since=since, until=until, pod=pod, container=container, node=node, severity=severity, q=q, attempt=attempt, limit=limit, cursor=cursor
  )
  try:
    # Validate filters/cursor before spending Kubernetes API calls.
    result = await asyncio.to_thread(log_store.query, run_id, **options)
    collection = None
    discovery_available = None
    events = []
    if refresh and not cursor:
      if data.demo_mode_enabled():
        detail = demo.demo_run_detail(run_id, 0)
        pods = (detail or {}).get("pods", [])
        discovery_available = True
      else:
        k8s = await asyncio.to_thread(data.k8s_snapshot)
        pods = data.model_pods(run_id, k8s["pods"])
        discovery_available = k8s.get("available", False)
      for item in pods:
        events.extend({**event, "pod": item["name"], "node": item.get("node")} for event in item.get("events", []))
      selected_pods = [item for item in pods if (not pod or item["name"] == pod) and (not node or item.get("node") == node)]
      if container:
        selected_pods = [{**item, "containers": [c for c in item.get("containers", []) if c["name"] == container]} for item in selected_pods]
      collection = await asyncio.to_thread(log_store.collect, run_id, selected_pods, data.demo_mode_enabled())
      result = await asyncio.to_thread(log_store.query, run_id, **options)
    return {
      **result,
      "demo": data.demo_mode_enabled(),
      "collection": collection,
      "discovery_available": discovery_available,
      "events": events,
      "collector_error": getattr(request.app.state, "log_collector_error", None),
    }
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})
  except Exception:
    return JSONResponse(status_code=503, content={"error": "Log archive unavailable"})


@router.post("/runs/{run_id}/stop")
async def dashboard_stop_run(run_id: str, request: Request):
  if data.demo_mode_enabled():
    return {"demo": True, "notice": demo.DEMO_NOTICE, "run_id": run_id, "stopped": True, "actions": ["demo mode — nothing was actually stopped"]}
  result = await data.stop_run(get_store(), request.app.state.fft_worker_manager, run_id)
  if not result["stopped"]:
    return JSONResponse(status_code=409, content={**result, "error": "nothing to stop for this run"})
  return result


@router.get("/health")
async def dashboard_health(request: Request):
  if data.demo_mode_enabled():
    return demo.demo_health()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  checks = await data.health_checks(get_store(), k8s)
  stats, queues = await data.operational_stats(get_store(), k8s, request.app.state.fft_worker_manager)
  return {"demo": False, "checks": checks, "stats": stats, "queues": queues}


@router.get("/problems")
async def dashboard_problems(request: Request):
  if data.demo_mode_enabled():
    return demo.demo_problems()
  k8s = await asyncio.to_thread(data.k8s_snapshot)
  store = get_store()
  try:
    queues, launch = await asyncio.gather(store.queue_stats(), store.worker_launch_stats())
  except Exception:
    queues, launch = [], {"depth": 0, "oldest_enqueued_at": None, "oldest_age_seconds": None}
  checks, runs, operational = await asyncio.gather(
    data.health_checks(store, k8s),
    data.runs_snapshot(store, request.app.state.fft_worker_manager, k8s["pods"], k8s.get("scheduler"), queues),
    data.operational_stats(store, k8s, request.app.state.fft_worker_manager, queues, launch),
  )
  stats, _ = operational
  return data.problems_payload(data.derive_problems(checks, k8s, stats, runs["runs"]))


@router.get("/pods/{pod}/logs")
async def dashboard_pod_logs(pod: str, container: str | None = None, tail: int = 500, previous: bool = False):
  if data.demo_mode_enabled():
    return demo.demo_pod_logs(pod)
  try:
    return await asyncio.to_thread(data.k8s_pod_logs, pod, container, min(max(tail, 1), 5000), previous)
  except Exception as exc:
    return JSONResponse(status_code=503, content={"error": str(exc)})


def mount_dashboard(app: FastAPI) -> None:
  # The gateway lifespan replaces this with the live manager once FFT workers exist.
  app.state.fft_worker_manager = None
  app.include_router(router)
  app.mount("/dashboard/static", StaticFiles(directory=STATIC_DIR), name="dashboard-static")

  @app.get("/dashboard", include_in_schema=False)
  async def dashboard_index():
    return FileResponse(STATIC_DIR / "index.html")
