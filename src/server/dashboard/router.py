"""The operator UI and the same read-only JSON API for agents."""

import asyncio
import math
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute
from pydantic import ValidationError

from server import telemetry
from server.dashboard import collector, inspection, metrics, sources
from server.dashboard.schema import Activity, CollectorMetrics, History, Inventory, Logs, Operation, PodLogs, Series
from server.dashboard.snapshot import METADATA_PREFIX, snapshot
from server.model_metadata import decode_model_metadata
from server.store import get_state_store

REQUEST_SECONDS = 12
# A long run holds tens of thousands of operation records; reading them needs more than a
# few seconds over a port-forward.
OPERATIONS_READ_SECONDS = 15
# One read and check of a run's or worker's records serves every page asking within this
# long, and a retry after a timeout joins the read still running instead of starting another.
# A finished run's records no longer change, so they, and its activity, are kept longer.
OPERATIONS_SECONDS = 10
FINISHED_SECONDS = 300
operations: OrderedDict[str, tuple[float, asyncio.Task]] = OrderedDict()
finished_activity: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()


class InspectionRoute(APIRoute):
  def get_route_handler(self):
    handler = super().get_route_handler()

    async def bounded(request):
      try:
        async with asyncio.timeout(REQUEST_SECONDS):
          return await handler(request)
      except TimeoutError as exc:
        raise HTTPException(504, "Inspection timed out; try again") from exc

    return bounded


router = APIRouter(tags=["inspection"], route_class=InspectionRoute)
STATIC = Path(__file__).parent / "static"
TimeBound = Annotated[
  str | None,
  Query(
    max_length=64,
    description="ISO-8601 timestamp, e.g. 2026-01-02T12:00:00Z; not Unix seconds. "
    "Defaults to the 30 minutes ending at inventory.observed_at. Maximum window: seven days.",
  ),
]


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


def time_range(since, until):
  try:
    start, end = sources.log_window(since, until)
    return start, end, datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 since/until timestamps and a positive window of at most seven days") from exc


def validate_bounds(since, until):
  # Validate before touching sources, but resolve an omitted end against the snapshot.
  if until is not None:
    time_range(since, until)
  elif since is not None:
    try:
      sources.timestamp(since)
    except (ValueError, OverflowError, OSError) as exc:
      raise HTTPException(400, "since must be an ISO-8601 timestamp") from exc


@router.get("/api/v1/inspect/inventory", operation_id="inventory", response_model=Inventory, response_model_exclude_unset=True)
async def inventory():
  """Current runs, placements, nodes and scheduler state, with an error field per unavailable source."""
  await sources.discover()
  state = await snapshot.current(get_state_store())
  return {key: value for key, value in state.items() if key != "processes"}


def is_finished(run: dict | None) -> bool:
  return bool(run) and (run.get("status") != "active" or run.get("display_status") in ("Ended", "Stopped"))


async def read_operations(identity: str, finished: bool = False) -> dict:
  now = time.monotonic()
  entry = operations.get(identity)
  if entry is None or (entry[0] <= now and entry[1].done()):
    task = asyncio.create_task(load_operations(identity))
    operations[identity] = entry = (now + (FINISHED_SECONDS if finished else OPERATIONS_SECONDS), task)
    operations.move_to_end(identity)
    while len(operations) > 64:
      operations.popitem(last=False)
  result = await asyncio.shield(entry[1])
  if result.get("error") and operations.get(identity) is entry:
    operations.pop(identity)
  return result


async def load_operations(identity: str) -> dict:
  try:
    records = await asyncio.wait_for(get_state_store().read_samples(telemetry.ops_key(identity)), timeout=OPERATIONS_READ_SECONDS)
    if not isinstance(records, list):
      raise ValueError("Expected telemetry records")
  except Exception:
    return {"samples": [], "error": "Operation records unavailable"}

  def validate():
    samples, invalid = [], 0
    for record in records:
      try:
        Operation.model_validate(record, strict=True)
        samples.append(record)
      except ValidationError:
        invalid += 1
    return {"samples": samples, **({"partial": f"Skipped {invalid} invalid operation records"} if invalid else {})}

  return await asyncio.to_thread(validate)


@router.get("/api/v1/inspect/activity", operation_id="activity", response_model=Activity, response_model_exclude_unset=True)
async def activity(since: TimeBound = None, until: TimeBound = None, run_id: str | None = None, node_id: str | None = None):
  """Per-process GPU turns and completed-op intervals, in Unix seconds.
  With run_id, also the run, its recorded ops and recent pod incidents."""
  validate_bounds(since, until)
  store = get_state_store()
  state, run = await run_of(run_id) if run_id else (await snapshot.current(store), None)
  start, end, low, high = time_range(since, until or state["observed_at"])
  finished = is_finished(run)
  cached = finished_activity.get((run_id, node_id, start, end)) if finished else None
  if cached and cached[0] > time.monotonic():
    return {**cached[1], "run": run, "observed_at": state["observed_at"]}
  selected = [p for p in inspection.segments(state, run_id, node_id) if p["start"] <= high and p["end"] >= low]
  # Each lane's runs, and its runtime for the worker-level ops recorded there.
  keys = sorted({k for p in selected for k in [*p["run_ids"], p.get("runtime_id")] if k} | ({run_id} if run_id else set()))

  records = dict(zip(keys, await asyncio.gather(*(read_operations(k, finished) for k in keys))))
  result = await asyncio.to_thread(inspection.activity, state, low, high, records, run_id, node_id)
  result.update(since=start, until=end, observed_at=state["observed_at"])
  result["coverage"] = (
    f"At most {telemetry.SAMPLE_LIMIT} completed records per run/worker, best effort. "
    "Missing records do not prove no requests arrived or that workers were idle. Queued and in-flight requests are in inventory runs[].progress."
  )
  if run_id:
    result.update(
      run=run,
      samples=[s for s in records[run_id]["samples"] if low <= s["at"] <= high],
      error=records[run_id].get("error"),
      partial=records[run_id].get("partial"),
      incidents=sorted([*inspection.incidents(run, low, high), *await sdk_incidents(store, run_id, low, high)], key=lambda e: e["at"], reverse=True),
    )
  if finished and not result.get("error"):
    finished_activity[(run_id, node_id, start, end)] = (time.monotonic() + FINISHED_SECONDS, result)
    while len(finished_activity) > 16:
      finished_activity.popitem(last=False)
  return result


async def sdk_incidents(store, run_id: str, low: float, high: float) -> list[dict]:
  """SDK errors from the run's own sessions, the newest of each reason with its count; none when the store cannot say."""
  try:
    row = await store.get_value(METADATA_PREFIX + run_id)
    sessions = decode_model_metadata(row).session_ids if row else []
    samples = [s for session in sessions for s in await store.read_samples(telemetry.sdk_errors_key(session))]
  except Exception:
    return []
  newest = {}
  for s in sorted(samples, key=lambda s: s.get("at", 0), reverse=True):
    if low <= s.get("at", 0) <= high:
      incident = {"at": s["at"], "pod": telemetry.CLIENT, "reason": s["reason"], "tone": "error", "message": s.get("message"), "count": 0}
      newest.setdefault(s["reason"], incident)["count"] += 1
  return list(newest.values())


@router.get("/api/v1/inspect/series", operation_id="series", response_model=Series, response_model_exclude_unset=True)
async def series(
  node_id: str | None = None,
  run_id: str | None = None,
  since: TimeBound = None,
  until: TimeBound = None,
  kind: Literal["all", "hardware", "devices", "samplers"] = "all",
):
  """GPU utilization/memory and worker CPU/memory over time, including idle devices and past placements.
  With run_id, the run's own GPUs and workers, plus throughput, queue, KV cache and latency of its vLLM samplers.
  kind=devices is the GPUs alone, for every node when no node or run is named."""
  validate_bounds(since, until)
  if kind == "samplers" and not run_id:
    raise HTTPException(400, "Sampler series require run_id")
  await sources.discover()
  state = await snapshot.current(get_state_store())
  start, end, _, _ = time_range(since, until or state["observed_at"])
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None) if run_id else None
  records = await read_operations(run_id, is_finished(run)) if run_id else {"samples": []}
  result = await metrics.series(state, node_id=node_id, since=start, until=end, run_id=run_id, kind=kind, samples=records["samples"])
  # Without its sampler records a run still has its current pods' series; say what is missing.
  if records.get("error") and run_id:
    result["samplers_reason"] = result.get("samplers_reason") or f"{records['error']}; samplers that are gone are not shown"
  return result


# Client step history that gateways kept before OpenTelemetry; read only, nothing writes it now.
HISTORY_PREFIX = "open_rl:run_metrics:"


@router.get("/api/v1/inspect/history", operation_id="history", response_model=History, response_model_exclude_unset=True)
async def history(run_id: str):
  """A run's client metrics by step, for runs from before OpenTelemetry; empty for newer runs."""
  try:
    records = await asyncio.wait_for(get_state_store().read_samples(HISTORY_PREFIX + run_id), timeout=10)
  except Exception:
    return {"available": False, "error": "Stored history unavailable", "rows": []}
  latest = {}
  for r in records:
    if isinstance(r, dict) and isinstance(r.get("step"), int) and isinstance(r.get("metrics"), dict):
      metrics = {k: float(v) for k, v in r["metrics"].items() if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)}
      latest[r["step"]] = {"step": r["step"], "metrics": metrics}
  return {"available": True, "rows": [latest[step] for step in sorted(latest)]}


@router.get("/api/v1/inspect/telemetry", operation_id="telemetry", response_model=CollectorMetrics, response_model_exclude_unset=True)
async def collector_metrics(
  run_id: str | None = None,
  node_id: str | None = None,
  since: TimeBound = None,
  until: TimeBound = None,
  metric: Annotated[str, Query(max_length=200)] = "",
):
  """Discovered client/server gauges sampled by Prometheus, with their original scope.
  Session/runtime-wide values are related to a model, not exclusively attributable to it."""
  if bool(run_id) == bool(node_id):
    raise HTTPException(400, "Choose exactly one run_id or node_id")
  _, _, start, end = time_range(since, until)
  sessions, runtimes = [], []
  if run_id:
    try:
      meta = decode_model_metadata(await get_state_store().get_value(METADATA_PREFIX + run_id))
    except Exception as exc:
      raise HTTPException(503, "Run metadata unavailable") from exc
    if meta is None:
      raise HTTPException(404, "Run not found")
    sessions = meta.session_ids
    if runtime := meta.runtime(run_id):
      runtimes = [runtime, meta.pool(run_id)]
  await sources.discover()
  return await collector.series(start, end, run_id=run_id, node_id=node_id, session_ids=sessions, runtime_ids=runtimes, metric=metric)


async def run_of(run_id: str) -> tuple[dict, dict]:
  store = get_state_store()
  state = await snapshot.current(store)
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
  if run is None:
    # A run created since the cached snapshot: read again rather than wait out the cache.
    try:
      created = await store.get_value(METADATA_PREFIX + run_id)
    except Exception as exc:
      raise HTTPException(503, "Run store unavailable") from exc
    if not created:
      raise HTTPException(404, "Run record not found")
    snapshot.invalidate()
    state = await snapshot.current(store)
    run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
    if run is None:
      raise HTTPException(503, "Run store unavailable")
  return state, run


@router.get("/api/v1/inspect/logs", operation_id="logs", response_model=Logs, response_model_exclude_unset=True)
async def run_logs(
  run_id: str,
  q: str = Query("", max_length=1024),
  pod: str | None = None,
  node: str | None = None,
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: TimeBound = None,
  until: TimeBound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: str | None = Query(None, max_length=sources.MAX_CURSOR),
):
  """Cloud Logging for every pod the run has owned, live or gone."""
  validate_bounds(since, until)
  await sources.discover()
  lifetimes, shared_runtime = [], False
  if not cursor:
    state, run = await run_of(run_id)
    records = await read_operations(run_id, is_finished(run))
    since, until, _, _ = time_range(since, until or state["observed_at"])
    lifetimes = sources.pod_sources(run, inspection.sources(state, run_id, samples=records["samples"]), state["observed_at"])
    shared_runtime = run["shared_runtime"]
    # Without its records a run's logs still cover the pods it is known to have had.
    note = f"{records['error']}; logs of pods known only from those records are not shown" if records.get("error") else None
  else:
    note = None
  try:
    result = await sources.run_logs(
      run_id,
      lifetimes,
      since=since,
      until=until,
      cursor=cursor,
      limit=limit,
      q=q,
      pod=pod,
      node=node,
      severity=severity,
      shared_runtime=shared_runtime,
    )
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  if not sources.configuration()["configured"] and pod in (None, sources.CLIENT) and not cursor:
    # Without Cloud Logging the run's client output still has its tail in the run store.
    try:
      client = await asyncio.wait_for(sources.client_tail(get_state_store(), run_id, limit), timeout=5)
    except Exception:
      client = []
    if client:
      result.update(records=client, available=True, error=None, note="Cloud Logging is not configured; showing the run's recent client output")
  if note and not result.get("note"):
    result["note"] = note
  return result


@router.get("/api/v1/inspect/pods/{pod}/logs", operation_id="pods_pod_logs", response_model=PodLogs, response_model_exclude_unset=True)
async def pod_logs(pod: str, container: str | None = None, previous: bool = False, tail: int = Query(200, ge=1, le=1000)):
  """Current or previous container output straight from the kubelet, for any pod in the namespace."""
  try:
    return await asyncio.to_thread(sources.pod_logs, pod, container, tail, previous)
  except Exception as exc:
    raise HTTPException(503, "Pod logs unavailable in the configured namespace") from exc
