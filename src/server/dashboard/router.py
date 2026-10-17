"""The operator UI and the same read-only JSON API for agents."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import ValidationError

from server.dashboard import inspection, metrics
from server.dashboard.schema import Activity, Inventory, Logs, Operation, PodLogs, Series
from server.dashboard.snapshot import METADATA_PREFIX, snapshot
from server.store import get_state_store
from server.telemetry import gke, kubernetes, ops

router = APIRouter(tags=["inspection"])
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
    start, end = gke.time_range(since, until)
    return start, end, datetime.fromisoformat(start).timestamp(), datetime.fromisoformat(end).timestamp()
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 since/until timestamps and a positive window of at most seven days") from exc


def validate_bounds(since, until):
  # Validate before touching sources, but resolve an omitted end against the snapshot.
  if until is not None:
    time_range(since, until)
  elif since is not None:
    try:
      gke.timestamp(since)
    except (ValueError, OverflowError, OSError) as exc:
      raise HTTPException(400, "since must be an ISO-8601 timestamp") from exc


@router.get("/api/v1/inspect/inventory", operation_id="inventory", response_model=Inventory, response_model_exclude_unset=True)
async def inventory():
  """Current runs, placements, nodes and scheduler state, with an error field per unavailable source."""
  await gke.discover()
  state = await snapshot.current(get_state_store())
  return {key: value for key, value in state.items() if key != "processes"}


@router.get("/api/v1/inspect/activity", operation_id="activity", response_model=Activity, response_model_exclude_unset=True)
async def activity(since: TimeBound = None, until: TimeBound = None, run_id: str | None = None, node_id: str | None = None):
  """Per-process GPU turns and completed-op intervals, in Unix seconds.
  With run_id, also the run, its recorded ops and recent pod incidents."""
  validate_bounds(since, until)
  store = get_state_store()
  state, run = await run_of(run_id) if run_id else (await snapshot.current(store), None)
  start, end, low, high = time_range(since, until or state["observed_at"])
  selected = [p for p in inspection.segments(state, run_id, node_id) if p["start"] <= high and p["end"] >= low]
  # Each lane's runs, and its runtime for the worker-level ops recorded there.
  keys = sorted({k for p in selected for k in [*p["run_ids"], p.get("runtime_id")] if k} | ({run_id} if run_id else set()))

  async def read(key):
    try:
      records = await asyncio.wait_for(store.read_samples(key), timeout=1)
      if not isinstance(records, list):
        raise ValueError("Expected telemetry records")
      return records
    except Exception:
      return None

  def validated(records):
    if records is None:
      return {"samples": [], "error": "Operation records unavailable"}
    samples, invalid = [], 0
    for record in records:
      try:
        samples.append(Operation.model_validate(record, strict=True).model_dump(exclude_unset=True))
      except ValidationError:
        invalid += 1
    return {"samples": samples, **({"partial": f"Skipped {invalid} invalid operation records"} if invalid else {})}

  def summarize(raw):
    records = {key: validated(r) for key, r in zip(keys, raw)}
    return records, inspection.activity(state, low, high, records, run_id, node_id)

  # Tens of thousands of records take seconds to check and join; a thread keeps other requests moving meanwhile.
  records, result = await asyncio.to_thread(summarize, await asyncio.gather(*(read(ops.key(k)) for k in keys)))
  result.update(since=start, until=end, observed_at=state["observed_at"])
  result["coverage"] = (
    f"At most {ops.SAMPLE_LIMIT} completed records per run/worker, best effort. "
    "Missing records do not prove no requests arrived or that workers were idle. Queued and in-flight requests are in inventory runs[].progress."
  )
  if run_id:
    result.update(
      run=run,
      samples=[s for s in records[run_id]["samples"] if low <= s["at"] <= high],
      error=records[run_id].get("error"),
      partial=records[run_id].get("partial"),
      incidents=inspection.incidents(run, low, high),
    )
  return result


@router.get("/api/v1/inspect/series", operation_id="series", response_model=Series, response_model_exclude_unset=True)
async def series(node_id: str | None = None, run_id: str | None = None, since: TimeBound = None, until: TimeBound = None):
  """GPU utilization/memory and worker CPU/memory over time, including idle devices and past placements.
  With run_id, the run's own GPUs and workers, plus throughput, queue, KV cache and latency of its vLLM samplers."""
  validate_bounds(since, until)
  await gke.discover()
  state = await snapshot.current(get_state_store())
  start, end, _, _ = time_range(since, until or state["observed_at"])
  return await metrics.series(state, node_id=node_id, since=start, until=end, run_id=run_id)


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
  cursor: str | None = Query(None, max_length=4096),
):
  """Cloud Logging for every pod the run has owned, live or gone."""
  validate_bounds(since, until)
  await gke.discover()
  state, run = await run_of(run_id)
  since, until, _, _ = time_range(since, until or state["observed_at"])
  sources = gke.pod_sources(run, inspection.segments(state, run_id), state["observed_at"])
  try:
    result = await gke.run_logs(run_id, sources, since=since, until=until, cursor=cursor, limit=limit, q=q, pod=pod, node=node, severity=severity)
  except ValueError as exc:
    raise HTTPException(400, str(exc)) from exc
  return {**result, "shared_runtime": run["shared_runtime"]}


@router.get("/api/v1/inspect/pods/{pod}/logs", operation_id="pods_pod_logs", response_model=PodLogs, response_model_exclude_unset=True)
async def pod_logs(pod: str, container: str | None = None, previous: bool = False, tail: int = Query(200, ge=1, le=1000)):
  """Current or previous container output straight from the kubelet, for any pod in the namespace."""
  try:
    return await asyncio.to_thread(kubernetes.pod_logs, pod, container, tail, previous)
  except Exception as exc:
    raise HTTPException(503, "Pod logs unavailable in the configured namespace") from exc
