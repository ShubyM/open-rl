"""The operator UI and the same read-only JSON API for agents."""

import asyncio
import json
import math
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.routing import APIRoute

from server import telemetry
from server.dashboard import facts, inspection, metrics, queries, sources
from server.dashboard.schema import (
  Accelerators,
  Activity,
  Commands,
  FleetRuns,
  Inventory,
  Logs,
  PodLogs,
  QueryResult,
  Series,
  Sessions,
  SessionSummary,
  Steps,
  Summary,
)
from server.dashboard.snapshot import METADATA_PREFIX, snapshot
from server.model_metadata import decode_model_metadata
from server.sampling_store import SamplingStore
from server.store import get_state_store

REQUEST_SECONDS = 12
# A long run holds tens of thousands of operation records; reading them needs more than a
# few seconds over a port-forward.
OPERATIONS_READ_SECONDS = 15
# One read of a run's or worker's records serves every page asking within this
# long, and a retry after a timeout joins the read still running instead of starting another.
# A finished run's records no longer change, so they are kept longer.
OPERATIONS_SECONDS = 10
FINISHED_SECONDS = 300
operations: OrderedDict[str, tuple[float, asyncio.Task]] = OrderedDict()


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
    description="ISO-8601 timestamp, e.g. 2026-01-02T12:00:00Z, not Unix seconds; since also takes a span before until, e.g. 1h, 30m, 2d. "
    "Defaults to the 30 minutes ending now. Maximum window: seven days.",
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


def time_range(since, until) -> sources.Window:
  try:
    return sources.window(since, until)
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 since/until timestamps and a positive window of at most seven days") from exc


@router.get("/api/v1/inspect/inventory", operation_id="inventory", response_model=Inventory, response_model_exclude_unset=True)
async def inventory():
  """Runs with their lifecycle, nodes with their GPUs, the GPUs held now, and pods with problems; an error field per unavailable source.
  A run's facts are /runs/{id}/summary."""
  await sources.discover()
  state = await snapshot.current(get_state_store())
  return {**state, "cluster": {**state["cluster"], "pods": [pod for pod in state["cluster"]["pods"] if pod["problem"]]}}


def is_finished(run: dict | None) -> bool:
  return bool(run) and run.get("lifecycle") == "ended"


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
    store = get_state_store()
    ops, samples = await asyncio.wait_for(
      asyncio.gather(store.read_samples(telemetry.ops_key(identity)), store.read_samples(telemetry.samples_key(identity))),
      timeout=OPERATIONS_READ_SECONDS,
    )
  except Exception:
    return {"samples": [], "error": "Operation records unavailable"}
  return {"samples": sorted(ops + samples, key=lambda r: r["at"])}


@router.get("/api/v1/inspect/activity", operation_id="activity", response_model=Activity, response_model_exclude_unset=True)
async def activity(since: TimeBound = None, until: TimeBound = None, run_id: str | None = None, node_id: str | None = None):
  """Per-process GPU turns and completed-op intervals, in Unix seconds.
  With run_id, also the run, its recorded ops and recent pod incidents."""
  start, end, low, high = time_range(since, until)
  store = get_state_store()
  state, run = await run_of(run_id) if run_id else (await snapshot.current(store), None)
  finished = is_finished(run)
  selected = [p for p in inspection.lanes(state, run_id, node_id) if p["start"] <= high and p["end"] >= low]
  # Each lane's runs, and its runtime for the worker-level ops recorded there.
  keys = sorted({k for p in selected for k in [*p["run_ids"], p.get("runtime_id")] if k} | ({run_id} if run_id else set()))

  records = dict(zip(keys, await asyncio.gather(*(read_operations(k, finished) for k in keys))))
  result = await asyncio.to_thread(inspection.activity, state, low, high, records, run_id, node_id)
  result.update(since=start, until=end, observed_at=state["observed_at"])
  if run_id:
    result.update(
      run=run,
      samples=inspection.chart_samples(records[run_id]["samples"], low, high),
      error=records[run_id].get("error"),
      incidents=sorted([*inspection.incidents(run, low, high), *await sdk_incidents(store, run_id, low, high)], key=lambda e: e["at"], reverse=True),
    )
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
  kind: Literal["hardware", "devices"] = "hardware",
):
  """GPU utilization/memory and worker CPU/memory over time, including idle devices and past placements.
  With run_id, the run's own GPUs and workers. kind=devices is the GPUs alone, for every node when no node or run is named.
  vLLM sampler charts are named queries: /query?name=vllm_..."""
  span = time_range(since, until)
  await sources.discover()
  state = await snapshot.current(get_state_store())
  return await metrics.series(state, span, node_id=node_id, run_id=run_id, kind=kind)


@router.get("/api/v1/inspect/query", operation_id="query", response_model=QueryResult, response_model_exclude_unset=True)
async def query(
  name: Annotated[str, Query(description="One of the named queries; an unknown name lists them")],
  run_id: str | None = None,
  node_id: str | None = None,
  since: TimeBound = None,
  until: TimeBound = None,
  points: int = Query(300, ge=1, le=1000),
):
  """A named PromQL query over a run's GPUs and samplers, or a node's GPUs, at `points` resolution.
  points=1 evaluates once over the whole window, which is how a duty cycle is one number per GPU."""
  if name not in queries.QUERIES:
    raise HTTPException(400, f"Unknown query; use one of {sorted(queries.QUERIES)}")
  if not (run_id or node_id):
    raise HTTPException(400, "Name a run_id or a node_id")
  span = time_range(since, until)
  await sources.discover()
  state = await snapshot.current(get_state_store())
  return await queries.run(name, state, run_id, node_id, span, points)


@router.get("/api/v1/inspect/accelerators", operation_id="accelerators", response_model=Accelerators, response_model_exclude_unset=True)
async def accelerators(run_id: str | None = None, node_id: str | None = None, since: TimeBound = None, until: TimeBound = None):
  """Every GPU in the fleet, or a node's, or those a run held in the window: node, model, holder, utilization now,
  mean utilization over the window (the last hour by default) and memory used of total."""
  if since is None and until is None:
    since = datetime.fromtimestamp(time.time() - 3600, UTC).isoformat()
  span = time_range(since, until)
  await sources.discover()
  state = await snapshot.current(get_state_store())
  return await queries.accelerators(state, run_id, node_id, span)


@router.get("/api/v1/inspect/runs/{run_id}/commands", operation_id="commands", response_model=Commands, response_model_exclude_unset=True)
async def commands(
  run_id: str,
  since: TimeBound = None,
  until: TimeBound = None,
  role: Literal["trainer", "sampler"] | None = None,
  limit: int = Query(100, ge=1, le=5000),
):
  """The run's command records in the window, newest first; without bounds, all of them."""
  _, _, low, high = time_range(since, until) if since or until else (None, None, 0.0, float("inf"))
  state = await snapshot.current(get_state_store())
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
  records = await read_operations(run_id, is_finished(run))
  matched = [s for s in records["samples"] if low <= s["at"] <= high and (role is None or (s.get("role") == "sampler") == (role == "sampler"))]
  matched.sort(key=lambda s: s["at"], reverse=True)
  extra = {"error": records["error"]} if records.get("error") else {}
  return {"run_id": run_id, "commands": matched[:limit], "total": len(matched), "truncated": len(matched) > limit, **extra}


@router.get("/api/v1/inspect/runs/{run_id}/steps", operation_id="steps", response_model=Steps, response_model_exclude_unset=True)
async def steps(
  run_id: str,
  last: int = Query(20, ge=1, le=1000),
  fields: Annotated[str | None, Query(description="Comma-separated prefixes of client metrics to keep, e.g. env/all/reward")] = None,
):
  """The run's last `last` steps: each step's window and trainer ops, with the client's metrics where stored."""
  state = await snapshot.current(get_state_store())
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
  records = await read_operations(run_id, is_finished(run))
  every = inspection.steps(records["samples"])
  found = every[-last:]
  keep = tuple(p for p in (fields or "").split(",") if p)
  client = {row["step"]: row["metrics"] for row in (await stored_history(run_id))["rows"]}
  if run and every:
    with suppress(Exception):  # client metrics are optional; steps stand without them
      client |= await queries.client_steps(run, every[0]["started_at"] - 300, (run.get("completed_at") or time.time()) + 300, keep)
  for step in found:
    step["client"] = {k: v for k, v in client.get(step["step"], {}).items() if not keep or k.startswith(keep)}
  return {"run_id": run_id, "total": len(every), "steps": found, **({"error": records["error"]} if records.get("error") else {})}


async def run_context(run_id: str, window: tuple[float, float] | None = None) -> dict:
  state = await snapshot.current(get_state_store())
  run = next((r for r in state["runs"] if r["run_id"] == run_id), None)
  if run is None:
    raise HTTPException(404, f"No run {run_id}")
  store, records = get_state_store(), await read_operations(run_id, is_finished(run))

  async def counters():
    found = {}
    for event in ("enqueued", "popped", "answered"):
      value = await store.get_value(telemetry.ledger_key(run_id, event))
      found[event] = int(value) if value is not None else None
    return found

  ledger = await counters()
  queued = ((run.get("progress") or {}).get("queued") or {}).get("trainer") or 0
  if ledger["enqueued"] is not None and ledger["enqueued"] - (ledger["popped"] or 0) - queued > 0:
    await asyncio.sleep(2)  # a pop in progress is counted a moment after it leaves the queue
    ledger = await counters()
  workers = [p for p in state.get("processes") or [] if p.get("runtime_id") in (run.get("runtime_id"), run_id) or p.get("pool") == run.get("pool")]
  now = datetime.fromisoformat(state["observed_at"]).timestamp()
  redis = getattr(store, "redis", None)  # pools need Redis; the in-memory store has none
  waiting, oldest = await SamplingStore(redis).backlog(run["pool"]) if redis is not None and run.get("pool") else (0, None)
  dispatcher = max((p for p in workers if p.get("role") == "dispatcher"), key=lambda p: p.get("heartbeat_at", 0), default=None)
  pool = {"name": run.get("pool"), "waiting": waiting, "oldest": oldest}
  if dispatcher:
    pool.update(dispatcher=dispatcher.get("pod") or dispatcher.get("id"), alive=telemetry.alive(dispatcher, now))
  created = run.get("created_at")
  low, high = window or (0.0, float("inf"))
  return {
    "run": run,
    "samples": [s for s in records["samples"] if low <= s["at"] <= high],
    "ledger": ledger,
    "workers": workers,
    "scheduler": bool(state["cluster"].get("scheduler", {}).get("installed")),
    "finished": is_finished(run),
    "now": now,
    "pool": pool,
    "placements": inspection.lanes(state, run_id=run_id),
    "since": created if isinstance(created, int | float) else 0.0,
    "state": state,
  }


@router.get("/api/v1/inspect/runs", operation_id="runs", response_model=FleetRuns, response_model_exclude_unset=True)
async def fleet_runs(ended: bool = False):
  """Every live run in one line each, longest since its last op first, so a stuck run is one read away.
  ended=true adds ended runs."""
  state = await snapshot.current(get_state_store())
  runs = [r for r in state["runs"] if ended or not is_finished(r)]
  found = [
    {"run_id": run["run_id"], "name": run.get("name"), **facts.line(ctx)}
    for run, ctx in zip(runs, await asyncio.gather(*(run_context(r["run_id"]) for r in runs)))
  ]
  # A run with no op at all waits longest of all.
  found.sort(key=lambda r: -(r["since_last_op_s"] if r["since_last_op_s"] is not None else math.inf))
  return {"runs": found}


@router.get("/api/v1/inspect/runs/{run_id}/summary", operation_id="summary", response_model=Summary, response_model_exclude_unset=True)
async def run_summary(run_id: str):
  """The run's facts in one read, each block only when it has content: steps and pace, goodput, what a live run waits on,
  unanswered requests and the request ledger, failed ops, worker relaunches, the sampling pool, saves, optimizer values,
  sampler latency, config drift, placement and Kubernetes incidents, with log reads for failures and relaunches.
  GPU duty by role covers the run's span; per GPU over any window is /accelerators?run_id=. No verdict; reason from the facts."""
  ctx = await run_context(run_id)
  state, found = ctx["state"], {"run_id": run_id, **facts.summary(ctx)}
  ops = sorted(s["at"] for s in ctx["samples"])
  if not ops:
    return found
  # Duty over the run's own span: averaging in the idle days after it ended would understate it.
  end = min(ctx["now"], ops[-1] + 60)
  window = {k: datetime.fromtimestamp(t, UTC).isoformat() for k, t in (("since", max(ops[0] - 60, end - 7 * 86400)), ("until", end))}
  duty = await queries.run("gpu_busy", state, run_id, None, sources.window(window["since"], window["until"]), 1)
  roles = {uuid: p["role"] for p in inspection.lanes(state, run_id=run_id) for uuid in p["devices"]}
  busy: dict[str, list[float]] = {}
  for series_ in duty.get("series", []):
    if series_["points"]:
      busy.setdefault(roles.get(series_["labels"].get("UUID")) or "other", []).append(series_["points"][-1][1])
  if busy:
    found.update(gpu_busy={role: round(sum(v) / len(v), 1) for role, v in busy.items()}, gpu_busy_window=window)
  return found


SESSION_LIVE_SECONDS = 120  # the live session key's expiry: a session silent this long is gone


async def session_context(session_id: str) -> dict:
  store = get_state_store()
  record = json.loads(await store.get_value(telemetry.session_key(session_id)) or "null")
  if record is None:
    raise HTTPException(404, f"No session {session_id}")
  live = await store.get_value(f"open_rl:session:{session_id}") is not None
  # A model's ops belong to this session while the session was alive; a backfilled record's last_seen_at is stale.
  window = (record["created_at"], math.inf if live else record["last_seen_at"] + SESSION_LIVE_SECONDS)
  models = {}
  for model_id in record["model_ids"]:
    with suppress(HTTPException):
      models[model_id] = await run_context(model_id, window)
  return {
    "record": record,
    "live": live,
    "models": models,
    "errors": await store.read_samples(telemetry.sdk_errors_key(session_id)),
  }


@router.get("/api/v1/inspect/sessions", operation_id="sessions", response_model=Sessions, response_model_exclude_unset=True)
async def sessions(limit: int = Query(50, ge=1, le=500)):
  """Client sessions, newest first: when each started and was last heard from, and the models it used."""
  store, found = get_state_store(), []
  live_ids = {key.removeprefix("open_rl:session:") for key in await store.scan_values("open_rl:session:sess")}
  for value in (await store.scan_values("open_rl:session_record:")).values():
    with suppress(ValueError, KeyError, TypeError):
      record = json.loads(value)
      live = record["session_id"] in live_ids
      metadata = record.get("metadata") or {}
      found.append(
        {**{k: record[k] for k in ("session_id", "created_at", "last_seen_at", "model_ids")}, "live": live, "recipe": metadata.get("recipe_name")}
      )
  found.sort(key=lambda r: r["created_at"], reverse=True)
  return {"total": len(found), "sessions": found[:limit]}


@router.get(
  "/api/v1/inspect/sessions/{session_id}/summary", operation_id="session_summary", response_model=SessionSummary, response_model_exclude_unset=True
)
async def session_summary(session_id: str):
  """The session's record, whether its registry key is live, its SDK errors, and each of its models in one line."""
  session = await session_context(session_id)
  errors = session["errors"]
  return {
    **session["record"],
    "live": session["live"],
    **({"sdk_errors": {"count": len(errors), "last": errors[-1].get("message")}} if errors else {}),
    "models": [{"model_id": model_id, **facts.line(ctx)} for model_id, ctx in session["models"].items()],
  }


# Client step history that gateways kept before OpenTelemetry; read only, nothing writes it now.
HISTORY_PREFIX = "open_rl:run_metrics:"


async def stored_history(run_id: str) -> dict:
  """Client metrics by step that gateways stored before OpenTelemetry, for older runs; /steps merges them."""
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


async def run_of(run_id: str) -> tuple[dict, dict]:
  state = await snapshot.current(get_state_store())
  if (run := next((r for r in state["runs"] if r["run_id"] == run_id), None)) is None:
    raise HTTPException(404, "Run record not found")
  return state, run


@router.get("/api/v1/inspect/logs", operation_id="logs", response_model=Logs, response_model_exclude_unset=True)
async def run_logs(
  run_id: str,
  q: str = Query("", max_length=1024),
  pod: str | None = None,
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: TimeBound = None,
  until: TimeBound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: Annotated[str | None, Query(max_length=sources.MAX_CURSOR, description="next_cursor of a page with the same filters and window")] = None,
):
  """Cloud Logging for the run: its worker pods by owner label, its sampling pool's pods, and its client."""
  span = time_range(since, until)
  await sources.discover()
  state, run = await run_of(run_id)
  pods = {lane["pod"]: {"pod": lane["pod"], "node": lane["node"], "role": lane["role"]} for lane in inspection.lanes(state, run_id) if lane["pod"]}
  try:
    result = await sources.run_logs(run, list(pods.values()), span, cursor=cursor, limit=limit, q=q, pod=pod, severity=severity)
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
  return result


@router.get("/api/v1/inspect/pods/{pod}/logs", operation_id="pods_pod_logs", response_model=PodLogs, response_model_exclude_unset=True)
async def pod_logs(pod: str, container: str | None = None, previous: bool = False, tail: int = Query(200, ge=1, le=1000)):
  """Current or previous container output straight from the kubelet, for any pod in the namespace."""
  try:
    return await asyncio.to_thread(sources.pod_logs, pod, container, tail, previous)
  except Exception as exc:
    raise HTTPException(503, "Pod logs unavailable in the configured namespace") from exc
