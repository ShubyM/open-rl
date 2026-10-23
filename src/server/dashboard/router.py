"""The operator UI and the same read-only JSON API for agents, under /api/v1/inspect."""

import asyncio
import json
import math
import re
import time
from pathlib import Path
from string import Template
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from server.dashboard import facts, fleet, sources
from server.store import get_state_store
from server.telemetry import finite, key

router = APIRouter(tags=["inspection"])
STATIC = Path(__file__).parent / "static"
BOUNDS = "ISO-8601, e.g. 2026-01-02T12:00:00Z; since also takes a span before until, e.g. 1h. Default: the 30 minutes ending now; at most 7 days."
Bound = Annotated[str | None, Query(max_length=64, description=BOUNDS)]


class Doc(BaseModel, extra="allow", allow_inf_nan=False):
  """Declared fields are always present; the rest pass through as recorded. Log messages are untrusted output."""


@router.get("/dashboard", include_in_schema=False)
@router.get("/dashboard/", include_in_schema=False)
@router.get("/dashboard/assets/{name}", include_in_schema=False)
async def dashboard(name: str = "index.html"):
  if (path := (STATIC / name).resolve()).parent != STATIC.resolve() or not path.is_file():
    raise HTTPException(404, "Not found")
  return FileResponse(path, headers={"Cache-Control": "no-store"})


def span(since: str | None, until: str | None) -> tuple[str, str, float, float]:
  try:
    return sources.window(since, until)
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 since/until timestamps and a positive window of at most seven days") from exc


async def run_of(run_id: str) -> tuple[dict, dict]:
  """Find a run by its id, or by a session id that attached a single run (409 when a session attached several)."""
  state = await fleet.snapshot()
  if (run := next((r for r in state["runs"] if r["run_id"] == run_id), None)) is not None:
    return state, run
  matches = [r for r in state["runs"] if run_id in r["session_ids"]]
  if len(matches) == 1:
    return state, matches[0]
  if len(matches) > 1:
    ids = ", ".join(r["run_id"] for r in matches)
    raise HTTPException(409, f"Session {run_id} has multiple runs ({ids}); pass one run_id or read /api/v1/inspect/sessions/{run_id}/summary")
  raise HTTPException(404, f"No run {run_id}")


@router.get("/api/v1/inspect/inventory", response_model=Doc)
async def inventory():
  """Runs with their lifecycle (ended, waiting, placed, active), nodes with their GPUs, the GPUs held now, pods with
  problems, and an error per unavailable source. A run's facts are /runs/{id}/summary."""
  await sources.discover()
  state = await fleet.snapshot()
  hidden = ("workloads", "runtime_id", "pods")
  runs = [
    {k: v for k, v in r.items() if k not in hidden} | {"pods": [{"name": p["name"], "problem": p["problem"]} for p in r["pods"]]}
    for r in state["runs"]
  ]
  fields = ("id", "label", "run_ids", "role", "node", "pod", "devices", "device_count")
  held = [{k: lane[k] for k in fields} for lane in fleet.lanes(state) if not lane["ended"]]
  pods = [{"name": p["name"], "problem": p["problem"], "restarts": p["restarts"]} for p in state["pods"] if p["problem"]]
  cluster = {"available": not any(e.startswith(("Kubernetes", "pods")) for e in state["errors"]), "nodes": state["nodes"], "pods": pods}
  return {"observed_at": facts.iso(state["observed_at"]), "errors": state["errors"], "runs": runs, "cluster": cluster, "placements": held}


@router.get("/api/v1/inspect/activity", response_model=Doc)
async def activity(since: Bound = None, until: Bound = None, run_id: str | None = None, node_id: str | None = None):
  """Each placement's GPU turns and recorded ops in the window (Unix seconds), fleet-wide, a node's or a run's.
  Gaps do not prove idleness."""
  since, until, low, high = span(since, until)
  state, run = await run_of(run_id) if run_id else (await fleet.snapshot(), None)
  placements = await fleet.activity(state, low, high, run["run_id"] if run else None, node_id)
  return {"placements": placements, "since": since, "until": until, "observed_at": facts.iso(state["observed_at"])}


VLLM = "{pod=~$pods$scope}"
QUERIES = {
  # By UUID and gpu: DCGM's slot index names a GPU after its node is gone.
  "gpu_util": "max by (UUID, gpu) (DCGM_FI_DEV_GPU_UTIL{UUID=~$gpus$scope})",
  "gpu_memory_mib": "max by (UUID, gpu) (DCGM_FI_DEV_FB_USED{UUID=~$gpus$scope})",
  "gpu_memory_total_mib": "max by (UUID) (DCGM_FI_DEV_FB_TOTAL{UUID=~$gpus$scope})",
  # Utilization comes in bursts; the last two minutes say whether a GPU is working now better than one sample.
  "gpu_util_now": "max by (Hostname, UUID, gpu, modelName) (avg_over_time(DCGM_FI_DEV_GPU_UTIL{UUID=~$gpus$scope}[2m]))",
  # Duty cycle over the window: the share of time a kernel ran, one number per GPU with points=1.
  "gpu_busy": "max by (UUID, gpu) (avg_over_time(DCGM_FI_DEV_GPU_UTIL{UUID=~$gpus$scope}[${window}s]))",
  # From vLLM's and llm-d's published Grafana dashboards.
  "vllm_generation_tokens_per_second": f"sum(rate(vllm:generation_tokens_total{VLLM}[2m]))",
  "vllm_prompt_tokens_per_second": f"sum(rate(vllm:prompt_tokens_total{VLLM}[2m]))",
  "vllm_requests_running": f"sum(vllm:num_requests_running{VLLM})",
  "vllm_requests_waiting": f"sum(vllm:num_requests_waiting{VLLM})",
  "vllm_kv_cache_usage": f"max(vllm:kv_cache_usage_perc{VLLM})",
  "vllm_preemptions_per_second": f"sum(rate(vllm:num_preemptions_total{VLLM}[2m]))",
  "vllm_time_to_first_token_p99_seconds": f"histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{VLLM}[2m])))",
  "vllm_request_latency_p99_seconds": f"histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{VLLM}[2m])))",
}


def subject(state: dict, run_id: str | None, node_id: str | None, low: float, high: float) -> tuple[list[str] | None, list[str]]:
  """The GPUs and pods a run's or a node's placements held in the window, plus a node's GPUs now; None GPUs is the fleet."""
  if not (run_id or node_id):
    return None, []
  held = [p for p in fleet.lanes(state, run_id, node_id) if p["start"] <= high and p["end"] >= low]
  now = [d["id"] for n in state["nodes"] if not run_id and n["name"] == node_id for d in n["devices"]]
  return sorted({uuid for p in held for uuid in p["devices"]} | set(now)), sorted({p["pod"] for p in held if p["pod"]})


@router.get("/api/v1/inspect/query", response_model=Doc)
async def query(
  name: Annotated[str, Query(description="One of the named queries; an unknown name lists them")],
  run_id: str | None = None,
  node_id: str | None = None,
  since: Bound = None,
  until: Bound = None,
  points: int = Query(300, ge=1, le=1000),
):
  """A named PromQL query over a run's GPUs and pods, a node's, or (GPU queries only) the fleet's, at `points` resolution.
  points=1 evaluates once over the whole window, which is how a duty cycle is one number per GPU."""
  if name not in QUERIES:
    raise HTTPException(400, f"Unknown query; use one of {sorted(QUERIES)}")
  since, until, low, high = span(since, until)
  await sources.discover()
  state, run = await run_of(run_id) if run_id else (await fleet.snapshot(), None)
  gpus, pods = subject(state, run["run_id"] if run else None, node_id, low, high)
  result = {"name": name, "since": since, "until": until, "series": [], "source": sources.prometheus_source()}
  if gpus == [] and "$gpus" in QUERIES[name] or not pods and "$pods" in QUERIES[name]:
    return {**result, "error": "No GPUs or pods recorded for this subject in this window"}
  values = {"gpus": '".+"' if gpus is None else sources.any_of(gpus), "pods": sources.any_of(pods), "window": max(1, int(high - low))}
  result["query"] = text = Template(QUERIES[name]).substitute(values, scope=sources.scope())
  try:
    rows = await sources.prom(text, high if points == 1 else low, high, points)
  except Exception as exc:
    return {**result, "error": str(exc) if isinstance(exc, LookupError) else f"Query failed: {type(exc).__name__}"}
  series = [{"labels": r["metric"], "points": [[float(t), float(v)] for t, v in r["values"] if math.isfinite(float(v))]} for r in rows]
  return result | {"series": series}


@router.get("/api/v1/inspect/accelerators", response_model=Doc)
async def accelerators(run_id: str | None = None, node_id: str | None = None, since: Bound = None, until: Bound = None):
  """Every GPU in the fleet, a node's, or a run's in the window (default the last hour): node, slot, model, holders (the
  newest placement on it and any time-sliced with it), util (mean of the last two minutes), busy (mean over the window), memory
  used and total in MiB."""
  since, until, low, high = span(since or ("1h" if until is None else None), until)
  state, run = await run_of(run_id) if run_id else (await fleet.snapshot(), None)
  resolved = run["run_id"] if run else None
  names = {"util": "gpu_util_now", "busy": "gpu_busy", "memory_used_mib": "gpu_memory_mib", "memory_total_mib": "gpu_memory_total_mib"}
  found = dict(zip(names, await asyncio.gather(*(query(name, resolved, node_id, since, until, 1) for name in names.values()))))
  result = {"since": since, "until": until, "gpus": [], "source": found["util"]["source"]}
  if error := next((r["error"] for r in found.values() if r.get("error")), None):
    return {**result, "error": error}
  value = {k: {s["labels"]["UUID"]: round(s["points"][-1][1], 1) for s in r["series"] if s["points"]} for k, r in found.items()}
  held = sorted((p for p in fleet.lanes(state, resolved) if p["start"] <= high and p["end"] >= low), key=lambda p: p["start"])
  placed: dict[str, list[dict]] = {}
  for lane in held:
    for uuid in lane["devices"]:
      placed.setdefault(uuid, []).append(lane)
  # A time-sliced GPU has several holders at once, every placement that overlaps the newest one.
  holders = {uuid: [p for p in lanes if p["end"] >= lanes[-1]["start"]] for uuid, lanes in placed.items()}
  # A GPU whose node scaled down in the window has no utilization now, only its duty; util's labels name node and model.
  labels_of = {series["labels"]["UUID"]: series["labels"] for series in found["busy"]["series"] + found["util"]["series"]}
  for uuid, labels in labels_of.items():
    node = labels.get("Hostname") or next((lane["node"] for lane in reversed(holders.get(uuid, []))), None)
    gpu = {"uuid": uuid, "node": node, "slot": labels.get("gpu"), "model": labels.get("modelName")}
    gpu |= {name: value[name][uuid] for name in names if uuid in value[name]}
    if uuid in holders:
      gpu["holders"] = [{k: lane[k] for k in ("run_ids", "role", "pod", "start", "end", "ended")} for lane in holders[uuid]]
    result["gpus"].append(gpu)
  result["gpus"].sort(key=lambda g: (g["node"] or "", int(g["slot"] or 0)))
  return result


HISTORY_ENTRIES = 64
history: dict[str, tuple[float, dict[int, dict[str, float]]]] = {}  # run and fields -> (read up to, client metrics by step)


async def client_steps(run: dict, low: float, high: float, prefixes: tuple[str, ...]) -> dict[int, dict[str, float]]:
  """The client's metrics by step: the older Redis history and the forwarded step events, else the OpenTelemetry gauges
  aligned by progress/batch. A step's values never change, so each gauge read asks only for the time after the last."""
  found, store = {}, get_state_store()
  for row in await store.read_samples(f"open_rl:run_metrics:{run['run_id']}"):
    found[row["step"]] = finite(row["metrics"])
  for session in run["session_ids"]:
    for row in await store.read_samples(key("client_steps", session)):
      # Only this model's training rows, since a session may train several and an eval may share a training step.
      ours = row.get("model_id") in (None, run["run_id"]) and row.get("phase") in (None, "train")
      if ours and row.get("step") is not None and low <= row["at"] <= high:
        found.setdefault(int(row["step"]), {}).update(row["metrics"])
  if found or not run["session_ids"] or high <= low or not sources.prometheus_source():
    return found
  memo = f"{run['run_id']} {prefixes}"
  seen, known = history.get(memo, (low, {}))
  start = max(low, seen - 300)  # overlap, so a step that landed during the last read is not missed
  wanted = "|".join(re.escape(p) + ".*" for p in ("progress/batch", *prefixes)) if prefixes else ".+"
  where = f"session_id=~{sources.any_of(run['session_ids'])},metric=~{json.dumps(wanted)}{sources.scope()}"
  where += f',openrl_phase=~"|train",openrl_model_id=~{json.dumps("|" + re.escape(run["run_id"]))}'  # an absent label matches ""
  rows = await sources.prom(f"max by (metric) (openrl_client_metric{{{where}}})", start, high, points=max(2, min(1000, int((high - start) / 15))))
  series = {r["metric"].get("metric"): {float(t): float(v) for t, v in r["values"] if math.isfinite(float(v))} for r in rows}
  first: dict[int, float] = {}
  for at, batch in sorted((series.pop("progress/batch", None) or {}).items()):
    first.setdefault(int(batch), at)
  known = dict(known)
  for step, at in first.items():  # a step read while its gauges landed gets the rest now; the newer values win
    known[step] = known.get(step, {}) | {name: points[at] for name, points in series.items() if at in points}
  history.pop(memo, None)
  history[memo] = (high, known)
  while len(history) > HISTORY_ENTRIES:  # the least recently read first
    history.pop(next(iter(history)))
  return found | known


@router.get("/api/v1/inspect/runs/{run_id}/steps", response_model=Doc)
async def steps(
  run_id: str,
  last: int = Query(20, ge=1, le=1000),
  fields: Annotated[str | None, Query(description="Comma-separated prefixes of client metrics to keep, e.g. env/all/reward")] = None,
):
  """The run's last steps, oldest first: window, completion, trainer ops (count, failures, seconds, metric spread), `client` metrics."""
  _, run = await run_of(run_id)
  run_id = run["run_id"]
  every, keep, result = fleet.steps(await fleet.operations(run_id)), tuple(p for p in (fields or "").split(",") if p), {}
  try:
    read = lambda: client_steps(run, every[0]["started_at"] - 300, (run["completed_at"] or time.time()) + 300, keep)  # noqa: E731
    client = await fleet.cached(f"client {run_id} {keep}", 30, read) if every else {}
  except Exception as exc:  # steps stand without the client's metrics
    client, result["error"] = {}, f"Client metrics unavailable: {type(exc).__name__}"
  for step in every[-last:]:
    step["client"] = {k: v for k, v in client.get(step["step"], {}).items() if not keep or k.startswith(keep)}
  return result | {"run_id": run_id, "total": len(every), "steps": every[-last:]}


@router.get("/api/v1/inspect/runs/{run_id}/commands", response_model=Doc)
async def commands(run_id: str, since: Bound = None, until: Bound = None, role: str | None = None, limit: int = Query(100, ge=1, le=5000)):
  """The run's raw command records (trainer ops and samples), newest first; without bounds, all of them."""
  _, run = await run_of(run_id)
  run_id = run["run_id"]
  low, high = span(since, until)[2:] if since or until else (0, math.inf)
  wanted = (lambda r: True) if role is None else (lambda r: (r.get("role") == "sampler") == (role == "sampler"))
  found = [r for r in reversed(await fleet.operations(run_id)) if low <= r["at"] <= high and wanted(r)]
  return {"run_id": run_id, "total": len(found), "commands": found[:limit]}


@router.get("/api/v1/inspect/runs", response_model=Doc)
async def runs(ended: bool = False):
  """Every live run in one line, longest since its last op first, so a stuck run is one read away. ended=true adds ended runs."""
  state = await fleet.snapshot()
  chosen = [r for r in state["runs"] if ended or r["lifecycle"] != "ended"]
  # A few runs at a time, since each read in flight holds a store connection.
  limit = asyncio.Semaphore(128)

  async def line(run):
    async with limit:
      return await facts.fleet_line(state, run)

  found = await asyncio.gather(*(line(r) for r in chosen))
  lines = [{"run_id": r["run_id"], "name": r["name"], **line} for r, line in zip(chosen, found)]
  return {"runs": sorted(lines, key=lambda r: -(r["since_last_op_s"] if r["since_last_op_s"] is not None else math.inf))}


@router.get("/api/v1/inspect/runs/{run_id}/summary", response_model=Doc)
async def summary(run_id: str):
  """The run's facts, each block only when it has content: pace, goodput, unanswered requests and their holders, the ledger
  (lost = enqueued - popped - queued), failures and relaunches with log reads, saves, optimizer, samplers, config,
  placement, incidents, and GPU duty by role over the run's span (gpu_busy, percent). No verdict; reason from the facts."""
  state, run = await run_of(run_id)
  run_id = run["run_id"]
  ctx = await facts.context(state, run)
  found = {"run_id": run_id, **facts.summary(ctx)}
  if ops := ctx["ops"]:
    # Duty over the run's own span: averaging in the idle days after it ended would understate it.
    end = min(state["observed_at"], ops[-1]["at"] + 60)
    duty = await query("gpu_busy", run_id, None, facts.iso(max(ops[0]["at"] - 60, end - 7 * 86400)), facts.iso(end), 1)
    roles: dict[str, set[str]] = {}
    for lane in ctx["lanes"]:
      for uuid in lane["devices"]:
        roles.setdefault(uuid, set()).add(lane["role"])
    busy: dict[str, list[float]] = {}
    for series in (s for s in duty["series"] if s["points"]):
      # A GPU that time-sliced roles shared counts under the roles together, e.g. sampler+trainer.
      busy.setdefault("+".join(sorted(roles.get(series["labels"].get("UUID")) or {"other"})), []).append(series["points"][-1][1])
    if busy:
      found["gpu_busy"] = {role: round(sum(v) / len(v), 1) for role, v in busy.items()}
      found["gpu_busy_window"] = {k: duty[k] for k in ("since", "until")}
  return found


@router.get("/api/v1/inspect/sessions/{session_id}/summary", response_model=Doc)
async def session_summary(session_id: str):
  """The session's record, whether its registry key is live, its SDK errors, and a line per model over the session's span."""
  store = get_state_store()
  if (record := json.loads(await store.get_value(key("session_record", session_id)) or "null")) is None:
    raise HTTPException(404, f"No session {session_id}")
  live = await store.get_value(key("session", session_id)) is not None
  # A model's ops belong to this session while it was alive, which ends when its live key expires after last_seen_at.
  low, high = record["created_at"], math.inf if live else record["last_seen_at"] + 120
  state, models = await fleet.snapshot(), []
  for run in (r for r in state["runs"] if r["run_id"] in record["model_ids"]):
    models.append({"model_id": run["run_id"], **facts.line(await facts.context(state, run, low, high))})
  errors = await store.read_samples(key("sdk_errors", session_id))
  return {**record, "live": live, "models": models} | ({"sdk_errors": {"count": len(errors), "last": errors[-1].get("message")}} if errors else {})


@router.get("/api/v1/inspect/logs", response_model=Doc)
async def logs(
  run_id: str,
  pod: Annotated[str | None, Query(description="One of the run's pods, or 'client' for its client's lines")] = None,
  q: str = Query("", max_length=1024),
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: Bound = None,
  until: Bound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: Annotated[str | None, Query(max_length=16384, description="next_cursor of a page with the same filters and window")] = None,
):
  """Cloud Logging, newest first: the run's worker pods by owner label and its client."""
  since, until, low, high = span(since, until)
  await sources.discover()
  state, run = await run_of(run_id)
  run_id = run["run_id"]
  pods = {lane["pod"]: {"pod": lane["pod"], "node": lane["node"], "role": lane["role"]} for lane in fleet.lanes(state, run_id) if lane["pod"]}
  result = {"run_id": run_id, "source": "gke", "since": since, "until": until, "sources": list(pods.values()), "records": [], "next_cursor": None}
  try:
    return {**result, **await sources.log_page(run, since, until, cursor=cursor, limit=limit, q=q, pod=pod, severity=severity), "available": True}
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 400 and cursor:
      raise HTTPException(400, "Invalid or expired log cursor; repeat the query without it") from None
    return {**result, "available": False, "error": f"Cloud Logging refused the read (HTTP {exc.response.status_code}); check credentials and IAM"}
  except Exception as exc:
    return {**result, "available": False, "error": str(exc) if isinstance(exc, LookupError) else "Cloud Logging unavailable"}
