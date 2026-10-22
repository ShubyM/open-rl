"""Dashboard projections over the same records and sources as the compact inspection API."""

import asyncio
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from server.dashboard import facts, fleet, sources
from server.dashboard.router import Bound, inventory_data, read_logs, read_steps, run_of, span
from server.store import get_state_store
from server.telemetry import CLIENT, iso, key

router = APIRouter(tags=["dashboard"])
STATIC = Path(__file__).parent / "static"
CHART_POINTS = 1000


@router.get("/dashboard", include_in_schema=False)
@router.get("/dashboard/", include_in_schema=False)
@router.get("/dashboard/assets/{name}", include_in_schema=False)
async def dashboard(name: str = "index.html"):
  if (path := (STATIC / name).resolve()).parent != STATIC.resolve() or not path.is_file():
    raise HTTPException(404, "Not found")
  return FileResponse(path, headers={"Cache-Control": "no-store"})


def incidents(run: dict, start: float, end: float) -> list[dict]:
  """Distinct timestamped failures/restarts, newest first; current conditions are not events."""
  errors = ("OOMKilled", "CrashLoopBackOff", "Evicted")
  warnings = ("WorkerRestarted", "FailedScheduling", "Unschedulable", "FailedMount", "FailedAttachVolume")
  events = []

  def add(pod, event, timestamp):
    try:
      at = datetime.fromisoformat(timestamp).timestamp()
    except (TypeError, ValueError):
      return
    if start <= at <= end:
      events.append({**event, "pod": pod["name"], "at": at, "tone": "error" if event["reason"] in errors else "warning"})

  for pod in run.get("pods", []):
    crashing = pod.get("restarts", 0) > 0 or (pod.get("problem") or "").split(":", 1)[0] == "CrashLoopBackOff"
    for event in pod.get("events", []):
      backoff = event.get("reason") == "BackOff" and event.get("type") == "Warning"
      backoff = backoff and (crashing or "back-off restarting failed container" in (event.get("message") or "").lower())
      if event.get("reason") in errors + warnings or backoff:
        add(pod, event, event.get("last_seen_at") or event.get("first_seen_at"))
    for container in pod.get("containers", []):
      name = container["name"]
      for termination in (container if container.get("state") == "terminated" else None, container.get("last_termination")):
        if termination and termination.get("reason") == "OOMKilled":
          code = termination.get("exit_code")
          event = {"container": name, "reason": "OOMKilled", "message": f"{name} exited with code {code if code is not None else 'unknown'}"}
          add(pod, event, termination.get("finished_at"))
      if container.get("state") == "running" and container.get("restart_count", 0) > 0:
        event = {"container": name, "reason": "WorkerRestarted", "restart_count": container["restart_count"]}
        event["message"] = f"{name} is running after a restart"
        add(pod, event, container.get("started_at"))
  events.sort(key=lambda event: event["at"], reverse=True)
  unique = {}
  for event in events:
    if not event.get("container") and any(
      other.get("container") and other["pod"] == event["pod"] and other["reason"] == event["reason"] and abs(other["at"] - event["at"]) < 2
      for other in events
    ):
      continue
    unique.setdefault((event["pod"], event.get("container"), event["reason"]), event)
  return list(unique.values())


def chart_samples(samples: list[dict], start: float, end: float) -> list[dict]:
  """A run's op records for its charts: every trainer op, which carries the metrics, and the slowest sampler
  request per time bucket, so a long run ships its peaks instead of tens of thousands of requests."""
  keep = ("at", "started_at", "operation", "role", "status", "elapsed_seconds", "metrics", "nonfinite")
  width = (end - start) / CHART_POINTS or 1
  charted, slowest = [], {}
  for sample in samples:
    if start <= sample["at"] <= end:
      sample = {k: sample[k] for k in keep if k in sample}
      bucket = int((sample["at"] - start) / width)
      if sample.get("role") != "sampler":
        charted.append(sample)
      elif sample.get("elapsed_seconds", 0) >= slowest.get(bucket, {}).get("elapsed_seconds", -1):
        slowest[bucket] = sample
  return sorted([*charted, *slowest.values()], key=lambda sample: sample["at"])


@router.get("/api/v1/dashboard/inventory")
async def inventory():
  await sources.discover()
  state = await fleet.snapshot()
  result = inventory_data(state)
  result["runs"] = state["runs"]
  result["cluster"]["pods"] = state["pods"]
  return result


@router.get("/api/v1/dashboard/activity")
async def activity(since: Bound = None, until: Bound = None, run_id: str | None = None, node_id: str | None = None):
  since, until, low, high = span(since, until)
  state, run = await run_of(run_id) if run_id else (await fleet.snapshot(), None)
  result = {"since": since, "until": until, "observed_at": facts.iso(state["observed_at"]), "errors": list(state["errors"])}
  try:
    result["placements"] = await fleet.activity(state, low, high, run_id, node_id)
  except Exception:
    result["placements"] = []
    result["errors"].append("Operation records unavailable")
  if run is not None:
    result.update(run=run, samples=[], incidents=incidents(run, low, high))
    try:
      result["samples"] = chart_samples(await fleet.operations(run_id), low, high)
      records = [r for session in run["session_ids"] for r in await get_state_store().read_samples(key("sdk_errors", session))]
      newest = {}
      for record in sorted(records, key=lambda r: r.get("at", 0), reverse=True):
        if low <= record.get("at", 0) <= high:
          event = {**record, "pod": CLIENT, "tone": "error", "count": 0}
          newest.setdefault(record["reason"], event)["count"] += 1
      result["incidents"] = sorted([*result["incidents"], *newest.values()], key=lambda r: r["at"], reverse=True)
    except Exception:
      result["errors"].append("Run operation or SDK history unavailable")
  return result


@router.get("/api/v1/dashboard/history")
async def history(run_id: str):
  try:
    records = await asyncio.wait_for(get_state_store().read_samples(f"open_rl:run_metrics:{run_id}"), timeout=10)
  except Exception:
    return {"available": False, "error": "Stored history unavailable", "rows": []}
  latest = {}
  for row in records:
    if isinstance(row, dict) and isinstance(row.get("step"), int) and isinstance(row.get("metrics"), dict):
      values = {k: float(v) for k, v in row["metrics"].items() if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)}
      latest[row["step"]] = {"step": row["step"], "metrics": values}
  return {"available": True, "rows": [latest[step] for step in sorted(latest)]}


@router.get("/api/v1/dashboard/runs/{run_id}/steps")
async def steps(run_id: str, last: int = Query(20, ge=1, le=1000), fields: str | None = None):
  result = await read_steps(run_id, last, fields)
  if "error" in result:
    result["client_error"] = result.pop("error")
  return result


@router.get("/api/v1/dashboard/pods/{pod}/logs")
async def pod_logs(pod: str, container: str | None = None, previous: bool = False, tail: int = Query(200, ge=1, le=1000)):
  try:
    return await sources.pod_logs(pod, container, tail, previous)
  except Exception as exc:
    raise HTTPException(503, "Pod logs unavailable in the configured namespace") from exc


@router.get("/api/v1/dashboard/logs")
async def logs(
  run_id: str,
  pod: str | None = None,
  q: str = Query("", max_length=1024),
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: Bound = None,
  until: Bound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: str | None = Query(None, max_length=16384),
):
  result = await read_logs(run_id, pod, q, severity, since, until, limit, cursor)
  if not sources.configuration()["configured"] and pod in (None, CLIENT) and not cursor:
    try:
      lines = await asyncio.wait_for(get_state_store().read_samples(f"open_rl:run_logs:{run_id}", last=limit), timeout=5)
    except Exception:
      lines = []
    if lines:
      records = []
      for line in reversed(lines):
        identity = hashlib.sha256(json.dumps([run_id, line.get("at"), line.get("message")]).encode()).hexdigest()
        records.append(
          {
            "id": identity,
            "run_id": run_id,
            "timestamp": iso(line.get("at", 0)),
            "pod": CLIENT,
            "role": CLIENT,
            "severity": line.get("severity", "INFO"),
            "message": line.get("message", ""),
          }
        )
      result.update(records=records, available=True, error=None, note="Cloud Logging is not configured; showing the run's recent client output")
  return result


@router.get("/api/v1/dashboard/telemetry")
async def telemetry(
  run_id: str | None = None, node_id: str | None = None, since: Bound = None, until: Bound = None, metric: str = Query("", max_length=200)
):
  if bool(run_id) == bool(node_id):
    raise HTTPException(400, "Choose exactly one run_id or node_id")
  _, _, low, high = span(since, until)
  selectors = [f"k8s_node_name={json.dumps(node_id)}"] if node_id else [f"openrl_model_id={json.dumps(run_id)}"]
  if run_id:
    _, run = await run_of(run_id)
    if run["session_ids"]:
      selectors.append(f'openrl_model_id="",session_id=~{sources.any_of(run["session_ids"])}')
    selectors.append(f'openrl_model_id="",session_id="",openrl_runtime_id=~{sources.any_of([run["runtime_id"]])}')
  await sources.discover()
  result = {"available": False, "series": [], "limit": 200, "truncated": False}
  if not sources.prometheus_source():
    return result | {"reason": "Collector metrics need a Prometheus backend. Configure OPEN_RL_PROMETHEUS_URL or GKE monitoring."}
  name, namespace = json.dumps("(?i).*" + re.escape(metric.strip()) + ".*"), json.dumps("|" + re.escape(sources.namespace()))
  common = f"metric=~{name},namespace=~{namespace}{sources.scope()}"
  query = " or ".join(
    "{" + f'__name__="{metric}",{common},{selector}' + "}" for metric in ("openrl_client_metric", "openrl_server_metric") for selector in selectors
  )
  result["source"] = sources.prometheus_source()
  try:
    rows = await sources.prom(query, low, high, 300)
  except Exception:
    return result | {"error": "Collector metrics query unavailable"}
  identities = (
    ("model", "openrl_model_id"),
    ("session", "session_id"),
    ("runtime", "openrl_runtime_id"),
    ("pod", "k8s_pod_name"),
    ("node", "k8s_node_name"),
  )
  for row in sorted(rows, key=lambda r: (r["metric"].get("metric", ""), json.dumps(r["metric"], sort_keys=True))):
    labels = row["metric"]
    points = [[float(at), float(v)] for at, v in row.get("values", []) if low <= float(at) <= high and math.isfinite(float(v))]
    if not points or not labels.get("metric"):
      continue
    if len(result["series"]) == result["limit"]:
      result["truncated"] = True
      break
    scope, identity = next(
      ((scope, labels[field]) for scope, field in identities if labels.get(field)), ("service", labels.get("service_name", "unknown"))
    )
    result["series"].append(
      {
        "id": hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()[:20],
        "name": labels["metric"],
        "source": "client" if labels["__name__"] == "openrl_client_metric" else "server",
        "scope": scope,
        "scope_id": identity,
        "phase": labels.get("openrl_phase"),
        "labels": labels,
        "points": points,
      }
    )
  return result | {"available": True}
