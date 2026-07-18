"""Human- and agent-facing read model for Open-RL runs and cluster state."""

import ast
import asyncio
import hashlib
import math
import os
import re
import socket
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Query

from server.run_metadata import MODEL_META_PREFIX, RUN_META_PREFIX, decode_metadata
from server.store import RequestStore, get_store

router = APIRouter(prefix="/api/v1/control", tags=["control"])
TRACKER_URL_FIELDS = ("tracker_url", "wandb_url", "experiment_url")
MAX_PROBLEMS = 200
ACTIVE_RUN_STATUSES = {"queued", "starting", "waiting", "running", "ready"}
TRANSITIONAL_STATUSES = {"pending", "queued", "starting", "waiting"}
TERMINAL_HISTORY_STATUSES = {"completed", "stopped"}
TERMINAL_WORKER_STATUSES = {"completed", "stopped", "succeeded"}
NODE_PRESSURE_CODES = {
  "MemoryPressure": "node_memory_pressure",
  "DiskPressure": "node_disk_pressure",
  "PIDPressure": "node_pid_pressure",
  "NetworkUnavailable": "node_network_unavailable",
}
SECRET_REDACTION_PATTERN = re.compile(r"(?i)\b(authorization|password|passwd|token|api[-_]?key|secret)\b\s*(?::|=|\s)\s*[^\s,;]+")
BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
URL_PATTERN = re.compile(r"(?i)https?://[^\s]+")
TIMESTAMPED_LOG_PATTERN = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\S+)\s(?P<message>.*)$")
POD_NAME_PATTERN = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*")


def safe_tracker_url(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  url = value.strip()
  if not url or len(url) > 2048 or "\\" in url or any(character.isspace() or ord(character) < 32 for character in url):
    return None
  try:
    parsed = urlsplit(url)
    if parsed.port is not None and not 0 <= parsed.port <= 65535:
      return None
  except ValueError:
    return None
  if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
    return None
  return url


def safe_text(value: Any, limit: int = 512) -> str | None:
  if value is None:
    return None
  text = " ".join(str(value).split())
  text = URL_PATTERN.sub("<redacted-url>", text)
  text = BEARER_TOKEN_PATTERN.sub("bearer <redacted>", text)
  text = SECRET_REDACTION_PATTERN.sub(lambda match: f"{match.group(1).lower()}=<redacted>", text)
  return text[:limit]


def safe_reference(value: Any) -> str | None:
  text = safe_text(value, limit=253)
  return text or None


def safe_evidence(value: Any, depth: int = 0) -> Any:
  if depth >= 3:
    return None
  if value is None or isinstance(value, bool | int):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, str):
    return safe_text(value)
  if isinstance(value, dict):
    return {
      (safe_text(key, limit=64) or "field"): cleaned
      for key, item in list(value.items())[:16]
      if (cleaned := safe_evidence(item, depth + 1)) is not None
    }
  if isinstance(value, list | tuple):
    return [cleaned for item in list(value)[:16] if (cleaned := safe_evidence(item, depth + 1)) is not None]
  return safe_text(value)


def age_seconds(value: Any, now: float) -> float | None:
  timestamp = epoch_seconds(value)
  if timestamp is None and isinstance(value, str):
    try:
      timestamp = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
      return None
  return max(0.0, now - timestamp) if timestamp is not None else None


def bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
  try:
    return min(maximum, max(minimum, int(os.getenv(name, str(default)))))
  except ValueError:
    return default


def resource_references(**values: Any) -> dict[str, str]:
  return {name: reference for name in ("run_id", "component", "pod_name", "node") if (reference := safe_reference(values.get(name))) is not None}


def problem_actions(resources: dict[str, str], run: dict[str, Any] | None) -> list[dict[str, Any]]:
  can_read_logs = bool(resources.get("run_id"))
  can_stop = bool(run and run.get("can_stop"))
  actions: list[dict[str, Any]] = [{"name": "inspect", "allowed": True}]
  logs: dict[str, Any] = {"name": "logs", "allowed": can_read_logs}
  if not can_read_logs:
    logs["reason"] = "No workload or pod is associated with this problem"
  actions.append(logs)
  stop: dict[str, Any] = {"name": "stop", "allowed": can_stop}
  if not can_stop:
    stop["reason"] = "No associated live run can be stopped"
  actions.append(stop)
  return actions


def build_problem(
  code: str,
  severity: str,
  summary: str,
  evidence: dict[str, Any],
  remediation: str,
  resources: dict[str, str],
  run: dict[str, Any] | None = None,
) -> dict[str, Any]:
  identity = "\0".join([code, *(f"{key}={resources.get(key, '')}" for key in ("run_id", "component", "pod_name", "node"))])
  problem_id = f"problem-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
  return {
    "id": problem_id,
    "severity": severity,
    "code": code,
    "summary": safe_text(summary) or code.replace("_", " "),
    "evidence": safe_evidence(evidence),
    "resources": resources,
    "remediation": remediation,
    "actions": problem_actions(resources, run),
  }


def worker_manager() -> Any:
  # Imported lazily so gateway can install the router before its lifespan has
  # constructed the worker manager.
  from server import gateway

  return gateway.fft_worker_manager


def iso_timestamp(value: Any) -> str | None:
  try:
    return datetime.fromtimestamp(float(value), UTC).isoformat().replace("+00:00", "Z")
  except (TypeError, ValueError, OSError):
    return None


def epoch_seconds(value: Any) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def format_event(event: dict[str, Any]) -> dict[str, Any]:
  public = event.copy()
  public["timestamp"] = iso_timestamp(event.get("timestamp")) or event.get("timestamp")
  public.setdefault("details", {})
  return public


def parse_log_line(line: str) -> dict[str, Any]:
  matched = TIMESTAMPED_LOG_PATTERN.match(line)
  return {
    "timestamp": matched.group("timestamp") if matched else None,
    "stream": "stdout",
    "message": matched.group("message") if matched else line,
  }


def decode_log_text(value: Any) -> str:
  data = getattr(value, "data", value)
  if isinstance(data, bytes | bytearray):
    return bytes(data).decode("utf-8", errors="replace")
  text = str(data or "")
  if len(text) <= 10 * 1024 * 1024 and text[:2] in {"b'", 'b"'}:
    try:
      literal = ast.literal_eval(text)
    except (SyntaxError, ValueError):
      literal = None
    if isinstance(literal, bytes):
      return literal.decode("utf-8", errors="replace")
  return text


async def load_run_metadata(store: RequestStore) -> dict[str, dict[str, Any]]:
  durable_values, active_values = await asyncio.gather(
    store.list_values(RUN_META_PREFIX),
    store.list_values(MODEL_META_PREFIX),
  )
  durable_by_id = {key.removeprefix(RUN_META_PREFIX): decode_metadata(raw) for key, raw in durable_values.items()}
  active_by_id = {key.removeprefix(MODEL_META_PREFIX): decode_metadata(raw) for key, raw in active_values.items()}
  result: dict[str, dict[str, Any]] = {}
  for run_id in durable_by_id.keys() | active_by_id.keys():
    result[run_id] = {
      **durable_by_id.get(run_id, {}),
      **active_by_id.get(run_id, {}),
      "active_model": run_id in active_by_id,
    }
  return result


def component_from_event(event: dict[str, Any]) -> dict[str, Any]:
  component = str(event.get("component") or "gateway")
  return {
    "id": component,
    "role": component,
    "status": event.get("status", "unknown"),
    "phase": event.get("phase", "unknown"),
    "message": event.get("message", ""),
    "pod_name": None,
    "namespace": None,
    "node": None,
    "ready": event.get("status") in {"ready", "completed"},
    "restarts": 0,
    "reason": None,
    "image": None,
    "updated_at": iso_timestamp(event.get("timestamp")),
  }


async def build_run(
  store: RequestStore,
  run_id: str,
  metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
  has_active_model = bool(metadata and metadata.get("active_model"))
  metadata = metadata or {}
  events, queue = await asyncio.gather(store.get_control_events(run_id, limit=1000), store.queue_depths(run_id))
  if not metadata and not events:
    return None

  for event in events:
    details = event.get("details") if isinstance(event.get("details"), dict) else {}
    for field in ("base_model", "training_kind"):
      if not metadata.get(field) and details.get(field):
        metadata[field] = details[field]

  latest_event = events[-1] if events else {}
  created_epoch = epoch_seconds(metadata.get("created_at")) or (epoch_seconds(events[0].get("timestamp")) if events else time.time()) or time.time()
  updated_epoch = epoch_seconds(latest_event.get("timestamp")) or created_epoch
  status = str(latest_event.get("status") or "queued")
  phase = str(latest_event.get("phase") or "submitted")
  message = str(latest_event.get("message") or "Waiting for the run to start")

  components: dict[str, dict[str, Any]] = {}
  for event in events:
    components[str(event.get("component") or "gateway")] = component_from_event(event)

  manager = worker_manager()
  if manager is not None and hasattr(manager, "describe_workers"):
    try:
      workers = await asyncio.to_thread(manager.describe_workers, run_id)
      for worker in workers:
        role = str(worker.get("role") or worker.get("id") or "worker")
        event_component = components.get(role)
        merged = {**(event_component or {}), **worker, "id": role, "role": role}
        if event_component:
          if event_component.get("status") == "failed":
            merged.update(
              {
                "status": "failed",
                "phase": event_component["phase"],
                "message": event_component["message"],
                "ready": False,
                "updated_at": event_component["updated_at"],
              }
            )
          elif worker.get("status") != "failed":
            merged["phase"] = event_component["phase"]
            merged["message"] = event_component["message"]
            merged["updated_at"] = event_component["updated_at"]
        components[role] = merged
    except Exception as exc:
      components["cluster"] = {
        **component_from_event(
          {
            "component": "cluster",
            "phase": "unavailable",
            "status": "degraded",
            "message": f"Could not inspect worker pods: {exc}",
            "timestamp": time.time(),
          }
        )
      }

  failed_components = {component_id for component_id, component in components.items() if component.get("status") == "failed"}
  if failed_components:
    status = "failed"
    failed_event = next(
      (event for event in reversed(events) if event.get("status") == "failed" and str(event.get("component") or "gateway") in failed_components),
      None,
    )
    if failed_event is not None:
      phase = str(failed_event.get("phase") or "failed")
      message = str(failed_event.get("message") or "Run component failed")
      updated_epoch = epoch_seconds(failed_event.get("timestamp")) or updated_epoch
    else:
      failed_component = next(component for component in components.values() if component.get("status") == "failed")
      phase = str(failed_component.get("phase") or "failed")
      message = str(failed_component.get("message") or failed_component.get("reason") or "Run component failed")
  stopped_at = epoch_seconds(metadata.get("stopped_at"))
  if stopped_at is not None:
    status = "stopped"
    phase = "stopped"
    message = "Run workers were asked to stop"
    updated_epoch = max(updated_epoch, stopped_at)
  elif not has_active_model and status != "failed":
    stopped_event = next((event for event in reversed(events) if event.get("status") == "stopped"), None)
    if stopped_event is not None:
      status = "stopped"
      phase = str(stopped_event.get("phase") or "stopped")
      message = str(stopped_event.get("message") or "Run workers were asked to stop")
      updated_epoch = epoch_seconds(stopped_event.get("timestamp")) or updated_epoch
  base_model = metadata.get("base_model")
  short_model = str(base_model or "run").rsplit("/", 1)[-1]
  terminal = status in {"completed", "failed", "stopped"}
  simulated = bool(metadata.get("simulated"))
  tracker_url = safe_tracker_url(metadata.get("tracker_url"))
  return {
    "id": run_id,
    "name": metadata.get("name") or f"{short_model} · {run_id[:8]}",
    "base_model": base_model,
    "training_kind": metadata.get("training_kind", "unknown"),
    "simulated": simulated,
    "can_stop": has_active_model and not simulated and not terminal,
    "tracker_url": tracker_url,
    "status": status,
    "phase": phase,
    "message": message,
    "created_at": iso_timestamp(created_epoch),
    "updated_at": iso_timestamp(updated_epoch),
    "elapsed_seconds": max(0.0, (time.time() if status not in {"completed", "failed", "stopped"} else updated_epoch) - created_epoch),
    "queue": queue,
    "components": list(components.values()),
  }


async def all_runs(store: RequestStore | None = None) -> list[dict[str, Any]]:
  store = store or get_store()
  metadata, control_run_ids = await asyncio.gather(
    load_run_metadata(store),
    store.list_control_run_ids(),
  )
  run_ids = set(metadata) | set(control_run_ids)
  runs = await asyncio.gather(*(build_run(store, run_id, metadata.get(run_id)) for run_id in run_ids))
  return sorted((run for run in runs if run is not None), key=lambda run: run.get("created_at") or "", reverse=True)


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
  return {"runs": await all_runs(), "generated_at": iso_timestamp(time.time())}


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
  store = get_store()
  metadata = await load_run_metadata(store)
  run = await build_run(store, run_id, metadata.get(run_id))
  if run is None:
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found")
  return run


@router.post("/runs/{run_id}/stop", status_code=202)
async def stop_run(run_id: str) -> dict[str, Any]:
  store = get_store()
  metadata, control_run_ids = await asyncio.gather(load_run_metadata(store), store.list_control_run_ids())
  if run_id not in metadata and run_id not in control_run_ids:
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found")
  run = await build_run(store, run_id, metadata.get(run_id))
  if run is None:
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found")
  if not run["can_stop"]:
    if run["status"] in {"completed", "failed", "stopped"}:
      return {"status": "noop", "run": run}
    raise HTTPException(status_code=409, detail="Only a live run with retained model metadata can be stopped")

  from server import gateway

  await gateway.request_model_stop(run_id, request_store=store, preserve_metadata=True)
  updated = await build_run(store, run_id, (await load_run_metadata(store)).get(run_id))
  return {"status": "accepted", "run": updated}


@router.get("/runs/{run_id}/events")
async def get_run_events(
  run_id: str,
  after: str | None = None,
  limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
  store = get_store()
  events = await store.get_control_events(run_id, after=after, limit=limit)
  if not events and run_id not in (await load_run_metadata(store)) and run_id not in await store.list_control_run_ids():
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found")
  return {
    "events": [format_event(event) for event in events],
    "next_cursor": events[-1].get("cursor") if events else after,
  }


@router.get("/runs/{run_id}/logs")
async def get_run_logs(
  run_id: str,
  component: str = "trainer",
  tail: int = Query(default=200, ge=1, le=5000),
  previous: bool = False,
) -> dict[str, Any]:
  store = get_store()
  valid_components = {"all", "client", "gateway", "scheduler", "trainer", "sampler", "timeslicer"}
  if component not in valid_components:
    raise HTTPException(status_code=400, detail=f"Unknown log component {component!r}")
  events = await store.get_control_events(run_id, limit=1000)
  if not events and await store.get_value(f"{MODEL_META_PREFIX}{run_id}") is None and run_id not in await store.list_control_run_ids():
    raise HTTPException(status_code=404, detail=f"Run {run_id!r} was not found")
  manager = worker_manager()
  result: dict[str, Any] = {"source": "events", "pod_name": None, "logs": "", "error": None}
  if component in {"gateway", "trainer", "sampler", "timeslicer"} and manager is not None and hasattr(manager, "read_logs"):
    result = await asyncio.to_thread(manager.read_logs, run_id, component, tail, previous)

  matching = [event for event in events if component in {"all", str(event.get("component"))}]
  event_lines = [
    f"{iso_timestamp(event.get('timestamp')) or '-'} [{event.get('component', 'gateway')}] {event.get('message', '')}" for event in matching[-tail:]
  ]
  raw_logs = decode_log_text(result.get("logs"))
  if not raw_logs:
    raw_logs = "\n".join(event_lines)
    if raw_logs:
      result["source"] = "events"
  lines = [parse_log_line(line) for line in raw_logs.splitlines()]
  return {
    "run_id": run_id,
    "component": component,
    "source": result.get("source", "events"),
    "pod_name": result.get("pod_name"),
    "logs": raw_logs,
    "lines": lines,
    "error": result.get("error"),
  }


@router.get("/pods/{pod_name}/logs")
async def get_pod_logs(
  pod_name: str,
  tail: int = Query(default=200, ge=1, le=5000),
  previous: bool = False,
) -> dict[str, Any]:
  if len(pod_name) > 253 or POD_NAME_PATTERN.fullmatch(pod_name) is None:
    raise HTTPException(status_code=400, detail="Invalid Kubernetes pod name")
  manager = worker_manager()
  if manager is None or not hasattr(manager, "read_pod_logs"):
    raise HTTPException(status_code=503, detail="Kubernetes pod logs are unavailable")
  result = await asyncio.to_thread(manager.read_pod_logs, pod_name, tail, previous)
  raw_logs = decode_log_text(result.get("logs"))
  return {
    "source": result.get("source", "kubernetes"),
    "pod_name": result.get("pod_name", pod_name),
    "logs": raw_logs,
    "lines": [parse_log_line(line) for line in raw_logs.splitlines()],
    "error": result.get("error"),
  }


def local_cluster_snapshot() -> dict[str, Any]:
  return {
    "mode": "local",
    "status": "healthy",
    "namespace": None,
    "summary": {
      "nodes": 1,
      "ready_nodes": 1,
      "pods": 0,
      "running_pods": 0,
      "pending_pods": 0,
      "actionable_pending_pods": 0,
      "failed_pods": 0,
    },
    "nodes": [
      {
        "name": socket.gethostname(),
        "status": "ready",
        "ready": True,
        "roles": ["local"],
        "capacity": {},
        "allocatable": {},
        "conditions": [],
        "taints": [],
        "pod_count": 0,
      }
    ],
    "pods": [],
    "errors": [],
    "generated_at": time.time(),
  }


async def cluster_snapshot() -> dict[str, Any]:
  manager = worker_manager()
  if manager is None or not hasattr(manager, "cluster_snapshot"):
    snapshot = local_cluster_snapshot()
  else:
    try:
      snapshot = await asyncio.to_thread(manager.cluster_snapshot)
    except Exception as exc:
      snapshot = local_cluster_snapshot()
      snapshot.update({"status": "unavailable", "errors": [str(exc)]})
  snapshot["generated_at"] = iso_timestamp(snapshot.get("generated_at") or time.time())
  return snapshot


@router.get("/cluster")
async def get_cluster() -> dict[str, Any]:
  return await cluster_snapshot()


def nonnegative_int(value: Any) -> int:
  if isinstance(value, bool):
    return 0
  try:
    return max(0, int(value))
  except (TypeError, ValueError, OverflowError):
    return 0


def derive_problems(snapshot: dict[str, Any], runs: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
  problems: list[dict[str, Any]] = []
  stuck_seconds = bounded_env_int("OPEN_RL_CONTROL_STUCK_SECONDS", 300, 30, 86400)
  backlog_depth = bounded_env_int("OPEN_RL_CONTROL_QUEUE_WARNING_DEPTH", 100, 1, 1000000)
  run_by_id = {str(run.get("id")): run for run in runs if isinstance(run, dict) and run.get("id") is not None}

  snapshot_errors = snapshot.get("errors") if isinstance(snapshot.get("errors"), list) else []
  if snapshot.get("status") == "unavailable" or snapshot_errors:
    unavailable = snapshot.get("status") == "unavailable"
    problems.append(
      build_problem(
        "cluster_snapshot_unavailable" if unavailable else "cluster_snapshot_incomplete",
        "error" if unavailable else "warning",
        "The Kubernetes cluster snapshot is unavailable" if unavailable else "The Kubernetes cluster snapshot is incomplete",
        {"status": snapshot.get("status"), "error_count": len(snapshot_errors)},
        "Check gateway Kubernetes credentials and API connectivity, then refresh the snapshot.",
        {},
      )
    )

  for node in snapshot.get("nodes") or []:
    if not isinstance(node, dict):
      continue
    node_name = safe_reference(node.get("name"))
    display_name = node_name or "an unknown node"
    resources = resource_references(node=node_name)
    conditions = [condition for condition in node.get("conditions") or [] if isinstance(condition, dict)]
    ready_condition = next((condition for condition in conditions if condition.get("type") == "Ready"), {})
    if not bool(node.get("ready")):
      problems.append(
        build_problem(
          "node_not_ready",
          "error",
          f"Node {display_name} is not ready",
          {
            "status": node.get("status"),
            "reason": ready_condition.get("reason"),
            "message": ready_condition.get("message"),
          },
          "Inspect the node and kubelet events; cordon or drain it if workloads need to be rescheduled.",
          resources,
        )
      )
    for condition in conditions:
      condition_type = str(condition.get("type") or "")
      code = NODE_PRESSURE_CODES.get(condition_type)
      if code is None or str(condition.get("status") or "").lower() != "true":
        continue
      label = condition_type.removesuffix("Pressure").replace("Unavailable", " unavailable").lower()
      problems.append(
        build_problem(
          code,
          "error",
          f"Node {display_name} reports {label}",
          {
            "condition": condition_type,
            "reason": condition.get("reason"),
            "message": condition.get("message"),
          },
          "Inspect node capacity and pressure sources, reclaim resources, or move workloads to a healthy node.",
          resources,
        )
      )

  pods_by_run: dict[str, list[dict[str, Any]]] = {}
  for pod in snapshot.get("pods") or []:
    if not isinstance(pod, dict):
      continue
    model_id = str(pod.get("model_id")) if pod.get("model_id") is not None else None
    run = run_by_id.get(model_id) if model_id is not None else None
    if model_id is not None:
      pods_by_run.setdefault(model_id, []).append(pod)
    if run and str(run.get("status") or "").lower() in TERMINAL_HISTORY_STATUSES:
      continue
    status = str(pod.get("status") or "unknown").lower()
    if status == "completed":
      continue
    pod_name = pod.get("pod_name") or pod.get("name")
    role = pod.get("role")
    resources = resource_references(run_id=model_id, component=role, pod_name=pod_name, node=pod.get("node"))
    display_name = resources.get("pod_name", "unknown pod")
    reason = str(pod.get("reason") or "")
    restarts = nonnegative_int(pod.get("restarts"))
    evidence = {
      "status": status,
      "reason": pod.get("reason"),
      "message": pod.get("message"),
      "restarts": restarts,
    }
    if status == "failed":
      problems.append(
        build_problem(
          "pod_failed",
          "error",
          f"Pod {display_name} failed" + (f" ({reason})" if reason else ""),
          evidence,
          "Inspect current and previous container logs plus Kubernetes events, then correct the failure before retrying.",
          resources,
          run,
        )
      )
    elif status == "pending":
      unschedulable = reason.lower() == "unschedulable"
      problems.append(
        build_problem(
          "pod_unschedulable" if unschedulable else "pod_pending",
          "error" if unschedulable else "warning",
          f"Pod {display_name} cannot be scheduled" if unschedulable else f"Pod {display_name} is pending",
          evidence,
          (
            "Compare the pod's resource requests and constraints with node allocatable capacity, taints, and affinity."
            if unschedulable
            else "Inspect pod events and container state to determine what is blocking startup."
          ),
          resources,
          run,
        )
      )
    if restarts > 0 and status not in {"completed", "stopped"}:
      problems.append(
        build_problem(
          "pod_restarting",
          "warning",
          f"Pod {display_name} has restarted {restarts} time" + ("s" if restarts != 1 else ""),
          evidence,
          "Inspect previous container logs and termination reasons before restart history is lost.",
          resources,
          run,
        )
      )

  for run_id, run in sorted(run_by_id.items()):
    status = str(run.get("status") or "unknown").lower()
    if status in TERMINAL_HISTORY_STATUSES:
      continue
    phase = str(run.get("phase") or "unknown").lower()
    run_resources = resource_references(run_id=run_id)
    updated_age = age_seconds(run.get("updated_at"), now)
    if status == "failed":
      problems.append(
        build_problem(
          "run_failed",
          "error",
          f"Run {run_id} failed in phase {phase}",
          {"status": status, "phase": phase, "message": run.get("message")},
          "Inspect the failed component, pod events, and logs; correct the cause before starting another run.",
          run_resources,
          run,
        )
      )

    components = [component for component in run.get("components") or [] if isinstance(component, dict)]
    component_by_role = {str(component.get("role") or component.get("id")): component for component in components}
    cluster_workers = [pod for pod in pods_by_run.get(run_id, []) if str(pod.get("role") or "").lower() in {"trainer", "sampler"}]
    observed_workers = cluster_workers or [
      component for component in components if str(component.get("role") or component.get("id") or "").lower() in {"trainer", "sampler"}
    ]
    worker_states = {
      str(worker.get("role") or worker.get("id") or worker.get("pod_name") or worker.get("name") or "worker"): str(
        worker.get("status") or "unknown"
      ).lower()
      for worker in observed_workers
    }
    stale_run_state = (
      status in ACTIVE_RUN_STATUSES
      and not bool(run.get("can_stop"))
      and bool(worker_states)
      and all(worker_status in TERMINAL_WORKER_STATUSES for worker_status in worker_states.values())
    )
    if stale_run_state:
      problems.append(
        build_problem(
          "stale_run_state",
          "warning",
          f"Run {run_id} reports {status}, but its workers are complete",
          {"reported_status": status, "reported_phase": phase, "worker_statuses": worker_states},
          "Treat the run as historical and inspect its final events; lifecycle state should be reconciled before automation acts on it.",
          run_resources,
          run,
        )
      )
    waiting = status == "waiting" or phase.startswith("waiting")
    if waiting and not stale_run_state:
      problems.append(
        build_problem(
          "run_waiting",
          "warning",
          f"Run {run_id} is waiting in phase {phase}",
          {"status": status, "phase": phase, "seconds_since_update": updated_age},
          "Inspect its worker state, queue depth, pod placement, and Kubernetes events to identify the dependency it is waiting for.",
          run_resources,
          run,
        )
      )
    elif not stale_run_state and status in TRANSITIONAL_STATUSES and updated_age is not None and updated_age >= stuck_seconds:
      problems.append(
        build_problem(
          "run_stuck",
          "warning",
          f"Run {run_id} has not progressed beyond {phase}",
          {"status": status, "phase": phase, "seconds_since_update": updated_age, "threshold_seconds": stuck_seconds},
          "Inspect component state, Kubernetes events, and logs; stop the run if it cannot make progress safely.",
          run_resources,
          run,
        )
      )

    for component in sorted(components, key=lambda item: str(item.get("role") or item.get("id") or "")):
      component_id = str(component.get("role") or component.get("id") or "unknown")
      component_status = str(component.get("status") or "unknown").lower()
      component_phase = str(component.get("phase") or "unknown").lower()
      resources = resource_references(
        run_id=run_id,
        component=component_id,
        pod_name=component.get("pod_name") or component.get("name"),
        node=component.get("node"),
      )
      component_age = age_seconds(component.get("updated_at"), now)
      if component_status == "failed":
        problems.append(
          build_problem(
            "component_failed",
            "error",
            f"The {component_id} component for run {run_id} failed",
            {
              "status": component_status,
              "phase": component_phase,
              "reason": component.get("reason"),
              "message": component.get("message"),
            },
            "Inspect the component's pod events and current or previous logs, then correct the failure before retrying.",
            resources,
            run,
          )
        )
      component_waiting = component_status == "waiting" or component_phase.startswith("waiting")
      if component_waiting and component_status != "failed":
        problems.append(
          build_problem(
            "component_waiting",
            "warning",
            f"The {component_id} component for run {run_id} is waiting",
            {"status": component_status, "phase": component_phase, "seconds_since_update": component_age},
            "Inspect this component's pod state, logs, and upstream dependencies.",
            resources,
            run,
          )
        )
      elif component_status in TRANSITIONAL_STATUSES and component_age is not None and component_age >= stuck_seconds:
        problems.append(
          build_problem(
            "component_stuck",
            "warning",
            f"The {component_id} component for run {run_id} appears stuck in {component_phase}",
            {
              "status": component_status,
              "phase": component_phase,
              "seconds_since_update": component_age,
              "threshold_seconds": stuck_seconds,
            },
            "Inspect this component's Kubernetes events and logs; stop the run if startup cannot recover.",
            resources,
            run,
          )
        )

    for queue_name, component_id in (("training", "trainer"), ("sampling", "sampler")):
      depth = nonnegative_int((run.get("queue") or {}).get(queue_name))
      if depth == 0:
        continue
      component = component_by_role.get(component_id)
      component_status = str(component.get("status") or "missing").lower() if component else "missing"
      component_age = age_seconds(component.get("updated_at"), now) if component else updated_age
      blocked = (
        status == "failed"
        or component_status in {"failed", "completed", "stopped"}
        or (component_status == "missing" and component_age is not None and component_age >= stuck_seconds)
        or (component_status in TRANSITIONAL_STATUSES and component_age is not None and component_age >= stuck_seconds)
      )
      resources = resource_references(run_id=run_id, component=component_id)
      if blocked:
        problems.append(
          build_problem(
            f"{queue_name}_queue_blocked",
            "error" if status == "failed" or component_status == "failed" else "warning",
            f"Run {run_id} has {depth} blocked {queue_name} request" + ("s" if depth != 1 else ""),
            {"depth": depth, "component_status": component_status, "seconds_since_update": component_age},
            f"Inspect the {component_id} component and its logs; stop the run if queued requests cannot be processed.",
            resources,
            run,
          )
        )
      elif depth >= backlog_depth:
        problems.append(
          build_problem(
            f"{queue_name}_queue_backlog",
            "warning",
            f"Run {run_id} has a {queue_name} queue depth of {depth}",
            {"depth": depth, "warning_depth": backlog_depth, "component_status": component_status},
            f"Inspect {component_id} throughput and resource availability before the backlog grows.",
            resources,
            run,
          )
        )

  unique = {problem["id"]: problem for problem in problems}
  severity_order = {"error": 0, "warning": 1}
  return sorted(
    unique.values(),
    key=lambda problem: (
      severity_order[problem["severity"]],
      problem["code"],
      *(problem["resources"].get(name, "") for name in ("run_id", "component", "pod_name", "node")),
      problem["id"],
    ),
  )[:MAX_PROBLEMS]


@router.get("/problems")
async def get_problems() -> dict[str, Any]:
  snapshot, runs = await asyncio.gather(cluster_snapshot(), all_runs())
  now = time.time()
  return {"generated_at": iso_timestamp(now), "problems": derive_problems(snapshot, runs, now)}


@router.get("/doctor")
async def doctor() -> dict[str, Any]:
  store = get_store()
  checks: list[dict[str, Any]] = [{"name": "gateway", "status": "pass", "message": "Gateway API is responding", "details": {}}]
  redis = getattr(store, "redis", None)
  if redis is None:
    checks.append(
      {
        "name": "store",
        "status": "warn",
        "message": "Using the in-memory store; worker state is not shared across processes",
        "details": {"backend": "memory"},
      }
    )
  else:
    try:
      latency_started = time.monotonic()
      await redis.ping()
      checks.append(
        {
          "name": "store",
          "status": "pass",
          "message": "Redis is reachable",
          "details": {"backend": "redis", "latency_seconds": time.monotonic() - latency_started},
        }
      )
    except Exception as exc:
      checks.append({"name": "store", "status": "fail", "message": f"Redis is unavailable: {exc}", "details": {}})

  cluster = await cluster_snapshot()
  cluster_check = "pass" if cluster["status"] == "healthy" else "warn" if cluster["status"] == "degraded" else "fail"
  checks.append(
    {
      "name": "cluster",
      "status": cluster_check,
      "message": f"{cluster['summary']['ready_nodes']}/{cluster['summary']['nodes']} nodes ready ({cluster['mode']})",
      "details": {"summary": cluster["summary"], "errors": cluster.get("errors", [])},
    }
  )
  metadata, control_run_ids = await asyncio.gather(load_run_metadata(store), store.list_control_run_ids())
  run_count = len(set(metadata) | set(control_run_ids))
  checks.append({"name": "runs", "status": "pass", "message": f"{run_count} runs visible", "details": {"count": run_count}})
  overall = (
    "unhealthy"
    if any(check["status"] == "fail" for check in checks)
    else "degraded"
    if any(check["status"] == "warn" for check in checks)
    else "healthy"
  )
  return {"status": overall, "checks": checks, "generated_at": iso_timestamp(time.time())}
