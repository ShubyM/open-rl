# Real data sources for the operational dashboard: gateway process state, Redis, the shared
# filesystem, and (when reachable) the Kubernetes API. Every accessor degrades to an explicit
# "unavailable" result instead of raising so the dashboard can always render something truthful.

import asyncio
import concurrent.futures
import functools
import json
import os
import platform
import shutil
import socket
import threading
import time
import urllib.parse
from datetime import UTC, datetime
from typing import Any

import httpx

from server.http_metrics import http_metrics
from server.store import InMemoryStore, RedisStore, RequestStore
from server.worker_launch_processor import FFTWorkerManager

START_TIME = time.time()
K8S_REQUEST_TIMEOUT = 4
NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
SCHEDULER_GROUP = "openrl.io"
SCHEDULER_VERSION = "v1alpha1"


def demo_mode_enabled() -> bool:
  return os.getenv("OPEN_RL_DASHBOARD_DEMO", "").lower() in {"1", "true", "yes"}


def tmp_dir() -> str:
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def iso_timestamp(ts: float | str | None) -> str | None:
  if ts is None:
    return None
  if isinstance(ts, str):
    return ts
  return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def safe_endpoint(value: str | None) -> str | None:
  """Return only scheme, host, and port. Diagnostic payloads must never echo credentials."""
  if not value:
    return None
  try:
    parsed = urllib.parse.urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
      return "[configured]"
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}"
  except ValueError:
    return "[configured]"


def build_summary() -> dict:
  return {
    "revision": os.getenv("OPEN_RL_BUILD_VERSION", "unknown"),
    "started_at": iso_timestamp(START_TIME),
    "uptime_seconds": max(0, int(time.time() - START_TIME)),
    "python_version": platform.python_version(),
    "hostname": socket.gethostname(),
  }


# *** Kubernetes ***


def k8s_namespace() -> str:
  if ns := os.getenv("OPEN_RL_WORKER_NAMESPACE"):
    return ns
  try:
    with open(NAMESPACE_FILE) as f:
      return f.read().strip()
  except OSError:
    return "default"


@functools.cache
def k8s_core_v1() -> tuple[Any, str | None]:
  """Return (CoreV1Api, None) or (None, reason). The client library and cluster credentials
  are both optional; the first outcome is cached for the lifetime of the process."""
  try:
    from kubernetes import client, config
  except ImportError:
    return None, "kubernetes python client not installed"
  try:
    config.load_incluster_config()
  except Exception:
    try:
      config.load_kube_config()
    except Exception as exc:
      return None, f"no cluster credentials: {exc}"
  return client.CoreV1Api(), None


@functools.cache
def k8s_custom_objects() -> tuple[Any, str | None]:
  _, err = k8s_core_v1()
  if err:
    return None, err
  from kubernetes import client

  return client.CustomObjectsApi(), None


@functools.cache
def k8s_workload_apis() -> tuple[Any, Any, str | None]:
  _, err = k8s_core_v1()
  if err:
    return None, None, err
  from kubernetes import client

  return client.AppsV1Api(), client.BatchV1Api(), None


def object_age_seconds(timestamp: str | None) -> int | None:
  if not timestamp:
    return None
  try:
    created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return max(0, int((datetime.now(tz=UTC) - created).total_seconds()))
  except (TypeError, ValueError):
    return None


def empty_scheduler_snapshot(*, installed: bool | None, error: str | None = None) -> dict:
  return {
    "installed": installed,
    "available": False,
    "error": error,
    "workloads": [],
    "ledgers": [],
    "summary": {"workloads": 0, "phase_counts": {}, "ledgers": 0, "seats": 0, "shared_ledgers": 0},
  }


def empty_resource_metrics(*, installed: bool | None, error: str | None = None) -> dict:
  return {
    "installed": installed,
    "available": False,
    "error": error,
    "pods_available": False,
    "nodes_available": False,
    "pods": {},
    "nodes": {},
  }


def empty_rollout_snapshot(*, available: bool = False, error: str | None = None) -> dict:
  return {
    "available": available,
    "error": error,
    "items": [],
    "sources": {},
    "summary": {"total": 0, "state_counts": {}, "kind_counts": {}, "problem_count": 0},
  }


def quantity_number(value: Any) -> float:
  if value in (None, ""):
    return 0.0
  from kubernetes.utils.quantity import parse_quantity

  return float(parse_quantity(str(value)))


def usage_values(usage: dict | None) -> dict:
  usage = usage or {}
  return {"cpu_cores": quantity_number(usage.get("cpu")), "memory_bytes": int(quantity_number(usage.get("memory")))}


def container_resource_summary(container: Any | None) -> dict:
  resources = getattr(container, "resources", None)
  requests = (getattr(resources, "requests", None) or {}) if resources else {}
  limits = (getattr(resources, "limits", None) or {}) if resources else {}
  return {
    "requests": {
      "cpu_cores": quantity_number(requests.get("cpu")),
      "memory_bytes": int(quantity_number(requests.get("memory"))),
    },
    "limits": {
      "cpu_cores": quantity_number(limits["cpu"]) if "cpu" in limits else None,
      "memory_bytes": int(quantity_number(limits["memory"])) if "memory" in limits else None,
    },
  }


def pod_resource_summary(pod: Any) -> dict:
  """Effective CPU/memory reservation with pod-level precedence and peak-init semantics.

  Requests omitted by a container are zero. A missing limit means the pod is unbounded for
  that resource and remains None instead of being presented as a zero-byte limit.
  """
  app = [container_resource_summary(container) for container in (pod.spec.containers or [])]
  init = [container_resource_summary(container) for container in (getattr(pod.spec, "init_containers", None) or [])]
  overhead = usage_values(getattr(pod.spec, "overhead", None))
  pod_resources = getattr(pod.spec, "resources", None)
  pod_requests = (getattr(pod_resources, "requests", None) or {}) if pod_resources else {}
  pod_limits = (getattr(pod_resources, "limits", None) or {}) if pod_resources else {}

  def effective_request(key: str) -> float:
    app_sum = sum(container["requests"][key] for container in app)
    init_peak = max((container["requests"][key] for container in init), default=0)
    return max(app_sum, init_peak) + overhead[key]

  def effective_limit(key: str) -> float | None:
    app_limits = [container["limits"][key] for container in app]
    init_limits = [container["limits"][key] for container in init]
    if any(value is None for value in (*app_limits, *init_limits)):
      return None
    app_sum = sum(app_limits)
    init_peak = max(init_limits, default=0)
    return max(app_sum, init_peak) + overhead[key]

  def request(key: str, resource_name: str) -> float:
    if resource_name in pod_requests:
      return quantity_number(pod_requests[resource_name]) + overhead[key]
    return effective_request(key)

  def limit(key: str, resource_name: str) -> float | None:
    if resource_name in pod_limits:
      return quantity_number(pod_limits[resource_name]) + overhead[key]
    return effective_limit(key)

  cpu_limit = limit("cpu_cores", "cpu")
  memory_limit = limit("memory_bytes", "memory")

  return {
    "requests": {
      "cpu_cores": request("cpu_cores", "cpu"),
      "memory_bytes": int(request("memory_bytes", "memory")),
    },
    "limits": {
      "cpu_cores": cpu_limit,
      "memory_bytes": int(memory_limit) if memory_limit is not None else None,
    },
  }


def resource_metrics_snapshot(namespace: str) -> dict:
  api, err = k8s_custom_objects()
  if api is None:
    return empty_resource_metrics(installed=None, error=err)

  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    pods_future = executor.submit(
      api.list_namespaced_custom_object,
      "metrics.k8s.io",
      "v1beta1",
      namespace,
      "pods",
      _request_timeout=K8S_REQUEST_TIMEOUT,
    )
    nodes_future = executor.submit(
      api.list_cluster_custom_object,
      "metrics.k8s.io",
      "v1beta1",
      "nodes",
      _request_timeout=K8S_REQUEST_TIMEOUT,
    )
    pod_error = node_error = None
    try:
      pod_items = pods_future.result().get("items", [])
    except Exception as exc:
      pod_items, pod_error = [], exc
    try:
      node_items = nodes_future.result().get("items", [])
    except Exception as exc:
      node_items, node_error = [], exc

  if pod_error is not None and node_error is not None and getattr(pod_error, "status", None) == getattr(node_error, "status", None) == 404:
    return empty_resource_metrics(installed=False)

  pods = {}
  for item in pod_items:
    name = item.get("metadata", {}).get("name")
    if not name:
      continue
    containers = {container["name"]: usage_values(container.get("usage")) for container in item.get("containers", []) if container.get("name")}
    pods[name] = {
      "cpu_cores": sum(container["cpu_cores"] for container in containers.values()),
      "memory_bytes": sum(container["memory_bytes"] for container in containers.values()),
      "containers": containers,
      "timestamp": item.get("timestamp"),
      "window": item.get("window"),
    }
  nodes = {}
  for item in node_items:
    name = item.get("metadata", {}).get("name")
    if name:
      nodes[name] = {**usage_values(item.get("usage")), "timestamp": item.get("timestamp"), "window": item.get("window")}
  errors = []
  if pod_error is not None:
    errors.append(f"pod metrics failed: {pod_error}")
  if node_error is not None:
    errors.append(f"node metrics failed: {node_error}")
  return {
    "installed": True,
    "available": pod_error is None or node_error is None,
    "error": "; ".join(errors) or None,
    "pods_available": pod_error is None,
    "nodes_available": node_error is None,
    "pods": pods,
    "nodes": nodes,
  }


def scheduler_snapshot(namespace: str) -> dict:
  """Read the optional scheduler CRDs as unstructured objects. A missing CRD is a supported
  configuration, while RBAC or API failures remain visible diagnostic facts."""
  api, err = k8s_custom_objects()
  if api is None:
    return empty_scheduler_snapshot(installed=None, error=err)

  def list_objects(plural: str) -> list[dict]:
    return api.list_namespaced_custom_object(
      SCHEDULER_GROUP,
      SCHEDULER_VERSION,
      namespace,
      plural,
      _request_timeout=K8S_REQUEST_TIMEOUT,
    ).get("items", [])

  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    workloads_future = executor.submit(list_objects, "workloads")
    ledgers_future = executor.submit(list_objects, "claimledgers")
    try:
      workload_items = workloads_future.result()
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return empty_scheduler_snapshot(installed=False)
      return empty_scheduler_snapshot(installed=True, error=f"workload list failed: {exc}")
    try:
      ledger_items = ledgers_future.result()
    except Exception as exc:
      return empty_scheduler_snapshot(installed=True, error=f"claim ledger list failed: {exc}")

  workloads = []
  for item in workload_items:
    metadata = item.get("metadata") or {}
    spec = item.get("spec") or {}
    accelerator = spec.get("accelerator") or {}
    status = item.get("status") or {}
    conditions = status.get("conditions") or []
    placed = next((condition for condition in conditions if condition.get("type") == "Placed"), None)
    workloads.append(
      {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "created_at": metadata.get("creationTimestamp"),
        "age_seconds": object_age_seconds(metadata.get("creationTimestamp")),
        "deleting": bool(metadata.get("deletionTimestamp")),
        "generation": metadata.get("generation"),
        "role": spec.get("role"),
        "model_id": spec.get("modelId"),
        "owner_id": spec.get("ownerId"),
        "training_kind": spec.get("trainingKind"),
        "requested_memory": accelerator.get("memory"),
        "max_devices": accelerator.get("maxDeviceCount", 1),
        "phase": status.get("phase") or "Pending",
        "reason": status.get("reason"),
        "claim_name": status.get("claimName"),
        "assignment_id": status.get("assignmentID"),
        "pod_name": status.get("podName"),
        "node_name": status.get("nodeName"),
        "device_count": status.get("deviceCount", 0),
        "memory_per_device": status.get("memoryPerDevice"),
        "observed_generation": status.get("observedGeneration"),
        "generation_current": status.get("observedGeneration") == metadata.get("generation"),
        "placed": None if placed is None else placed.get("status") == "True",
        "placed_reason": placed.get("reason") if placed else None,
        "placed_message": placed.get("message") if placed else None,
        "placed_transition_at": placed.get("lastTransitionTime") if placed else None,
      }
    )

  ledgers = []
  for item in ledger_items:
    metadata = item.get("metadata") or {}
    spec = item.get("spec") or {}
    seats = spec.get("seats") or []
    ledgers.append(
      {
        "name": metadata.get("name"),
        "created_at": metadata.get("creationTimestamp"),
        "age_seconds": object_age_seconds(metadata.get("creationTimestamp")),
        "claim_name": spec.get("claimName"),
        "seat_count": len(seats),
        "owners": sorted({owner for seat in seats if (owner := seat.get("owner") or seat.get("workload"))}),
        "seats": [
          {
            "workload": seat.get("workload"),
            "workload_uid": seat.get("workloadUID"),
            "assignment_id": seat.get("assignmentID"),
            "owner": seat.get("owner"),
            "host_request": seat.get("hostRequest"),
          }
          for seat in seats
        ],
      }
    )

  phase_counts: dict[str, int] = {}
  for workload in workloads:
    phase_counts[workload["phase"]] = phase_counts.get(workload["phase"], 0) + 1
  return {
    "installed": True,
    "available": True,
    "error": None,
    "workloads": workloads,
    "ledgers": ledgers,
    "summary": {
      "workloads": len(workloads),
      "phase_counts": phase_counts,
      "ledgers": len(ledgers),
      "seats": sum(ledger["seat_count"] for ledger in ledgers),
      "shared_ledgers": sum(ledger["seat_count"] > 1 for ledger in ledgers),
    },
  }


def k8s_timestamp(value: Any) -> str | None:
  return value.isoformat() if value and hasattr(value, "isoformat") else str(value) if value else None


def controller_conditions(resource: Any) -> list[dict]:
  return [
    {
      "type": condition.type,
      "status": condition.status,
      "reason": condition.reason,
      "message": condition.message,
      "last_transition_at": k8s_timestamp(condition.last_transition_time),
    }
    for condition in (getattr(resource.status, "conditions", None) or [])
  ]


def condition_matching(conditions: list[dict], condition_type: str, status: str = "True") -> dict | None:
  return next((condition for condition in conditions if condition["type"] == condition_type and condition["status"] == status), None)


def controller_to_dict(resource: Any, kind: str) -> dict:
  metadata = resource.metadata
  spec = resource.spec
  status = resource.status
  conditions = controller_conditions(resource)
  created_at = k8s_timestamp(metadata.creation_timestamp)
  generation = metadata.generation or 0
  observed_generation = getattr(status, "observed_generation", None) or 0
  current = observed_generation >= generation
  reason = message = None
  desired = ready = updated = available = active = succeeded = failed = 0

  if kind == "Deployment":
    desired = spec.replicas or 0
    ready = status.ready_replicas or 0
    updated = status.updated_replicas or 0
    available = status.available_replicas or 0
    if stalled := condition_matching(conditions, "Progressing", "False"):
      state, reason, message = "failed", stalled["reason"], stalled["message"]
    elif unavailable := condition_matching(conditions, "Available", "False"):
      state, reason, message = "degraded", unavailable["reason"], unavailable["message"]
    elif current and ready >= desired and available >= desired and updated >= desired:
      state = "healthy"
    else:
      state = "progressing"
  elif kind == "DaemonSet":
    desired = status.desired_number_scheduled or 0
    ready = status.number_ready or 0
    updated = status.updated_number_scheduled or 0
    available = status.number_available or 0
    if current and updated >= desired and ready >= desired and available >= desired:
      state = "healthy"
    elif current and updated >= desired and ready < desired:
      state = "degraded"
    else:
      state = "progressing"
  elif kind == "StatefulSet":
    desired = spec.replicas or 0
    ready = status.ready_replicas or 0
    updated = status.updated_replicas or 0
    available = getattr(status, "available_replicas", None) or ready
    current_replicas = status.current_replicas or 0
    if current and ready >= desired and current_replicas >= desired and updated >= desired:
      state = "healthy"
    elif current and updated >= desired and ready < desired:
      state = "degraded"
    else:
      state = "progressing"
  else:
    desired = spec.completions or 1
    active = status.active or 0
    succeeded = status.succeeded or 0
    failed = status.failed or 0
    if terminal := condition_matching(conditions, "Failed") or condition_matching(conditions, "FailureTarget"):
      state, reason, message = "failed", terminal["reason"], terminal["message"]
    elif condition_matching(conditions, "Complete") or succeeded >= desired:
      state = "complete"
    elif active:
      state = "running"
    else:
      state = "pending"

  return {
    "kind": kind,
    "name": metadata.name,
    "state": state,
    "reason": reason,
    "message": message,
    "desired": desired,
    "ready": ready,
    "updated": updated,
    "available": available,
    "active": active,
    "succeeded": succeeded,
    "failed": failed,
    "generation": generation,
    "observed_generation": observed_generation,
    "current": current,
    "created_at": created_at,
    "age_seconds": object_age_seconds(created_at),
    "conditions": conditions,
  }


def workload_controllers_snapshot(namespace: str) -> dict:
  apps, batch, err = k8s_workload_apis()
  if err:
    return empty_rollout_snapshot(error=err)

  calls = {
    "deployments": (apps.list_namespaced_deployment, "Deployment"),
    "daemonsets": (apps.list_namespaced_daemon_set, "DaemonSet"),
    "statefulsets": (apps.list_namespaced_stateful_set, "StatefulSet"),
    "jobs": (batch.list_namespaced_job, "Job"),
  }
  items = []
  sources = {}
  with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as executor:
    futures = {
      name: (executor.submit(measured_call, call, namespace, _request_timeout=K8S_REQUEST_TIMEOUT), kind) for name, (call, kind) in calls.items()
    }
    for name, (future, kind) in futures.items():
      result, error, collection_ms = future.result()
      observed = [] if error is not None else [controller_to_dict(resource, kind) for resource in result.items]
      sources[name] = {
        "available": error is None,
        "error": str(error) if error is not None else None,
        "collection_ms": collection_ms,
        "count": len(observed),
      }
      items.extend(observed)

  state_counts: dict[str, int] = {}
  kind_counts: dict[str, int] = {}
  for item in items:
    state_counts[item["state"]] = state_counts.get(item["state"], 0) + 1
    kind_counts[item["kind"]] = kind_counts.get(item["kind"], 0) + 1
  errors = [f"{name}: {source['error']}" for name, source in sources.items() if source["error"]]
  return {
    "available": any(source["available"] for source in sources.values()),
    "error": "; ".join(errors) or None,
    "items": sorted(items, key=lambda item: (item["kind"], item["name"])),
    "sources": sources,
    "summary": {
      "total": len(items),
      "state_counts": state_counts,
      "kind_counts": kind_counts,
      "problem_count": sum(state in {"degraded", "failed"} for state in (item["state"] for item in items)),
    },
  }


def terminated_state(state: Any) -> dict | None:
  if state is None:
    return None
  return {
    "reason": state.reason,
    "message": state.message,
    "exit_code": state.exit_code,
    "signal": state.signal,
    "started_at": k8s_timestamp(state.started_at),
    "finished_at": k8s_timestamp(state.finished_at),
  }


def pod_container_statuses(pod: Any) -> list[Any]:
  return [*(pod.status.init_container_statuses or []), *(pod.status.container_statuses or [])]


def pod_problem(pod: Any) -> str | None:
  phase = pod.status.phase or "Unknown"
  if phase == "Failed":
    detail = ": ".join(part for part in (pod.status.reason, pod.status.message) if part)
    return f"Failed: {detail or 'see logs'}"
  for cs in pod_container_statuses(pod):
    waiting = cs.state.waiting if cs.state else None
    if waiting and waiting.reason not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting.reason}: {waiting.message or ''}".strip(": ")
    terminated = cs.state.terminated if cs.state else None
    if terminated and (terminated.exit_code or terminated.reason not in (None, "Completed")):
      return f"{terminated.reason or 'Terminated'}: exit code {terminated.exit_code}{f' — {terminated.message}' if terminated.message else ''}"
    previous = cs.last_state.terminated if cs.last_state else None
    if previous and (cs.restart_count or 0) and previous.reason in {"OOMKilled", "Error", "ContainerCannotRun"}:
      return f"{previous.reason}: {cs.name} exited {previous.exit_code} and restarted"
  if phase == "Pending":
    for cond in pod.status.conditions or []:
      if cond.type == "PodScheduled" and cond.status != "True":
        return f"Unschedulable: {cond.message or cond.reason or 'no node available'}"
    return "Pending"
  return None


def pod_gpu_count(pod: Any) -> int:
  """GPUs a pod claims: nvidia.com/gpu requests/limits, or DRA resource claims (1 device each)."""
  gpus = 0
  for container in pod.spec.containers or []:
    resources = container.resources
    for source in (resources.requests if resources else None, resources.limits if resources else None):
      if source and "nvidia.com/gpu" in source:
        gpus += int(float(source["nvidia.com/gpu"]))
        break
  if gpus == 0:
    gpus = len(pod.spec.resource_claims or [])
  return gpus


def pod_to_dict(pod: Any) -> dict:
  statuses = pod.status.container_statuses or []
  specs = {
    (kind, container.name): container
    for kind, items in (("app", pod.spec.containers or []), ("init", getattr(pod.spec, "init_containers", None) or []))
    for container in items
  }
  containers = []
  for kind, items in (("app", statuses), ("init", pod.status.init_container_statuses or [])):
    for cs in items:
      state = "unknown"
      reason = None
      message = None
      started_at = None
      finished_at = None
      exit_code = None
      signal = None
      if cs.state:
        if cs.state.running:
          state = "running"
          started_at = k8s_timestamp(cs.state.running.started_at)
        elif cs.state.waiting:
          state = "waiting"
          reason = cs.state.waiting.reason
          message = cs.state.waiting.message
        elif cs.state.terminated:
          state = "terminated"
          terminated = terminated_state(cs.state.terminated) or {}
          reason = terminated.get("reason")
          message = terminated.get("message")
          started_at = terminated.get("started_at")
          finished_at = terminated.get("finished_at")
          exit_code = terminated.get("exit_code")
          signal = terminated.get("signal")
      containers.append(
        {
          "name": cs.name,
          "kind": kind,
          "image": cs.image,
          "image_id": getattr(cs, "image_id", None),
          "resources": container_resource_summary(specs.get((kind, cs.name))),
          "ready": bool(cs.ready),
          "state": state,
          "reason": reason,
          "message": message,
          "started_at": started_at,
          "finished_at": finished_at,
          "exit_code": exit_code,
          "signal": signal,
          "restart_count": cs.restart_count or 0,
          "last_termination": terminated_state(cs.last_state.terminated if cs.last_state else None),
        }
      )
  if not containers:
    containers = [
      {
        "name": c.name,
        "kind": "app",
        "image": c.image,
        "image_id": None,
        "resources": container_resource_summary(c),
        "ready": False,
        "state": "unknown",
        "reason": None,
        "message": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "signal": None,
        "restart_count": 0,
        "last_termination": None,
      }
      for c in pod.spec.containers or []
    ]
  ready_count = sum(1 for c in statuses if c.ready)
  return {
    "name": pod.metadata.name,
    "phase": pod.status.phase or "Unknown",
    "node": pod.spec.node_name,
    "app": (pod.metadata.labels or {}).get("app"),
    "labels": pod.metadata.labels or {},
    "ready": f"{ready_count}/{len(pod.spec.containers or [])}",
    "restarts": sum(cs.restart_count or 0 for cs in pod_container_statuses(pod)),
    "created_at": pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else None,
    "reason": pod.status.reason,
    "message": pod.status.message,
    "problem": pod_problem(pod),
    "containers": containers,
    "conditions": [
      {
        "type": condition.type,
        "status": condition.status,
        "reason": condition.reason,
        "message": condition.message,
        "last_transition_at": k8s_timestamp(condition.last_transition_time),
      }
      for condition in pod.status.conditions or []
    ],
    "events": [],
    "gpus": pod_gpu_count(pod),
    "resources": pod_resource_summary(pod),
  }


def event_to_dict(event: Any) -> dict:
  series = event.series
  source = event.source
  return {
    "reason": event.reason,
    "message": event.message,
    "type": event.type,
    "count": event.count or (series.count if series else None) or 1,
    "source": event.reporting_component or (source.component if source else None),
    "first_seen_at": k8s_timestamp(event.first_timestamp or event.metadata.creation_timestamp),
    "last_seen_at": k8s_timestamp((series.last_observed_time if series else None) or event.event_time or event.last_timestamp),
    "pod_name": event.involved_object.name,
  }


def node_to_dict(node: Any) -> dict:
  labels = node.metadata.labels or {}
  capacity = node.status.capacity or {}
  allocatable = node.status.allocatable or {}
  conditions = node.status.conditions or []
  return {
    "name": node.metadata.name,
    "ready": any(c.type == "Ready" and c.status == "True" for c in conditions),
    "memory_pressure": any(c.type == "MemoryPressure" and c.status == "True" for c in conditions),
    "disk_pressure": any(c.type == "DiskPressure" and c.status == "True" for c in conditions),
    "unschedulable": bool(node.spec.unschedulable),
    "instance_type": labels.get("node.kubernetes.io/instance-type") or labels.get("beta.kubernetes.io/instance-type"),
    "accelerator": labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"),
    "gpu_capacity": int(capacity.get("nvidia.com/gpu", 0)),
    "gpu_allocatable": int(allocatable.get("nvidia.com/gpu", 0)),
    "cpu_capacity_cores": quantity_number(capacity.get("cpu")),
    "cpu_allocatable_cores": quantity_number(allocatable.get("cpu")),
    "memory_capacity_bytes": int(quantity_number(capacity.get("memory"))),
    "memory_allocatable_bytes": int(quantity_number(allocatable.get("memory"))),
    "usage": None,
  }


def measured_call(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, Exception | None, float]:
  """Run one blocking observation and return its result, error, and elapsed milliseconds."""
  started = time.perf_counter()
  try:
    return func(*args, **kwargs), None, round((time.perf_counter() - started) * 1000, 3)
  except Exception as exc:
    return None, exc, round((time.perf_counter() - started) * 1000, 3)


def _collect_k8s_snapshot() -> dict:
  """List pods in our namespace and (when RBAC allows) cluster nodes. Blocking; call in a thread."""
  api, err = k8s_core_v1()
  namespace = k8s_namespace()
  if api is None:
    return {
      "available": False,
      "namespace": namespace,
      "error": err,
      "pods": [],
      "nodes": [],
      "metrics": empty_resource_metrics(installed=None, error=err),
      "scheduler": empty_scheduler_snapshot(installed=None, error=err),
      "rollouts": empty_rollout_snapshot(error=err),
    }
  pod_list, pod_error, pod_ms = measured_call(api.list_namespaced_pod, namespace, _request_timeout=K8S_REQUEST_TIMEOUT)
  component_ms = {"pods": pod_ms}
  if pod_error is not None:
    error = f"pod list failed: {pod_error}"
    return {
      "available": False,
      "namespace": namespace,
      "error": error,
      "pods": [],
      "nodes": [],
      "metrics": empty_resource_metrics(installed=None, error=error),
      "scheduler": empty_scheduler_snapshot(installed=None, error=error),
      "rollouts": empty_rollout_snapshot(error=error),
      "_component_ms": component_ms,
    }
  pods = [pod_to_dict(p) for p in pod_list.items]
  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    nodes_future = executor.submit(measured_call, api.list_node, _request_timeout=K8S_REQUEST_TIMEOUT)
    scheduler_future = executor.submit(measured_call, scheduler_snapshot, namespace)
    rollouts_future = executor.submit(measured_call, workload_controllers_snapshot, namespace)
    events_future = executor.submit(
      measured_call,
      api.list_namespaced_event,
      namespace,
      field_selector="involvedObject.kind=Pod",
      limit=200,
      _request_timeout=K8S_REQUEST_TIMEOUT,
    )
    metrics_future = executor.submit(measured_call, resource_metrics_snapshot, namespace)

    node_list, node_error, component_ms["nodes"] = nodes_future.result()
    if node_error is not None:
      # Namespaced service accounts often cannot list nodes; pods alone are still useful.
      nodes = []
      nodes_error = f"node list failed: {node_error}"
    else:
      nodes = [node_to_dict(n) for n in node_list.items]
      nodes_error = None

    scheduler, scheduler_error, component_ms["scheduler"] = scheduler_future.result()
    if scheduler_error is not None:
      scheduler = empty_scheduler_snapshot(installed=True, error=f"scheduler snapshot failed: {scheduler_error}")

    rollouts, rollouts_error, component_ms["rollouts"] = rollouts_future.result()
    if rollouts_error is not None:
      rollouts = empty_rollout_snapshot(error=f"workload controller snapshot failed: {rollouts_error}")

    event_list, event_error, component_ms["events"] = events_future.result()
    if event_error is not None:
      events, events_error = [], f"event list failed: {event_error}"
    else:
      events = [event_to_dict(event) for event in event_list.items]
      events_error = None

    metrics, metrics_error, component_ms["metrics"] = metrics_future.result()
    if metrics_error is not None:
      metrics = empty_resource_metrics(installed=True, error=f"resource metrics snapshot failed: {metrics_error}")
  events_by_pod: dict[str, list[dict]] = {}
  for event in events:
    events_by_pod.setdefault(event["pod_name"], []).append(event)
  for pod in pods:
    pod["events"] = sorted(events_by_pod.get(pod["name"], []), key=lambda event: event["last_seen_at"] or "")[-10:]
    pod["usage"] = metrics["pods"].get(pod["name"])
  for node in nodes:
    node["usage"] = metrics["nodes"].get(node["name"])
  return {
    "available": True,
    "namespace": namespace,
    "error": None,
    "pods": pods,
    "nodes": nodes,
    "nodes_error": nodes_error,
    "events_error": events_error,
    "metrics": metrics,
    "scheduler": scheduler,
    "rollouts": rollouts,
    "_component_ms": component_ms,
  }


class K8sObservationCache:
  """Short-lived, single-flight cache for API-server observations.

  Every dashboard surface can request Kubernetes state independently. Serializing refreshes
  behind one lock prevents a burst of agents and browser tabs from issuing the same expensive
  pod/node/event/custom-resource lists in parallel.
  """

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._snapshot: dict | None = None
    self._observed_monotonic = 0.0

  def clear(self) -> None:
    with self._lock:
      self._snapshot = None
      self._observed_monotonic = 0.0

  def get(self, *, force: bool = False) -> dict:
    try:
      ttl = max(0.0, float(os.getenv("OPEN_RL_K8S_CACHE_SECONDS", "1")))
    except ValueError:
      ttl = 1.0
    with self._lock:
      now = time.monotonic()
      age = now - self._observed_monotonic
      if not force and self._snapshot is not None and age < ttl:
        return {
          **self._snapshot,
          "observation": {**self._snapshot["observation"], "source": "cache", "age_seconds": round(age, 6)},
        }

      started = time.perf_counter()
      snapshot = dict(_collect_k8s_snapshot())
      component_ms = snapshot.pop("_component_ms", {})
      completed = time.time()
      self._observed_monotonic = time.monotonic()
      self._snapshot = {
        **snapshot,
        "observation": {
          "observed_at": iso_timestamp(completed),
          "collection_ms": round((time.perf_counter() - started) * 1000, 3),
          "components_ms": component_ms,
          "source": "live",
          "age_seconds": 0.0,
        },
      }
      return {**self._snapshot, "observation": dict(self._snapshot["observation"])}


k8s_observation_cache = K8sObservationCache()


def k8s_snapshot(*, force: bool = False) -> dict:
  return k8s_observation_cache.get(force=force)


def k8s_pod_logs(pod: str, container: str | None, tail: int, previous: bool = False) -> dict:
  api, err = k8s_core_v1()
  if api is None:
    raise RuntimeError(err or "kubernetes unavailable")
  text = api.read_namespaced_pod_log(
    pod,
    k8s_namespace(),
    container=container,
    previous=previous,
    tail_lines=tail,
    _request_timeout=K8S_REQUEST_TIMEOUT + 4,
  )
  return {"demo": False, "pod": pod, "container": container, "previous": previous, "text": text}


# *** GPU duty cycle ***
#
# Allocation duty per pool, broken down by job: GPUs claimed by non-terminal pods / pool GPU
# capacity, attributed to runs via the timeslice job-id labels the k8s worker manager stamps,
# sampled into an in-memory ring buffer whenever the cluster is polled. This is truthful
# scheduler state, not device utilization — DCGM owns that. History lives for the gateway's
# lifetime. A series sample is [unix_ts, {job: claimed_gpus}].

TERMINAL_POD_PHASES = {"Succeeded", "Failed"}


def pool_gpu_capacity(pool: dict) -> int:
  return sum(node["gpu_capacity"] for node in pool["nodes"])


def pod_job(pod: dict) -> str:
  """The run a pod's GPUs belong to: the model id from its timeslice job-id label, else its
  app label, else 'other'. Trainer and sampler pods of one run share one job."""
  job_id = (pod.get("labels") or {}).get("timeslice.io/job-id", "")
  for role in ("trainer-", "sampler-"):
    if job_id.startswith(role):
      return job_id.removeprefix(role)
  return pod.get("app") or "other"


class DutyTracker:
  """Ring buffer of per-job allocation-duty samples per GPU pool."""

  def __init__(self, max_samples: int = 120, min_interval_s: float = 5.0):
    self.max_samples = max_samples
    self.min_interval_s = min_interval_s
    self.series: dict[str, list[list]] = {}
    self.last_sample_at = 0.0

  def record(self, pools: list[dict], pods: list[dict], now: float | None = None) -> None:
    """Append one duty sample per GPU pool, throttled so overlapping polls don't stack points."""
    now = now if now is not None else time.time()
    if now - self.last_sample_at < self.min_interval_s:
      return
    self.last_sample_at = now
    claims_by_node: dict[str, dict[str, int]] = {}
    for pod in pods:
      gpus = pod.get("gpus", 0)
      if not gpus or not pod["node"] or pod["phase"] in TERMINAL_POD_PHASES:
        continue
      node_claims = claims_by_node.setdefault(pod["node"], {})
      job = pod_job(pod)
      node_claims[job] = node_claims.get(job, 0) + gpus
    for pool in pools:
      if not pool_gpu_capacity(pool):
        continue
      claims: dict[str, int] = {}
      for node in pool["nodes"]:
        for job, gpus in claims_by_node.get(node["name"], {}).items():
          claims[job] = claims.get(job, 0) + gpus
      series = self.series.setdefault(pool["id"], [])
      series.append([int(now), claims])
      del series[: -self.max_samples]

  def duty(self, pool: dict) -> dict | None:
    capacity = pool_gpu_capacity(pool)
    if not capacity:
      return None
    series = self.series.get(pool["id"], [])
    jobs: list[str] = []
    for _, claims in series:
      for job in claims:
        if job not in jobs:
          jobs.append(job)
    # Deliberately unclamped: time-sliced pools can be overcommitted, and current > 1 is the
    # honest way to report that.
    current = sum(series[-1][1].values()) / capacity if series else 0.0
    return {"capacity": capacity, "current": round(current, 4), "jobs": jobs, "series": series}


duty_tracker = DutyTracker()


# *** Cluster assembly ***


def gateway_summary(http_observation: dict | None = None) -> dict:
  from server import gateway

  return {
    "title": "open-rl gateway",
    "mode": "single-process" if gateway.is_single_process_mode() else "distributed",
    "fft_enabled": gateway.is_fft_enabled(),
    "redis_configured": bool(os.getenv("REDIS_URL")),
    "vllm_url": safe_endpoint(gateway.VLLM_URL) if gateway.get_sampler_backend() == "vllm" else None,
    "sampler_backend": gateway.get_sampler_backend(),
    "build": build_summary(),
    "http": http_observation if http_observation is not None else http_metrics.snapshot(),
  }


async def ping_redis(store: RequestStore) -> bool | None:
  """True/False for a Redis-backed store, None when the store is in-memory."""
  if not isinstance(store, RedisStore):
    return None
  try:
    return bool(await store.redis.ping())
  except Exception:
    return False


async def cluster_snapshot(store: RequestStore, k8s: dict, http_observation: dict | None = None) -> dict:
  gateway_card = gateway_summary(http_observation)
  redis_ok = await ping_redis(store)

  shared = tmp_dir()
  services = [
    {
      "id": "redis",
      "label": "Redis",
      "configured": gateway_card["redis_configured"],
      "ok": redis_ok,
      "detail": safe_endpoint(os.getenv("REDIS_URL")) or "not set — in-memory store",
    },
    {
      "id": "storage",
      "label": "Shared storage",
      "configured": True,
      "ok": os.path.isdir(shared) and os.access(shared, os.W_OK),
      "detail": shared,
    },
  ]
  if gateway_card["vllm_url"]:
    services.append({"id": "vllm", "label": "vLLM worker", "configured": True, "ok": None, "detail": gateway_card["vllm_url"]})

  # Edges only where we know the gateway actually connects: its configured Redis and vLLM URLs.
  edges = []
  if gateway_card["redis_configured"]:
    edges.append({"from": "gateway", "to": "redis", "reason": "REDIS_URL configured"})
  if gateway_card["vllm_url"]:
    edges.append({"from": "gateway", "to": "vllm", "reason": "VLLM_URL configured"})

  pools: dict[str, dict] = {}
  pods_by_node: dict[str, list[str]] = {}
  for pod in k8s["pods"]:
    if pod["node"]:
      pods_by_node.setdefault(pod["node"], []).append(pod["name"])
  for node in k8s["nodes"]:
    pool_id = node["accelerator"] or ("gpu" if node["gpu_capacity"] else "cpu")
    pool = pools.setdefault(pool_id, {"id": pool_id, "label": pool_id, "nodes": []})
    pool["nodes"].append(
      {
        **{
          k: node.get(k)
          for k in (
            "name",
            "ready",
            "instance_type",
            "gpu_capacity",
            "gpu_allocatable",
            "cpu_capacity_cores",
            "cpu_allocatable_cores",
            "memory_capacity_bytes",
            "memory_allocatable_bytes",
            "usage",
          )
        },
        "pods": pods_by_node.get(node["name"], []),
      }
    )
  duty_tracker.record(list(pools.values()), k8s["pods"])
  for pool in pools.values():
    pool["duty"] = duty_tracker.duty(pool)

  known_nodes = {n["name"] for n in k8s["nodes"]}
  unplaced = [p["name"] for p in k8s["pods"] if not p["node"] or p["node"] not in known_nodes]
  if unplaced:
    pools["unscheduled"] = {
      "id": "unscheduled",
      "label": "not scheduled",
      "duty": None,
      "nodes": [{"name": "—", "ready": None, "instance_type": None, "gpu_capacity": 0, "gpu_allocatable": 0, "pods": unplaced}],
    }

  metrics = k8s.get("metrics") or empty_resource_metrics(installed=None)
  return {
    "demo": False,
    "kubernetes": {
      "available": k8s["available"],
      "namespace": k8s["namespace"],
      "error": k8s["error"],
      "nodes_error": k8s.get("nodes_error"),
      "observation": k8s.get("observation"),
      "metrics": {
        "installed": metrics["installed"],
        "available": metrics["available"],
        "error": metrics["error"],
        "pods_available": metrics["pods_available"],
        "nodes_available": metrics["nodes_available"],
        "pods_observed": len(metrics["pods"]),
        "nodes_observed": len(metrics["nodes"]),
      },
    },
    "gateway": gateway_card,
    "scheduler": k8s.get("scheduler") or empty_scheduler_snapshot(installed=None),
    "rollouts": k8s.get("rollouts") or empty_rollout_snapshot(),
    "services": services,
    "edges": edges,
    "pools": sorted(pools.values(), key=lambda p: (p["id"] == "cpu", p["id"] == "unscheduled", p["id"])),
    "pods": k8s["pods"],
  }


async def diagnostic_snapshot(store: RequestStore, worker_manager: FFTWorkerManager | None, k8s: dict) -> dict:
  """Build one coherent dashboard payload from one Kubernetes observation."""
  http_observation = http_metrics.snapshot()
  try:
    queues, launch = await asyncio.gather(store.queue_stats(), store.worker_launch_stats())
  except Exception:
    queues, launch = [], {"depth": 0, "oldest_enqueued_at": None, "oldest_age_seconds": None}
  cluster, runs, checks, operational = await asyncio.gather(
    cluster_snapshot(store, k8s, http_observation),
    runs_snapshot(store, worker_manager, k8s["pods"], k8s.get("scheduler"), queues),
    health_checks(store, k8s),
    operational_stats(store, k8s, worker_manager, queues, launch, http_observation),
  )
  stats, queues = operational
  return {
    "demo": False,
    "generated_at": iso_timestamp(time.time()),
    "cluster": cluster,
    "runs": runs,
    "health": {"demo": False, "checks": checks, "stats": stats, "queues": queues},
    "problems": {"demo": False, "problems": derive_problems(checks, k8s, stats, runs["runs"])},
  }


# *** Runs ***


def filesystem_runs() -> dict[str, dict]:
  found: dict[str, dict] = {}
  peft_dir = os.path.join(tmp_dir(), "peft")
  if os.path.isdir(peft_dir):
    for entry in os.scandir(peft_dir):
      if not entry.is_dir():
        continue
      info = found.setdefault(entry.name, {"sources": set()})
      info["sources"].add("adapter")
      info.setdefault("created_at", iso_timestamp(entry.stat().st_ctime))
      metadata_path = os.path.join(entry.path, "metadata.json")
      if os.path.exists(metadata_path):
        try:
          with open(metadata_path) as f:
            meta = json.load(f)
          info.setdefault("name", meta.get("alias"))
          info.setdefault("base_model", meta.get("base_model"))
          info.setdefault("wandb_url", meta.get("wandb_url"))
        except Exception:
          pass
  ckpt_dir = os.path.join(tmp_dir(), "checkpoints")
  if os.path.isdir(ckpt_dir):
    for entry in os.scandir(ckpt_dir):
      if entry.is_dir():
        info = found.setdefault(entry.name, {"sources": set()})
        info["sources"].add("checkpoint")
        info.setdefault("created_at", iso_timestamp(entry.stat().st_ctime))
  return found


async def redis_runs(store: RequestStore, queues: list[dict] | None = None) -> dict[str, dict]:
  found: dict[str, dict] = {}
  try:
    for model_id, meta in (await store.list_model_metadata()).items():
      info = found.setdefault(model_id, {"sources": set()})
      info["sources"].add("registered")
      for field in (
        "base_model",
        "created_at",
        "wandb_url",
        "status",
        "error",
        "operation",
        "state_path",
        "ready_at",
        "failed_at",
        "stopped_at",
        "telemetry",
      ):
        if meta.get(field) is not None:
          info[field] = iso_timestamp(meta[field]) if field in {"created_at", "ready_at", "failed_at", "stopped_at"} else meta[field]
  except Exception:
    pass
  try:
    for queue in queues if queues is not None else await store.queue_stats():
      model_id = queue["model_id"]
      if model_id != "default":
        info = found.setdefault(model_id, {"sources": set()})
        info["sources"].add("queue")
        info["queue_depth"] = queue["depth"]
        info["queue_oldest_at"] = iso_timestamp(queue["oldest_enqueued_at"])
        info["queue_oldest_seconds"] = queue["oldest_age_seconds"]
  except Exception:
    pass
  return found


def worker_processes(worker_manager: FFTWorkerManager | None) -> dict[str, bool]:
  """model_id -> process alive, for gateway-launched local FFT workers."""
  if worker_manager is None:
    return {}
  return {model_id: proc.poll() is None for model_id, proc in worker_manager.processes.items()}


def sanitize_job_id(model_id: str) -> str:
  return "".join(c if c.isalnum() else "-" for c in model_id.lower()).strip("-")


def model_pods(model_id: str, pods: list[dict]) -> list[dict]:
  """Pods launched for this model, matched exactly on the timeslice job-id labels the k8s
  worker manager stamps. Exact match only: prefix matching would cross-match runs that share
  an id prefix, and stop_run deletes what this returns. (Branches whose sanitize_job_id
  hash-truncates long ids should swap that helper in here.)"""
  wanted = {f"{role}-{sanitize_job_id(model_id)}" for role in ("trainer", "sampler")}
  return [p for p in pods if (p.get("labels") or {}).get("timeslice.io/job-id") in wanted]


def model_workloads(model_id: str, scheduler: dict | None) -> list[dict]:
  return [workload for workload in (scheduler or {}).get("workloads", []) if workload.get("model_id") == model_id]


def summarize_run(
  pods: list[dict],
  queue_depth: int,
  worker_alive: bool | None,
  sources: set[str] | list[str],
  model_status: str | None = None,
  model_error: str | None = None,
  workloads: list[dict] | None = None,
) -> dict:
  """Derive a compact lifecycle verdict from observable state without pretending an artifact
  is a currently running process. The verdict is shared by the list and detail endpoints."""
  phases: dict[str, int] = {}
  for pod in pods:
    phase = pod.get("phase") or "Unknown"
    phases[phase] = phases.get(phase, 0) + 1

  failed = next(
    (
      pod
      for pod in pods
      if pod.get("phase") == "Failed" or any(marker in (pod.get("problem") or "") for marker in ("BackOff", "OOM", "Error", "Failed"))
    ),
    None,
  )
  waiting = next((pod for pod in pods if pod.get("problem") or pod.get("phase") == "Pending"), None)
  workloads = workloads or []
  workload_phases: dict[str, int] = {}
  for workload in workloads:
    workload_phases[workload["phase"]] = workload_phases.get(workload["phase"], 0) + 1
  failed_workload = next((workload for workload in workloads if workload["phase"] == "Failed"), None)
  waiting_workload = next((workload for workload in workloads if workload["phase"] in {"Pending", "Placing"}), None)
  running = phases.get("Running", 0)
  if failed or failed_workload or model_status == "failed":
    if failed:
      reason = failed.get("problem") or f"pod {failed['name']} failed"
    elif failed_workload:
      reason = failed_workload.get("reason") or f"workload {failed_workload['name']} failed"
    else:
      reason = model_error or "model creation failed"
    phase, status = "failed", "error"
  elif waiting and running:
    phase, status, reason = "degraded", "warn", waiting.get("problem") or f"pod {waiting['name']} is pending"
  elif waiting:
    phase, status, reason = "waiting", "warn", waiting.get("problem") or f"pod {waiting['name']} is pending"
  elif waiting_workload:
    phase, status, reason = (
      "waiting",
      "warn",
      waiting_workload.get("reason") or f"workload {waiting_workload['name']} is {waiting_workload['phase'].lower()}",
    )
  elif worker_alive or running or workload_phases.get("Running"):
    count = running or 1
    if running:
      reason = f"{count} running {'pod' if count == 1 else 'pods'}"
    elif workload_phases.get("Running"):
      count = workload_phases["Running"]
      reason = f"{count} scheduler {'workload is' if count == 1 else 'workloads are'} running"
    else:
      reason = "local worker process is alive"
    phase, status = "running", "ok"
  elif queue_depth:
    phase, status, reason = "queued", "ok", f"{queue_depth} queued {'request' if queue_depth == 1 else 'requests'}"
  elif model_status == "queued":
    phase, status, reason = "starting", "ok", "model creation is in progress"
  elif model_status == "ready":
    phase, status, reason = "ready", "ok", "model is ready in the gateway"
  elif model_status == "stopped":
    phase, status, reason = "stopped", "off", "run was stopped by an operator"
  elif phases.get("Succeeded"):
    phase, status, reason = "succeeded", "ok", f"{phases['Succeeded']} pod{'s' if phases['Succeeded'] != 1 else ''} succeeded"
  elif set(sources) & {"adapter", "checkpoint"}:
    phase, status, reason = "saved", "off", "saved artifacts are present; no active worker is visible"
  else:
    phase, status, reason = "inactive", "off", "no active worker, queued work, or pod is visible"
  return {
    "phase": phase,
    "status": status,
    "reason": reason,
    "pod_phase_counts": phases,
    "workload_phase_counts": workload_phases,
  }


def current_gpu_claims(pods: list[dict], nodes: list[dict]) -> dict[str, int]:
  """Current GPU claims for one run grouped by the same pool identifiers as Cluster."""
  pools_by_node = {node["name"]: node["accelerator"] or ("gpu" if node["gpu_capacity"] else "cpu") for node in nodes}
  claims: dict[str, int] = {}
  for pod in pods:
    if not pod.get("node") or pod.get("phase") in TERMINAL_POD_PHASES or not pod.get("gpus"):
      continue
    pool = pools_by_node.get(pod["node"], "unknown")
    claims[pool] = claims.get(pool, 0) + pod["gpus"]
  return claims


def aggregate_pod_resources(pods: list[dict]) -> dict:
  requests = {
    "cpu_cores": sum(((pod.get("resources") or {}).get("requests") or {}).get("cpu_cores") or 0 for pod in pods),
    "memory_bytes": sum(((pod.get("resources") or {}).get("requests") or {}).get("memory_bytes") or 0 for pod in pods),
  }

  def aggregate_limit(key: str) -> float | None:
    values = [((pod.get("resources") or {}).get("limits") or {}).get(key) for pod in pods]
    return None if any(value is None for value in values) else sum(values)

  measured = [pod for pod in pods if pod.get("usage")]
  return {
    "requests": requests,
    "limits": {
      "cpu_cores": aggregate_limit("cpu_cores"),
      "memory_bytes": aggregate_limit("memory_bytes"),
    },
    "usage": {
      "cpu_cores": sum(pod["usage"]["cpu_cores"] for pod in measured),
      "memory_bytes": sum(pod["usage"]["memory_bytes"] for pod in measured),
      "measured_pods": len(measured),
      "total_pods": len(pods),
    },
  }


def diagnostic_entry(
  code: str,
  severity: str,
  source: str,
  message: str,
  *,
  resource: dict,
  evidence: dict | None = None,
  actions: list[dict] | None = None,
) -> dict:
  resource_id = f"{resource['kind'].lower()}/{resource['name']}"
  return {
    "id": f"{code}:{resource_id}",
    "code": code,
    "severity": severity,
    "source": source,
    "resource": resource,
    "message": message,
    "evidence": evidence or {},
    "actions": actions or [],
  }


def pod_diagnostic(pod: dict, namespace: str | None = None) -> dict | None:
  source = f"pod/{pod['name']}"
  namespace = namespace or k8s_namespace()
  resource = {"kind": "Pod", "name": pod["name"], "namespace": namespace}
  action = {
    "label": "Read pod logs",
    "method": "GET",
    "path": f"/api/v1/dashboard/pods/{pod['name']}/logs?tail=500",
    "command": f"make ops logs {pod['name']}",
  }
  actions = [
    action,
    {
      "label": "Describe pod and events",
      "command": f"kubectl describe pod {pod['name']} -n {namespace}",
    },
  ]
  containers = pod.get("containers") or []
  affected = next(
    (
      container for container in containers if (container.get("last_termination") or {}).get("reason") in {"OOMKilled", "Error", "ContainerCannotRun"}
    ),
    None,
  ) or next((container for container in containers if container.get("reason")), None)
  if affected and affected.get("last_termination"):
    actions.append(
      {
        "label": f"Read previous {affected['name']} logs",
        "command": f"kubectl logs {pod['name']} -n {namespace} -c {affected['name']} --previous --tail=500",
      }
    )
  problem = pod.get("problem")
  reasons = {
    reason for container in containers for reason in (container.get("reason"), (container.get("last_termination") or {}).get("reason")) if reason
  }
  event_reasons = {event.get("reason") for event in pod.get("events") or []}
  combined = reasons | event_reasons | ({pod.get("reason")} if pod.get("reason") else set())
  problem_lower = (problem or "").lower()
  resource_evidence = {"resources": pod.get("resources") or {}, "usage": pod.get("usage")}
  code = None
  severity = "warn"
  if "OOMKilled" in combined or "oom" in problem_lower or "out of memory" in problem_lower:
    code, severity = "pod.oom_killed", "error"
  elif "Evicted" in combined:
    code, severity = "pod.evicted", "error"
  elif combined & {"ImagePullBackOff", "ErrImagePull", "InvalidImageName"}:
    code, severity = "pod.image_pull", "error"
  elif "CrashLoopBackOff" in combined or "CrashLoopBackOff" in (problem or ""):
    code, severity = "pod.crash_loop", "error"
  elif combined & {"FailedMount", "FailedAttachVolume", "FailedMapVolume"}:
    code = "pod.volume_mount"
  elif "FailedScheduling" in combined or "Unschedulable" in (problem or ""):
    code = "pod.unschedulable"
  if problem:
    severity = (
      "error" if severity == "error" or pod.get("phase") == "Failed" or any(marker in problem for marker in ("BackOff", "OOM", "Error")) else "warn"
    )
    code = code or ("pod.failed" if severity == "error" else "pod.waiting")
    return diagnostic_entry(
      code,
      severity,
      source,
      problem,
      resource=resource,
      evidence={
        "phase": pod.get("phase"),
        "reason": pod.get("reason"),
        "message": pod.get("message"),
        "node": pod.get("node"),
        "restarts": pod.get("restarts", 0),
        "containers": containers,
        "conditions": pod.get("conditions") or [],
        "events": pod.get("events") or [],
        **resource_evidence,
      },
      actions=actions,
    )
  memory_limit = ((pod.get("resources") or {}).get("limits") or {}).get("memory_bytes")
  memory_used = (pod.get("usage") or {}).get("memory_bytes")
  if memory_limit and memory_used is not None and memory_used / memory_limit >= 0.9:
    ratio = memory_used / memory_limit
    return diagnostic_entry(
      "pod.memory_limit_near",
      "warn",
      source,
      f"memory working set is {ratio:.0%} of the pod limit",
      resource=resource,
      evidence={"utilization": ratio, **resource_evidence},
      actions=actions,
    )
  if pod.get("restarts", 0) >= 3:
    return diagnostic_entry(
      "pod.restarts",
      "warn",
      source,
      f"{pod['restarts']} container restarts",
      resource=resource,
      evidence={"phase": pod.get("phase"), "node": pod.get("node"), "restarts": pod["restarts"], "containers": containers, **resource_evidence},
      actions=actions,
    )
  return None


def run_diagnostics(
  run_id: str,
  state: dict,
  pods: list[dict],
  queue_depth: int,
  k8s: dict,
  queue_age_seconds: float | None = None,
  telemetry: dict | None = None,
) -> list[dict]:
  diagnostics = []
  if not k8s["available"]:
    diagnostics.append(
      diagnostic_entry(
        "kubernetes.unavailable",
        "warn",
        "kubernetes",
        k8s.get("error") or "Kubernetes is unavailable; pod state cannot be inspected",
        resource={"kind": "Cluster", "name": k8s.get("namespace") or "unknown"},
        evidence={"available": False, "namespace": k8s.get("namespace")},
        actions=[{"label": "Check dashboard health", "method": "GET", "path": "/api/v1/dashboard/health", "command": "make ops health"}],
      )
    )
  for pod in pods:
    if diagnostic := pod_diagnostic(pod, k8s.get("namespace")):
      diagnostics.append(diagnostic)
  if state["phase"] == "queued" and queue_depth:
    warn_after = float(os.getenv("OPEN_RL_QUEUE_WARN_SECONDS", "300"))
    stalled = queue_age_seconds is not None and queue_age_seconds >= warn_after
    diagnostics.append(
      diagnostic_entry(
        "run.queue_stalled" if stalled else "run.queued",
        "warn" if stalled else "info",
        f"run/{run_id}",
        f"oldest request has waited {format_duration(queue_age_seconds)}" if stalled else state["reason"],
        resource={"kind": "Run", "name": run_id},
        evidence={"queue_depth": queue_depth, "oldest_age_seconds": queue_age_seconds, "warn_after_seconds": warn_after},
        actions=[
          {"label": "Refresh run inspection", "method": "GET", "path": f"/api/v1/dashboard/runs/{run_id}", "command": f"make ops run {run_id}"}
        ],
      )
    )
  diagnostics.extend(run_telemetry_diagnostics(run_id, telemetry or {}))
  return diagnostics


def run_telemetry_diagnostics(run_id: str, telemetry: dict) -> list[dict]:
  diagnostics = []
  completed = int(telemetry.get("requests_completed") or 0)
  failures = int(telemetry.get("requests_failed") or 0)
  last_failed = telemetry.get("last_outcome") == "error"
  elevated_failures = completed >= 5 and failures / completed >= 0.2
  action = {
    "label": "Inspect run and pod logs",
    "method": "GET",
    "path": f"/api/v1/dashboard/runs/{run_id}?logs=100",
    "command": f"make ops run {run_id} 100",
  }
  if last_failed or elevated_failures:
    diagnostics.append(
      diagnostic_entry(
        "run.request_failed" if last_failed else "run.request_errors",
        "error" if last_failed else "warn",
        f"run/{run_id}",
        telemetry.get("last_error") or f"{failures} of {completed} completed requests failed",
        resource={"kind": "Run", "name": run_id},
        evidence={
          "requests_completed": completed,
          "requests_failed": failures,
          "failure_rate": failures / completed if completed else 0,
          "last_operation": telemetry.get("last_operation"),
          "last_error_at": iso_timestamp(telemetry.get("last_error_at")),
        },
        actions=[action],
      )
    )
  active = telemetry.get("active_request") or {}
  active_age = active.get("age_seconds") or 0
  warn_after = float(os.getenv("OPEN_RL_OPERATION_WARN_SECONDS", "600"))
  if active and active_age >= warn_after:
    diagnostics.append(
      diagnostic_entry(
        "run.request_stalled",
        "warn",
        f"run/{run_id}",
        f"{active.get('operation') or 'request'} has been executing for {format_duration(active_age)}",
        resource={"kind": "Run", "name": run_id},
        evidence={**active, "warn_after_seconds": warn_after},
        actions=[action],
      )
    )
  return diagnostics


def telemetry_snapshot(raw: dict | None) -> dict:
  telemetry = {**(raw or {})}
  if active := telemetry.get("active_request"):
    active = {**active}
    try:
      active["age_seconds"] = max(0.0, time.time() - float(active["started_at"]))
    except (KeyError, TypeError, ValueError):
      active["age_seconds"] = None
    telemetry["active_request"] = active
  return telemetry


def scheduler_run_diagnostics(workloads: list[dict], ledgers: list[dict], k8s: dict) -> list[dict]:
  if not workloads:
    return []
  workload_names = {workload["name"] for workload in workloads}
  ledger_names = {ledger["name"] for ledger in ledgers}
  scheduler_only = {**k8s, "pods": [], "nodes": []}
  return [
    problem
    for problem in derive_problems([], scheduler_only)
    if (problem["resource"]["kind"] == "Workload" and problem["resource"]["name"] in workload_names)
    or (problem["resource"]["kind"] == "ClaimLedger" and problem["resource"]["name"] in ledger_names)
  ]


async def runs_snapshot(
  store: RequestStore,
  worker_manager: FFTWorkerManager | None,
  pods: list[dict],
  scheduler: dict | None = None,
  queues: list[dict] | None = None,
) -> dict:
  found = await redis_runs(store, queues)
  for workload in (scheduler or {}).get("workloads", []):
    model_id = workload.get("model_id")
    if not model_id:
      continue
    info = found.setdefault(model_id, {"sources": set()})
    info["sources"].add("scheduler")
    if info.get("created_at") is None:
      info["created_at"] = workload.get("created_at")
  for model_id, info in filesystem_runs().items():
    merged = found.setdefault(model_id, {"sources": set()})
    merged["sources"] |= info.pop("sources")
    for key, value in info.items():
      if merged.get(key) is None:
        merged[key] = value
  workers = worker_processes(worker_manager)
  for model_id, alive in workers.items():
    info = found.setdefault(model_id, {"sources": set()})
    info["sources"].add("worker")
    info["worker_alive"] = alive

  runs = []
  for model_id, info in found.items():
    run_pods = model_pods(model_id, pods)
    active_resource_pods = [pod for pod in run_pods if pod.get("phase") not in TERMINAL_POD_PHASES]
    resource_pods = active_resource_pods or run_pods
    run_workloads = model_workloads(model_id, scheduler)
    queue_depth = int(info.get("queue_depth") or 0)
    worker_alive = info.get("worker_alive")
    stoppable = bool(info.get("worker_alive") or "queue" in info["sources"] or run_pods)
    runs.append(
      {
        "run_id": model_id,
        "name": info.get("name") or f"run-{model_id[:8]}",
        "base_model": info.get("base_model"),
        "created_at": info.get("created_at"),
        "wandb_url": info.get("wandb_url"),
        "stoppable": stoppable,
        "sources": sorted(info["sources"]),
        "pods": [p["name"] for p in run_pods],
        "workloads": [workload["name"] for workload in run_workloads],
        "placement": {
          "workloads": len(run_workloads),
          "device_count": sum(workload.get("device_count") or 0 for workload in run_workloads),
          "phase_counts": {
            phase: sum(workload["phase"] == phase for workload in run_workloads)
            for phase in sorted({workload["phase"] for workload in run_workloads})
          },
        },
        "resources": {
          **aggregate_pod_resources(resource_pods),
          "scope": "active" if active_resource_pods else "terminal" if run_pods else "none",
        },
        "queue_depth": queue_depth,
        "queue_oldest_at": info.get("queue_oldest_at"),
        "queue_oldest_seconds": info.get("queue_oldest_seconds"),
        "telemetry": telemetry_snapshot(info.get("telemetry")),
        "worker_alive": worker_alive,
        "model_status": info.get("status"),
        "lifecycle": {
          "operation": info.get("operation"),
          "status": info.get("status"),
          "error": info.get("error"),
          "ready_at": info.get("ready_at"),
          "failed_at": info.get("failed_at"),
          "stopped_at": info.get("stopped_at"),
        },
        "state": summarize_run(run_pods, queue_depth, worker_alive, info["sources"], info.get("status"), info.get("error"), run_workloads),
      }
    )
  runs.sort(key=lambda r: (r["created_at"] is not None, r["created_at"] or "", r["run_id"]), reverse=True)
  return {"demo": False, "runs": runs}


async def run_detail(
  store: RequestStore,
  worker_manager: FFTWorkerManager | None,
  run_id: str,
  k8s: dict,
  log_tail: int = 0,
) -> dict | None:
  """Everything about one run in a single payload: its record, full pod state, queue depth,
  current GPU claims per pool, and (when log_tail > 0) a log tail per pod."""
  scheduler = k8s.get("scheduler")
  snapshot = await runs_snapshot(store, worker_manager, k8s["pods"], scheduler)
  run = next((r for r in snapshot["runs"] if r["run_id"] == run_id), None)
  if run is None:
    return None

  pods = model_pods(run_id, k8s["pods"])
  workloads = model_workloads(run_id, scheduler)
  claim_names = {workload["claim_name"] for workload in workloads if workload.get("claim_name")}
  ledgers = [ledger for ledger in (scheduler or {}).get("ledgers", []) if ledger.get("claim_name") in claim_names]
  queue_depth = run["queue_depth"]
  diagnostics = run_diagnostics(run_id, run["state"], pods, queue_depth, k8s, run.get("queue_oldest_seconds"), run.get("telemetry"))
  diagnostics.extend(scheduler_run_diagnostics(workloads, ledgers, k8s))
  gpu_claims = current_gpu_claims(pods, k8s["nodes"])
  detail = {
    **run,
    "demo": False,
    "pods": pods,
    "workloads": workloads,
    "claim_ledgers": ledgers,
    "queue_depth": queue_depth,
    "gpu_claims": gpu_claims,
    "gpu_devices": sum(gpu_claims.values()),
    "scheduled_devices": sum(workload.get("device_count") or 0 for workload in workloads),
    "diagnostics": diagnostics,
  }
  if log_tail:

    async def read_log(pod: dict) -> tuple[str, str]:
      try:
        text = (await asyncio.to_thread(k8s_pod_logs, pod["name"], None, log_tail))["text"]
      except Exception as exc:
        text = f"(logs unavailable: {exc})"
      return pod["name"], text

    detail["logs"] = dict(await asyncio.gather(*(read_log(pod) for pod in pods)))
  return detail


async def stop_run(store: RequestStore, worker_manager: FFTWorkerManager | None, model_id: str) -> dict:
  """Stop everything we can truthfully stop for a run: the gateway-launched worker process,
  queued work in Redis, and any pods labeled for the model. Reports each action taken."""
  actions = []
  errors = []
  changed = False

  proc = worker_manager.processes.get(model_id) if worker_manager is not None else None
  if proc is not None and proc.poll() is None:
    proc.terminate()
    actions.append("terminated local worker process")
    changed = True

  if isinstance(store, RedisStore):
    try:
      removed = await store.redis.delete(
        f"open_rl:queue:{model_id}",
        f"open_rl:sampler_queue:{model_id}",
        f"open_rl:sampler_ready:{model_id}",
      )
      await store.redis.lrem(store.active_list, 0, model_id)
      await store.redis.srem(store.active_set, model_id)
      if removed:
        actions.append(f"cleared {removed} queue key(s) in Redis")
        changed = True
    except Exception as exc:
      errors.append(f"redis cleanup failed: {exc}")
  elif isinstance(store, InMemoryStore):
    async with store.active_tenants_cv:
      if model_id in store.queues:
        del store.queues[model_id]
        store.queue_oldest_at.pop(model_id, None)
        actions.append("cleared in-memory queue")
        changed = True
      if model_id in store.active_tenants:
        store.active_tenants.remove(model_id)

  api, _ = k8s_core_v1()
  if api is not None:
    try:
      k8s = k8s_snapshot(force=True)
      for pod in model_pods(model_id, k8s["pods"]):
        api.delete_namespaced_pod(pod["name"], k8s["namespace"], _request_timeout=K8S_REQUEST_TIMEOUT)
        actions.append(f"deleted pod {pod['name']}")
        changed = True
    except Exception as exc:
      errors.append(f"pod deletion failed: {exc}")
    finally:
      # The forced read happened before deletion; never serve that pre-mutation
      # observation to the next agent or browser refresh.
      k8s_observation_cache.clear()

  if changed:
    try:
      await store.set_model_metadata(model_id, {"status": "stopped", "stopped_at": time.time()})
    except Exception as exc:
      errors.append(f"lifecycle update failed: {exc}")
  return {"run_id": model_id, "stopped": changed, "actions": actions, "errors": errors}


# *** Operational load stats ***


def format_bytes(n: float) -> str:
  for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
    if n < 1024 or unit == "TiB":
      return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
    n /= 1024
  return f"{n:.1f} TiB"


def format_duration(seconds: float | None) -> str:
  seconds = max(0, int(seconds or 0))
  if seconds < 60:
    return f"{seconds}s"
  if seconds < 3600:
    return f"{seconds // 60}m {seconds % 60}s"
  return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def gateway_rss_bytes() -> int | None:
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) * 1024
  except (OSError, ValueError, IndexError):
    pass
  return None


def stat_entry(
  stat_id: str,
  label: str,
  value: str,
  detail: str = "",
  *,
  value_number: int | float,
  unit: str,
  context: dict | None = None,
  status: str = "ok",
) -> dict:
  return {
    "id": stat_id,
    "label": label,
    "value": value,
    "value_number": value_number,
    "unit": unit,
    "detail": detail,
    "context": context or {},
    "status": status,
  }


async def operational_stats(
  store: RequestStore,
  k8s: dict,
  worker_manager: FFTWorkerManager | None,
  queues: list[dict] | None = None,
  launch: dict | None = None,
  http_observation: dict | None = None,
) -> tuple[list[dict], list[dict]]:
  """Load numbers for the Health screen: queue depths, active runs, Redis and gateway memory,
  disk, and pod totals. Everything is measured, never estimated."""
  queues_provided = queues is not None
  launch_provided = launch is not None
  queues = queues or []
  launch = launch or {"depth": 0, "oldest_enqueued_at": None, "oldest_age_seconds": None}
  redis_stats: list[dict] = []
  try:
    if not queues_provided:
      queues = await store.queue_stats()
    if not launch_provided:
      launch = await store.worker_launch_stats()
  except Exception:
    pass
  if isinstance(store, RedisStore):
    try:
      memory = await store.redis.info("memory")
      used = memory.get("used_memory", 0)
      maxmemory = memory.get("maxmemory", 0)
      value = f"{used / maxmemory:.0%} of {format_bytes(maxmemory)}" if maxmemory else format_bytes(used)
      limit = f"limit {format_bytes(maxmemory)}" if maxmemory else "no maxmemory limit"
      peak = memory.get("used_memory_peak", 0)
      utilization = used / maxmemory if maxmemory else None
      redis_stats.append(
        stat_entry(
          "redis.memory",
          "Redis memory",
          value,
          f"peak {format_bytes(peak)} · {limit}",
          value_number=used,
          unit="bytes",
          context={"peak_bytes": peak, "limit_bytes": maxmemory or None, "utilization": utilization},
          status="warn" if utilization is not None and utilization >= 0.8 else "ok",
        )
      )
      clients = await store.redis.info("clients")
      connected = clients.get("connected_clients", 0)
      redis_stats.append(stat_entry("redis.clients", "Redis clients", str(connected), "connected", value_number=connected, unit="clients"))
    except Exception:
      pass
  queues.sort(key=lambda q: -q["depth"])

  workers = worker_processes(worker_manager)
  active = {model_id for model_id, alive in workers.items() if alive} | {q["model_id"] for q in queues if q["model_id"] != "default"}
  queue_warn_seconds = float(os.getenv("OPEN_RL_QUEUE_WARN_SECONDS", "300"))
  launch_warn_seconds = float(os.getenv("OPEN_RL_LAUNCH_WARN_SECONDS", "60"))
  oldest_queue = max(queues, key=lambda queue: queue["oldest_age_seconds"] or 0, default=None)
  oldest_queue_seconds = (oldest_queue or {}).get("oldest_age_seconds") or 0
  launch_depth = launch["depth"]
  launch_age = launch["oldest_age_seconds"] or 0

  stats = [
    stat_entry("runs.active", "Active runs", str(len(active)), "live worker or queued work", value_number=len(active), unit="runs"),
    stat_entry(
      "queue.requests",
      "Queued requests",
      str(sum(q["depth"] for q in queues)),
      f"across {len(queues)} queue{'' if len(queues) == 1 else 's'}",
      value_number=sum(q["depth"] for q in queues),
      unit="requests",
      context={
        "queue_count": len(queues),
        "oldest_model_id": oldest_queue["model_id"] if oldest_queue else None,
        "oldest_age_seconds": oldest_queue_seconds,
      },
      status="warn" if oldest_queue_seconds >= queue_warn_seconds else "ok",
    ),
    stat_entry(
      "queue.request_age",
      "Oldest request wait",
      format_duration(oldest_queue_seconds),
      oldest_queue["model_id"] if oldest_queue else "no queued requests",
      value_number=oldest_queue_seconds,
      unit="seconds",
      context={"model_id": oldest_queue["model_id"] if oldest_queue else None, "warn_after_seconds": queue_warn_seconds},
      status="warn" if oldest_queue_seconds >= queue_warn_seconds else "ok",
    ),
    stat_entry(
      "queue.launch",
      "Launches pending",
      str(launch_depth),
      f"oldest waiting {format_duration(launch_age)}" if launch_depth else "worker launch queue",
      value_number=launch_depth,
      unit="runs",
      context={"oldest_age_seconds": launch_age},
      status="warn" if launch_age >= launch_warn_seconds else "ok",
    ),
    stat_entry(
      "queue.launch_age",
      "Oldest launch wait",
      format_duration(launch_age),
      "worker launch queue" if launch_depth else "no pending launches",
      value_number=launch_age,
      unit="seconds",
      context={"warn_after_seconds": launch_warn_seconds},
      status="warn" if launch_age >= launch_warn_seconds else "ok",
    ),
    *redis_stats,
  ]
  http_observation = http_observation if http_observation is not None else http_metrics.snapshot()
  application_http = http_observation["groups"]["application"]
  request_count = application_http["requests"]
  p95_seconds = application_http["p95_latency_seconds"]
  p95_ms = (p95_seconds or 0.0) * 1000
  latency_warn_seconds = float(os.getenv("OPEN_RL_HTTP_LATENCY_WARN_SECONDS", "2"))
  server_errors = application_http["in_window_server_errors"]
  recent_application_errors = [error for error in http_observation["recent_server_errors"] if error["group"] == "application"]
  stats.extend(
    [
      stat_entry(
        "gateway.http_requests",
        "Gateway requests",
        str(request_count),
        (
          f"latest {http_observation['sample_capacity']} samples; window truncated"
          if http_observation["window_truncated"]
          else f"last {format_duration(http_observation['window_seconds'])} · {http_observation['in_flight']} in flight"
        ),
        value_number=request_count,
        unit="requests",
        context={
          "window_seconds": http_observation["window_seconds"],
          "requests_per_second": application_http["requests_per_second"],
          "in_flight": http_observation["in_flight"],
          "sample_capacity": http_observation["sample_capacity"],
          "sample_count": http_observation["sample_count"],
          "dropped_samples": http_observation["dropped_samples"],
          "window_truncated": http_observation["window_truncated"],
          "group_requests": {name: group["requests"] for name, group in http_observation["groups"].items()},
        },
        status="warn" if http_observation["window_truncated"] else "ok",
      ),
      stat_entry(
        "gateway.http_latency",
        "Gateway p95 latency",
        f"{p95_ms:.1f} ms" if p95_seconds is not None else "—",
        f"p50 {((application_http['p50_latency_seconds'] or 0) * 1000):.1f} ms · max {((application_http['max_latency_seconds'] or 0) * 1000):.1f} ms"
        if request_count
        else "no application requests in the window",
        value_number=p95_ms,
        unit="milliseconds",
        context={
          "requests": request_count,
          "p50_latency_seconds": application_http["p50_latency_seconds"],
          "p95_latency_seconds": p95_seconds,
          "max_latency_seconds": application_http["max_latency_seconds"],
          "warn_after_seconds": latency_warn_seconds,
          "routes": [route for route in http_observation["routes"] if route["group"] == "application"][:20],
        },
        status="warn" if request_count >= 5 and p95_seconds is not None and p95_seconds >= latency_warn_seconds else "ok",
      ),
      stat_entry(
        "gateway.http_errors",
        "Gateway server errors",
        str(server_errors),
        f"{application_http['server_error_rate']:.1%} of application requests",
        value_number=server_errors,
        unit="errors",
        context={
          "requests": request_count,
          "server_error_rate": application_http["server_error_rate"],
          "client_errors": application_http["in_window_client_errors"],
          "recent_errors": recent_application_errors,
        },
        status="warn" if server_errors else "ok",
      ),
    ]
  )
  if observation := k8s.get("observation"):
    collection_ms = observation["collection_ms"]
    components_ms = observation.get("components_ms") or {}
    slowest_component = max(components_ms, key=components_ms.get) if components_ms else None
    warn_after_ms = float(os.getenv("OPEN_RL_K8S_WARN_MS", "1000"))
    detail = f"{observation['source']} observation"
    if slowest_component is not None:
      detail += f" · slowest {slowest_component} {components_ms[slowest_component]:.1f} ms"
    stats.append(
      stat_entry(
        "kubernetes.collection",
        "Kubernetes collection",
        f"{collection_ms:.1f} ms",
        detail,
        value_number=collection_ms,
        unit="milliseconds",
        context={
          "source": observation["source"],
          "age_seconds": observation["age_seconds"],
          "observed_at": observation["observed_at"],
          "components_ms": components_ms,
          "slowest_component": slowest_component,
          "warn_after_ms": warn_after_ms,
        },
        status="warn" if collection_ms >= warn_after_ms else "ok",
      )
    )
  rss = gateway_rss_bytes()
  if rss is not None:
    stats.append(stat_entry("gateway.rss", "Gateway memory", format_bytes(rss), "resident set size", value_number=rss, unit="bytes"))
  shared = tmp_dir()
  if os.path.isdir(shared):
    usage = shutil.disk_usage(shared)
    free_ratio = usage.free / usage.total if usage.total else 0
    stats.append(
      stat_entry(
        "storage.disk",
        "Disk free",
        format_bytes(usage.free),
        f"of {format_bytes(usage.total)} at {shared}",
        value_number=usage.free,
        unit="bytes",
        context={"total_bytes": usage.total, "free_ratio": free_ratio, "path": shared},
        status="warn" if usage.free < 20 * 2**30 else "ok",
      )
    )
  if k8s["available"]:
    phases: dict[str, int] = {}
    for pod in k8s["pods"]:
      phases[pod["phase"]] = phases.get(pod["phase"], 0) + 1
    running = phases.pop("Running", 0)
    others = " · ".join(f"{count} {phase.lower()}" for phase, count in sorted(phases.items())) or "no other phases"
    stats.append(
      stat_entry(
        "pods.running",
        "Pods running",
        str(running),
        others,
        value_number=running,
        unit="pods",
        context={"phase_counts": {"Running": running, **phases}},
        status="warn" if phases.get("Failed", 0) else "ok",
      )
    )
    total_gpus = sum(node["gpu_capacity"] for node in k8s["nodes"])
    if total_gpus:
      claimed = sum(pod.get("gpus", 0) for pod in k8s["pods"] if pod["node"] and pod["phase"] not in TERMINAL_POD_PHASES)
      ratio = claimed / total_gpus
      detail = "across all pools" if ratio <= 1 else f"{ratio:.1f}× allocation overcommit across all pools"
      stats.append(
        stat_entry(
          "gpus.claimed",
          "GPUs claimed",
          f"{claimed}/{total_gpus}",
          detail,
          value_number=claimed,
          unit="devices",
          context={"capacity_devices": total_gpus, "allocation_ratio": ratio, "overcommitted": ratio > 1},
          status="warn" if ratio > 1 else "ok",
        )
      )
    measured_nodes = [node for node in k8s["nodes"] if node.get("usage")]
    if measured_nodes:
      cpu_used = sum(node["usage"]["cpu_cores"] for node in measured_nodes)
      cpu_available = sum(node.get("cpu_allocatable_cores") or 0 for node in measured_nodes)
      cpu_ratio = cpu_used / cpu_available if cpu_available else None
      stats.append(
        stat_entry(
          "cluster.cpu",
          "Cluster CPU",
          f"{cpu_used:.2f} cores",
          f"{cpu_ratio:.0%} of {cpu_available:.2f} allocatable" if cpu_ratio is not None else "Metrics Server usage",
          value_number=cpu_used,
          unit="cores",
          context={"allocatable_cores": cpu_available or None, "utilization": cpu_ratio, "measured_nodes": len(measured_nodes)},
          status="warn" if cpu_ratio is not None and cpu_ratio >= 0.9 else "ok",
        )
      )
      memory_used = sum(node["usage"]["memory_bytes"] for node in measured_nodes)
      memory_available = sum(node.get("memory_allocatable_bytes") or 0 for node in measured_nodes)
      memory_ratio = memory_used / memory_available if memory_available else None
      stats.append(
        stat_entry(
          "cluster.memory",
          "Cluster memory",
          format_bytes(memory_used),
          f"{memory_ratio:.0%} of {format_bytes(memory_available)} allocatable" if memory_ratio is not None else "Metrics Server usage",
          value_number=memory_used,
          unit="bytes",
          context={"allocatable_bytes": memory_available or None, "utilization": memory_ratio, "measured_nodes": len(measured_nodes)},
          status="warn" if memory_ratio is not None and memory_ratio >= 0.9 else "ok",
        )
      )
    visible_nodes = {node.get("name") for node in k8s["nodes"] if node.get("name")}
    scheduled_pods = [pod for pod in k8s["pods"] if pod.get("node") in visible_nodes and pod.get("phase") not in TERMINAL_POD_PHASES]
    unscheduled_pods = [pod for pod in k8s["pods"] if not pod.get("node") and pod.get("phase") not in TERMINAL_POD_PHASES]
    reservations = aggregate_pod_resources(scheduled_pods)
    waiting_reservations = aggregate_pod_resources(unscheduled_pods)
    cpu_allocatable = sum(node.get("cpu_allocatable_cores") or 0 for node in k8s["nodes"])
    if cpu_allocatable:
      cpu_requested = reservations["requests"]["cpu_cores"]
      cpu_waiting = waiting_reservations["requests"]["cpu_cores"]
      cpu_request_ratio = cpu_requested / cpu_allocatable
      stats.append(
        stat_entry(
          "cluster.cpu_requests",
          "CPU requested",
          f"{cpu_requested:.2f}/{cpu_allocatable:.2f} cores",
          f"{cpu_request_ratio:.0%} reserved" + (f" · {cpu_waiting:.2f} cores unscheduled" if cpu_waiting else ""),
          value_number=cpu_requested,
          unit="cores",
          context={
            "allocatable_cores": cpu_allocatable,
            "request_ratio": cpu_request_ratio,
            "scheduled_pods": len(scheduled_pods),
            "unscheduled_request_cores": cpu_waiting,
          },
          status="warn" if cpu_request_ratio >= 0.9 else "ok",
        )
      )
    memory_allocatable = sum(node.get("memory_allocatable_bytes") or 0 for node in k8s["nodes"])
    if memory_allocatable:
      memory_requested = reservations["requests"]["memory_bytes"]
      memory_waiting = waiting_reservations["requests"]["memory_bytes"]
      memory_request_ratio = memory_requested / memory_allocatable
      stats.append(
        stat_entry(
          "cluster.memory_requests",
          "Memory requested",
          f"{format_bytes(memory_requested)}/{format_bytes(memory_allocatable)}",
          f"{memory_request_ratio:.0%} reserved" + (f" · {format_bytes(memory_waiting)} unscheduled" if memory_waiting else ""),
          value_number=memory_requested,
          unit="bytes",
          context={
            "allocatable_bytes": memory_allocatable,
            "request_ratio": memory_request_ratio,
            "scheduled_pods": len(scheduled_pods),
            "unscheduled_request_bytes": memory_waiting,
          },
          status="warn" if memory_request_ratio >= 0.9 else "ok",
        )
      )
    scheduler = k8s.get("scheduler") or {}
    if scheduler.get("available"):
      summary = scheduler["summary"]
      failed = summary["phase_counts"].get("Failed", 0)
      waiting = summary["phase_counts"].get("Pending", 0) + summary["phase_counts"].get("Placing", 0)
      stats.extend(
        [
          stat_entry(
            "scheduler.workloads",
            "Scheduler workloads",
            str(summary["workloads"]),
            f"{waiting} waiting · {failed} failed",
            value_number=summary["workloads"],
            unit="workloads",
            context={"phase_counts": summary["phase_counts"]},
            status="warn" if failed else "ok",
          ),
          stat_entry(
            "scheduler.seats",
            "Claim ledger seats",
            str(summary["seats"]),
            f"across {summary['ledgers']} ledgers · {summary['shared_ledgers']} shared",
            value_number=summary["seats"],
            unit="seats",
            context={"ledgers": summary["ledgers"], "shared_ledgers": summary["shared_ledgers"]},
          ),
        ]
      )
    rollouts = k8s.get("rollouts") or {}
    if rollouts.get("available"):
      summary = rollouts["summary"]
      states = " · ".join(f"{count} {state}" for state, count in sorted(summary["state_counts"].items())) or "none observed"
      stats.append(
        stat_entry(
          "kubernetes.rollouts",
          "Kubernetes rollouts",
          f"{summary['problem_count']}/{summary['total']} issues",
          states,
          value_number=summary["problem_count"],
          unit="controllers",
          context={"total": summary["total"], "state_counts": summary["state_counts"], "kind_counts": summary["kind_counts"]},
          status="warn" if summary["problem_count"] else "ok",
        )
      )
  return stats, queues


# *** Health ***


def check_entry(check_id: str, group: str, label: str, status: str, detail: str) -> dict:
  return {"id": check_id, "group": group, "label": label, "status": status, "detail": detail}


async def health_checks(store: RequestStore, k8s: dict) -> list[dict]:
  from server import gateway

  checks = []

  uptime = int(time.time() - START_TIME)
  mode = "single-process" if gateway.is_single_process_mode() else "distributed"
  fft = "FFT enabled" if gateway.is_fft_enabled() else "LoRA mode"
  revision = build_summary()["revision"]
  revision_label = revision[:12] if revision not in {"", "unknown"} else "unknown build"
  checks.append(
    check_entry("gateway", "Gateway", "Gateway process", "ok", f"{mode}, {fft}, build {revision_label}, up {uptime // 3600}h {uptime % 3600 // 60}m")
  )

  if isinstance(store, RedisStore):
    started = time.perf_counter()
    ok = await ping_redis(store)
    latency_ms = (time.perf_counter() - started) * 1000
    if ok:
      checks.append(check_entry("storage.redis", "Storage", "Redis", "ok", f"PING {latency_ms:.1f} ms — {safe_endpoint(os.getenv('REDIS_URL'))}"))
    else:
      checks.append(check_entry("storage.redis", "Storage", "Redis", "error", f"PING failed — {safe_endpoint(os.getenv('REDIS_URL'))}"))
  else:
    checks.append(check_entry("storage.redis", "Storage", "Redis", "off", "REDIS_URL not set — using in-memory store"))

  shared = tmp_dir()
  if os.path.isdir(shared) and os.access(shared, os.W_OK):
    free_gib = shutil.disk_usage(shared).free / 2**30
    status = "warn" if free_gib < 20 else "ok"
    checks.append(check_entry("storage.shared", "Storage", "Shared filesystem", status, f"{shared} writable, {free_gib:.0f} GiB free"))
  else:
    checks.append(check_entry("storage.shared", "Storage", "Shared filesystem", "warn", f"{shared} missing or not writable"))

  if k8s["available"]:
    detail = f"{len(k8s['pods'])} pods visible in namespace {k8s['namespace']}"
    if observation := k8s.get("observation"):
      detail += f" · {observation['source']} observation collected in {observation['collection_ms']:.1f} ms"
    checks.append(check_entry("kubernetes", "Kubernetes", "API server", "ok", detail))
    if nodes_error := k8s.get("nodes_error"):
      checks.append(check_entry("visibility.nodes", "Visibility", "Cluster nodes", "warn", nodes_error))
    else:
      checks.append(check_entry("visibility.nodes", "Visibility", "Cluster nodes", "ok", f"{len(k8s['nodes'])} nodes visible"))
    if event_error := k8s.get("events_error"):
      checks.append(check_entry("visibility.events", "Visibility", "Pod events", "warn", event_error))
    else:
      event_count = sum(len(pod.get("events") or []) for pod in k8s["pods"])
      checks.append(check_entry("visibility.events", "Visibility", "Pod events", "ok", f"{event_count} recent events visible"))
    metrics = k8s.get("metrics") or empty_resource_metrics(installed=None)
    if metrics["installed"] is False:
      checks.append(check_entry("visibility.metrics", "Visibility", "Resource metrics", "off", "metrics.k8s.io is not installed"))
    elif metrics["available"]:
      status = "warn" if metrics["error"] else "ok"
      detail = f"usage visible for {len(metrics['pods'])} pods and {len(metrics['nodes'])} nodes"
      if metrics["error"]:
        detail += f"; {metrics['error']}"
      checks.append(
        check_entry(
          "visibility.metrics",
          "Visibility",
          "Resource metrics",
          status,
          detail,
        )
      )
    elif metrics["installed"] is True:
      checks.append(check_entry("visibility.metrics", "Visibility", "Resource metrics", "warn", metrics["error"] or "metrics.k8s.io unavailable"))
  else:
    status = "off" if "not installed" in (k8s["error"] or "") or "credentials" in (k8s["error"] or "") else "error"
    checks.append(check_entry("kubernetes", "Kubernetes", "API server", status, k8s["error"] or "unavailable"))

  scheduler = k8s.get("scheduler") or empty_scheduler_snapshot(installed=None)
  if scheduler["installed"] is False:
    checks.append(check_entry("scheduler", "Scheduler", "Placement API", "off", "Workload CRD is not installed"))
  elif scheduler["available"]:
    summary = scheduler["summary"]
    checks.append(
      check_entry(
        "scheduler",
        "Scheduler",
        "Placement API",
        "ok",
        f"{summary['workloads']} workloads, {summary['ledgers']} claim ledgers, {summary['seats']} seats",
      )
    )
  elif scheduler["installed"] is True:
    checks.append(check_entry("scheduler", "Scheduler", "Placement API", "error", scheduler["error"] or "scheduler API unavailable"))

  rollouts = k8s.get("rollouts") or empty_rollout_snapshot()
  if rollouts["available"]:
    summary = rollouts["summary"]
    status = "warn" if rollouts["error"] else "ok"
    detail = f"{summary['total']} workload controllers · {summary['problem_count']} degraded or failed"
    if rollouts["error"]:
      detail += f" · partial visibility: {rollouts['error']}"
    checks.append(check_entry("visibility.rollouts", "Visibility", "Workload controllers", status, detail))
  elif not k8s["available"]:
    checks.append(check_entry("visibility.rollouts", "Visibility", "Workload controllers", "off", "Kubernetes is unavailable"))
  else:
    checks.append(check_entry("visibility.rollouts", "Visibility", "Workload controllers", "warn", rollouts["error"] or "unavailable"))

  if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
    checks.append(check_entry("visibility.trace", "Visibility", "Trace export", "ok", "GCP Cloud Trace exporter configured"))
  else:
    checks.append(check_entry("visibility.trace", "Visibility", "Trace export", "off", "ENABLE_GCP_TRACE=0 — tracing not configured"))

  if gateway.get_sampler_backend() == "vllm" and gateway.is_single_process_mode():
    healthz = f"{gateway.VLLM_URL.rstrip('/')}/healthz"
    try:
      async with httpx.AsyncClient(timeout=2.0) as client:
        (await client.get(healthz)).raise_for_status()
      checks.append(check_entry("visibility.sampler", "Visibility", "vLLM worker", "ok", f"reachable at {safe_endpoint(gateway.VLLM_URL)}"))
    except Exception:
      checks.append(check_entry("visibility.sampler", "Visibility", "vLLM worker", "error", f"unreachable at {safe_endpoint(gateway.VLLM_URL)}"))

  return checks


def derive_problems(checks: list[dict], k8s: dict, stats: list[dict] | None = None, runs: list[dict] | None = None) -> list[dict]:
  problems = []
  for check in checks:
    if check["status"] in {"warn", "error"}:
      actions = [{"label": "Refresh health checks", "method": "GET", "path": "/api/v1/dashboard/health", "command": "make ops health"}]
      if check["id"] == "kubernetes":
        actions.append({"label": "Verify pod visibility", "command": f"kubectl auth can-i list pods -n {k8s['namespace']}"})
      elif check["id"] == "visibility.events":
        actions.append({"label": "Verify event visibility", "command": f"kubectl auth can-i list events -n {k8s['namespace']}"})
      elif check["id"] == "visibility.nodes":
        actions.append({"label": "Verify node visibility", "command": "kubectl auth can-i list nodes"})
      elif check["id"] == "visibility.rollouts":
        actions.append({"label": "Verify rollout visibility", "command": f"kubectl auth can-i list deployments -n {k8s['namespace']}"})
      elif check["id"] == "visibility.metrics":
        actions.append({"label": "Verify Metrics API visibility", "command": f"kubectl auth can-i list pods.metrics.k8s.io -n {k8s['namespace']}"})
      problems.append(
        diagnostic_entry(
          f"check.{check['id']}",
          check["status"],
          check["label"],
          check["detail"],
          resource={"kind": "Check", "name": check["id"]},
          evidence={"group": check["group"], "status": check["status"]},
          actions=actions,
        )
      )
  alerting_stats = {
    "queue.request_age",
    "queue.launch_age",
    "redis.memory",
    "storage.disk",
    "cluster.cpu",
    "cluster.memory",
    "cluster.cpu_requests",
    "cluster.memory_requests",
    "kubernetes.collection",
    "gateway.http_latency",
    "gateway.http_errors",
    "gateway.http_requests",
  }
  for stat in stats or []:
    if stat["status"] != "warn" or stat["id"] not in alerting_stats:
      continue
    actions = [{"label": "Inspect load metrics", "method": "GET", "path": "/api/v1/dashboard/health", "command": "make ops health"}]
    if (model_id := stat["context"].get("model_id")) and model_id != "default":
      actions.append(
        {
          "label": "Inspect waiting run",
          "method": "GET",
          "path": f"/api/v1/dashboard/runs/{model_id}",
          "command": f"make ops run {model_id}",
        }
      )
    elif stat["id"] == "kubernetes.collection":
      actions.append(
        {
          "label": "Inspect Kubernetes observation",
          "method": "GET",
          "path": "/api/v1/dashboard/cluster",
          "command": "make ops inspect",
        }
      )
    elif stat["id"] in {"gateway.http_latency", "gateway.http_errors", "gateway.http_requests"}:
      actions.append(
        {
          "label": "Inspect gateway traffic",
          "method": "GET",
          "path": "/api/v1/dashboard/snapshot",
          "command": "make ops diagnose",
        }
      )
    problems.append(
      diagnostic_entry(
        f"metric.{stat['id'].replace('.', '_')}",
        "warn",
        stat["label"],
        f"{stat['value']} — {stat['detail']}",
        resource={"kind": "Metric", "name": stat["id"]},
        evidence={"value_number": stat["value_number"], "unit": stat["unit"], **stat["context"]},
        actions=actions,
      )
    )
  for run in runs or []:
    problems.extend(run_telemetry_diagnostics(run["run_id"], run.get("telemetry") or {}))
  for pod in k8s["pods"]:
    if diagnostic := pod_diagnostic(pod, k8s.get("namespace")):
      problems.append(diagnostic)
  for node in k8s["nodes"]:
    conditions = [
      (not node.get("ready"), "node.not_ready", "Node not ready"),
      (node.get("memory_pressure"), "node.memory_pressure", "Node under memory pressure"),
      (node.get("disk_pressure"), "node.disk_pressure", "Node under disk pressure"),
    ]
    for active, code, message in conditions:
      if not active:
        continue
      problems.append(
        diagnostic_entry(
          code,
          "warn",
          f"node/{node['name']}",
          message,
          resource={"kind": "Node", "name": node["name"]},
          evidence={
            "ready": node.get("ready"),
            "memory_pressure": bool(node.get("memory_pressure")),
            "disk_pressure": bool(node.get("disk_pressure")),
          },
          actions=[
            {
              "label": "Describe node",
              "method": "GET",
              "path": "/api/v1/dashboard/cluster",
              "command": f"kubectl describe node {node['name']}",
            }
          ],
        )
      )
  scheduler = k8s.get("scheduler") or {}
  if scheduler.get("available"):
    namespace = k8s["namespace"]
    workloads_by_name = {workload["name"]: workload for workload in scheduler["workloads"]}
    ledgers_by_claim = {ledger["claim_name"]: ledger for ledger in scheduler["ledgers"]}
    for workload in scheduler["workloads"]:
      source = f"workload/{workload['name']}"
      resource = {"kind": "Workload", "name": workload["name"], "namespace": namespace}
      actions = [
        {
          "label": "Inspect workload",
          "method": "GET",
          "path": "/api/v1/dashboard/snapshot",
          "command": f"kubectl get workload {workload['name']} -n {namespace} -o yaml",
        }
      ]
      evidence = {
        "phase": workload["phase"],
        "reason": workload["reason"],
        "age_seconds": workload["age_seconds"],
        "claim_name": workload["claim_name"],
        "assignment_id": workload["assignment_id"],
        "pod_name": workload["pod_name"],
        "node_name": workload["node_name"],
      }
      if workload["phase"] == "Failed":
        problems.append(
          diagnostic_entry(
            "scheduler.workload_failed",
            "error",
            source,
            workload["reason"] or "scheduler marked the workload failed",
            resource=resource,
            evidence=evidence,
            actions=actions,
          )
        )
      elif workload["phase"] in {"Pending", "Placing"} and (workload["age_seconds"] or 0) >= (300 if workload["phase"] == "Pending" else 120):
        problems.append(
          diagnostic_entry(
            f"scheduler.workload_{workload['phase'].lower()}_slow",
            "warn",
            source,
            workload["reason"] or f"workload has remained {workload['phase'].lower()} for {workload['age_seconds']} seconds",
            resource=resource,
            evidence=evidence,
            actions=actions,
          )
        )
      if workload["observed_generation"] is not None and not workload["generation_current"]:
        problems.append(
          diagnostic_entry(
            "scheduler.generation_stale",
            "warn",
            source,
            f"controller observed generation {workload['observed_generation']}, current generation is {workload['generation']}",
            resource=resource,
            evidence=evidence,
            actions=actions,
          )
        )
      if not workload["deleting"] and workload["claim_name"] and workload["assignment_id"] and (workload["age_seconds"] or 0) >= 60:
        ledger = ledgers_by_claim.get(workload["claim_name"])
        seat = next((seat for seat in (ledger or {}).get("seats", []) if seat["workload"] == workload["name"]), None)
        if seat is None or seat["assignment_id"] != workload["assignment_id"]:
          problems.append(
            diagnostic_entry(
              "scheduler.assignment_mismatch",
              "error",
              source,
              "workload status assignment does not match its ClaimLedger seat",
              resource=resource,
              evidence={**evidence, "ledger": ledger["name"] if ledger else None, "seat_assignment_id": seat["assignment_id"] if seat else None},
              actions=actions,
            )
          )
    for ledger in scheduler["ledgers"]:
      for seat in ledger["seats"]:
        if seat["workload"] in workloads_by_name or (ledger["age_seconds"] or 0) < 60:
          continue
        problems.append(
          diagnostic_entry(
            "scheduler.stale_seat",
            "warn",
            f"claimledger/{ledger['name']}",
            f"seat references missing workload {seat['workload']}",
            resource={"kind": "ClaimLedger", "name": ledger["name"], "namespace": namespace},
            evidence={"claim_name": ledger["claim_name"], "workload": seat["workload"], "assignment_id": seat["assignment_id"]},
            actions=[
              {
                "label": "Inspect claim ledger",
                "method": "GET",
                "path": "/api/v1/dashboard/snapshot",
                "command": f"kubectl get claimledger {ledger['name']} -n {namespace} -o yaml",
              }
            ],
          )
        )

  rollouts = k8s.get("rollouts") or {}
  for rollout in rollouts.get("items") or []:
    if rollout["state"] not in {"degraded", "failed"}:
      continue
    kind = rollout["kind"]
    name = rollout["name"]
    problems.append(
      diagnostic_entry(
        f"kubernetes.{kind.lower()}_{rollout['state']}",
        "error" if rollout["state"] == "failed" else "warn",
        f"{kind.lower()}/{name}",
        rollout["message"] or rollout["reason"] or f"{rollout['ready']} of {rollout['desired']} replicas ready",
        resource={"kind": kind, "name": name, "namespace": k8s["namespace"]},
        evidence={key: rollout[key] for key in rollout if key not in {"kind", "name", "message"}},
        actions=[
          {
            "label": f"Describe {kind}",
            "method": "GET",
            "path": "/api/v1/dashboard/cluster",
            "command": f"kubectl describe {kind.lower()} {name} -n {k8s['namespace']}",
          }
        ],
      )
    )
  priority = {"error": 0, "warn": 1, "info": 2}
  problems.sort(key=lambda problem: (priority.get(problem["severity"], 3), problem["id"]))
  return problems
