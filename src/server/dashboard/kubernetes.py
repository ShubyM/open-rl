# Real data sources for the operational dashboard: gateway process state, Redis, the shared
# filesystem, and (when reachable) the Kubernetes API. Every accessor degrades to an explicit
# "unavailable" result instead of raising so the dashboard can always render something truthful.

import concurrent.futures
import functools
import os
import platform
import socket
import time
from datetime import UTC, datetime
from typing import Any

START_TIME = time.time()
K8S_REQUEST_TIMEOUT = 4
NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
SCHEDULER_GROUP = "openrl.io"
SCHEDULER_VERSION = "v1alpha1"


def iso_timestamp(ts: float | str | None) -> str | None:
  if ts is None:
    return None
  if isinstance(ts, str):
    return ts
  return datetime.fromtimestamp(ts, tz=UTC).isoformat()


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
        "model_id": spec.get("modelID"),
        "owner_id": spec.get("ownerID"),
        "training_kind": spec.get("trainingKind"),
        "exclusive": spec.get("exclusive", False),
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
        "owners": sorted({owner for seat in seats if (owner := seat.get("ownerID") or seat.get("workload"))}),
        "seats": [
          {
            "workload": seat.get("workload"),
            "workload_uid": seat.get("workloadUID"),
            "assignment_id": seat.get("assignmentID"),
            "owner": seat.get("ownerID"),
            "exclusive": seat.get("exclusive", False),
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
    "uid": str(pod.metadata.uid) if getattr(pod.metadata, "uid", None) else None,
    "phase": pod.status.phase or "Unknown",
    "node": pod.spec.node_name,
    "app": (pod.metadata.labels or {}).get("app"),
    "labels": pod.metadata.labels or {},
    "owner_references": [{"uid": str(owner.uid), "name": owner.name, "kind": owner.kind} for owner in (pod.metadata.owner_references or [])],
    "resource_claims": sorted(
      {
        claim.resource_claim_name
        for claim in [*(getattr(pod.status, "resource_claim_statuses", None) or []), *(getattr(pod.spec, "resource_claims", None) or [])]
        if claim.resource_claim_name
      }
    ),
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
    "pod_uid": event.involved_object.uid,
  }


def attach_pod_events(pods: list[dict], events: list[dict]) -> None:
  """Keep observations from replaced pods out of the current run evidence."""
  events_by_pod: dict[str, list[dict]] = {}
  for event in events:
    events_by_pod.setdefault(event["pod_uid"], []).append(event)
  for pod in pods:
    pod["events"] = sorted(events_by_pod.get(pod["uid"], []), key=lambda event: event["last_seen_at"] or "")[-10:]


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
  attach_pod_events(pods, events)
  for pod in pods:
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


def k8s_snapshot() -> dict:
  """Collect once; Observations owns the shared dashboard cache and refresh lock."""
  started = time.perf_counter()
  snapshot = _collect_k8s_snapshot()
  snapshot["observation"] = {
    "observed_at": iso_timestamp(time.time()),
    "collection_ms": round((time.perf_counter() - started) * 1000, 3),
    "components_ms": snapshot.pop("_component_ms", {}),
    "source": "live",
    "age_seconds": 0.0,
  }
  return snapshot


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
    timestamps=True,
    limit_bytes=128 * 1024,
    _request_timeout=K8S_REQUEST_TIMEOUT + 4,
  )
  return {"demo": False, "pod": pod, "container": container, "previous": previous, "text": text}
