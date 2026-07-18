"""Kubernetes manager for dedicated per-model trainer workers.

Cluster-mode counterpart of FFTWorkerManager: instead of a local subprocess, each
FFT model gets its own trainer worker pod, labeled with a stable per-model id.
The pod spec comes from a ConfigMap-mounted YAML template; this class only stamps the
per-model name, labels, job-id env, and model_id argument. The labels follow
the time-slicing convention used by the node-local snapshot agent. DRA pinning
is handled by the shared ResourceClaim in the pod template; the accel-timeslicer
coordinates which colocated worker process may access CUDA.

This module is part of the cluster extra; importing it assumes Kubernetes
dependencies are installed.
"""

import copy
import hashlib
import json
import os
import re
import socket
import time
import uuid
from datetime import datetime
from typing import Any

import yaml
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

POD_NAME_PREFIX = "open-rl-trainer-"
TERMINAL_POD_PHASES = {"Succeeded", "Failed"}
FAILED_CONTAINER_REASONS = {
  "CrashLoopBackOff",
  "CreateContainerConfigError",
  "ErrImagePull",
  "ImagePullBackOff",
  "InvalidImageName",
  "OOMKilled",
}
BENIGN_PENDING_REASONS = {None, "ContainerCreating", "ContainersNotReady", "PodInitializing"}
WORKER_INSTANCE_ANNOTATION = "open-rl.dev/worker-instance"
WORKER_REVISION_ANNOTATION = "open-rl.dev/worker-revision"
WORKER_MODEL_ANNOTATION = "open-rl.dev/model-id"
WORKER_SOURCE_ANNOTATION = "open-rl.dev/source-revision"
# Label values allow at most 63 chars of [a-z0-9A-Z-_.]; we also reuse the
# sanitized id in the pod name, which is stricter (lowercase DNS).
_LABEL_SAFE = re.compile(r"[^a-z0-9-]+")


def sanitize_job_id(model_id: str) -> str:
  cleaned = _LABEL_SAFE.sub("-", model_id.lower()).strip("-")
  if not cleaned:
    raise ValueError(f"model_id {model_id!r} has no label-safe characters")
  return cleaned[:63]


class KubernetesFFTWorkerManager:
  """Runs one trainer worker pod per FFT model."""

  def __init__(self, core_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    trainer_path = os.getenv("OPEN_RL_TRAINER_POD_TEMPLATE") or os.getenv("OPEN_RL_WORKER_POD_TEMPLATE")
    if not trainer_path:
      raise RuntimeError("OPEN_RL_WORKER_MANAGER=kubernetes requires OPEN_RL_TRAINER_POD_TEMPLATE or OPEN_RL_WORKER_POD_TEMPLATE")
    with open(trainer_path, encoding="utf-8") as f:
      self.trainer_template: dict[str, Any] = yaml.safe_load(f)

    sampler_path = os.getenv("OPEN_RL_SAMPLER_POD_TEMPLATE") or trainer_path
    with open(sampler_path, encoding="utf-8") as f:
      self.sampler_template: dict[str, Any] = yaml.safe_load(f)

    self.pod_template = self.trainer_template

    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "default")
    self.worker_image = os.getenv("OPEN_RL_WORKER_IMAGE")

    if core_api is None:
      config.load_incluster_config()
      core_api = client.CoreV1Api()
    self.core_api = core_api

  def launch(self, model_id: str, base_model: str | None = None) -> str:
    return self.launch_trainer(model_id, base_model)

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> str:
    return self.launch_pod(model_id, role="trainer", base_model=base_model)

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> str:
    return self.launch_pod(model_id, role="sampler", base_model=base_model)

  def launch_pod(self, model_id: str, role: str, base_model: str | None = None) -> str:
    job_id = sanitize_job_id(model_id)
    prefix = "open-rl-trainer-" if role == "trainer" else "open-rl-sampler-"
    pod_name = prefix + job_id
    revision = self.worker_revision(role)

    existing = self.read_pod(pod_name)
    if existing is not None:
      if instance_id := reusable_worker_instance(existing, revision):
        return instance_id
      self.delete_pod_and_wait(pod_name)

    instance_id = uuid.uuid4().hex
    pod_body = self.render_pod(
      pod_name,
      model_id,
      job_id,
      role=role,
      base_model=base_model,
      instance_id=instance_id,
      revision=revision,
    )
    try:
      self.core_api.create_namespaced_pod(namespace=self.namespace, body=pod_body)
    except Exception as exc:
      if getattr(exc, "status", None) != 409:
        raise
      existing = self.read_pod(pod_name)
      if existing is None:
        raise RuntimeError(f"pod {pod_name} reported a create conflict but could not be read") from exc
      instance_id = reusable_worker_instance(existing, revision)
      if instance_id is None:
        raise RuntimeError(f"pod {pod_name} already exists with an unexpected worker revision") from exc

    return instance_id

  def shutdown(self, model_id: str) -> None:
    job_id = sanitize_job_id(model_id)
    for prefix in ("open-rl-trainer-", "open-rl-sampler-"):
      pod_name = prefix + job_id
      try:
        self.core_api.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
      except Exception as exc:
        if getattr(exc, "status", None) != 404:
          raise

  def shutdown_all(self) -> None:
    # Workers outlive gateway rollouts and are replaced lazily when their
    # revision changes. Explicit model deletion owns worker teardown.
    return None

  def render_pod(
    self,
    pod_name: str,
    model_id: str,
    job_id: str,
    role: str = "trainer",
    base_model: str | None = None,
    instance_id: str | None = None,
    revision: str | None = None,
  ) -> dict[str, Any]:
    from server.worker_manager import _fetch_metadata_from_store

    stored_base_model, weight_sync_strategy = _fetch_metadata_from_store(model_id)
    base_model = base_model or stored_base_model
    base_tmpl = self.trainer_template if role == "trainer" else self.sampler_template
    pod = copy.deepcopy(base_tmpl)
    metadata = pod.setdefault("metadata", {})
    metadata["name"] = pod_name
    revision = revision or self.worker_revision(role)
    instance_id = instance_id or uuid.uuid4().hex
    metadata.setdefault("annotations", {}).update(
      {
        WORKER_INSTANCE_ANNOTATION: instance_id,
        WORKER_REVISION_ANNOTATION: revision,
        WORKER_MODEL_ANNOTATION: model_id,
      }
    )
    source_revision = os.getenv("OPEN_RL_SOURCE_REVISION")
    if source_revision:
      metadata["annotations"][WORKER_SOURCE_ANNOTATION] = source_revision
    app_label = "open-rl-trainer-worker" if role == "trainer" else "open-rl-sampler-worker"
    role_group = TRAINER_TIME_SLICE_GROUP if role == "trainer" else SAMPLER_TIME_SLICE_GROUP
    role_job_id = workload_job_id(role, job_id)
    metadata.setdefault("labels", {}).update(
      {
        "app": app_label,
        "accel-timeslicer": "true",
        "timeslice.io/group": role_group,
        "timeslice.io/job-id": role_job_id,
      }
    )

    container = pod["spec"]["containers"][0]
    if self.worker_image:
      container["image"] = self.worker_image
    if role == "sampler":
      container["command"] = ["uv", "run", "--no-sync", "python", "-u", "-m", "server.vllm_sampler"]
    container.setdefault("args", []).append(f"model_id={model_id}")
    if base_model:
      set_env(container, "BASE_MODEL", base_model)
      set_env(container, "OPEN_RL_BASE_MODEL", base_model)
      if "gemma-4" in base_model.lower() or "gemma4" in base_model.lower():
        set_env(container, "VLLM_ARCHITECTURE_OVERRIDE", "Gemma4ForCausalLM")
    arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
    if arch_override:
      set_env(container, "VLLM_ARCHITECTURE_OVERRIDE", arch_override)
    # Keep env aligned with labels so process discovery and llm-d target the
    # same workload identity.
    set_env(container, "OPEN_RL_TIME_SLICE_JOB_ID", role_job_id)
    set_env(container, "OPEN_RL_TIME_SLICE_GROUP", role_group)
    weight_sync = weight_sync_strategy or os.getenv("OPEN_RL_WEIGHT_SYNC_STRATEGY", "delta")
    if weight_sync:
      set_env(container, "OPEN_RL_WEIGHT_SYNC_STRATEGY", weight_sync)
    set_env(container, "OPEN_RL_WORKER_INSTANCE_ID", instance_id)
    if source_revision:
      source_path = os.getenv("PYTHONPATH") or f"/mnt/shared/open-rl/source/{source_revision}"
      set_env(container, "PYTHONPATH", source_path)
      set_env(container, "OPEN_RL_SOURCE_REVISION", source_revision)
    return pod

  def worker_revision(self, role: str) -> str:
    template = self.trainer_template if role == "trainer" else self.sampler_template
    template_bytes = yaml.safe_dump(template, sort_keys=True).encode("utf-8")
    template_digest = hashlib.sha256(template_bytes).hexdigest()
    image_revision = os.getenv("OPEN_RL_WORKER_REVISION") or self.worker_image or "unversioned"
    source_revision = os.getenv("OPEN_RL_SOURCE_REVISION", "image")
    return hashlib.sha256(f"{image_revision}\0{source_revision}\0{template_digest}".encode()).hexdigest()[:16]

  def read_pod(self, pod_name: str) -> Any | None:
    try:
      return self.core_api.read_namespaced_pod(name=pod_name, namespace=self.namespace)
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return None
      raise

  def delete_pod_and_wait(self, pod_name: str, timeout: float = 60.0) -> None:
    self.core_api.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
    deadline = time.monotonic() + timeout
    while self.read_pod(pod_name) is not None:
      if time.monotonic() > deadline:
        raise RuntimeError(f"pod {pod_name} did not terminate within {timeout:.0f}s; cannot relaunch worker")
      time.sleep(0.5)

  def describe_workers(self, model_id: str) -> list[dict[str, Any]]:
    workers = []
    job_id = sanitize_job_id(model_id)
    for role in ("trainer", "sampler"):
      pod = self.read_pod(f"open-rl-{role}-{job_id}")
      if pod is not None:
        workers.append({**pod_summary(pod), "id": role, "role": role, "model_id": model_id})
    return workers

  def cluster_snapshot(self) -> dict[str, Any]:
    errors: list[str] = []
    nodes: list[dict[str, Any]] = []
    pods: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    try:
      nodes = [node_summary(node) for node in list_response_items(self.core_api.list_node())]
    except Exception as exc:
      errors.append(f"nodes: {exc}")
    try:
      max_pods = max(1, int(os.getenv("OPEN_RL_CONTROL_MAX_PODS", "500")))
      pods = [pod_summary(pod) for pod in list_response_items(self.core_api.list_namespaced_pod(namespace=self.namespace, limit=max_pods))[:max_pods]]
    except Exception as exc:
      errors.append(f"pods: {exc}")
    try:
      max_events = max(1, int(os.getenv("OPEN_RL_CONTROL_MAX_EVENTS", "100")))
      event_items = list_response_items(self.core_api.list_namespaced_event(namespace=self.namespace, limit=max_events))
      events = [event_summary(event) for event in event_items[:max_events]]
      events.sort(key=lambda event: event.get("last_seen_at") or 0, reverse=True)
    except Exception as exc:
      errors.append(f"events: {exc}")

    for node in nodes:
      node["workloads"] = [pod for pod in pods if pod.get("node") == node["name"]]
      node["pod_count"] = len(node["workloads"])
      has_time_slicer = any(label in node.get("labels", {}) for label in ("group.timeslice.io/trainers", "group.timeslice.io/samplers"))
      if has_time_slicer and (host := node.get("internal_ip")):
        try:
          node["time_slicer"] = read_time_slicer_status(host)
        except Exception as exc:
          node["time_slicer"] = {"ok": False, "error": str(exc), "workloads": []}
          errors.append(f"time-slicer {node['name']}: {exc}")

    ready_nodes = sum(bool(node.get("ready")) for node in nodes)
    failed_pods = sum(pod.get("status") == "failed" for pod in pods)
    pending_pods = sum(pod.get("status") == "pending" for pod in pods)
    actionable_pending_pods = sum(
      pod.get("status") == "pending" and bool(pod.get("reason")) and pod.get("reason") not in BENIGN_PENDING_REASONS for pod in pods
    )
    running_pods = sum(pod.get("status") in {"running", "ready"} for pod in pods)
    status = "unavailable" if not nodes else "degraded" if errors or ready_nodes < len(nodes) or failed_pods or actionable_pending_pods else "healthy"
    return {
      "mode": "kubernetes",
      "status": status,
      "namespace": self.namespace,
      "summary": {
        "nodes": len(nodes),
        "ready_nodes": ready_nodes,
        "pods": len(pods),
        "running_pods": running_pods,
        "pending_pods": pending_pods,
        "actionable_pending_pods": actionable_pending_pods,
        "failed_pods": failed_pods,
      },
      "nodes": nodes,
      "pods": pods,
      "events": events,
      "errors": errors,
      "generated_at": time.time(),
    }

  def read_logs(self, model_id: str, component: str, tail_lines: int = 200, previous: bool = False) -> dict[str, Any]:
    if component in {"trainer", "sampler"}:
      pod_name = f"open-rl-{component}-{sanitize_job_id(model_id)}"
    elif component in {"gateway", "timeslicer"}:
      app_label = "open-rl-gateway" if component == "gateway" else "open-rl-accel-timeslicer"
      try:
        pods = list_response_items(self.core_api.list_namespaced_pod(namespace=self.namespace, label_selector=f"app={app_label}", limit=1))
      except Exception as exc:
        return {"source": "kubernetes", "pod_name": None, "logs": "", "error": str(exc)}
      if not pods:
        return {"source": "kubernetes", "pod_name": None, "logs": "", "error": f"No {component} pod was found in namespace {self.namespace!r}."}
      pod_name = str(get_field(get_field(pods[0], "metadata", {}) or {}, "name"))
    else:
      return {"source": "kubernetes", "pod_name": None, "logs": "", "error": f"Unknown log component {component!r}"}
    return self.read_pod_logs(pod_name, tail_lines, previous)

  def read_pod_logs(self, pod_name: str, tail_lines: int = 200, previous: bool = False) -> dict[str, Any]:
    try:
      logs = self.core_api.read_namespaced_pod_log(
        name=pod_name,
        namespace=self.namespace,
        tail_lines=max(1, min(int(tail_lines), 5000)),
        timestamps=True,
        previous=bool(previous),
      )
      return {"source": "kubernetes", "pod_name": pod_name, "logs": logs or "", "error": None}
    except Exception as exc:
      if isinstance(exc, ApiException) and exc.status == 404:
        return {"source": "kubernetes", "pod_name": pod_name, "logs": "", "error": f"Pod or logs for {pod_name} were not found."}
      return {"source": "kubernetes", "pod_name": pod_name, "logs": "", "error": str(exc)}


def reusable_worker_instance(pod: Any, revision: str) -> str | None:
  if pod.status.phase in TERMINAL_POD_PHASES:
    return None
  annotations = pod.metadata.annotations or {}
  if annotations.get(WORKER_REVISION_ANNOTATION) != revision:
    return None
  return annotations.get(WORKER_INSTANCE_ANNOTATION)


def set_env(container: dict[str, Any], name: str, value: str) -> None:
  env = container.setdefault("env", [])
  for item in env:
    if item.get("name") == name:
      item.clear()
      item.update({"name": name, "value": value})
      return
  env.append({"name": name, "value": value})


def get_field(value: Any, name: str, default: Any = None) -> Any:
  if isinstance(value, dict):
    return value.get(name, default)
  return getattr(value, name, default)


def list_response_items(value: Any) -> list[Any]:
  return list(get_field(value, "items", []) or [])


def datetime_to_timestamp(value: Any) -> float | None:
  if isinstance(value, datetime):
    return value.timestamp()
  return None


def pod_summary(pod: Any) -> dict[str, Any]:
  metadata = get_field(pod, "metadata", {}) or {}
  spec = get_field(pod, "spec", {}) or {}
  status_obj = get_field(pod, "status", {}) or {}
  labels = dict(get_field(metadata, "labels", {}) or {})
  annotations = dict(get_field(metadata, "annotations", {}) or {})
  raw_phase = str(get_field(status_obj, "phase", "Unknown") or "Unknown")
  container_statuses = list(get_field(status_obj, "container_statuses", []) or [])
  init_container_statuses = list(get_field(status_obj, "init_container_statuses", []) or [])
  diagnostic_statuses = [*init_container_statuses, *container_statuses]
  ready = bool(container_statuses) and all(bool(get_field(item, "ready", False)) for item in container_statuses)
  restarts = sum(int(get_field(item, "restart_count", 0) or 0) for item in diagnostic_statuses)
  reason = get_field(status_obj, "reason")
  message = get_field(status_obj, "message")
  container_failed = False
  for condition in list(get_field(status_obj, "conditions", []) or []):
    if str(get_field(condition, "status", "Unknown")).lower() != "false":
      continue
    reason = get_field(condition, "reason") or reason
    message = get_field(condition, "message") or message
    if get_field(condition, "type") == "PodScheduled":
      break
  for container_status in diagnostic_statuses:
    state = get_field(container_status, "state", {}) or {}
    waiting = get_field(state, "waiting")
    terminated = get_field(state, "terminated")
    if waiting:
      reason = get_field(waiting, "reason") or reason
      message = get_field(waiting, "message") or message
      container_failed = container_failed or reason in FAILED_CONTAINER_REASONS
      if container_failed:
        break
      continue
    if terminated:
      terminated_reason = get_field(terminated, "reason")
      exit_code = get_field(terminated, "exit_code", 0)
      if terminated_reason in FAILED_CONTAINER_REASONS or exit_code not in {None, 0}:
        reason = terminated_reason or reason
        message = get_field(terminated, "message") or message
        container_failed = True

  containers = list(get_field(spec, "containers", []) or [])
  resource_claims = []
  for claim in list(get_field(spec, "resource_claims", []) or []):
    resource_claims.append(
      {
        "name": get_field(claim, "name"),
        "resource_claim_name": get_field(claim, "resource_claim_name"),
        "resource_claim_template_name": get_field(claim, "resource_claim_template_name"),
      }
    )
  image = get_field(containers[0], "image") if containers else None
  role = (
    "trainer" if labels.get("app") == "open-rl-trainer-worker" else "sampler" if labels.get("app") == "open-rl-sampler-worker" else labels.get("app")
  )
  normalized_status = {
    "Pending": "pending",
    "Running": "ready" if ready else "running",
    "Succeeded": "completed",
    "Failed": "failed",
  }.get(raw_phase, "unknown")
  if container_failed or reason in FAILED_CONTAINER_REASONS:
    normalized_status = "failed"
    ready = False
  return {
    "name": get_field(metadata, "name"),
    "pod_name": get_field(metadata, "name"),
    "namespace": get_field(metadata, "namespace"),
    "node": get_field(spec, "node_name"),
    "status": normalized_status,
    "phase": raw_phase.lower(),
    "message": message or reason or raw_phase,
    "reason": reason,
    "ready": ready,
    "restarts": restarts,
    "image": image,
    "role": role,
    "model_id": annotations.get(WORKER_MODEL_ANNOTATION),
    "source_revision": annotations.get(WORKER_SOURCE_ANNOTATION),
    "labels": labels,
    "resource_claims": resource_claims,
    "started_at": datetime_to_timestamp(get_field(status_obj, "start_time")),
    "updated_at": time.time(),
  }


def node_summary(node: Any) -> dict[str, Any]:
  metadata = get_field(node, "metadata", {}) or {}
  spec = get_field(node, "spec", {}) or {}
  status_obj = get_field(node, "status", {}) or {}
  labels = dict(get_field(metadata, "labels", {}) or {})
  conditions = []
  ready = False
  for condition in list(get_field(status_obj, "conditions", []) or []):
    item = {
      "type": get_field(condition, "type"),
      "status": str(get_field(condition, "status", "Unknown")),
      "reason": get_field(condition, "reason"),
      "message": get_field(condition, "message"),
    }
    conditions.append(item)
    if item["type"] == "Ready":
      ready = item["status"] == "True"
  roles = sorted(key.removeprefix("node-role.kubernetes.io/") or "worker" for key in labels if key.startswith("node-role.kubernetes.io/"))
  taints = [
    {"key": get_field(taint, "key"), "value": get_field(taint, "value"), "effect": get_field(taint, "effect")}
    for taint in list(get_field(spec, "taints", []) or [])
  ]
  addresses = {get_field(address, "type"): get_field(address, "address") for address in list(get_field(status_obj, "addresses", []) or [])}
  return {
    "name": get_field(metadata, "name"),
    "status": "ready" if ready else "not_ready",
    "ready": ready,
    "roles": roles or ["worker"],
    "capacity": dict(get_field(status_obj, "capacity", {}) or {}),
    "allocatable": dict(get_field(status_obj, "allocatable", {}) or {}),
    "conditions": conditions,
    "taints": taints,
    "labels": labels,
    "internal_ip": addresses.get("InternalIP"),
  }


def event_summary(event: Any) -> dict[str, Any]:
  metadata = get_field(event, "metadata", {}) or {}
  involved = get_field(event, "involved_object", {}) or {}
  raw_type = str(get_field(event, "type", "Normal") or "Normal")
  return {
    "name": get_field(metadata, "name"),
    "type": raw_type.lower(),
    "reason": get_field(event, "reason"),
    "message": get_field(event, "message"),
    "count": int(get_field(event, "count", 1) or 1),
    "object_kind": get_field(involved, "kind"),
    "object_name": get_field(involved, "name"),
    "first_seen_at": datetime_to_timestamp(get_field(event, "first_timestamp")),
    "last_seen_at": datetime_to_timestamp(get_field(event, "last_timestamp")) or datetime_to_timestamp(get_field(metadata, "creation_timestamp")),
  }


def read_time_slicer_status(host: str) -> dict[str, Any]:
  port = int(os.getenv("OPEN_RL_TIMESLICER_STATUS_PORT", "9753"))
  timeout = float(os.getenv("OPEN_RL_TIMESLICER_STATUS_TIMEOUT_SECONDS", "0.3"))
  if timeout <= 0:
    raise ValueError("OPEN_RL_TIMESLICER_STATUS_TIMEOUT_SECONDS must be positive")
  with socket.create_connection((host, port), timeout=timeout) as connection:
    connection.settimeout(timeout)
    connection.sendall(b'{"command":"STATUS"}\n')
    response = connection.makefile("rb").readline(1024 * 1024)
  if not response:
    raise RuntimeError("status connection closed without a response")
  payload = json.loads(response)
  if not isinstance(payload, dict):
    raise RuntimeError("status response was not an object")
  return payload
