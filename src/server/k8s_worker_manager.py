"""Kubernetes manager for dedicated per-model trainer workers.

Cluster-mode counterpart of FFTWorkerManager: instead of a local subprocess, each
FFT model gets its own trainer worker pod, labeled with a stable per-model id.
The pod spec comes from a ConfigMap-mounted YAML template; this class only stamps the
per-model name, labels, job-id env, and --model-id argument. The labels follow
the time-slicing convention used by the node-local snapshot agent. DRA pinning
is handled by the shared ResourceClaim in the pod template; the accel-timeslicer
coordinates which colocated worker process may access CUDA.

This module is part of the cluster extra; importing it assumes Kubernetes
dependencies are installed.
"""

import copy
import hashlib
import os
import re
import time
import uuid
from typing import Any

import yaml
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

TERMINAL_POD_PHASES = {"Succeeded", "Failed"}
WORKER_INSTANCE_ANNOTATION = "open-rl.dev/worker-instance"
WORKER_REVISION_ANNOTATION = "open-rl.dev/worker-revision"
# Label values allow at most 63 chars of [a-z0-9A-Z-_.]; we also reuse the
# sanitized id in the pod name, which is stricter (lowercase DNS).
_LABEL_SAFE = re.compile(r"[^a-z0-9-]+")
_MAX_JOB_ID_LENGTH = 47


def sanitize_job_id(model_id: str) -> str:
  cleaned = _LABEL_SAFE.sub("-", model_id.lower()).strip("-")
  if not cleaned:
    raise ValueError(f"model_id {model_id!r} has no label-safe characters")
  if len(cleaned) <= _MAX_JOB_ID_LENGTH:
    return cleaned
  digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:8]
  return f"{cleaned[: _MAX_JOB_ID_LENGTH - len(digest) - 1].rstrip('-')}-{digest}"


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
      revision=revision,
      instance_id=instance_id,
    )
    try:
      self.core_api.create_namespaced_pod(namespace=self.namespace, body=pod_body)
    except ApiException as exc:
      if exc.status != 409:
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
      except ApiException as exc:
        if exc.status != 404:
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
    revision: str | None = None,
    instance_id: str | None = None,
  ) -> dict[str, Any]:
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
      }
    )
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
    container.setdefault("args", []).extend(["--model-id", model_id])
    if base_model:
      set_env(container, "BASE_MODEL", base_model)
    # Keep env aligned with labels so process discovery and llm-d target the
    # same workload identity.
    set_env(container, "OPEN_RL_TIME_SLICE_JOB_ID", role_job_id)
    set_env(container, "OPEN_RL_TIME_SLICE_GROUP", role_group)
    set_env(container, "OPEN_RL_WORKER_INSTANCE_ID", instance_id)
    return pod

  def worker_revision(self, role: str) -> str:
    template = self.trainer_template if role == "trainer" else self.sampler_template
    template_bytes = yaml.safe_dump(template, sort_keys=True).encode("utf-8")
    template_digest = hashlib.sha256(template_bytes).hexdigest()
    image_revision = os.getenv("OPEN_RL_WORKER_REVISION") or self.worker_image or "unversioned"
    return hashlib.sha256(f"{image_revision}\0{template_digest}".encode()).hexdigest()[:16]

  def read_pod(self, pod_name: str) -> client.V1Pod | None:
    try:
      return self.core_api.read_namespaced_pod(name=pod_name, namespace=self.namespace)
    except ApiException as exc:
      if exc.status == 404:
        return None
      raise

  def delete_pod_and_wait(self, pod_name: str, timeout: float = 60.0) -> None:
    self.core_api.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
    deadline = time.monotonic() + timeout
    while self.read_pod(pod_name) is not None:
      if time.monotonic() > deadline:
        raise RuntimeError(f"pod {pod_name} did not terminate within {timeout:.0f}s; cannot relaunch worker")
      time.sleep(0.5)


def reusable_worker_instance(pod: client.V1Pod, revision: str) -> str | None:
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
