"""Kubernetes launcher for dedicated per-model FFT workers.

Cluster-mode counterpart of FFTWorkerManager: instead of a local subprocess, each
FFT model gets its own pod, labeled so the llm-d time-slicing stack picks it up
(timeslice.io/group + timeslice.io/job-id for the accelerator orchestrator,
snapshot-agent=true for the node-local snapshot agent DaemonSet). The pod spec
comes from a ConfigMap-mounted YAML template; this class only stamps the
per-model name, labels, job-id env, and --model-id argument.

Kept separate from worker_manager.py so the `kubernetes` dependency (the
`cluster` extra) is only imported when this launcher is selected.
"""

import copy
import os
import re
import time
from typing import Any

import yaml

POD_NAME_PREFIX = "open-rl-fft-"
TERMINAL_POD_PHASES = {"Succeeded", "Failed"}
# Label values allow at most 63 chars of [a-z0-9A-Z-_.]; we also reuse the
# sanitized id in the pod name, which is stricter (lowercase DNS).
_LABEL_SAFE = re.compile(r"[^a-z0-9-]+")


def sanitize_job_id(model_id: str) -> str:
  cleaned = _LABEL_SAFE.sub("-", model_id.lower()).strip("-")
  if not cleaned:
    raise ValueError(f"model_id {model_id!r} has no label-safe characters")
  return cleaned[:63]


class KubernetesFFTWorkerManager:
  """Runs one worker pod per FFT model."""

  def __init__(self, core_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    template_path = os.getenv("OPEN_RL_WORKER_POD_TEMPLATE")
    if not template_path:
      raise RuntimeError("OPEN_RL_WORKER_LAUNCHER=kubernetes requires OPEN_RL_WORKER_POD_TEMPLATE pointing at the worker pod YAML")
    with open(template_path, encoding="utf-8") as f:
      self.pod_template: dict[str, Any] = yaml.safe_load(f)

    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "default")
    self.group_id = os.getenv("OPEN_RL_TIME_SLICE_GROUP", "trainers")

    if core_api is None:
      from kubernetes import client, config

      config.load_incluster_config()
      core_api = client.CoreV1Api()
    self.core_api = core_api

  def launch(self, model_id: str) -> None:
    job_id = sanitize_job_id(model_id)
    pod_name = POD_NAME_PREFIX + job_id

    existing = self.read_pod(pod_name)
    if existing is not None:
      if existing.status.phase not in TERMINAL_POD_PHASES:
        return
      self.delete_pod_and_wait(pod_name)

    try:
      self.core_api.create_namespaced_pod(namespace=self.namespace, body=self.render_pod(pod_name, model_id, job_id))
    except Exception as exc:
      # Another gateway replica created it between our read and create.
      if getattr(exc, "status", None) != 409:
        raise

  def shutdown(self, model_id: str) -> None:
    pod_name = POD_NAME_PREFIX + sanitize_job_id(model_id)
    try:
      self.core_api.delete_namespaced_pod(name=pod_name, namespace=self.namespace)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise

  def shutdown_all(self) -> None:
    # Worker pods deliberately outlive gateway restarts; Kubernetes owns them.
    pass

  def render_pod(self, pod_name: str, model_id: str, job_id: str) -> dict[str, Any]:
    pod = copy.deepcopy(self.pod_template)
    metadata = pod.setdefault("metadata", {})
    metadata["name"] = pod_name
    metadata.setdefault("labels", {}).update(
      {
        "app": "open-rl-fft-worker",
        "snapshot-agent": "true",
        "timeslice.io/group": self.group_id,
        "timeslice.io/job-id": job_id,
      }
    )

    container = pod["spec"]["containers"][0]
    container.setdefault("args", []).extend(["--model-id", model_id])
    # The orchestrator matches on the (possibly sanitized) label value, so the
    # worker must Acquire with exactly that job id, not the raw model_id.
    container.setdefault("env", []).append({"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": job_id})
    return pod

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
