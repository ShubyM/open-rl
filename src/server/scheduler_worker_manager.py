"""Worker manager that creates one Workload per worker process and lets the
scheduler place it. Part of the cluster extra: importing it assumes the
Kubernetes client is installed.

Two FFT jobs and two LoRA jobs on the same base model come in:

  model_id (job)   owner               Workload (process)          shares
  fft-job 1a2b     1a2b                fft-1a2b-trainer            nothing
  fft-job 3c4d     3c4d                fft-3c4d-trainer            nothing
  lora-job 5e6f    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   one runtime
  lora-job 7a8b    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   with 5e6f

The Workload name is what the pod label and the time-slicer call job_id.
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from kubernetes import client, config

from server.estimator import Footprint, footprint
from server.worker_manager import base_model_of, runtime_of, worker_args, worker_env, worker_module

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"

# Owners double as label values and pod name stems.
LABEL_UNSAFE = re.compile(r"[^a-z0-9-]+")


def owner_of(runtime: str) -> str:
  cleaned = LABEL_UNSAFE.sub("-", runtime.lower()).strip("-")
  if not cleaned:
    raise ValueError(f"model_id {runtime!r} has no label-safe characters")
  return cleaned[:63]


def workload_name(role: str, owner: str, is_lora: bool) -> str:
  # A second compatible LoRA request renders the same name, and the create's
  # AlreadyExists is the reuse. The instance index stays 0 until adapter
  # capacity accounting exists.
  if is_lora:
    return f"lora-{owner}-0-{role}"
  return f"fft-{owner}-{role}"


@dataclass(frozen=True)
class Worker:
  """Everything the gateway knows about one worker process before placement."""

  role: str
  runtime: str
  base_model: str
  is_lora: bool
  meta: Any
  footprint: Footprint

  @property
  def owner(self) -> str:
    return owner_of(self.runtime)

  @property
  def name(self) -> str:
    return workload_name(self.role, self.owner, self.is_lora)


def describe_worker(model_id: str, role: str) -> Worker:
  meta, runtime, is_lora = runtime_of(model_id)
  base_model = base_model_of(meta, runtime)
  return Worker(role, runtime, base_model, is_lora, meta, footprint(base_model, meta.fine_tuning_type, role))


def pod_env(worker: Worker) -> list[dict[str, Any]]:
  """The shared worker env plus what only the cluster knows. The time-slice
  group is placement's and the scheduler stamps it."""
  tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl")
  values = {
    "REDIS_URL": os.environ["REDIS_URL"],
    "OPEN_RL_TMP_DIR": tmp_dir,
    "HF_HOME": os.getenv("HF_HOME", f"{tmp_dir}/huggingface"),
    **worker_env(worker.meta, worker.base_model, worker.runtime, worker.is_lora, worker.role),
    "OPEN_RL_WORKLOAD_ID": worker.name,
    # The llmd snapshot agent still discovers processes by the older name.
    "OPEN_RL_TIME_SLICE_JOB_ID": worker.name,
    "OPEN_RL_ACCEL_TIMESLICER_PORT": os.getenv("OPEN_RL_ACCEL_TIMESLICER_PORT", "9753"),
  }
  if os.getenv("VLLM_GPU_MEMORY_UTILIZATION"):
    values["VLLM_GPU_MEMORY_UTILIZATION"] = os.environ["VLLM_GPU_MEMORY_UTILIZATION"]
  env: list[dict[str, Any]] = [{"name": name, "value": value} for name, value in values.items()]
  env.append({"name": "OPEN_RL_ACCEL_TIMESLICER_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}})
  return env


def pod_template(worker: Worker) -> dict[str, Any]:
  """The complete worker pod minus placement. Node selection and claims are
  the scheduler's; it rejects a template that carries them."""
  return {
    "spec": {
      "restartPolicy": "OnFailure",
      "containers": [
        {
          "name": "worker",
          "image": os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest"),
          "command": ["uv", "run", "python", "-u", "-m", worker_module(worker.role, worker.is_lora)],
          "args": worker_args(worker.runtime, worker.role, worker.is_lora),
          "env": pod_env(worker),
          "resources": worker.footprint.resources,
          "volumeMounts": [{"name": "shared-storage", "mountPath": "/mnt/shared"}],
        }
      ],
      "volumes": [
        {
          "name": "shared-storage",
          "persistentVolumeClaim": {"claimName": os.getenv("OPEN_RL_SHARED_PVC", "open-rl-shared-pvc")},
        }
      ],
      "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
    },
  }


def workload_body(worker: Worker) -> dict[str, Any]:
  return {
    "apiVersion": f"{GROUP}/{VERSION}",
    "kind": "Workload",
    "metadata": {"name": worker.name, "labels": {"app.kubernetes.io/managed-by": "open-rl-gateway"}},
    "spec": {
      "role": worker.role,
      "trainingKind": "lora" if worker.is_lora else "fft",
      "modelID": worker.runtime,
      "ownerID": worker.owner,
      "accelerator": {"mode": "SingleGPU", "memory": worker.footprint.accelerator},
      "workerContainerName": "worker",
      "template": pod_template(worker),
    },
  }


class SchedulerWorkerManager:
  """Runs trainer and sampler workers by creating Workload objects."""

  def __init__(self, custom_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    if custom_api is None:
      config.load_incluster_config()
      custom_api = client.CustomObjectsApi()
    self.custom_api = custom_api

  def ensure(self, model_id: str, role: str) -> None:
    worker = describe_worker(model_id, role)
    try:
      self.custom_api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, workload_body(worker))
      logger.info("requested %s workload %s (%s, owner %s)", role, worker.name, worker.footprint.accelerator, worker.owner)
    except Exception as exc:
      # AlreadyExists is the reuse: this runtime was requested before.
      if getattr(exc, "status", None) != 409:
        raise

  def render_workload(self, model_id: str, role: str) -> dict[str, Any]:
    return workload_body(describe_worker(model_id, role))

  def release(self, model_id: str) -> None:
    try:
      _, runtime, is_lora = runtime_of(model_id)
    except Exception:
      runtime, is_lora = model_id, False
    if is_lora:
      return  # a shared runtime outlives any one job
    for role in ("trainer", "sampler"):
      self.delete_workload(workload_name(role, owner_of(runtime), is_lora))

  def close(self) -> None:
    pass  # Workloads outlive the gateway; the scheduler owns them from here

  def delete_workload(self, name: str) -> None:
    try:
      self.custom_api.delete_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise
