"""Worker manager that asks the scheduler for placement instead of doing it.

Where KubernetesWorkerManager renders a pod itself -- picking the claim, the
node selector, the time-slice group, the container resources -- this manager
creates one Workload object per runtime process and stops. The
placement controller (controller/) selects or cuts the ResourceClaim, stamps
placement onto the pod rendered from the workload's inline template, and
kube-scheduler and DRA do the rest.

What stays here is exactly what only the API server knows: which runtime
processes must exist (identity and reuse), roughly how much accelerator
memory each needs, and the complete pod template -- image, entrypoint,
identity env, resources, volumes. Everything placement-shaped (claims,
groups, node selectors) is deliberately absent from the template; the
controller rejects it rather than merging.

Identity is the name. FFT jobs own their processes: fft-<job>-<role>. LoRA
jobs on one base model share them: lora-<base>-<instance>-<role>, so two
compatible requests render the same name and the second create's
AlreadyExists is the reuse. The instance index is fixed at 0 until adapter
capacity accounting exists.

This module is part of the cluster extra; importing it assumes Kubernetes
dependencies are installed.
"""

import logging
import os
from typing import Any

from kubernetes import client, config

from server.k8s_worker_manager import sanitize_job_id
from server.worker_manager import estimate_worker_footprint, get_model_target_info, vllm_gpu_memory_utilization

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"


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

  def launch(self, model_id: str) -> None:
    self.launch_trainer(model_id)

  def launch_trainer(self, model_id: str) -> None:
    self._launch(model_id, role="trainer")

  def launch_sampler(self, model_id: str) -> None:
    self._launch(model_id, role="sampler")

  def _launch(self, model_id: str, role: str) -> None:
    body = self.render_workload(model_id, role)
    name = body["metadata"]["name"]
    try:
      self.custom_api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, body)
      logger.info(
        "requested %s workload %s (%s, owner %s)",
        role,
        name,
        body["spec"]["accelerator"]["memory"],
        body["spec"]["ownerId"],
      )
    except Exception as exc:
      # AlreadyExists is the reuse: the runtime this name selects has been
      # requested before, and the scheduler owns it from here.
      if getattr(exc, "status", None) != 409:
        raise

  def workload_name(self, role: str, is_lora: bool, target_id: str) -> str:
    """Identity is the name: fft names embed the job (unique per job), lora
    names embed the base model (shared across compatible jobs). The lora
    instance index is fixed at 0 until adapter-capacity accounting exists."""
    key = sanitize_job_id(target_id)
    if is_lora:
      return f"lora-{key}-0-{role}"
    return f"fft-{key}-{role}"

  def render_workload(self, model_id: str, role: str) -> dict[str, Any]:
    meta, target_id, is_lora = get_model_target_info(model_id)
    job_id = sanitize_job_id(target_id)
    base_model = (meta.base_model if meta and meta.base_model else None) or os.getenv("BASE_MODEL") or target_id
    name = self.workload_name(role, is_lora, target_id)

    # One estimator call: the accelerator figure and the host request it
    # parks into are one fact.
    footprint = estimate_worker_footprint(base_model, meta.fine_tuning_type, role)

    spec: dict[str, Any] = {
      "role": role,
      "trainingKind": "lora" if is_lora else "fft",
      "modelId": target_id,
      # Owner = fairness unit: one per FFT job, shared per LoRA base runtime.
      "ownerId": job_id,
      "accelerator": {
        "memory": footprint["accelerator_memory"],
        "maxDeviceCount": 1,
      },
      "workerContainerName": "worker",
      "template": self.render_template(name, role, target_id, is_lora, base_model, meta, job_id, footprint["resources"]),
    }

    return {
      "apiVersion": f"{GROUP}/{VERSION}",
      "kind": "Workload",
      "metadata": {"name": name, "labels": {"app.kubernetes.io/managed-by": "open-rl-gateway"}},
      "spec": spec,
    }

  def render_template(
    self, name: str, role: str, target_id: str, is_lora: bool, base_model: str, meta: Any, job_id: str, resources: dict[str, Any]
  ) -> dict[str, Any]:
    """The complete worker pod, minus placement: every field knowable at
    render time. The group env is placement's and the controller stamps it."""
    if role == "sampler":
      module = "server.lora_sampler" if is_lora else "server.vllm_sampler"
    else:
      module = "server.training_requests_processor"

    args = ["--model-id", target_id]
    if role == "trainer" and is_lora:
      args += ["--active-tenant-set-id", f"{target_id}-1"]

    tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl")
    env: list[dict[str, Any]] = [
      {"name": "REDIS_URL", "value": os.environ["REDIS_URL"]},
      {"name": "BASE_MODEL", "value": base_model},
      {"name": "OPEN_RL_BASE_MODEL", "value": base_model},
      {"name": "OPEN_RL_ENABLE_FFT", "value": "false" if is_lora else "true"},
      {"name": "OPEN_RL_FINE_TUNING_TYPE", "value": "lora" if is_lora else "full"},
      {"name": "OPEN_RL_TMP_DIR", "value": tmp_dir},
      {"name": "HF_HOME", "value": os.getenv("HF_HOME", f"{tmp_dir}/huggingface")},
      # Identity: knowable at render time, so the API server writes it. The
      # group is placement's and is stamped by the controller.
      {"name": "OPEN_RL_TIME_SLICE_OWNER", "value": job_id},
      {"name": "OPEN_RL_WORKLOAD_ID", "value": name},
      # The llmd snapshot agent discovers processes by the older spelling;
      # carry both until the agent reads OPEN_RL_WORKLOAD_ID.
      {"name": "OPEN_RL_TIME_SLICE_JOB_ID", "value": name},
      {
        "name": "OPEN_RL_ACCEL_TIMESLICER_HOST",
        "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}},
      },
      {"name": "OPEN_RL_ACCEL_TIMESLICER_PORT", "value": os.getenv("OPEN_RL_ACCEL_TIMESLICER_PORT", "9753")},
    ]

    if role == "trainer":
      env.append({"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"})
    else:
      env += [
        {
          "name": "VLLM_GPU_MEMORY_UTILIZATION",
          "value": os.getenv("VLLM_GPU_MEMORY_UTILIZATION") or vllm_gpu_memory_utilization(base_model, "lora" if is_lora else "full"),
        },
        {"name": "VLLM_SERVER_DEV_MODE", "value": "1"},
        {"name": "VLLM_ALLOW_INSECURE_SERIALIZATION", "value": "1"},
      ]
      if not is_lora and ("gemma-4" in base_model.lower() or "gemma4" in base_model.lower()):
        env.append({"name": "VLLM_ARCHITECTURE_OVERRIDE", "value": "Gemma4ForCausalLM"})

    weight_sync = getattr(meta, "weight_sync_config", None)
    if weight_sync is not None:
      env.append({"name": "OPEN_RL_WEIGHT_SYNC_STRATEGY", "value": weight_sync.strategy})
      if weight_sync.strategy == "delta":
        env.append({"name": "OPEN_RL_WEIGHT_SYNC_DELTA_FORMAT", "value": weight_sync.delta_format})
        env.append({"name": "OPEN_RL_WEIGHT_SYNC_DELTA_APPLY_METHOD", "value": weight_sync.delta_apply_method})

    image = os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest")

    return {
      "spec": {
        "restartPolicy": "OnFailure",
        "containers": [
          {
            "name": "worker",
            "image": image,
            "command": ["uv", "run", "python", "-u", "-m", module],
            "args": args,
            "env": env,
            "resources": resources,
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

  def shutdown(self, model_id: str) -> None:
    try:
      _, target_id, is_lora = get_model_target_info(model_id)
    except Exception:
      target_id, is_lora = model_id, False
    for role in ("trainer", "sampler"):
      self._delete(self.workload_name(role, is_lora, target_id))

  def shutdown_all(self) -> None:
    listing = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL)
    for item in listing.get("items", []):
      self._delete(item["metadata"]["name"])

  def _delete(self, name: str) -> None:
    try:
      self.custom_api.delete_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise
