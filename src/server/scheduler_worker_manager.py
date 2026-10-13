"""Worker manager that creates one Workload per worker process and lets the
scheduler place it. Part of the cluster extra: importing it assumes the
Kubernetes client is installed.

Two FFT jobs and two LoRA jobs on the same base model come in:

  model_id (job)   owner               Workload (process)          shares
  fft-job 1a2b     1a2b                fft-1a2b-trainer            nothing
  fft-job 3c4d     3c4d                fft-3c4d-trainer            nothing
  lora-job 5e6f    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   one runtime
  lora-job 7a8b    qwen-qwen2-5-0-5b   lora-qwen-qwen2-5-0-5b-0-trainer   with 5e6f
  automodel 9c0d   automodel-qwen-qwen2-5-0-5b   lora-automodel-qwen-qwen2-5-0-5b-0-trainer   one Automodel
                                                                                  runtime, apart from PyTorch's
  image 1d2e       image-<hash>-qwen-qwen2-5-0-5b   lora-image-<hash>-qwen-qwen2-5-0-5b-0-trainer   one runtime
                                                                                  per trainer image

The Workload name is what the pod label and the time-slicer call job_id.
"""

import dataclasses
import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from kubernetes import client, config

from server.estimator import Footprint, footprint
from server.sampler_routing import SAMPLER_SET_LABEL, sampler_set
from server.worker_manager import base_model_of, owner_id, runtime_of, worker_args, worker_env, worker_module

logger = logging.getLogger(__name__)

GROUP = "openrl.io"
VERSION = "v1alpha1"
PLURAL = "workloads"

# With openrl.sampler_router=llmd every sampler of a set is stock `vllm serve` on
# SAMPLER_PORT, labelled into its set, and the set gets a router from ROUTER_TEMPLATE.
SAMPLER_PORT = 8000
ROUTER_TEMPLATE = Path(__file__).with_name("llmd_router.yaml")
# The API server's vLLM settings, passed to routed samplers as flags.
VLLM_FLAGS = {
  "VLLM_MAX_MODEL_LEN": "--max-model-len",
  "VLLM_GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
  "VLLM_MAX_NUM_SEQS": "--max-num-seqs",
  "VLLM_MAX_NUM_BATCHED_TOKENS": "--max-num-batched-tokens",
}


def workload_name(role: str, owner: str, is_lora: bool, index: int = 0) -> str:
  # A second compatible request renders the same name, and the create's
  # AlreadyExists is the reuse. The index numbers sampler replicas; the first
  # keeps the name it always had.
  if is_lora:
    return f"lora-{owner}-{index}-{role}"
  return f"fft-{owner}-{role}" if index == 0 else f"fft-{owner}-{role}-{index}"


@dataclass(frozen=True)
class Worker:
  """Everything the API server knows about one worker process before placement."""

  role: str
  runtime: str
  base_model: str
  is_lora: bool
  exclusive: bool
  meta: Any
  footprint: Footprint
  # GPUs the worker drives as one torchrun group.
  devices: int = 1

  # Which replica of its role this is.
  index: int = 0

  @property
  def owner(self) -> str:
    return owner_id(self.runtime)

  @property
  def name(self) -> str:
    return workload_name(self.role, self.owner, self.is_lora, self.index)


def replicas_of(model_id: str, role: str) -> int:
  """How many workers of this role the model asked for. Each sampler replica
  is its own Workload draining the shared sampling queue."""
  if role != "sampler":
    return 1
  meta, _, _ = runtime_of(model_id)
  return meta.sampler_replicas


def describe_worker(model_id: str, role: str) -> Worker:
  meta, runtime, is_lora = runtime_of(model_id)
  base_model = base_model_of(meta, runtime)
  exclusive = not meta.shares_gpu()
  devices = meta.trainer_gpus if role == "trainer" else 1
  size = footprint(base_model, meta.fine_tuning_type, role)
  # Each torchrun rank is its own process with its own host memory.
  size = replace(size, host_request_bytes=size.host_request_bytes * devices, host_limit_bytes=size.host_limit_bytes * devices)
  return Worker(role, runtime, base_model, is_lora, exclusive, meta, size, devices)


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
  # MAX_JOBS caps FlashInfer's JIT build, which otherwise runs one ~3GB
  # compiler per core and blows through the pod's host memory limit.
  for name in ("VLLM_GPU_MEMORY_UTILIZATION", "VLLM_MAX_MODEL_LEN", "OPEN_RL_TRAIN_TOKEN_BUDGET", "MAX_JOBS"):
    if os.getenv(name):
      values[name] = os.environ[name]
  # No other worker shares an exclusive worker's GPUs, so it never parks.
  if worker.exclusive:
    values["OPEN_RL_TIME_SLICING"] = "off"
  env: list[dict[str, Any]] = [{"name": name, "value": value} for name, value in values.items()]
  env.append({"name": "OPEN_RL_ACCEL_TIMESLICER_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}})
  return env


def worker_container(worker: Worker) -> tuple[str, list[str]]:
  """The image and command. An Automodel trainer, or one from an image the job
  names, runs the python on its image's PATH."""
  if worker.role == "trainer" and worker.meta.trainer_backend != "pytorch":
    image = worker.meta.trainer_image() or os.getenv("OPEN_RL_AUTOMODEL_IMAGE", "ghcr.io/gke-labs/open-rl/automodel:latest")
    if worker.devices > 1:
      torchrun = ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={worker.devices}"]
      return image, ["python", "-u", *torchrun, "-m", worker_module(worker.role)]
    return image, ["python", "-u", "-m", worker_module(worker.role)]
  image = os.getenv("OPEN_RL_WORKER_IMAGE", "ghcr.io/gke-labs/open-rl/server:latest")
  return image, ["uv", "run", "python", "-u", "-m", worker_module(worker.role)]


def pod_template(worker: Worker) -> dict[str, Any]:
  """The complete worker pod minus placement. Node selection and claims are
  the scheduler's; it rejects a template that carries them."""
  image, command = worker_container(worker)
  template = {
    "spec": {
      "restartPolicy": "OnFailure",
      "containers": [
        {
          "name": "worker",
          "image": image,
          "command": command,
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

  # NCCL moves data between ranks through /dev/shm, which is 64Mi by default.
  if worker.devices > 1:
    template["spec"]["containers"][0]["volumeMounts"].append({"name": "dshm", "mountPath": "/dev/shm"})
    template["spec"]["volumes"].append({"name": "dshm", "emptyDir": {"medium": "Memory"}})

  if worker.role == "sampler" and worker.meta.sampler_router:
    join_set(template, worker)

  if pull_policy := os.getenv("OPEN_RL_WORKER_IMAGE_PULL_POLICY"):
    template["spec"]["containers"][0]["imagePullPolicy"] = pull_policy
  return template


def join_set(template: dict[str, Any], worker: Worker) -> None:
  """Run the sampler as stock `vllm serve`, labelled into its set. Requests name a
  saved adapter by its snapshot directory, which vLLM's filesystem resolver loads
  on first use."""
  sampler = template["spec"]["containers"][0]
  sampler["image"] = os.getenv("OPEN_RL_VLLM_IMAGE", "vllm/vllm-openai:v0.30.0")
  sampler["command"] = ["vllm", "serve"]
  sampler["args"] = [
    worker.base_model,
    f"--port={SAMPLER_PORT}",
    "--enable-lora",
    "--max-lora-rank=64",
    "--max-loras=8",
    # Sample as the queue's samplers do, without the model's generation defaults.
    "--generation-config=vllm",
    '--limit-mm-per-prompt={"image": 0, "video": 0}',
    *(f"{flag}={os.environ[name]}" for name, flag in VLLM_FLAGS.items() if os.getenv(name)),
  ]
  tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/mnt/shared/open-rl")
  sampler["env"] += [
    {"name": "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "value": "true"},
    {"name": "VLLM_LORA_RESOLVER_CACHE_DIR", "value": f"{tmp_dir}/sampler_lora"},
  ]
  # Named, so the picker's target port and Prometheus's scrape both find it.
  sampler["ports"] = [{"name": "http", "containerPort": SAMPLER_PORT}]
  sampler["readinessProbe"] = {"httpGet": {"path": "/health", "port": SAMPLER_PORT}, "periodSeconds": 10}
  template["metadata"] = {"labels": {SAMPLER_SET_LABEL: sampler_set(worker.runtime)}}


def router_objects(set_id: str, owner: str) -> list[dict[str, Any]]:
  """The set's router Deployment and Service."""
  return list(yaml.safe_load_all(ROUTER_TEMPLATE.read_text().replace("SET_ID", set_id).replace("OWNER", owner)))


def accelerator_spec(worker: Worker) -> dict[str, Any]:
  """A torchrun group asks for its devices, each sized for a whole replica."""
  if worker.devices > 1:
    return {"mode": "MultiGPU", "devices": worker.devices, "memory": worker.footprint.accelerator}
  return {"mode": "SingleGPU", "memory": worker.footprint.accelerator}


def workload_body(worker: Worker) -> dict[str, Any]:
  return {
    "apiVersion": f"{GROUP}/{VERSION}",
    "kind": "Workload",
    "metadata": {"name": worker.name, "labels": {"app.kubernetes.io/managed-by": "open-rl-api-server"}},
    "spec": {
      "role": worker.role,
      "trainingKind": "lora" if worker.is_lora else "fft",
      "exclusive": worker.exclusive,
      "modelID": worker.runtime,
      "ownerID": worker.owner,
      "accelerator": accelerator_spec(worker),
      "workerContainerName": "worker",
      "template": pod_template(worker),
    },
  }


class SchedulerWorkerManager:
  """Runs trainer and sampler workers by creating Workload objects."""

  def __init__(self, custom_api: Any = None, apps_api: Any = None, core_api: Any = None):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")
    self.namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    if custom_api is None:
      config.load_incluster_config()
      custom_api = client.CustomObjectsApi()
    self.custom_api = custom_api
    # Only routed sampler sets need these; the config loaded above serves them too.
    self.apps_api = apps_api
    self.core_api = core_api

  def apps(self) -> Any:
    if self.apps_api is None:
      self.apps_api = client.AppsV1Api()
    return self.apps_api

  def core(self) -> Any:
    if self.core_api is None:
      self.core_api = client.CoreV1Api()
    return self.core_api

  def ensure(self, model_id: str, role: str) -> None:
    worker = describe_worker(model_id, role)
    for index in range(replicas_of(model_id, role)):
      self.ensure_workload(dataclasses.replace(worker, index=index))
    if role == "sampler" and worker.meta.sampler_router:
      self.ensure_router(sampler_set(worker.runtime), worker.owner)

  def ensure_router(self, set_id: str, owner: str) -> None:
    """Create the set's router and its Service, waiting out a router still being deleted."""
    deployment, service = router_objects(set_id, owner)
    deadline = time.monotonic() + 180
    while True:
      try:
        self.apps().create_namespaced_deployment(self.namespace, deployment)
        break
      except Exception as exc:
        if getattr(exc, "status", None) != 409:
          raise
      try:
        if not self.apps().read_namespaced_deployment(deployment["metadata"]["name"], self.namespace).metadata.deletion_timestamp:
          break
      except Exception as exc:
        if getattr(exc, "status", None) != 404:
          raise
        continue
      if time.monotonic() > deadline:
        raise RuntimeError(f"router for {set_id} has been terminating for over three minutes")
      time.sleep(2)
    try:
      self.core().create_namespaced_service(self.namespace, service)
    except Exception as exc:
      if getattr(exc, "status", None) != 409:
        raise

  def ensure_workload(self, worker: Worker) -> None:
    role = worker.role
    deadline = time.monotonic() + 180
    while True:
      try:
        self.custom_api.create_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, workload_body(worker))
        logger.info("requested %s workload %s (%s, owner %s)", role, worker.name, worker.footprint.accelerator, worker.owner)
        return
      except Exception as exc:
        if getattr(exc, "status", None) != 409:
          raise
      # AlreadyExists is the reuse, unless the old one is still being deleted.
      if not self.still_deleting(worker.name):
        return
      if time.monotonic() > deadline:
        raise RuntimeError(f"workload {worker.name} has been terminating for over three minutes")
      time.sleep(2)

  def still_deleting(self, name: str) -> bool:
    try:
      workload = self.custom_api.get_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return True  # gone since the create failed, so the next create will go through
      raise
    return bool(workload["metadata"].get("deletionTimestamp"))

  def release(self, model_id: str) -> None:
    try:
      meta, runtime, _ = runtime_of(model_id)
      shared = meta.shares_runtime()
    except Exception:
      runtime, shared = model_id, False
    if shared:
      return  # a shared runtime outlives any one job
    self.release_owner(owner_id(runtime))

  def release_owner(self, owner: str) -> set[str]:
    """Delete the owner's workloads and routers. The scheduler's finalizer frees the seats."""
    found = self.apps().list_namespaced_deployment(self.namespace, label_selector=f"openrl.io/owner={owner},app=openrl-sampler-router")
    for deployment in found.items:
      for delete in (self.apps().delete_namespaced_deployment, self.core().delete_namespaced_service):
        try:
          delete(deployment.metadata.name, self.namespace)
        except Exception as exc:
          if getattr(exc, "status", None) != 404:
            raise
    selector = "app.kubernetes.io/managed-by=open-rl-api-server"
    found = self.custom_api.list_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, label_selector=selector)
    ours = [item for item in found["items"] if item["spec"]["ownerID"] == owner]
    for item in ours:
      self.delete_workload(item["metadata"]["name"])
    return {item["spec"]["modelID"] for item in ours}

  def close(self) -> None:
    pass  # Workloads outlive the API server; the scheduler owns them from here

  def delete_workload(self, name: str) -> None:
    try:
      self.custom_api.delete_namespaced_custom_object(GROUP, VERSION, self.namespace, PLURAL, name)
    except Exception as exc:
      if getattr(exc, "status", None) != 404:
        raise
