"""Worker managers for dedicated per-model trainer workers.

The gateway ensures a model's worker exists before enqueueing its create request:
locally by spawning a subprocess, on Kubernetes by creating a pod. There is no
separate launch queue: the subprocess table / the Kubernetes API already hold
the launched-worker state, and both launchers are idempotent per model_id.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Protocol

from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, TRAINER_TIME_SLICE_GROUP, workload_job_id

PROJECT_DIR = Path(__file__).resolve().parents[2]


def local_gpu_inventory() -> list[dict[str, Any]]:
  try:
    result = subprocess.run(
      ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
      check=True,
      capture_output=True,
      text=True,
      timeout=2,
    )
  except (FileNotFoundError, subprocess.SubprocessError):
    return []
  gpus = []
  for line in result.stdout.splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 5:
      continue
    try:
      index, name, total, used, utilization = fields
      gpus.append(
        {
          "index": int(index),
          "name": name,
          "memory_total_mib": int(total),
          "memory_used_mib": int(used),
          "utilization_percent": int(utilization),
        }
      )
    except ValueError:
      continue
  return gpus


def local_time_slicer_status() -> dict[str, Any] | None:
  socket_path = os.getenv("OPEN_RL_ACCEL_TIMESLICER_SOCKET", "/tmp/open-rl/accel-timeslicer.sock")
  if not os.path.exists(socket_path):
    return None
  timeout = float(os.getenv("OPEN_RL_TIMESLICER_STATUS_TIMEOUT_SECONDS", "0.3"))
  with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.settimeout(timeout)
    connection.connect(socket_path)
    connection.sendall(b'{"command":"STATUS"}\n')
    response = connection.makefile("rb").readline(1024 * 1024)
  return json.loads(response) if response else {"ok": False, "error": "status connection closed", "workloads": []}


def tail_log(path: Path, tail_lines: int) -> str:
  limit = max(1, min(int(tail_lines), 5000))
  with path.open(encoding="utf-8", errors="replace") as log_file:
    return "".join(deque(log_file, maxlen=limit))


def _py_cmd(extras: list[str], module: str, model_id: str) -> list[str]:
  if shutil.which("uv"):
    extra_args = []
    for e in extras:
      extra_args.extend(["--extra", e])
    return ["uv", "run", "--no-sync", *extra_args, "python", "-u", "-m", module, f"model_id={model_id}"]
  return [sys.executable, "-u", "-m", module, f"model_id={model_id}"]


def _fetch_metadata_from_store(model_id: str) -> tuple[str | None, str | None]:
  """Retrieve base_model and weight_sync_strategy from canonical open_rl:model_meta:<model_id>."""
  import json

  from server.store import get_store

  try:
    val = get_store().get_value_sync(f"open_rl:model_meta:{model_id}")
    if val:
      meta = json.loads(val) if isinstance(val, str) else val
      if isinstance(meta, dict):
        return meta.get("base_model"), meta.get("weight_sync_strategy")
  except Exception:
    pass
  return None, None


class WorkerManager(Protocol):
  def launch(self, model_id: str, base_model: str | None = None) -> str | None:
    """Ensure the model's worker exists and return its instance id when available."""
    ...

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> str | None:
    """Ensure the trainer worker exists."""
    ...

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> str | None:
    """Ensure the sampler worker exists and return its readiness instance id."""
    ...

  def shutdown(self, model_id: str) -> None:
    """Tear down the model's worker, if any. The idempotent launch can revive it later."""
    ...

  def shutdown_all(self) -> None: ...

  def describe_workers(self, model_id: str) -> list[dict[str, Any]]: ...

  def cluster_snapshot(self) -> dict[str, Any]: ...

  def read_logs(self, model_id: str, component: str, tail_lines: int = 200, previous: bool = False) -> dict[str, Any]: ...


class FFTWorkerManager:
  """Runs local trainer and sampler subprocesses per FFT model."""

  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    self.project_dir = project_dir
    self.train_processes: dict[str, subprocess.Popen] = {}
    self.sampler_processes: dict[str, subprocess.Popen] = {}
    self.sampler_instances: dict[str, str] = {}

  def launch(self, model_id: str, base_model: str | None = None) -> str | None:
    return self.launch_trainer(model_id, base_model)

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> None:
    proc = self.train_processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    stored_base_model, weight_sync_strategy = _fetch_metadata_from_store(model_id)
    base_model = base_model or stored_base_model
    env = {
      **os.environ,
      "OPEN_RL_ENABLE_FFT": "true",
      "OPEN_RL_TIME_SLICE_JOB_ID": workload_job_id("trainer", model_id),
      "OPEN_RL_TIME_SLICE_GROUP": TRAINER_TIME_SLICE_GROUP,
    }
    if base_model:
      env["BASE_MODEL"] = base_model
    if weight_sync_strategy:
      env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = weight_sync_strategy
    self.train_processes[model_id] = subprocess.Popen(
      _py_cmd(["gpu"], "server.training_requests_processor", model_id),
      cwd=self.project_dir,
      env=env,
      start_new_session=True,
    )

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> str | None:
    proc = self.sampler_processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return self.sampler_instances[model_id]

    stored_base_model, weight_sync_strategy = _fetch_metadata_from_store(model_id)
    base_model = base_model or stored_base_model
    env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
    if base_model:
      env["BASE_MODEL"] = base_model
    sampling_backend = os.getenv("SAMPLING_BACKEND", "vllm").lower()
    if sampling_backend == "vllm":
      instance_id = uuid.uuid4().hex
      sampler_env = env.copy()
      sampler_env["OPEN_RL_MODEL_ID"] = model_id
      sampler_env["OPEN_RL_WORKER_INSTANCE_ID"] = instance_id
      sampler_env["OPEN_RL_TIME_SLICE_JOB_ID"] = workload_job_id("sampler", model_id)
      sampler_env["OPEN_RL_TIME_SLICE_GROUP"] = SAMPLER_TIME_SLICE_GROUP
      if weight_sync_strategy:
        sampler_env["OPEN_RL_WEIGHT_SYNC_STRATEGY"] = weight_sync_strategy
      sampler_gpu = os.getenv("SAMPLER_CUDA_VISIBLE_DEVICES")
      if sampler_gpu:
        sampler_env["CUDA_VISIBLE_DEVICES"] = sampler_gpu

      self.sampler_processes[model_id] = subprocess.Popen(
        _py_cmd(["gpu", "vllm"], "server.vllm_sampler", model_id),
        cwd=self.project_dir,
        env=sampler_env,
        start_new_session=True,
      )
      self.sampler_instances[model_id] = instance_id
      return instance_id
    return None

  def shutdown(self, model_id: str) -> None:
    proc = self.train_processes.pop(model_id, None)
    if proc is not None and proc.poll() is None:
      proc.terminate()
    proc_s = self.sampler_processes.pop(model_id, None)
    self.sampler_instances.pop(model_id, None)
    if proc_s is not None and proc_s.poll() is None:
      proc_s.terminate()

  def shutdown_all(self) -> None:
    for model_id in set(list(self.train_processes) + list(self.sampler_processes)):
      self.shutdown(model_id)

  def describe_workers(self, model_id: str) -> list[dict[str, Any]]:
    workers = []
    for role, processes in (("trainer", self.train_processes), ("sampler", self.sampler_processes)):
      proc = processes.get(model_id)
      if proc is None:
        continue
      return_code = proc.poll()
      workers.append(
        {
          "id": role,
          "role": role,
          "status": "running" if return_code is None else ("completed" if return_code == 0 else "failed"),
          "phase": "running" if return_code is None else "exited",
          "message": f"local process {proc.pid}" if return_code is None else f"process exited with code {return_code}",
          "pid": proc.pid,
          "ready": return_code is None,
          "restarts": 0,
          "updated_at": time.time(),
        }
      )
    return workers

  def cluster_snapshot(self) -> dict[str, Any]:
    pods = []
    for model_id in sorted(set(self.train_processes) | set(self.sampler_processes)):
      for worker in self.describe_workers(model_id):
        pods.append({**worker, "model_id": model_id, "name": f"local-{worker['role']}-{model_id}"})
    running = sum(1 for worker in pods if worker["status"] == "running")
    failed = sum(1 for worker in pods if worker["status"] == "failed")
    gpus = local_gpu_inventory()
    errors = []
    time_slicer = None
    try:
      time_slicer = local_time_slicer_status()
    except Exception as exc:
      errors.append(f"time-slicer: {exc}")
    return {
      "mode": "local",
      "status": "healthy" if not errors and all(worker["status"] != "failed" for worker in pods) else "degraded",
      "namespace": None,
      "summary": {
        "nodes": 1,
        "ready_nodes": 1,
        "pods": len(pods),
        "running_pods": running,
        "pending_pods": 0,
        "actionable_pending_pods": 0,
        "failed_pods": failed,
      },
      "nodes": [
        {
          "name": socket.gethostname(),
          "status": "ready",
          "ready": True,
          "capacity": {"nvidia.com/gpu": str(len(gpus))},
          "allocatable": {"nvidia.com/gpu": str(len(gpus))},
          "conditions": [],
          "taints": [],
          "pod_count": len(pods),
          "workloads": pods,
          "gpus": gpus,
          "configured_cuda_devices": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
          "time_slicer": time_slicer,
        }
      ],
      "pods": pods,
      "events": [],
      "errors": errors,
      "generated_at": time.time(),
    }

  def read_logs(self, model_id: str, component: str, tail_lines: int = 200, previous: bool = False) -> dict[str, Any]:
    del previous
    if component not in {"gateway", "trainer", "sampler", "timeslicer"}:
      return {"source": "local", "pod_name": None, "logs": "", "error": f"Unknown log component {component!r}"}
    log_dir = os.getenv("OPEN_RL_LOG_DIR")
    log_name = "timeslicer.log" if component == "timeslicer" else "gateway.log"
    log_path = Path(log_dir) / log_name if log_dir else None
    if log_path is not None and log_path.is_file():
      return {
        "source": "file",
        "pod_name": None,
        "logs": tail_log(log_path, tail_lines),
        "error": None,
      }
    if component in {"gateway", "timeslicer"}:
      return {
        "source": "local",
        "pod_name": None,
        "logs": "",
        "error": f"Local {component} logs are written to the terminal; set OPEN_RL_LOG_DIR to expose retained logs.",
      }
    process = (self.train_processes if component == "trainer" else self.sampler_processes).get(model_id)
    return {
      "source": "local",
      "pod_name": None,
      "logs": "",
      "error": "Local workers inherit the gateway terminal; start the gateway with redirected output to retain raw logs."
      if process is not None
      else f"No local {component} worker exists for this run.",
    }


def create_fft_worker_manager() -> WorkerManager:
  mode = os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower()
  if mode in {"kubernetes", "k8s"}:
    from server.k8s_worker_manager import KubernetesFFTWorkerManager

    return KubernetesFFTWorkerManager()
  return FFTWorkerManager()
