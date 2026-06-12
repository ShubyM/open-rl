"""Launchers for dedicated per-model FFT workers.

The gateway ensures a model's worker exists before enqueueing its create request:
locally by spawning a subprocess, on Kubernetes by creating a pod. There is no
separate launch queue — the subprocess table / the Kubernetes API already hold
the launched-worker state, and both launchers are idempotent per model_id.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

PROJECT_DIR = Path(__file__).resolve().parents[2]


class WorkerManager(Protocol):
  def launch(self, model_id: str) -> None:
    """Ensure the model's worker exists; idempotent per model_id."""
    ...

  def shutdown(self, model_id: str) -> None:
    """Tear down the model's worker, if any. The idempotent launch can revive it later."""
    ...

  def shutdown_all(self) -> None: ...


class FFTWorkerManager:
  """Runs one local worker subprocess per FFT model."""

  def __init__(self, project_dir: Path = PROJECT_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    self.project_dir = project_dir
    self.processes: dict[str, subprocess.Popen] = {}

  def launch(self, model_id: str) -> None:
    proc = self.processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
    self.processes[model_id] = subprocess.Popen(
      [sys.executable, "-m", "server.training_requests_processor", "--model-id", model_id],
      cwd=self.project_dir,
      env=env,
    )

  def shutdown(self, model_id: str) -> None:
    proc = self.processes.pop(model_id, None)
    if proc is not None and proc.poll() is None:
      proc.terminate()

  def shutdown_all(self) -> None:
    for model_id in list(self.processes):
      self.shutdown(model_id)


def create_fft_worker_manager() -> WorkerManager:
  if os.getenv("OPEN_RL_WORKER_LAUNCHER", "subprocess").lower() == "kubernetes":
    from server.k8s_worker_manager import KubernetesFFTWorkerManager

    return KubernetesFFTWorkerManager()
  return FFTWorkerManager()
