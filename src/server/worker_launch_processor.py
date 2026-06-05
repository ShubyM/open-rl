import asyncio
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from store import RequestStore

SERVER_DIR = Path(__file__).resolve().parent


class CreateModelWorkerLaunchRequest(BaseModel):
  req_id: str
  model_id: str
  type: Literal["create_model"]
  base_model: str
  lora_config: dict[str, Any] = Field(default_factory=dict)
  full_config: dict[str, Any] = Field(default_factory=dict)
  trace_context: dict[str, str] = Field(default_factory=dict)


class CreateModelFromStateWorkerLaunchRequest(BaseModel):
  req_id: str
  model_id: str
  type: Literal["create_model_from_state"]
  state_path: str
  restore_optimizer: bool = False
  trace_context: dict[str, str] = Field(default_factory=dict)


WorkerLaunchRequest = CreateModelWorkerLaunchRequest | CreateModelFromStateWorkerLaunchRequest


class FFTWorkerManager:
  def __init__(self, server_dir: Path = SERVER_DIR):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("OPEN_RL_ENABLE_FFT=true requires REDIS_URL so launched workers can share queues and futures")

    self.server_dir = server_dir
    self.processes: dict[str, subprocess.Popen] = {}

  def launch(self, model_id: str) -> None:
    proc = self.processes.get(model_id)
    if proc is not None and proc.poll() is None:
      return

    env = {**os.environ, "OPEN_RL_ENABLE_FFT": "true"}
    self.processes[model_id] = subprocess.Popen(
      [sys.executable, "-m", "clock_cycle", "--model-id", model_id],
      cwd=self.server_dir,
      env=env,
    )

  def shutdown_all(self) -> None:
    for proc in self.processes.values():
      if proc.poll() is None:
        proc.terminate()


class WorkerLaunchProcessor:
  """Drain the worker launch queue and start FFT workers before enqueueing training requests."""

  def __init__(self, store: RequestStore, worker_manager: FFTWorkerManager):
    self.store = store
    self.worker_manager = worker_manager

  async def process_request(self, request: dict) -> None:
    try:
      model_id = request.get("model_id")
      if not model_id:
        raise ValueError("worker launch request requires model_id")

      self.worker_manager.launch(model_id)
      await self.store.put_request(request)
    except Exception as exc:
      traceback.print_exc()
      await self.store.set_future(request["req_id"], {"type": "RequestFailedResponse", "error_message": str(exc)})

  async def process_batch(self, requests: list[dict]) -> None:
    for request in requests:
      await self.process_request(request)

  async def run(self) -> None:
    while True:
      try:
        batch = await self.store.get_worker_launch_requests()
        if not batch:
          await asyncio.sleep(0.1)
          continue
        await self.process_batch(batch)
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in worker launch processor: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)
