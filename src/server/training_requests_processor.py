# This file contains the training request processor implementation for Open-RL.

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import threading
import traceback
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import TimeSlicerClient, TimeSlicerFault, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_CLAIM, local_workload_name
from server.model_metadata import get_model_metadata
from server.store import RequestStore, get_state_store, get_store
from training import commands
from training.backend import Suspendable, TrainingBackend
from training.commands import parse_command

tracer = trace.get_tracer(__name__)


async def call_backend(operation: Callable[..., Any], *args, **kwargs) -> Any:
  """A cancelled waiter must not leave native work running across sleep/release."""
  task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
  cancellation = None
  while not task.done():
    try:
      await asyncio.shield(task)
    except asyncio.CancelledError as exc:
      # Repeated cancellation must not cancel the task while its native
      # thread is still running. Drain it before parking the process.
      cancellation = exc
    except Exception:
      break
  if cancellation is not None:
    if not task.cancelled():
      task.exception()
    raise cancellation
  return task.result()


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def describe_requests(batch: list[dict[str, Any]]) -> str:
  """`op:request_id` per request, matching the API server's enqueue log line."""
  return ", ".join(f"{r.get('op')}:{r.get('request_id')}" for r in batch)


# Sampler weight versions kept on the volume. The sampler applies each delta
# as it lands, so older versions are dead weight; an 8B run otherwise leaves
# 3 GiB per step behind.
SAMPLER_VERSIONS_KEPT = int(os.getenv("OPEN_RL_SAMPLER_VERSIONS_KEPT", "3"))


def older_versions(path: str, keep: int) -> list[str]:
  """Sibling version directories of `path` beyond the newest `keep`, oldest first."""
  parent = os.path.dirname(path)
  if not os.path.isdir(parent):
    return []
  versions = sorted((p for p in (os.path.join(parent, name) for name in os.listdir(parent)) if os.path.isdir(p)), key=os.path.getmtime)
  return versions[: max(0, len(versions) - keep)]


class TrainingRequestsProcessor:
  """Drains training commands for one worker.

  A model_id selects one dedicated queue; otherwise the worker serves a
  shared active set. An optional time slicer controls GPU leases separately
  from queue ownership. Resident workers on exclusive GPUs need no slicer.
  """

  def __init__(
    self,
    store: RequestStore,
    worker: TrainingBackend,
    model_id: str | None = None,
    active_tenant_set_id: str | None = None,
    time_slicer: TimeSlicerClient | None = None,
  ):
    if model_id is not None and active_tenant_set_id is not None:
      raise ValueError("Choose a dedicated model_id or a shared active_tenant_set_id, not both")
    if time_slicer is not None and not model_id:
      raise ValueError("A time-sliced trainer needs a dedicated model_id")
    if time_slicer is not None and not isinstance(worker, Suspendable):
      raise ValueError("This backend cannot suspend; launch it with exclusive resources")

    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.active_tenant_set_id = active_tenant_set_id
    self.time_slicer = time_slicer
    self.suspension = worker if time_slicer is not None else None
    self.workload = workload_from_env(os.getpid(), name=local_workload_name("trainer", model_id), claim=TRAINER_CLAIM) if time_slicer else None
    self.snapshot_registered = False
    self.stopping = False

  async def run(self) -> None:
    print(f"[WORKER] Training requests processor started (model={self.model_id} active_set={self.active_tenant_set_id}).")
    try:
      if self.time_slicer is not None:
        await self.time_slicer.register(self.workload)
        self.snapshot_registered = True
      while not self.stopping:
        try:
          await self.run_once()
        except asyncio.CancelledError:
          break
        except Exception as exc:
          print(f"Error in training requests processor: {exc}")
          traceback.print_exc()
          await asyncio.sleep(1)
    finally:
      if self.time_slicer is not None:
        try:
          if self.snapshot_registered:
            await self.time_slicer.unregister(self.workload)
        finally:
          await self.time_slicer.close()

  async def exit_gracefully(self, unregister: bool = True) -> None:
    print(f"[WORKER] Initiating immediate exit for model {self.model_id} trainer worker...")
    if unregister and self.time_slicer is not None and self.snapshot_registered:
      try:
        await self.time_slicer.unregister(self.workload)
        self.snapshot_registered = False
      except Exception as exc:
        print(f"[WORKER] Failed to unregister: {exc}")
    if self.time_slicer is not None:
      try:
        await self.time_slicer.close()
      except Exception:
        pass
    os._exit(0)

  async def next_batch(self) -> list[dict[str, Any]]:
    if self.model_id is not None:
      return await self.store.get_requests_for_model(self.model_id)
    return await self.store.get_requests(active_set_id=self.active_tenant_set_id)

  async def run_once(self) -> None:
    batch = await self.next_batch()
    if not batch:
      await asyncio.sleep(0.1)
      return

    shutdown_index = next((index for index, req in enumerate(batch) if req.get("op") == "shutdown_workers"), len(batch))
    shutdown = shutdown_index < len(batch)
    work = batch[:shutdown_index]
    model_id = batch[0].get("model_id", "default")

    results: list[tuple[str | None, dict[str, Any]]] = []
    failure: Exception | None = None
    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(work))
      batch_span.set_attribute("model_id", model_id)
      if work:
        print(f"\n[TRAINING REQUESTS] Popped {len(work)} requests for model: {model_id}: {describe_requests(work)}")
        results, failure = await self.answer_batch(work)

    for request_id, result in results:
      if request_id is not None:
        await self.store.set_future(request_id, result)
    if shutdown:
      # A dedicated process ends after its model is deleted, whether or not
      # it uses leases. Answer every popped request before any fatal exit.
      self.stopping = self.model_id is not None
      for request in batch[shutdown_index + 1 :]:
        if request_id := request.get("request_id"):
          await self.store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": "Trainer model has been shut down"})
    if self.time_slicer is not None and self.time_slicer.faulted:
      # This process still holds the accelerator. Exit without unregistering so
      # the grant moves on only once the memory is gone. Exit 0 keeps the pod
      # from restarting on fresh weights mid-run; the run fails on its next call.
      print(f"[WORKER] Time slicer could not park this process: {self.time_slicer.faulted}. Exiting to free the accelerator.")
      await self.exit_gracefully(unregister=False)
    if failure is not None:
      raise failure

  async def answer_batch(self, requests: list[dict[str, Any]]) -> tuple[list[tuple[str | None, dict[str, Any]]], Exception | None]:
    """Every request gets an answer: its result, or the failure that stopped the batch."""
    results: list[tuple[str | None, dict[str, Any]]] = []
    try:
      async with self.execution():
        for request in requests:
          result = await self.handle_request(request)
          request_id, response = result
          if self.time_slicer is None and request_id is not None:
            await self.store.set_future(request_id, response)
          results.append(result)
    except Exception as exc:
      published = len(results) if self.time_slicer is None else 0
      answered = {request_id for request_id, _ in results}
      for request in requests:
        request_id = request.get("request_id")
        if request_id and request_id not in answered:
          results.append((request_id, {"type": "RequestFailedResponse", "error_message": f"Trainer worker error: {exc}"}))
      return results[published:], exc
    # Resident workers publish immediately; leased workers publish only once
    # the device has been released. Return just the answers still to publish.
    return ([] if self.time_slicer is None else results), None

  @asynccontextmanager
  async def execution(self):
    """Every operation, including creation and export, runs while the backend is awake."""
    if self.time_slicer is None:
      yield
      return
    async with self.time_slicer.acquire(self.workload):
      try:
        await call_backend(self.suspension.wake_up)
        yield
      finally:
        await call_backend(self.suspension.sleep)
    if self.time_slicer.faulted:
      raise TimeSlicerFault(self.time_slicer.faulted)

  async def process_request(self, raw_request: dict[str, Any]) -> None:
    request_id, result = await self.handle_request(raw_request)
    if request_id is not None:
      await self.store.set_future(request_id, result)

  async def handle_request(self, raw_request: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    request_id = raw_request.get("request_id")
    token = None

    try:
      command = parse_command(raw_request)
      request_id = command.request_id
      if self.model_id is not None and command.model_id != self.model_id:
        raise ValueError(f"This trainer serves model {self.model_id}, not {command.model_id}")

      ctx = propagate.extract(command.trace_context) if command.trace_context else None
      token = otel_context.attach(ctx) if ctx else None

      result = await self.dispatch_operation(command)
      return request_id, result
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
      return request_id, {"type": "RequestFailedResponse", "error_message": str(exc)}
    finally:
      if token:
        otel_context.detach(token)

  async def dispatch_operation(self, command: commands.TrainingCommand) -> dict[str, Any]:
    match command:
      case commands.CreateModel():
        is_lora = command.fine_tuning_type == "lora"
        config = command.lora_config if is_lora else command.full_config
        await call_backend(self.worker.create_model, command.base_model, command.model_id, config)
        result = {
          "base_model": command.base_model,
          "model_id": command.model_id,
          "fine_tuning_type": command.fine_tuning_type,
          "type": "model_created",
        }
        if is_lora:
          result["rank"] = command.lora_config.rank
        return result
      case commands.CreateModelFromState():
        result = await call_backend(self.worker.load_from_state, command.model_id, command.state_path, command.restore_optimizer)
        return {
          "base_model": result.get("base_model"),
          "model_id": result.get("model_id", command.model_id),
          "fine_tuning_type": command.fine_tuning_type,
          "type": "model_loaded_from_state",
        }
      case commands.ForwardBackward():
        result = await call_backend(
          self.worker.forward_backward,
          command.data,
          command.loss_fn,
          command.loss_config,
          command.model_id,
          forward_only=command.forward_only,
        )
        result["type"] = "forward_backward_completed"
        return result
      case commands.OptimStep():
        result = await call_backend(self.worker.optim_step, command.adam_params, command.model_id)
        result["type"] = "optim_step_completed"
        return result
      case commands.Sample():
        result = await call_backend(
          self.worker.generate,
          command.prompt_tokens,
          command.max_tokens,
          command.num_samples,
          command.temperature,
          command.model_id,
          command.prompt_logprobs,
        )
        result["type"] = "sample_completed"
        return result
      case commands.SaveState():
        result = await call_backend(self.worker.save_state, command.model_id, command.state_path, command.include_optimizer, command.kind)
        return {"path": result.get("path", command.state_path), "type": "state_saved"}
      case commands.LoadWeights():
        await call_backend(self.worker.load_from_state, command.model_id, command.state_path, command.restore_optimizer)
        return {"path": command.state_path, "type": "weights_loaded"}
      case commands.SaveWeightsForSampler():
        ref = command.path or command.sampling_session_id
        checkpoint = await call_backend(self.worker.save_for_sampler, command.model_id, command.alias, ref)
        if checkpoint:
          await self.publish_checkpoint(command.model_id, checkpoint)
        return {"path": command.path, "sampling_session_id": command.sampling_session_id, "type": "sampler_weights_saved"}
      case commands.Shutdown():
        return {"status": "ok", "type": "shutdown_acknowledged"}
      case _:
        raise NotImplementedError(f"Training request op {command.op!r} is not supported")

  async def publish_checkpoint(self, model_id: str, local_path: str) -> None:
    """Tell the samplers about a new checkpoint and drop the versions they no longer need."""
    if hasattr(self.store, "redis"):
      num_subs = await self.store.redis.publish(f"open_rl:weight_update:{model_id}", json.dumps({"weights_path": local_path}))
      print(f"[Trainer] Published weight update signal to {num_subs} subscribers for version path: {local_path}")
    older = older_versions(local_path, SAMPLER_VERSIONS_KEPT)
    for path in older:
      shutil.rmtree(path, ignore_errors=True)
    if older:
      print(f"[Trainer] Removed {len(older)} sampler weight versions older than the newest {SAMPLER_VERSIONS_KEPT}")


async def run_training_requests_processor(
  worker: TrainingBackend,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
  active_tenant_set_id: str | None = None,
  *,
  store: RequestStore | None = None,
) -> None:
  store = get_store() if store is None else store
  await TrainingRequestsProcessor(store, worker, model_id, active_tenant_set_id, time_slicer).run()


async def main_async(args: argparse.Namespace) -> None:
  from training.fft_trainer_worker import FFTTrainingWorker
  from training.lora_trainer_worker import LoraTrainingWorker

  fine_tuning_type = os.getenv("OPEN_RL_FINE_TUNING_TYPE") or ("full" if is_fft_enabled() else "lora")
  if args.model_id:
    metadata = await get_model_metadata(get_state_store(), args.model_id)
    if metadata is not None:
      fine_tuning_type = metadata.fine_tuning_type

  is_lora = fine_tuning_type == "lora"
  if not is_lora:
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the API server")
    if not args.model_id:
      raise RuntimeError("A dedicated trainer worker needs --model-id")
  print(f"-> Fine-Tuning Type: {fine_tuning_type} (Is LoRA: {is_lora})\n")

  worker: TrainingBackend
  if is_lora:
    worker = LoraTrainingWorker()
    if preload_target := os.getenv("BASE_MODEL"):
      worker.load_base_model(preload_target)
  else:
    # Allocation happens when create_model executes under its resource lease.
    worker = FFTTrainingWorker()

  if is_lora:
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      return {"status": "ready"}

    def run_probe_server():
      try:
        uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")
      except Exception as exc:
        print(f"[WORKER] Probe server on port 8000 skipped: {exc}")

    threading.Thread(target=run_probe_server, daemon=True).start()

  await run_training_requests_processor(
    worker,
    args.model_id,
    time_slicer=None if is_lora else time_slicer_client_from_env(),
    active_tenant_set_id=getattr(args, "active_tenant_set_id", None),
  )


def start_request_processing_loop() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", help="Model id whose per-model request queue this dedicated trainer worker drains.")
  parser.add_argument("--active-tenant-set-id", help="Active tenant rotation set ID for LoRA workers (e.g. Qwen/Qwen3-0.6B-1).")
  args = parser.parse_args()

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")

  asyncio.run(main_async(args))


if __name__ == "__main__":
  start_request_processing_loop()
