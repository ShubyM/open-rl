# This file contains the training request processor implementation for Open-RL.

import argparse
import asyncio
import json
import os
import shutil
import threading
import traceback
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_CLAIM, local_workload_name
from server.model_metadata import WeightSyncConfig, get_model_metadata
from server.store import RequestStore, get_state_store, get_store
from training import commands
from training.commands import parse_command
from training.trained_weights import park, unpark, write
from training.trainer_worker import TrainerWorker
from training.weight_delta import SparseWeightDelta

tracer = trace.get_tracer(__name__)


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def tmp_dir() -> str:
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


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

  With a time slicer the worker is dedicated to one model: it drains that
  model's queue under GPU leases and exits when the model is deleted. Without
  one it serves every model in a shared active set, as LoRA trainers do.
  """

  def __init__(
    self,
    store: RequestStore,
    worker: TrainerWorker,
    model_id: str | None = None,
    active_tenant_set_id: str | None = None,
    time_slicer: TimeSlicerClient | None = None,
  ):
    if time_slicer is not None:
      if not os.getenv("REDIS_URL"):
        raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the API server")
      if not model_id:
        raise RuntimeError("A dedicated trainer worker needs --model-id so it knows which per-model queue to drain")

    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.active_tenant_set_id = active_tenant_set_id
    self.time_slicer = time_slicer
    self.workload = workload_from_env(os.getpid(), name=local_workload_name("trainer", model_id), claim=TRAINER_CLAIM) if time_slicer else None
    self.snapshot_registered = False
    # Between time-slicer turns the worker's tensors are parked on the host.
    # FFTConfig.cpu_offload=False keeps them on the device instead.
    self.offload = True
    self.weight_sync = WeightSyncConfig.from_env()
    self.deltas: dict[str, SparseWeightDelta] = {}

  async def run(self) -> None:
    print(f"[WORKER] Training requests processor started (model={self.model_id} active_set={self.active_tenant_set_id}).")
    try:
      if self.time_slicer is not None:
        await self.time_slicer.register(self.workload)
        self.snapshot_registered = True
      while True:
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
    if unregister and self.snapshot_registered:
      try:
        await self.time_slicer.unregister(self.workload)
        self.snapshot_registered = False
      except Exception as exc:
        print(f"[WORKER] Failed to unregister: {exc}")
    try:
      await self.time_slicer.close()
    except Exception:
      pass
    os._exit(0)

  async def next_batch(self) -> list[dict[str, Any]]:
    if self.time_slicer is not None:
      return await self.store.get_requests_for_model(self.model_id)
    return await self.store.get_requests(active_set_id=self.active_tenant_set_id)

  async def run_once(self) -> None:
    batch = await self.next_batch()
    if not batch:
      await asyncio.sleep(0.1)
      return

    shutdown = any(req.get("op") == "shutdown_workers" for req in batch)
    work = [req for req in batch if req.get("op") != "shutdown_workers"]
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
    if failure is not None:
      raise failure

    if self.time_slicer is None:
      return
    if self.time_slicer.faulted:
      # This process still holds the accelerator. Exit without unregistering so
      # the grant moves on only once the memory is gone. Exit 0 keeps the pod
      # from restarting on fresh weights mid-run; the run fails on its next call.
      print(f"[WORKER] Time slicer could not park this process: {self.time_slicer.faulted}. Exiting to free the accelerator.")
      await self.exit_gracefully(unregister=False)
    if shutdown:
      await self.exit_gracefully()

  async def answer_batch(self, requests: list[dict[str, Any]]) -> tuple[list[tuple[str | None, dict[str, Any]]], Exception | None]:
    """Every request gets an answer: its result, or the failure that stopped the batch."""
    results: list[tuple[str | None, dict[str, Any]]] = []
    try:
      await self.handle_batch(requests, results)
    except Exception as exc:
      answered = {request_id for request_id, _ in results}
      for request in requests:
        request_id = request.get("request_id")
        if request_id and request_id not in answered:
          results.append((request_id, {"type": "RequestFailedResponse", "error_message": f"Trainer worker error: {exc}"}))
      return results, exc
    return results, None

  async def handle_batch(self, requests: list[dict[str, Any]], results: list[tuple[str | None, dict[str, Any]]]) -> None:
    """GPU work under one time-slicer turn; saves need the device only when the worker is not offloaded."""
    if self.time_slicer is None:
      # No lease to hold, so each answer goes out as soon as it is ready.
      for request in requests:
        await self.process_request(request)
      return

    save_ops = {"save_state", "save_weights_for_sampler"}
    gpu_reqs = [r for r in requests if r.get("op") not in save_ops]
    save_reqs = [r for r in requests if r.get("op") in save_ops]

    if gpu_reqs:
      async with self.time_slicer.acquire(self.workload):
        if self.offload:
          await asyncio.to_thread(unpark, self.worker.tensors())
        try:
          for request in gpu_reqs:
            results.append(await self.handle_request(request))
        finally:
          if self.offload:
            await asyncio.to_thread(park, self.worker.tensors())

    if not save_reqs:
      return
    if self.offload:
      for request in save_reqs:
        results.append(await self.handle_request(request))
    else:
      async with self.time_slicer.acquire(self.workload):
        for request in save_reqs:
          results.append(await self.handle_request(request))

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
        if not is_lora:
          self.offload = command.full_config.cpu_offload
          self.set_weight_sync_strategy(command.full_config.weight_sync_strategy)
        config = command.lora_config if is_lora else command.full_config
        await asyncio.to_thread(self.worker.create_model, command.base_model, command.model_id, command.lora_config if is_lora else None, config.seed)
        await asyncio.to_thread(self.track_new_weights, command.model_id)
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
        result = await asyncio.to_thread(self.worker.load_state, command.model_id, command.state_path, command.restore_optimizer)
        await asyncio.to_thread(self.track_new_weights, command.model_id)
        return {
          "base_model": result.get("base_model"),
          "model_id": result.get("model_id", command.model_id),
          "fine_tuning_type": command.fine_tuning_type,
          "type": "model_loaded_from_state",
        }
      case commands.ForwardBackward():
        result = await asyncio.to_thread(
          self.worker.forward_backward,
          command.model_id,
          command.data,
          command.loss_fn,
          command.loss_config,
          forward_only=command.forward_only,
        )
        result["type"] = "forward_backward_completed"
        return result
      case commands.OptimStep():
        result = await asyncio.to_thread(self.worker.optim_step, command.model_id, command.adam_params)
        await asyncio.to_thread(self.track_step, command.model_id)
        result["type"] = "optim_step_completed"
        return result
      case commands.Sample():
        raise NotImplementedError("Sampling from the trainer is not supported; sample through the vLLM sampler.")
      case commands.SaveState():
        await asyncio.to_thread(self.save_state, command.model_id, command.state_path, command.include_optimizer, command.kind)
        return {"path": command.state_path, "type": "state_saved"}
      case commands.LoadWeights():
        await asyncio.to_thread(self.worker.load_state, command.model_id, command.state_path, command.restore_optimizer)
        await asyncio.to_thread(self.track_new_weights, command.model_id)
        return {"path": command.state_path, "type": "weights_loaded"}
      case commands.SaveWeightsForSampler():
        ref = command.path or command.sampling_session_id
        checkpoint = await asyncio.to_thread(self.save_for_sampler, command.model_id, ref)
        if checkpoint:
          await self.publish_checkpoint(command.model_id, checkpoint)
        return {"path": command.path, "sampling_session_id": command.sampling_session_id, "type": "sampler_weights_saved"}
      case commands.Shutdown():
        return {"status": "ok", "type": "shutdown_acknowledged"}
      case _:
        raise NotImplementedError(f"Training request op {command.op!r} is not supported")

  # -- getting weights to the samplers --------------------------------------------------

  def set_weight_sync_strategy(self, strategy: str | None) -> None:
    if strategy is None:
      return
    if strategy not in ("full", "delta"):
      raise ValueError(f"Invalid weight_sync_strategy '{strategy}'. Must be 'full' or 'delta'.")
    self.weight_sync.strategy = strategy

  def track_new_weights(self, model_id: str) -> None:
    """A model_id just got weights. Export a LoRA adapter so the sampler can load it, or start diffing full weights."""
    trained = self.worker.models[model_id]
    if trained.lora_config is not None:
      self.export_adapter(model_id)
    elif self.weight_sync.strategy == "delta":
      self.deltas[model_id] = SparseWeightDelta(trained.weights, getattr(self.worker.base.module, "config", None), self.weight_sync.delta_format)

  def track_step(self, model_id: str) -> None:
    trained = self.worker.models[model_id]
    if trained.lora_config is not None:
      self.export_adapter(model_id)
    elif model_id in self.deltas:
      self.deltas[model_id].record(trained.weights)

  def export_adapter(self, model_id: str) -> None:
    """The LoRA sampler hot-loads adapters from this directory, so every change lands there."""
    write(self.worker.models[model_id], os.path.join(tmp_dir(), "peft", model_id), base_model=self.worker.base.name, model_id=model_id, kind="weights")

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool, kind: str) -> None:
    trained = self.worker.models[model_id]
    full = trained.lora_config is None
    write(
      trained,
      state_path,
      base_model=self.worker.base.name,
      model_id=model_id,
      with_adam=include_optimizer,
      kind=kind,
      hf_config=getattr(self.worker.base.module, "config", None) if full else None,
      tokenizer=self.worker.base.tokenizer if full else None,
    )
    print(f"Saved state for '{model_id}' to {state_path}")

  def save_for_sampler(self, model_id: str, ref: str | None) -> str | None:
    """Full weights go under a versioned path the samplers reload from. A LoRA
    adapter is hot-loaded from its export directory, so there is no checkpoint to announce."""
    if self.worker.models[model_id].lora_config is not None:
      self.export_adapter(model_id)
      return None
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    rel_path = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
    local_path = os.path.join(tmp_dir(), "sampler_full", rel_path)
    if model_id in self.deltas:
      self.deltas[model_id].write(local_path, base_model=self.worker.base.name, model_id=model_id)
    else:
      self.save_state(model_id, local_path, False, "sampler")
    return local_path

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
  worker: TrainerWorker,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
  active_tenant_set_id: str | None = None,
  *,
  store: RequestStore | None = None,
  time_sliced: bool = False,
) -> None:
  """time_sliced is how full fine-tuning trainers run, one model per process taking turns on the accelerator."""
  store = get_store() if store is None else store
  if time_sliced and time_slicer is None:
    time_slicer = time_slicer_client_from_env()
  await TrainingRequestsProcessor(store, worker, model_id, active_tenant_set_id, time_slicer).run()


async def main_async(args: argparse.Namespace) -> None:
  fine_tuning_type = os.getenv("OPEN_RL_FINE_TUNING_TYPE") or ("full" if is_fft_enabled() else "lora")
  if args.model_id:
    metadata = await get_model_metadata(get_state_store(), args.model_id)
    if metadata is not None:
      fine_tuning_type = metadata.fine_tuning_type

  is_lora = fine_tuning_type == "lora"
  print(f"-> Fine-Tuning Type: {fine_tuning_type} (Is LoRA: {is_lora})\n")

  worker = TrainerWorker()
  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target and is_lora:
    worker.load_base_model(preload_target)
    is_ready = True
  else:
    if not is_lora:
      print("[WORKER] Full fine-tuning mode loads its model from the create_model request.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if is_lora:
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    def run_probe_server():
      try:
        uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")
      except Exception as exc:
        print(f"[WORKER] Probe server on port 8000 skipped: {exc}")

    threading.Thread(target=run_probe_server, daemon=True).start()

  await run_training_requests_processor(
    worker,
    args.model_id,
    active_tenant_set_id=getattr(args, "active_tenant_set_id", None),
    time_sliced=not is_lora,
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
