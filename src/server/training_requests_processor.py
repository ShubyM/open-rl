# This file contains the training request processor implementation for Open-RL.

import asyncio
import json
import os
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any, Protocol

import chz
import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_TIME_SLICE_GROUP, workload_job_id
from server.protocol import SamplerSnapshot, TrainingCommand, TrainingOperation
from server.store import (
  InMemoryStore,
  RequestStore,
  bump_model_revision,
  get_model_revision,
  get_sampler_artifact,
  get_store,
  prune_sampler_snapshots,
  put_sampler_artifact,
  put_sampler_snapshot,
  report_control_event,
  sampler_revision_path,
)
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import Datum

tracer = trace.get_tracer(__name__)


TrainingWorker = FFTTrainingWorker | LoraTrainingWorker


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def parse_datum(raw: dict[str, Any]) -> Datum:
  """Convert Tinker wire-format datum with chunks to the flat Datum type."""
  tokens: list[int] = []
  for chunk in raw.get("model_input", {}).get("chunks", []):
    tokens.extend(chunk.get("tokens", []))

  loss_fn_inputs = {
    key: value if isinstance(value, dict) and "data" in value else {"data": value} for key, value in raw.get("loss_fn_inputs", {}).items()
  }
  return Datum(model_input=tokens, loss_fn_inputs=loss_fn_inputs)


def sampler_local_path(storage_ref: str) -> str:
  relative = storage_ref[len("tinker://") :] if storage_ref.startswith("tinker://") else storage_ref.lstrip("/")
  return os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_full", relative)


async def save_sampler_snapshot(
  store: RequestStore,
  worker: TrainingWorker,
  payload: dict[str, Any],
  model_id: str,
) -> dict[str, Any]:
  sampling_session_id = payload.get("sampling_session_id")
  if not sampling_session_id:
    raise ValueError("save_weights_for_sampler requires sampling_session_id")

  revision = await get_model_revision(store, model_id)
  storage_ref = await get_sampler_artifact(store, model_id, revision)
  checkpoint_created = storage_ref is None
  if storage_ref is None:
    storage_ref = sampler_revision_path(model_id, revision)
    if not payload.get("skip_checkpoint"):
      await asyncio.to_thread(worker.save_state, model_id, sampler_local_path(storage_ref), False, "sampler")
    else:
      print(f"[WORKER] Explicit mock sampler: skipping unused sampler checkpoint for {storage_ref}")
    await put_sampler_artifact(store, model_id, revision, storage_ref)
    if redis_client := getattr(store, "redis", None):
      local_path = sampler_local_path(storage_ref)
      subscribers = await redis_client.publish(
        f"open_rl:weight_update:{model_id}",
        json.dumps({"weights_path": local_path, "weights_revision": revision}),
      )
      print(f"[Trainer] Published weight update for revision {revision} to {subscribers} subscribers: {local_path}")

  ttl_seconds = payload.get("ttl_seconds")
  if ttl_seconds is not None:
    ttl_seconds = int(ttl_seconds)
    if ttl_seconds < 1:
      raise ValueError("ttl_seconds must be positive")
  now = time.time()
  named = bool(payload.get("alias"))
  snapshot = SamplerSnapshot(
    sampling_session_id=sampling_session_id,
    model_id=model_id,
    revision=revision,
    storage_path=storage_ref,
    named=named,
    created_at=now,
    expires_at=now + ttl_seconds if ttl_seconds is not None else None,
  )
  await put_sampler_snapshot(store, snapshot)
  for orphaned_ref in await prune_sampler_snapshots(store, model_id):
    orphaned_path = sampler_local_path(orphaned_ref)
    if os.path.isdir(orphaned_path):
      await asyncio.to_thread(shutil.rmtree, orphaned_path)

  return {
    "path": payload.get("path") if named else None,
    "sampling_session_id": sampling_session_id,
    "revision": revision,
    "checkpoint_created": checkpoint_created,
    "mock": bool(payload.get("skip_checkpoint")),
    "type": "sampler_weights_saved",
  }


class TrainingRequestsProcessor(Protocol):
  store: RequestStore

  async def process_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> None:
    request_id, result = await self.handle_request(raw_request, model_id)
    if request_id is not None:
      await self.store.set_future(request_id, result)

  async def handle_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> tuple[str | None, dict[str, Any]]:
    request_id = raw_request.get("request_id")
    resolved_model_id = model_id or raw_request.get("model_id") or "default"
    token = None

    try:
      command = TrainingCommand.model_validate(raw_request)
      op = command.op
      request_id = command.request_id
      resolved_model_id = model_id or command.model_id or "default"
      carrier = command.trace_context
      ctx = propagate.extract(carrier) if carrier else None
      token = otel_context.attach(ctx) if ctx else None

      started = time.monotonic()
      await report_control_event(
        self.store,
        resolved_model_id,
        component="trainer",
        phase=op,
        status="running",
        message=f"Running {op.replace('_', ' ')}",
        details={"request_id": request_id, "operation": op},
      )
      result = await self.dispatch_operation(op, command.payload, resolved_model_id)
      if op == "optim_step":
        result["revision"] = await bump_model_revision(self.store, resolved_model_id)
      elapsed = time.monotonic() - started
      details: dict[str, Any] = {"request_id": request_id, "operation": op}
      if isinstance(result.get("metrics"), dict):
        details["metrics"] = result["metrics"]
      if result.get("mock") is True:
        details["mock"] = True
      await report_control_event(
        self.store,
        resolved_model_id,
        component="trainer",
        phase=f"{op}_complete",
        status="ready",
        message=f"Completed {op.replace('_', ' ')} in {elapsed:.1f}s",
        duration_seconds=elapsed,
        details=details,
      )
      return request_id, result
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
      await report_control_event(
        self.store,
        resolved_model_id,
        component="trainer",
        phase="operation_failed",
        status="failed",
        level="error",
        message=str(exc),
        details={"request_id": request_id, "operation": raw_request.get("op")},
      )
      return request_id, {"type": "RequestFailedResponse", "error_message": str(exc)}
    finally:
      if token:
        otel_context.detach(token)

  async def dispatch_operation(self, op: str, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    match op:
      case "create_model":
        return await self.create_model(payload, model_id)
      case "create_model_from_state":
        return await self.create_model_from_state(payload, model_id)
      case "forward_backward":
        return await self.forward_backward(payload, model_id)
      case "optim_step":
        return await self.optim_step(payload, model_id)
      case "sample":
        return await self.sample(payload, model_id)
      case "save_state":
        return await self.save_state(payload, model_id)
      case "load_weights":
        return await self.load_weights(payload, model_id)
      case "save_weights_for_sampler":
        return await self.save_weights_for_sampler(payload, model_id)
      case "save_weights":
        return await self.save_weights(payload, model_id)
      case _:
        raise NotImplementedError(f"Training request op {op!r} is not supported")

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def sample(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def save_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]: ...


async def _fetch_model_meta(
  store: RequestStore,
  model_id: str,
  payload: dict[str, Any],
  default_kind: str = "full",
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
  val = None
  if hasattr(store, "get_value"):
    try:
      val = await store.get_value(f"open_rl:model_meta:{model_id}")
    except Exception:
      pass
  if val:
    try:
      meta = json.loads(val) if isinstance(val, str) else val
      if isinstance(meta, dict):
        base_model = meta.get("base_model") or payload.get("base_model") or ""
        full_config = meta.get("full_config") or payload.get("full_config") or {}
        lora_config = meta.get("lora_config") or payload.get("lora_config") or {}
        training_kind = meta.get("training_kind") or ("lora" if "lora_config" in meta or "lora_config" in payload else default_kind)
        return base_model, full_config, lora_config, training_kind
    except Exception:
      pass
  return (
    payload.get("base_model", ""),
    payload.get("full_config") or {},
    payload.get("lora_config") or {},
    "lora" if "lora_config" in payload or default_kind == "lora" else "full",
  )


class LoraTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(self, store: RequestStore, worker: LoraTrainingWorker):
    self.store = store
    self.worker = worker

  async def run(self) -> None:
    print("[WORKER] LoRA training requests processor started.")

    while True:
      try:
        await self.run_once()
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in training requests processor: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)

  async def run_once(self) -> None:
    batch = await self.store.get_requests()
    if not batch:
      await asyncio.sleep(0.1)
      return

    model_id = batch[0].get("model_id", "default")

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", model_id)

      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {model_id}")
      for request in batch:
        await self.process_request(request, model_id)

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, raw_config, training_kind = await _fetch_model_meta(self.store, model_id, payload, default_kind="lora")
    lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, lora_config)
    return {
      "base_model": base_model,
      "model_id": model_id,
      "rank": lora_config.rank,
      "training_kind": training_kind,
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, training_kind = await _fetch_model_meta(self.store, model_id, payload, default_kind="lora")
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model") or base_model,
      "model_id": result.get("model_id", model_id),
      "training_kind": training_kind,
      "type": "model_loaded_from_state",
    }

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    typed_data = [parse_datum(item) for item in payload.get("data", [])]
    result = await asyncio.to_thread(
      self.worker.forward_backward,
      typed_data,
      payload.get("loss_fn", "cross_entropy"),
      payload.get("loss_config"),
      model_id,
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
    return result

  async def sample(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.generate,
      payload.get("prompt_tokens", []),
      payload.get("max_tokens", 20),
      payload.get("num_samples", 1),
      payload.get("temperature", 0.0),
      model_id,
      bool(payload.get("prompt_logprobs", False)),
    )
    result["type"] = "sample_completed"
    return result

  async def save_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.save_state,
      model_id,
      payload["state_path"],
      bool(payload.get("include_optimizer", False)),
      payload.get("kind", "state"),
    )
    return {"path": result.get("path", payload["state_path"]), "type": "state_saved"}

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {"path": payload["state_path"], "type": "weights_loaded"}

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    return await save_sampler_snapshot(self.store, self.worker, payload, model_id)

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"))
    return {"status": "ok", "type": "weights_saved"}


class TrainingRequestFailed(RuntimeError):
  """Raised when an in-process training command returns a failed response."""


class InProcessLoraBackend:
  """Exercise the LoRA training boundary without starting infrastructure.

  The HTTP gateway and Redis worker loop are transports around the same
  ``TrainingCommand`` contract. Local behavior tests can use this backend to
  run that contract against the real worker in one Python process.
  """

  def __init__(self, store: RequestStore | None = None, worker: LoraTrainingWorker | None = None):
    self.store = store or InMemoryStore()
    self.worker = worker or LoraTrainingWorker()
    self.processor = LoraTrainingRequestsProcessor(self.store, self.worker)
    self.request_sequence = 0

  async def request(
    self,
    op: TrainingOperation,
    payload: dict[str, Any] | None = None,
    model_id: str = "default",
  ) -> dict[str, Any]:
    self.request_sequence += 1
    command = TrainingCommand(
      request_id=f"local:{model_id}:{self.request_sequence}",
      model_id=model_id,
      op=op,
      payload=payload or {},
    )
    _, response = await self.processor.handle_request(command.model_dump(), model_id)
    if response.get("type") == "RequestFailedResponse":
      raise TrainingRequestFailed(response.get("error_message", "Training request failed"))
    return response


class FFTTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: FFTTrainingWorker,
    model_id: str | None,
    time_slicer: TimeSlicerClient,
  ):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the gateway")
    if not model_id:
      raise RuntimeError("A dedicated trainer worker needs model_id=<id> so it knows which per-model queue to drain")

    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.workload = workload_from_env(os.getpid(), job_id=workload_job_id("trainer", model_id), group=TRAINER_TIME_SLICE_GROUP)
    self.time_slicer = time_slicer
    self.snapshot_registered = False

  async def exit_gracefully(self) -> None:
    print(f"[WORKER] Initiating immediate exit for model {self.model_id} trainer worker...")
    if self.snapshot_registered:
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

  async def run(self) -> None:
    print("[WORKER] Full fine-tuning training requests processor started.")

    try:
      await report_control_event(
        self.store,
        self.model_id,
        component="trainer",
        phase="registering",
        status="starting",
        message="Trainer process started; registering with the accelerator scheduler",
      )
      await self.time_slicer.register(self.workload)
      self.snapshot_registered = True
      await report_control_event(
        self.store,
        self.model_id,
        component="trainer",
        phase="idle",
        status="ready",
        message="Trainer is registered and waiting for work",
      )
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
      try:
        if self.snapshot_registered:
          await self.time_slicer.unregister(self.workload)
      finally:
        await self.time_slicer.close()

  async def run_once(self) -> None:
    batch = await self.store.get_requests_for_model(self.model_id)
    if not batch:
      await asyncio.sleep(0.1)
      return

    has_shutdown = False
    training_reqs = []
    for req in batch:
      if req.get("request_id") == "SHUTDOWN_SENTINEL" or req.get("op") in {"shutdown", "shutdown_workers"}:
        has_shutdown = True
      else:
        training_reqs.append(req)

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(training_reqs))
      batch_span.set_attribute("model_id", self.model_id)

      if training_reqs:
        print(f"\n[TRAINING REQUESTS] Popped {len(training_reqs)} requests for model: {self.model_id}")
        results = []
        save_ops = {"save_state", "save_weights", "save_weights_for_sampler"}
        request_index = 0
        while request_index < len(training_reqs):
          request = training_reqs[request_index]
          if request.get("op") in save_ops:
            if hasattr(self.worker, "cpu_offload") and not self.worker.cpu_offload:
              async with self.time_slicer.acquire(self.workload):
                results.append(await self.handle_request(request, self.model_id))
            else:
              results.append(await self.handle_request(request, self.model_id))
            request_index += 1
            continue

          gpu_reqs = []
          while request_index < len(training_reqs) and training_reqs[request_index].get("op") not in save_ops:
            gpu_reqs.append(training_reqs[request_index])
            request_index += 1

          lease_started = time.monotonic()
          await report_control_event(
            self.store,
            self.model_id,
            component="trainer",
            phase="waiting_for_gpu",
            status="waiting",
            message="Waiting for an accelerator time slice",
            details={"queued_operations": [queued.get("op") for queued in gpu_reqs]},
          )
          async with self.time_slicer.acquire(self.workload):
            lease_wait = time.monotonic() - lease_started
            await report_control_event(
              self.store,
              self.model_id,
              component="trainer",
              phase="gpu_acquired",
              status="running",
              message=f"Accelerator acquired after {lease_wait:.1f}s",
              duration_seconds=lease_wait,
            )
            include_optimizer = any(queued.get("op") == "optim_step" for queued in gpu_reqs)
            await self.transition_worker("wake_up", lambda include_optimizer=include_optimizer: self.worker.wake_up(include_optimizer))
            try:
              for gpu_request in gpu_reqs:
                results.append(await self.handle_request(gpu_request, self.model_id))
            finally:
              await self.transition_worker("sleep", self.worker.sleep)

        for request_id, result in results:
          if request_id is not None:
            await self.store.set_future(request_id, result)

    if has_shutdown:
      await self.exit_gracefully()

  async def transition_worker(self, phase: str, transition: Callable[[], None]) -> None:
    started = time.monotonic()
    await report_control_event(
      self.store,
      self.model_id,
      component="trainer",
      phase=phase,
      status="running",
      message=f"Trainer {phase.replace('_', ' ')}",
    )
    await asyncio.to_thread(transition)
    elapsed = time.monotonic() - started
    await report_control_event(
      self.store,
      self.model_id,
      component="trainer",
      phase=f"{phase}_complete",
      status="running",
      message=f"Trainer {phase.replace('_', ' ')} completed in {elapsed:.1f}s",
      duration_seconds=elapsed,
    )

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, raw_config, _, training_kind = await _fetch_model_meta(self.store, model_id, payload, default_kind="full")
    full_config = FFTConfig(**{k: v for k, v in raw_config.items() if k in FFTConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, full_config)
    return {
      "base_model": base_model,
      "model_id": model_id,
      "training_kind": training_kind,
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, training_kind = await _fetch_model_meta(self.store, model_id, payload, default_kind="full")
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model") or base_model,
      "model_id": result.get("model_id", model_id),
      "training_kind": training_kind,
      "type": "model_loaded_from_state",
    }

  async def forward_backward(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    typed_data = [parse_datum(item) for item in payload.get("data", [])]
    result = await asyncio.to_thread(
      self.worker.forward_backward,
      typed_data,
      payload.get("loss_fn", "cross_entropy"),
      payload.get("loss_config"),
      model_id,
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
    return result

  async def sample(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.generate,
      payload.get("prompt_tokens", []),
      payload.get("max_tokens", 20),
      payload.get("num_samples", 1),
      payload.get("temperature", 0.0),
      model_id,
      bool(payload.get("prompt_logprobs", False)),
    )
    result["type"] = "sample_completed"
    return result

  async def save_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(
      self.worker.save_state,
      model_id,
      payload["state_path"],
      bool(payload.get("include_optimizer", False)),
      payload.get("kind", "state"),
    )
    return {"path": result.get("path", payload["state_path"]), "type": "state_saved"}

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {"path": payload["state_path"], "type": "weights_loaded"}

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    return await save_sampler_snapshot(self.store, self.worker, payload, model_id)

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_model, payload.get("alias") or model_id)
    return {"status": "ok", "type": "weights_saved"}


async def run_training_requests_processor(
  worker: TrainingWorker,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
) -> None:
  store = get_store()
  if isinstance(worker, FFTTrainingWorker):
    time_slicer = time_slicer or time_slicer_client_from_env()
    processor = FFTTrainingRequestsProcessor(store, worker, model_id, time_slicer)
  else:
    processor = LoraTrainingRequestsProcessor(store, worker)
  await processor.run()


@chz.chz
class WorkerArgs:
  model_id: str | None = chz.field(default=None, doc="Model id whose dedicated request queue this worker drains.")


def start_request_processing_loop() -> None:
  args = chz.entrypoint(WorkerArgs, allow_hyphens=True)

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> FFT enabled: {is_fft_enabled()}\n")

  worker: TrainingWorker = FFTTrainingWorker() if is_fft_enabled() else LoraTrainingWorker()
  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target and not is_fft_enabled():
    worker.load_base_model(preload_target)
    is_ready = True
  else:
    if is_fft_enabled():
      print("[WORKER] Full fine-tuning mode loads its model from the create_model request.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if not is_fft_enabled():
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    def run_probe_server():
      uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")

    threading.Thread(target=run_probe_server, daemon=True).start()
  asyncio.run(run_training_requests_processor(worker, args.model_id))


if __name__ == "__main__":
  start_request_processing_loop()
