# This file contains the training request processor implementation for Open-RL.

import argparse
import asyncio
import json
import os
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Protocol

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_TIME_SLICE_GROUP, workload_job_id
from server.store import RequestStore, get_store
from training import paths
from training.distributed import barrier, broadcast_object, is_distributed, is_primary, local_rank
from training.distributed import close as close_distributed
from training.distributed import initialize as initialize_distributed
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.megatron_worker import MegatronTrainingWorker
from training.trainer_worker import Datum

tracer = trace.get_tracer(__name__)


TrainingWorker = FFTTrainingWorker | LoraTrainingWorker | MegatronTrainingWorker

# Full-parameter backends. Both load their model from create_model rather than
# BASE_MODEL, both run one torchrun process per GPU, and both take the
# time-sliced GPU lease -- everything start_request_processing_loop and
# run_training_requests_processor branch on.
FULL_PARAMETER_WORKERS = (FFTTrainingWorker, MegatronTrainingWorker)


def trainer_backend() -> str:
  """Which trainer worker to run: "lora", "fft", or "megatron"."""
  backend = os.getenv("OPEN_RL_TRAINER_BACKEND", "").lower()
  if backend:
    if backend not in ("lora", "fft", "megatron"):
      raise RuntimeError(f"Unknown OPEN_RL_TRAINER_BACKEND={backend!r}; expected lora, fft, or megatron")
    return backend
  return "fft" if is_fft_enabled() else "lora"


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


class TrainingRequestsProcessor(Protocol):
  store: RequestStore

  async def process_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> None:
    request_id, result = await self.handle_request(raw_request, model_id)
    if request_id is not None and is_primary():
      await self.store.set_future(request_id, result)

  async def handle_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> tuple[str | None, dict[str, Any]]:
    request_id = raw_request.get("request_id")
    token = None

    try:
      op = raw_request["op"]
      request_id = raw_request["request_id"]
      resolved_model_id = model_id or raw_request.get("model_id") or "default"

      carrier = raw_request.get("trace_context")
      ctx = propagate.extract(carrier) if carrier else None
      token = otel_context.attach(ctx) if ctx else None

      result = await self.dispatch_operation(op, raw_request.get("payload", {}), resolved_model_id)
      return request_id, result
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
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
      case "shutdown_workers":
        return {"status": "ok", "type": "shutdown_acknowledged"}
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
        fine_tuning_type = meta.get("fine_tuning_type") or ("lora" if "lora_config" in meta or "lora_config" in payload else default_kind)
        return base_model, full_config, lora_config, fine_tuning_type
    except Exception:
      pass
  return (
    payload.get("base_model", ""),
    payload.get("full_config") or {},
    payload.get("lora_config") or {},
    "lora" if "lora_config" in payload or default_kind == "lora" else "full",
  )


class LoraTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: LoraTrainingWorker,
    model_id: str | None = None,
    active_tenant_set_id: str | None = None,
  ):
    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.active_tenant_set_id = active_tenant_set_id or (f"{model_id}-1" if model_id else None)

  async def run(self) -> None:
    print(f"[WORKER] LoRA training requests processor started (Active Set ID: {self.active_tenant_set_id}).")

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
    batch = await self.store.get_requests(active_set_id=self.active_tenant_set_id) if is_primary() else None
    if is_distributed():
      # Every rank must execute the same request sequence (collectives are
      # positional); rank 0 owns the queue and fans the batch out.
      batch = await asyncio.to_thread(broadcast_object, batch)
    if not batch:
      if is_primary():
        await asyncio.sleep(0.1)
      return

    model_id = batch[0].get("model_id", "default")

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", model_id)

      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {model_id}")
      for request in batch:
        target_model_id = request.get("adapter_id") or request.get("model_id") or model_id
        await self.process_request(request, target_model_id)

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, raw_config, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="lora")
    lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, lora_config)
    return {
      "base_model": base_model,
      "model_id": model_id,
      "rank": lora_config.rank,
      "fine_tuning_type": fine_tuning_type,
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="lora")
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model") or base_model,
      "model_id": result.get("model_id", model_id),
      "fine_tuning_type": fine_tuning_type,
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
      bool(payload.get("forward_only")),
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
    await asyncio.to_thread(self.worker.save_adapter, model_id)
    if hasattr(self, "store") and self.store:
      try:
        raw_meta = await self.store.get_value(f"open_rl:model_meta:{model_id}")
        current_step = json.loads(raw_meta).get("total_steps_completed", 0) if raw_meta else 0
        await self.store.update_job_metadata(model_id, {"total_steps_completed": current_step + 1, "updated_at": time.time()})
      except Exception as exc:
        print(f"[PROCESSOR] Failed to update step metadata for model {model_id}: {exc}")
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
    return {"path": payload.get("public_path") or result.get("path", payload["state_path"]), "type": "state_saved"}

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {"path": payload["state_path"], "type": "weights_loaded"}

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    # The session ref's last segment (e.g. "sampler-<seq>") names this
    # snapshot's immutable adapter directory.
    session_id = payload.get("sampling_session_id") or ""
    session_label = session_id.rsplit("/", 1)[-1] or None
    await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"), session_label)
    return {
      "path": payload.get("path"),
      "sampling_session_id": payload.get("sampling_session_id"),
      "type": "sampler_weights_saved",
    }

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"))
    return {"status": "ok", "type": "weights_saved"}


class FFTTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: FFTTrainingWorker | MegatronTrainingWorker,
    model_id: str | None,
    time_slicer: TimeSlicerClient | None,
  ):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the gateway")

    self.store = store
    self.worker = worker
    # Two deployments, distinguished by whether --model-id was given. With one,
    # this is a per-model worker the gateway launched and it drains that model's
    # queue. Without one -- how launch_work.sh starts the Megatron trainer -- it
    # is the only trainer on the box, so it drains the shared round-robin queue
    # like the LoRA worker does and takes whatever model_id the requests carry.
    # Nothing else changes: the per-model launcher is off in that mode
    # (gateway.gateway_launches_trainers), so there is no second trainer to
    # contend with, and the time slicer is a no-op on a box it owns outright.
    self.model_id = model_id
    self.workload = workload_from_env(
      os.getpid(), job_id=workload_job_id("trainer", model_id or "shared"), group=TRAINER_TIME_SLICE_GROUP
    )
    self.time_slicer = time_slicer
    self.snapshot_registered = False

  async def exit_gracefully(self) -> None:
    print(f"[WORKER] Initiating immediate exit for model {self.model_id} trainer worker...")
    if self.snapshot_registered and self.time_slicer is not None:
      try:
        await self.time_slicer.unregister(self.workload)
        self.snapshot_registered = False
      except Exception as exc:
        print(f"[WORKER] Failed to unregister: {exc}")
    try:
      if self.time_slicer is not None:
        await self.time_slicer.close()
    except Exception:
      pass
    os._exit(0)

  async def run(self) -> None:
    print("[WORKER] Full fine-tuning training requests processor started.")

    try:
      if is_primary():
        assert self.time_slicer is not None
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
      try:
        if self.snapshot_registered and self.time_slicer is not None:
          await self.time_slicer.unregister(self.workload)
      finally:
        if self.time_slicer is not None:
          await self.time_slicer.close()
        close_distributed()

  async def run_once(self) -> None:
    if is_primary():
      batch = await (
        self.store.get_requests_for_model(self.model_id) if self.model_id else self.store.get_requests()
      )
    else:
      batch = None
    if is_distributed():
      batch = await asyncio.to_thread(broadcast_object, batch)
    if not batch:
      if is_primary():
        await asyncio.sleep(0.1)
      return

    # get_requests() batches one tenant at a time, so a batch is single-model
    # either way and the ops below can share one id.
    model_id = self.model_id or batch[0].get("model_id", "default")
    has_shutdown = False
    training_reqs = []
    for req in batch:
      if req.get("request_id") == "SHUTDOWN_SENTINEL" or req.get("op") in {"shutdown", "shutdown_workers"}:
        has_shutdown = True
      else:
        training_reqs.append(req)

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(training_reqs))
      batch_span.set_attribute("model_id", model_id)

      if training_reqs:
        print(f"\n[TRAINING REQUESTS] Popped {len(training_reqs)} requests for model: {model_id}")
        results = []
        # FSDP checkpoint consolidation is collective GPU work, so distributed
        # saves remain inside the trainer lease. Single-GPU CPU-offloaded saves
        # retain the cheaper outside-lease behavior.
        save_ops = set() if is_distributed() else {"save_state", "save_weights", "save_weights_for_sampler"}
        gpu_reqs = [r for r in training_reqs if r.get("op") not in save_ops]
        save_reqs = [r for r in training_reqs if r.get("op") in save_ops]

        if gpu_reqs:
          lease_started = time.monotonic()
          async with self.gpu_lease():
            lease_wait = time.monotonic() - lease_started
            batch_span.set_attribute("gpu_lease_wait_seconds", lease_wait)
            print(f"[TIMING] model={model_id} phase=gpu_lease_wait duration={lease_wait:.3f}s")
            if hasattr(self.worker, "wake_up"):
              await self.transition_worker("wake_up", self.worker.wake_up)
            try:
              for request in gpu_reqs:
                results.append(await self.handle_request(request, model_id))
            finally:
              if hasattr(self.worker, "sleep"):
                await self.transition_worker("sleep", self.worker.sleep)

        if hasattr(self.worker, "cpu_offload") and not self.worker.cpu_offload and save_reqs:
          async with self.time_slicer.acquire(self.workload):
            for request in save_reqs:
              results.append(await self.handle_request(request, model_id))
        else:
          for request in save_reqs:
            results.append(await self.handle_request(request, model_id))

        for request_id, result in results:
          if is_primary() and request_id is not None:
            await self.store.set_future(request_id, result)

    if has_shutdown:
      await self.exit_gracefully()

  @asynccontextmanager
  async def gpu_lease(self):
    lease = None
    if is_primary():
      assert self.time_slicer is not None
      lease = self.time_slicer.acquire(self.workload)
      await lease.__aenter__()
    await asyncio.to_thread(barrier)
    try:
      yield
    finally:
      await asyncio.to_thread(barrier)
      if lease is not None:
        await lease.__aexit__(None, None, None)

  async def transition_worker(self, phase: str, transition: Callable[..., None], *args: Any) -> None:
    started = time.monotonic()
    with tracer.start_as_current_span(f"training.{phase}") as span:
      await asyncio.to_thread(transition, *args)
      elapsed = time.monotonic() - started
      span.set_attribute("duration_seconds", elapsed)
      span.set_attribute("model_id", self.model_id or "shared")
      print(f"[TIMING] model={self.model_id or 'shared'} phase={phase} duration={elapsed:.3f}s")

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, raw_config, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="full")
    config_class = type(self.worker).config_class
    full_config = config_class(**{k: v for k, v in raw_config.items() if k in config_class.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, full_config)
    return {
      "base_model": base_model,
      "model_id": model_id,
      "fine_tuning_type": fine_tuning_type,
      "type": "model_created",
    }

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind="full")
    result = await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {
      "base_model": result.get("base_model") or base_model,
      "model_id": result.get("model_id", model_id),
      "fine_tuning_type": fine_tuning_type,
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
      bool(payload.get("forward_only")),
    )
    result["type"] = "forward_backward_completed"
    return result

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    result["type"] = "optim_step_completed"
    if hasattr(self, "store") and self.store:
      try:
        raw_meta = await self.store.get_value(f"open_rl:model_meta:{model_id}")
        current_step = json.loads(raw_meta).get("total_steps_completed", 0) if raw_meta else 0
        await self.store.update_job_metadata(model_id, {"total_steps_completed": current_step + 1, "updated_at": time.time()})
      except Exception as exc:
        print(f"[PROCESSOR] Failed to update step metadata for model {model_id}: {exc}")
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
    return {"path": payload.get("public_path") or result.get("path", payload["state_path"]), "type": "state_saved"}

  async def load_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(
      self.worker.load_from_state,
      model_id,
      payload["state_path"],
      bool(payload.get("restore_optimizer", False)),
    )
    return {"path": payload["state_path"], "type": "weights_loaded"}

  async def save_weights_for_sampler(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    ref = payload.get("path") or payload.get("sampling_session_id")
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    if self.worker.publishes_sampler_adapter():
      # A LoRA-trained full-parameter worker (the Megatron backend) publishes
      # the adapter alone, into the same peft/<model>/<label> layout the LoRA
      # processor above writes and gateway.sampler_adapter_path resolves.
      session_label = (payload.get("sampling_session_id") or "").rsplit("/", 1)[-1] or None
      await asyncio.to_thread(self.worker.write_adapter, model_id, payload.get("alias"), session_label)
    else:
      rel_path = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
      local_path = os.path.join(paths.tmp_dir(), "sampler_full", rel_path)
      await asyncio.to_thread(self.worker.save_state, model_id, local_path, False, "sampler")
      if hasattr(self.store, "redis"):
        num_subs = await self.store.redis.publish(
          f"open_rl:weight_update:{model_id}",
          json.dumps({"weights_path": local_path}),
        )
        print(f"[Trainer] Published weight update signal to {num_subs} subscribers for version path: {local_path}")
    return {
      "path": payload.get("path"),
      "sampling_session_id": payload.get("sampling_session_id"),
      "type": "sampler_weights_saved",
    }

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    await asyncio.to_thread(self.worker.save_model, payload.get("alias") or model_id)
    return {"status": "ok", "type": "weights_saved"}


def pin_worker_threads_to_this_rank() -> None:
  """Give every executor thread this rank's CUDA device.

  Torch's current device is thread-local, and every worker call here is handed
  to a thread with asyncio.to_thread. set_device() is only reached once, deep
  inside create_model, so exactly one pool thread -- whichever served that
  request -- ends up pointing at this rank's GPU. Any thread the pool spawns
  afterwards still points at cuda:0, and a device-less allocation on one of them
  lands there: correct on rank 0, wrong on every other rank.

  run33 died of this. Four steps ran on the single warm thread that had run
  create_model; the pool then grew a second thread and rank 2 hit
  `cuda:0 and cuda:2` in a forward. The cuda:0 buffer then reached a NCCL
  communicator bound to cuda:2, which is an illegal access, and the watchdog
  aborted the rank -- so it reads as a NCCL fault rather than a device bug.
  """
  if not torch.cuda.is_available():
    return
  device = local_rank()
  torch.cuda.set_device(device)
  # An initializer, not a call at each entry point: the pool creates threads
  # lazily and on demand, so the only place guaranteed to run once per thread is
  # the thread's own startup.
  asyncio.get_running_loop().set_default_executor(
    ThreadPoolExecutor(thread_name_prefix="trainer-worker", initializer=torch.cuda.set_device, initargs=(device,))
  )


async def run_training_requests_processor(
  worker: TrainingWorker,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
  active_tenant_set_id: str | None = None,
) -> None:
  pin_worker_threads_to_this_rank()
  store = get_store()
  if isinstance(worker, FULL_PARAMETER_WORKERS):
    time_slicer = (time_slicer or time_slicer_client_from_env()) if is_primary() else None
    processor = FFTTrainingRequestsProcessor(store, worker, model_id, time_slicer)
  else:
    processor = LoraTrainingRequestsProcessor(store, worker, model_id, active_tenant_set_id)
  await processor.run()


async def main_async(args: argparse.Namespace) -> None:
  fine_tuning_type = os.getenv("OPEN_RL_FINE_TUNING_TYPE") or ("full" if is_fft_enabled() else "lora")
  if args.model_id:
    try:
      store = get_store()
      raw_meta = await store.get_value(f"open_rl:model_meta:{args.model_id}")
      if raw_meta:
        meta_dict = json.loads(raw_meta)
        fine_tuning_type = meta_dict.get("fine_tuning_type", fine_tuning_type)
    except Exception as exc:
      print(f"[WORKER] Failed to fetch model metadata for {args.model_id}: {exc}")

  is_lora = fine_tuning_type == "lora"
  # OPEN_RL_TRAINER_BACKEND overrides the metadata-derived choice (it is how
  # launch_work.sh selects Megatron); otherwise the fine-tuning type decides.
  backend = trainer_backend() if os.getenv("OPEN_RL_TRAINER_BACKEND") else ("lora" if is_lora else "fft")
  print(f"-> Fine-Tuning Type: {fine_tuning_type} (Is LoRA: {is_lora}), trainer backend: {backend}\n")

  worker: TrainingWorker = {
    "megatron": MegatronTrainingWorker,
    "fft": FFTTrainingWorker,
    "lora": LoraTrainingWorker,
  }[backend]()
  full_parameter = isinstance(worker, FULL_PARAMETER_WORKERS)
  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target and not full_parameter:
    worker.load_base_model(preload_target)
    is_ready = True
  else:
    if full_parameter:
      print(f"[WORKER] {backend} mode loads its model from the create_model request.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if not full_parameter and is_primary():
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    # Configurable so a dedicated trainer can coexist with a vLLM server on
    # the same box (both defaulted to 8000); non-primary ranks skip the probe
    # entirely or every rank would fight over the port.
    probe_port = int(os.getenv("OPEN_RL_WORKER_PROBE_PORT", "8000"))

    def run_probe_server():
      try:
        uvicorn.run(probe_app, host="0.0.0.0", port=probe_port, log_level="warning")
      except Exception as exc:
        print(f"[WORKER] Probe server on port {probe_port} skipped: {exc}")

    threading.Thread(target=run_probe_server, daemon=True).start()

  await run_training_requests_processor(
    worker,
    args.model_id,
    active_tenant_set_id=getattr(args, "active_tenant_set_id", None),
  )


def start_request_processing_loop() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", help="Model id whose per-model request queue this dedicated trainer worker drains.")
  parser.add_argument("--active-tenant-set-id", help="Active tenant rotation set ID for LoRA workers (e.g. Qwen/Qwen3-0.6B-1).")
  args = parser.parse_args()
  initialize_distributed()

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> {paths.describe_roots()}")

  asyncio.run(main_async(args))


if __name__ == "__main__":
  start_request_processing_loop()
