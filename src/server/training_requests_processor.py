# This file contains the training request processor implementation for Open-RL.

import argparse
import asyncio
import json
import os
import threading
import time
import traceback
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace

from accel_timeslicer.time_slicer import NoOpTimeSlicer, TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import TRAINER_CLAIM, local_workload_name
from server.store import RequestStore, get_store
from training.distributed import barrier, broadcast_object, is_distributed, is_primary, pin_executor_threads
from training.distributed import close as close_distributed
from training.distributed import initialize as initialize_distributed
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import BaseTrainerWorker, Datum, tmp_dir

tracer = trace.get_tracer(__name__)


TrainingWorker = BaseTrainerWorker


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


def describe_requests(batch: list[dict[str, Any]]) -> str:
  """`op:request_id` per request, matching the gateway's enqueue log line."""
  return ", ".join(f"{r.get('op')}:{r.get('request_id')}" for r in batch)


class TrainingRequestsProcessor:
  store: RequestStore
  worker: TrainingWorker
  default_kind: str = "full"

  async def run_loop(self) -> None:
    while True:
      try:
        await self.run_once()
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in training requests processor: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)

  async def run_once(self) -> None: ...

  async def process_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> None:
    request_id, result = await self.handle_request(raw_request, model_id)
    await self.publish_result(request_id, result)

  async def publish_result(self, request_id: str | None, result: dict[str, Any]) -> None:
    if request_id is not None and is_primary():
      await self.store.set_future(request_id, result)

  async def next_batch(self, fetch: Callable[[], Awaitable[list[dict[str, Any]] | None]]) -> list[dict[str, Any]] | None:
    """The next request batch, identical on every rank."""
    batch = await fetch() if is_primary() else None
    if is_distributed():
      batch = await asyncio.to_thread(broadcast_object, batch)
    if not batch and is_primary():
      await asyncio.sleep(0.1)
    return batch or None

  async def bump_step_count(self, model_id: str) -> None:
    if not is_primary():
      return
    try:
      raw_meta = await self.store.get_value(f"open_rl:model_meta:{model_id}")
      current_step = json.loads(raw_meta).get("total_steps_completed", 0) if raw_meta else 0
      await self.store.update_job_metadata(model_id, {"total_steps_completed": current_step + 1, "updated_at": time.time()})
    except Exception as exc:
      print(f"[PROCESSOR] Failed to update step metadata for model {model_id}: {exc}")

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

      return request_id, await self.dispatch_operation(op, raw_request.get("payload", {}), resolved_model_id)
    except Exception as exc:
      traceback.print_exc()
      if request_id is None:
        raise
      return request_id, {"type": "RequestFailedResponse", "error_message": str(exc)}
    finally:
      if token:
        otel_context.detach(token)

  async def dispatch_operation(self, op: str, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    if op == "shutdown_workers":
      return {"status": "ok", "type": "shutdown_acknowledged"}
    allowed = {
      "create_model",
      "create_model_from_state",
      "forward_backward",
      "optim_step",
      "sample",
      "save_state",
      "load_weights",
      "save_weights_for_sampler",
      "save_weights",
    }
    if op not in allowed:
      raise NotImplementedError(f"Training request op {op!r} is not supported")
    return await getattr(self, op)(payload, model_id)

  async def create_model(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, full_raw, lora_raw, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind=self.default_kind)
    is_lora = self.default_kind == "lora"
    cfg_cls, raw = (LoraConfig, lora_raw) if is_lora else (FFTConfig, full_raw)
    config = cfg_cls(**{k: v for k, v in raw.items() if k in cfg_cls.model_fields})
    await asyncio.to_thread(self.worker.create_model, base_model, model_id, config)
    res = {"base_model": base_model, "model_id": model_id, "fine_tuning_type": fine_tuning_type, "type": "model_created"}
    if is_lora:
      res["rank"] = config.rank
    return res

  async def create_model_from_state(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    base_model, _, _, fine_tuning_type = await _fetch_model_meta(self.store, model_id, payload, default_kind=self.default_kind)
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
    )
    return {**result, "type": "forward_backward_completed"}

  async def optim_step(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    result = await asyncio.to_thread(self.worker.optim_step, payload.get("adam_params", {}), model_id)
    if self.default_kind == "lora":
      await asyncio.to_thread(self.worker.save_adapter, model_id)
    await self.bump_step_count(model_id)
    return {**result, "type": "optim_step_completed"}

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
    return {**result, "type": "sample_completed"}

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
    if self.default_kind == "lora":
      await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"))
    else:
      ref = payload.get("path") or payload.get("sampling_session_id")
      if not ref:
        raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
      local_path = os.path.join(tmp_dir(), "sampler_full", ref.removeprefix("tinker://").lstrip("/"))
      await asyncio.to_thread(self.worker.save_state, model_id, local_path, False, "sampler")
      if is_primary() and hasattr(self.store, "redis"):
        num_subs = await self.store.redis.publish(f"open_rl:weight_update:{model_id}", json.dumps({"weights_path": local_path}))
        print(f"[Trainer] Published weight update signal to {num_subs} subscribers for version path: {local_path}")
    return {"path": payload.get("path"), "sampling_session_id": payload.get("sampling_session_id"), "type": "sampler_weights_saved"}

  async def save_weights(self, payload: dict[str, Any], model_id: str) -> dict[str, Any]:
    if self.default_kind == "lora":
      await asyncio.to_thread(self.worker.save_adapter, model_id, payload.get("alias"))
    else:
      await asyncio.to_thread(self.worker.save_model, payload.get("alias") or model_id)
    return {"status": "ok", "type": "weights_saved"}


async def _fetch_model_meta(
  store: RequestStore,
  model_id: str,
  payload: dict[str, Any],
  default_kind: str = "full",
) -> tuple[str, dict[str, Any], dict[str, Any], str]:
  meta: dict[str, Any] = {}
  if hasattr(store, "get_value"):
    with suppress(Exception):
      val = await store.get_value(f"open_rl:model_meta:{model_id}")
      if isinstance(val, str):
        val = json.loads(val)
      if isinstance(val, dict):
        meta = val

  def get(key: str, default: Any) -> Any:
    return meta.get(key) or payload.get(key) or default

  ft_type = get("fine_tuning_type", "lora" if "lora_config" in meta or "lora_config" in payload or default_kind == "lora" else "full")
  return get("base_model", ""), get("full_config", {}), get("lora_config", {}), ft_type


class LoraTrainingRequestsProcessor(TrainingRequestsProcessor):
  default_kind = "lora"

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
    await self.run_loop()

  async def run_once(self) -> None:
    batch = await self.next_batch(lambda: self.store.get_requests(active_set_id=self.active_tenant_set_id))
    if not batch:
      return

    model_id = batch[0].get("model_id", "default")
    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", model_id)
      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {model_id}: {describe_requests(batch)}")
      for request in batch:
        target_model_id = request.get("adapter_id") or request.get("model_id") or model_id
        await self.process_request(request, target_model_id)


class FFTTrainingRequestsProcessor(TrainingRequestsProcessor):
  def __init__(
    self,
    store: RequestStore,
    worker: BaseTrainerWorker,
    model_id: str | None,
    time_slicer: TimeSlicerClient,
  ):
    if not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the gateway")
    self.store = store
    self.worker = worker
    self.model_id = model_id
    self.workload = workload_from_env(os.getpid(), name=local_workload_name("trainer", model_id or "shared"), claim=TRAINER_CLAIM)
    self.time_slicer = time_slicer
    self.snapshot_registered = False

  async def exit_gracefully(self) -> None:
    print(f"[WORKER] Initiating immediate exit for model {self.model_id} trainer worker...")
    if self.snapshot_registered:
      with suppress(Exception):
        await self.time_slicer.unregister(self.workload)
        self.snapshot_registered = False
    with suppress(Exception):
      await self.time_slicer.close()
    os._exit(0)

  async def run(self) -> None:
    print("[WORKER] Full fine-tuning training requests processor started.")
    try:
      await self.time_slicer.register(self.workload)
      self.snapshot_registered = True
      await self.run_loop()
    finally:
      try:
        if self.snapshot_registered:
          await self.time_slicer.unregister(self.workload)
      finally:
        await self.time_slicer.close()
        close_distributed()

  @asynccontextmanager
  async def gpu_lease(self):
    """Rank 0 holds the lease; the other ranks enter and leave with it."""
    async with AsyncExitStack() as stack:
      await stack.enter_async_context(self.time_slicer.acquire(self.workload))
      await asyncio.to_thread(barrier)
      try:
        yield
      finally:
        await asyncio.to_thread(barrier)

  async def run_once(self) -> None:
    batch = await self.next_batch(lambda: self.store.get_requests_for_model(self.model_id) if self.model_id else self.store.get_requests())
    if not batch:
      return

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
        print(f"\n[TRAINING REQUESTS] Popped {len(training_reqs)} requests for model: {model_id}: {describe_requests(training_reqs)}")
        results = []
        save_ops = set() if self.worker.save_needs_gpu() else {"save_state", "save_weights", "save_weights_for_sampler"}
        gpu_reqs = [r for r in training_reqs if r.get("op") not in save_ops]
        save_reqs = [r for r in training_reqs if r.get("op") in save_ops]

        if gpu_reqs:
          async with self.gpu_lease():
            await asyncio.to_thread(self.worker.wake_up)
            try:
              for request in gpu_reqs:
                results.append(await self.handle_request(request, model_id))
            finally:
              await asyncio.to_thread(self.worker.sleep)

        for request in save_reqs:
          results.append(await self.handle_request(request, model_id))

        for request_id, result in results:
          await self.publish_result(request_id, result)

    if has_shutdown:
      await self.exit_gracefully()


async def run_training_requests_processor(
  worker: BaseTrainerWorker,
  model_id: str | None = None,
  time_slicer: TimeSlicerClient | None = None,
  active_tenant_set_id: str | None = None,
) -> None:
  pin_executor_threads()
  store = get_store()
  if worker.full_parameter:
    # Rank 0 holds the GPU lease for the whole group; the other ranks follow it through barriers.
    time_slicer = time_slicer or (time_slicer_client_from_env() if is_primary() else NoOpTimeSlicer())
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
  print(f"-> Fine-Tuning Type: {fine_tuning_type} (Is LoRA: {is_lora})\n")

  worker: BaseTrainerWorker
  if os.getenv("OPEN_RL_TRAINER_BACKEND", "").lower() == "automodel":
    from training.automodel_worker import AutomodelTrainingWorker

    worker = AutomodelTrainingWorker(full_parameter=not is_lora)
  else:
    worker = LoraTrainingWorker() if is_lora else FFTTrainingWorker()
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

  if is_lora and is_primary():
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    # Configurable so a trainer can share a box with a vLLM server on 8000.
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

  asyncio.run(main_async(args))


if __name__ == "__main__":
  start_request_processing_loop()
