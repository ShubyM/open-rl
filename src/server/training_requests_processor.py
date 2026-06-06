"""Process queued training requests for an OpenRL worker process."""

import argparse
import asyncio
import os
import threading
import traceback
from typing import Any

import redis
import store as store_mod
import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from store import RequestStore, get_store
from training_request_types import (
  CreateModelFromStateRequest,
  CreateModelRequest,
  ForwardBackwardRequest,
  LoadWeightsRequest,
  OptimStepRequest,
  SampleRequest,
  SaveStateRequest,
  SaveWeightsForSamplerRequest,
  SaveWeightsRequest,
  parse_training_request,
)

from snapshot_agent.client import SnapshotAgentClient
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import Datum

tracer = trace.get_tracer(__name__)


TrainingWorker = FFTTrainingWorker | LoraTrainingWorker


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def create_training_worker() -> TrainingWorker:
  return FFTTrainingWorker() if is_fft_enabled() else LoraTrainingWorker()


def create_snapshot_agent_client(socket_path: str) -> SnapshotAgentClient:
  return SnapshotAgentClient(socket_path)


def parse_datum(raw: dict[str, Any]) -> Datum:
  """Convert wire-format datum with chunks to the flat trainer datum type."""
  chunks = raw.get("model_input", {}).get("chunks", [])
  tokens: list[int] = []
  for chunk in chunks:
    tokens.extend(chunk.get("tokens", []))

  return Datum(model_input=tokens, loss_fn_inputs=raw.get("loss_fn_inputs", {}))


class TrainingRequestsProcessor:
  def __init__(
    self,
    store: RequestStore,
    worker: TrainingWorker,
    model_id: str | None = None,
    snapshot_client: SnapshotAgentClient | None = None,
  ):
    self.store = store
    self.worker = worker
    self.is_full_training_worker = isinstance(worker, FFTTrainingWorker)
    self.model_id = model_id
    self.pid = os.getpid()
    self.snapshot_client = snapshot_client
    self.snapshot_registered = False

    if self.is_full_training_worker and not os.getenv("REDIS_URL"):
      raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the gateway")
    if self.is_full_training_worker and not self.model_id:
      raise RuntimeError("A dedicated FFT worker needs --model-id so it knows which per-model queue to drain")

  def get_snapshot_client(self) -> SnapshotAgentClient:
    if self.snapshot_client is None:
      snapshot_socket = os.getenv("OPEN_RL_SNAPSHOT_AGENT_SOCKET", "/tmp/open-rl/snapshot-agent.sock")
      self.snapshot_client = create_snapshot_agent_client(snapshot_socket)
    return self.snapshot_client

  async def run(self) -> None:
    print(f"[WORKER] Training worker started. Full fine-tuning: {self.is_full_training_worker}.")

    try:
      if self.is_full_training_worker:
        await self.get_snapshot_client().register(self.pid)
        self.snapshot_registered = True

      while True:
        try:
          await self.run_once()
        except asyncio.CancelledError:
          break
        except Exception as exc:
          print(f"Error in training requests processor: {exc}")
          traceback.print_exc()
          self.reconnect_store_if_needed(exc)
          await asyncio.sleep(1)
    finally:
      if self.is_full_training_worker and self.snapshot_client is not None:
        try:
          if self.snapshot_registered:
            await self.snapshot_client.unregister(self.pid)
        finally:
          await self.snapshot_client.close()

  async def run_once(self) -> None:
    batch = await self.get_next_batch()
    if not batch:
      await asyncio.sleep(0.1)
      return

    model_id = self.model_id or batch[0].get("model_id", "default")

    with tracer.start_as_current_span("training_requests_batch") as batch_span:
      batch_span.set_attribute("batch_size", len(batch))
      batch_span.set_attribute("model_id", model_id)

      print(f"\n[TRAINING REQUESTS] Popped {len(batch)} requests for model: {model_id}")
      await self.process_batch(batch, model_id)

  async def get_next_batch(self) -> list[dict[str, Any]]:
    if self.is_full_training_worker:
      if self.model_id is None:
        raise RuntimeError("A dedicated FFT worker needs model_id so it knows which per-model queue to drain")
      return await self.store.get_requests_for_model(self.model_id)
    return await self.store.get_requests()

  async def process_batch(self, batch: list[dict[str, Any]], model_id: str) -> None:
    if self.is_full_training_worker:
      async with self.get_snapshot_client().acquire(self.pid):
        for request in batch:
          await self.process_request(request, model_id)
      return

    for request in batch:
      await self.process_request(request, model_id)

  async def process_request(self, raw_request: dict[str, Any], model_id: str | None = None) -> None:
    token = None
    req_id = raw_request.get("req_id")

    try:
      request = parse_training_request(raw_request)
      req_id = request.req_id
      model_id = model_id or self.model_id or request.model_id or "default"

      carrier = request.trace_context
      ctx = propagate.extract(carrier) if carrier else None
      token = otel_context.attach(ctx) if ctx else None

      if self.is_full_training_worker and request.model_id != model_id:
        raise ValueError(f"Full worker for {model_id!r} received request for {request.model_id!r}")

      match request:
        case CreateModelRequest():
          if self.is_full_training_worker:
            raw_config = request.full_config
            full_config = FFTConfig(**{k: v for k, v in raw_config.items() if k in FFTConfig.model_fields})
            await asyncio.to_thread(self.worker.create_model, request.base_model, model_id, full_config)
            await self.store.set_future(
              req_id,
              {
                "model_id": model_id,
                "lora_rank": 16,
                "base_model": request.base_model,
                "type": "create_model_result",
              },
            )
          else:
            raw_config = request.lora_config
            lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})
            await asyncio.to_thread(self.worker.create_model, request.base_model, model_id, lora_config)
            await self.store.set_future(
              req_id,
              {
                "model_id": model_id,
                "lora_rank": lora_config.rank,
                "type": "create_model_result",
              },
            )

        case CreateModelFromStateRequest():
          result = await asyncio.to_thread(self.worker.load_from_state, model_id, request.state_path, request.restore_optimizer)
          internal_result = {
            "model_id": result.get("model_id", model_id),
            "base_model": result.get("base_model"),
            "type": "create_model_from_state_result",
          }
          if self.is_full_training_worker:
            internal_result["lora_rank"] = 16
          await self.store.set_future(req_id, internal_result)

        case ForwardBackwardRequest():
          typed_data = [parse_datum(item) for item in request.data]

          result = await asyncio.to_thread(self.worker.forward_backward, typed_data, request.loss_fn, request.loss_config, model_id)
          result["type"] = "forward_backward"
          await self.store.set_future(req_id, result)

        case OptimStepRequest():
          result = await asyncio.to_thread(self.worker.optim_step, request.adam_params, model_id)
          result["type"] = "optim_step"
          await self.store.set_future(req_id, result)

        case SampleRequest():
          result = await asyncio.to_thread(
            self.worker.generate,
            request.prompt_tokens,
            request.max_tokens,
            request.num_samples,
            request.temperature,
            model_id,
            request.prompt_logprobs,
          )
          result["type"] = "sample"
          await self.store.set_future(req_id, result)

        case SaveStateRequest():
          result = await asyncio.to_thread(self.worker.save_state, model_id, request.state_path, request.include_optimizer, request.kind)
          # SDK's save_state() returns SaveWeightsResponse which requires type="save_weights".
          result["type"] = "save_weights"
          await self.store.set_future(req_id, result)

        case LoadWeightsRequest():
          await asyncio.to_thread(self.worker.load_from_state, model_id, request.state_path, request.restore_optimizer)
          await self.store.set_future(req_id, {"path": request.state_path, "type": "load_weights"})

        case SaveWeightsForSamplerRequest():
          if self.is_full_training_worker:
            ref = request.path or request.sampling_session_id
            if not ref:
              raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
            rel_path = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
            local_path = os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_full", rel_path)
            await asyncio.to_thread(self.worker.save_state, model_id, local_path, False, "sampler")
          else:
            await asyncio.to_thread(self.worker.save_adapter, model_id, request.alias)
          await self.store.set_future(
            req_id,
            {
              "path": request.path,
              "sampling_session_id": request.sampling_session_id,
              "type": "save_weights_for_sampler",
            },
          )

        case SaveWeightsRequest():
          if self.is_full_training_worker:
            await asyncio.to_thread(self.worker.save_model, request.alias or model_id)
          else:
            await asyncio.to_thread(self.worker.save_adapter, model_id, request.alias)
          await self.store.set_future(req_id, {"status": "ok", "type": request.type})

        case _:
          print(f"Warning: Unhandled request type: {request.type}")
          await self.store.set_future(
            req_id,
            {"type": "RequestFailedResponse", "error_message": f"Unknown request type: {request.type}"},
          )

    except Exception as exc:
      traceback.print_exc()
      if req_id is not None:
        await self.store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
      else:
        raise
    finally:
      if token:
        otel_context.detach(token)

  def reconnect_store_if_needed(self, exc: Exception) -> None:
    if not isinstance(exc, redis.exceptions.ConnectionError):
      return

    print("[worker] Destroying StateStore singleton to force Redis reconnection...")
    store_mod._store_instance = None
    self.store = store_mod.get_store()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-id", help="Model id whose per-model request queue this dedicated FFT worker drains.")
  args = parser.parse_args()

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> FFT enabled: {is_fft_enabled()}\n")

  worker = create_training_worker()
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

  processor = TrainingRequestsProcessor(get_store(), worker, args.model_id)
  asyncio.run(processor.run())


if __name__ == "__main__":
  main()
