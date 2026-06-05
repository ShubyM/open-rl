# This file contains the clock cycle loop implementation for orchestrating training requests in Open-RL.

import asyncio
import os
import threading
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from store import get_store
from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import Datum

tracer = trace.get_tracer(__name__)

WORKER_MODE = os.getenv("OPEN_RL_WORKER_MODE")

worker = FFTTrainingWorker() if WORKER_MODE == "full" else LoraTrainingWorker()


def is_full_worker_mode() -> bool:
  return WORKER_MODE == "full"


def parse_datum(raw: dict) -> Datum:
  """Convert wire-format datum (with chunks) to our flat Datum type."""
  chunks = raw.get("model_input", {}).get("chunks", [])
  tokens: list[int] = []
  for chunk in chunks:
    tokens.extend(chunk.get("tokens", []))

  loss_inputs = raw.get("loss_fn_inputs", {})
  return Datum(model_input=tokens, loss_fn_inputs=loss_inputs)


async def handle_request(store, request: dict, model_id: str) -> None:
  req_id = request["req_id"]
  req_type = request["type"]

  carrier = request.get("trace_context", {})
  ctx = propagate.extract(carrier) if carrier else None
  token = otel_context.attach(ctx) if ctx else None

  try:
    if is_full_worker_mode() and request.get("model_id") != model_id:
      raise ValueError(f"Full worker for {model_id!r} received request for {request.get('model_id')!r}")

    match req_type:
      case "create_model":
        base_model = request["base_model"]

        if is_full_worker_mode():
          raw_config = request.get("full_config") or {}
          full_config = FFTConfig(**{k: v for k, v in raw_config.items() if k in FFTConfig.model_fields})
          await asyncio.to_thread(worker.create_model, base_model, model_id, full_config)
          # SDK compatibility: the public client currently expects LoRA-shaped training metadata,
          # even when this process is running a full fine-tuning worker.
          await store.set_future(
            req_id,
            {
              "model_id": model_id,
              "is_lora": True,
              "lora_rank": 16,
              "base_model": base_model,
              "type": "create_model",
            },
          )
        else:
          raw_config = request.get("lora_config") or {}
          lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})
          await asyncio.to_thread(worker.create_model, base_model, model_id, lora_config)
          await store.set_future(
            req_id,
            {
              "model_id": model_id,
              "is_lora": True,
              "lora_rank": lora_config.rank,
              "type": "create_model",
            },
          )

      case "create_model_from_state":
        state_path = request["state_path"]
        restore_optimizer = bool(request.get("restore_optimizer", False))
        result = await asyncio.to_thread(worker.load_from_state, model_id, state_path, restore_optimizer)
        result["type"] = "create_model_from_state"
        await store.set_future(req_id, result)

      case "forward_backward":
        raw_data = request["data"]
        loss_fn = request["loss_fn"]
        loss_config = request.get("loss_config")
        typed_data = [parse_datum(item) for item in raw_data]

        result = await asyncio.to_thread(worker.forward_backward, typed_data, loss_fn, loss_config, model_id)
        result["type"] = "forward_backward"
        await store.set_future(req_id, result)

      case "optim_step":
        adam_params = request["adam_params"]
        result = await asyncio.to_thread(worker.optim_step, adam_params, model_id)
        result["type"] = "optim_step"
        await store.set_future(req_id, result)

      case "sample":
        prompt_tokens = request["prompt_tokens"]
        max_tokens = request["max_tokens"]
        num_samples = request["num_samples"]
        temperature = request.get("temperature", 0.0)
        prompt_logprobs = bool(request.get("prompt_logprobs", False))

        result = await asyncio.to_thread(
          worker.generate,
          prompt_tokens,
          max_tokens,
          num_samples,
          temperature,
          model_id,
          prompt_logprobs,
        )
        result["type"] = "sample"
        await store.set_future(req_id, result)

      case "save_state":
        state_path = request["state_path"]
        include_optimizer = bool(request.get("include_optimizer", False))
        kind = request.get("kind", "state")

        result = await asyncio.to_thread(worker.save_state, model_id, state_path, include_optimizer, kind)
        # SDK's save_state() returns SaveWeightsResponse which requires type="save_weights".
        result["type"] = "save_weights"
        await store.set_future(req_id, result)

      case "load_weights":
        state_path = request["state_path"]
        restore_optimizer = bool(request.get("restore_optimizer", False))
        await asyncio.to_thread(worker.load_from_state, model_id, state_path, restore_optimizer)
        await store.set_future(req_id, {"path": state_path, "type": "load_weights"})

      case "save_weights_for_sampler":
        if is_full_worker_mode():
          ref = request.get("path") or request.get("sampling_session_id")
          if not ref:
            raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
          rel_path = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
          local_path = os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_full", rel_path)
          await asyncio.to_thread(worker.save_state, model_id, local_path, False, "sampler")
        else:
          await asyncio.to_thread(worker.save_adapter, model_id, request.get("alias"))
        await store.set_future(
          req_id,
          {
            "path": request.get("path"),
            "sampling_session_id": request.get("sampling_session_id"),
            "type": "save_weights_for_sampler",
          },
        )

      case "save_weights":
        if is_full_worker_mode():
          await asyncio.to_thread(worker.save_model, request.get("alias") or model_id)
        else:
          await asyncio.to_thread(worker.save_adapter, model_id, request.get("alias"))
        await store.set_future(req_id, {"status": "ok", "type": req_type})

      case _:
        print(f"Warning: Unhandled request type: {req_type}")
        await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": f"Unknown request type: {req_type}"})

  except Exception as e:
    traceback.print_exc()
    await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})
  finally:
    if token:
      otel_context.detach(token)


async def clock_cycle_loop() -> None:
  if is_full_worker_mode() and not os.getenv("REDIS_URL"):
    raise RuntimeError("Full fine-tuning workers require REDIS_URL so they can share queues and futures with the gateway")

  store = get_store()
  worker_model_id = os.getenv("OPEN_RL_WORKER_MODEL_ID") if is_full_worker_mode() else None
  if is_full_worker_mode() and not worker_model_id:
    raise RuntimeError("OPEN_RL_WORKER_MODEL_ID is required when OPEN_RL_WORKER_MODE=full")

  print(f"[WORKER] Training worker started in {WORKER_MODE} mode.")

  while True:
    try:
      batch = await store.get_requests_for_model(worker_model_id) if worker_model_id else await store.get_requests()
      if not batch:
        await asyncio.sleep(0.1)
        continue

      m_id = worker_model_id or batch[0].get("model_id", "default")

      with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
        batch_span.set_attribute("batch_size", len(batch))
        batch_span.set_attribute("model_id", m_id)

        print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests for tenant: {m_id}")

        for r in batch:
          await handle_request(store, r, m_id)

    except asyncio.CancelledError:
      break
    except Exception as e:
      print(f"Error in clock cycle loop: {e}")
      traceback.print_exc()

      import redis

      if isinstance(e, redis.exceptions.ConnectionError):
        print("[worker] Destroying StateStore singleton to force Redis reconnection...")
        import store as store_mod

        store_mod._store_instance = None
        store = store_mod.get_store()

      await asyncio.sleep(1)


def main() -> None:
  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> Worker mode: {WORKER_MODE}\n")

  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target and not is_full_worker_mode():
    worker.load_base_model(preload_target)
    is_ready = True
  else:
    if is_full_worker_mode():
      print("[WORKER] Full fine-tuning mode loads its model from the create_model request.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if not is_full_worker_mode():
    probe_app = FastAPI()

    @probe_app.get("/healthz")
    def healthz():
      if is_ready:
        return {"status": "ready"}
      raise HTTPException(status_code=503, detail="Model Loading")

    def run_probe_server():
      uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")

    threading.Thread(target=run_probe_server, daemon=True).start()
  asyncio.run(clock_cycle_loop())


if __name__ == "__main__":
  main()
