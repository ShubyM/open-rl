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
from trainer import Datum, LoraConfig, TrainerEngine
from trainer_scheduler import TrainerSchedulerClient

tracer = trace.get_tracer(__name__)

engine = TrainerEngine()


def parse_datum(raw: dict) -> Datum:
  """Convert wire-format datum (with chunks) to our flat Datum type."""
  chunks = raw.get("model_input", {}).get("chunks", [])
  tokens: list[int] = []
  for chunk in chunks:
    tokens.extend(chunk.get("tokens", []))

  loss_inputs = raw.get("loss_fn_inputs", {})
  return Datum(model_input=tokens, loss_fn_inputs=loss_inputs)


async def process_batch(store, batch: list[dict]) -> None:
  m_id = batch[0].get("model_id", "default")

  with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
    batch_span.set_attribute("batch_size", len(batch))
    batch_span.set_attribute("model_id", m_id)

    print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests for tenant: {m_id}")

    skip_adapter_switch = {"create_model", "create_model_from_state"}
    if not any(r.get("type") in skip_adapter_switch for r in batch):
      try:
        await asyncio.to_thread(engine.set_active_adapter, m_id)
      except Exception as e:
        print(f"Failed to set adapter {m_id}: {e}")
        for r in batch:
          await store.set_future(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})
        return

    for r in batch:
      req_id = r["req_id"]
      req_type = r["type"]

      carrier = r.get("trace_context", {})
      ctx = propagate.extract(carrier) if carrier else None
      token = otel_context.attach(ctx) if ctx else None

      try:
        match req_type:
          case "create_model":
            base_model = r["base_model"]
            training_mode = r.get("training_mode", "lora")
            if training_mode == "full":
              await asyncio.to_thread(engine.create_full_model, m_id, base_model)
              await store.set_future(
                req_id,
                {
                  "model_id": m_id,
                  "is_lora": False,
                  "type": "create_model",
                },
              )
            else:
              raw_config = r.get("lora_config") or {}
              lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})

              await asyncio.to_thread(engine.load_base_model, base_model)
              await asyncio.to_thread(engine.create_adapter, m_id, lora_config)

              await store.set_future(
                req_id,
                {
                  "model_id": m_id,
                  "is_lora": True,
                  "lora_rank": lora_config.rank,
                  "type": "create_model",
                },
              )

          case "create_model_from_state":
            state_path = r["state_path"]
            restore_optimizer = bool(r.get("restore_optimizer", False))
            result = await asyncio.to_thread(engine.load_from_state, m_id, state_path, restore_optimizer)
            result["type"] = "create_model_from_state"
            await store.set_future(req_id, result)

          case "forward_backward":
            raw_data = r["data"]
            loss_fn = r["loss_fn"]
            loss_config = r.get("loss_config")

            typed_data = [parse_datum(item) for item in raw_data]

            result = await asyncio.to_thread(engine.forward_backward, typed_data, loss_fn, loss_config, m_id)
            result["type"] = "forward_backward"
            await store.set_future(req_id, result)

          case "optim_step":
            adam_params = r["adam_params"]
            result = await asyncio.to_thread(engine.optim_step, adam_params, m_id)
            result["type"] = "optim_step"
            await store.set_future(req_id, result)

          case "sample":
            prompt_tokens = r["prompt_tokens"]
            max_tokens = r["max_tokens"]
            num_samples = r["num_samples"]
            temperature = r.get("temperature", 0.0)
            prompt_logprobs = bool(r.get("prompt_logprobs", False))

            result = await asyncio.to_thread(
              engine.generate,
              prompt_tokens,
              max_tokens,
              num_samples,
              temperature,
              m_id,
              prompt_logprobs,
            )
            result["type"] = "sample"
            await store.set_future(req_id, result)

          case "save_state":
            state_path = r["state_path"]
            response_path = r.get("response_path")
            include_optimizer = bool(r.get("include_optimizer", False))
            kind = r.get("kind", "state")

            result = await asyncio.to_thread(engine.save_state, m_id, state_path, include_optimizer, kind)
            if response_path is not None:
              result["path"] = response_path
            result["type"] = "save_weights"
            await store.set_future(req_id, result)

          case "load_weights":
            state_path = r["state_path"]
            restore_optimizer = bool(r.get("restore_optimizer", False))
            await asyncio.to_thread(engine.load_from_state, m_id, state_path, restore_optimizer)
            await store.set_future(req_id, {"path": state_path, "type": "load_weights"})

          case "save_weights_for_sampler":
            response_path = r.get("path")
            if engine.is_full_model(m_id):
              ref = response_path or ""
              rel = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
              local_path = os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_full", rel)
              await asyncio.to_thread(engine.save_state, m_id, local_path, False, "sampler")
            else:
              await asyncio.to_thread(engine.save_adapter, m_id, r.get("alias"))
            await store.set_future(
              req_id,
              {
                "path": response_path,
                "sampling_session_id": r.get("sampling_session_id"),
                "type": "save_weights_for_sampler",
              },
            )

          case "save_weights":
            await asyncio.to_thread(engine.save_adapter, m_id, r.get("alias"))
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
  store = get_store()
  scheduler_socket = os.getenv("OPEN_RL_SCHEDULER_SOCKET")
  worker_run_id = os.getenv("OPEN_RL_WORKER_RUN_ID") if scheduler_socket else None
  scheduler_client: TrainerSchedulerClient | None = None

  if scheduler_socket:
    if not worker_run_id:
      raise RuntimeError("OPEN_RL_WORKER_RUN_ID is required when OPEN_RL_SCHEDULER_SOCKET is set")
    scheduler_client = TrainerSchedulerClient(scheduler_socket)
    await scheduler_client.register(worker_run_id, os.getpid())
    print(f"[WORKER] Registered run '{worker_run_id}' with trainer scheduler at {scheduler_socket}.")

  print("[WORKER] Training worker started.")

  try:
    while True:
      try:
        batch = await store.get_requests_for_model(worker_run_id) if worker_run_id else await store.get_requests()
        if not batch:
          await asyncio.sleep(0.1)
          continue

        processed = False
        try:
          if scheduler_client is not None:
            async with scheduler_client.acquire(worker_run_id):
              await process_batch(store, batch)
              processed = True
          else:
            await process_batch(store, batch)
            processed = True
        except Exception as e:
          traceback.print_exc()
          if not processed:
            for r in batch:
              await store.set_future(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})

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
  finally:
    if scheduler_client is not None and worker_run_id:
      try:
        await scheduler_client.unregister(worker_run_id)
      finally:
        await scheduler_client.close()


def main() -> None:
  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}\n")

  preload_target = os.getenv("BASE_MODEL")
  scheduled_worker = bool(os.getenv("OPEN_RL_SCHEDULER_SOCKET") and os.getenv("OPEN_RL_WORKER_RUN_ID"))
  is_ready = False
  if preload_target and not scheduled_worker:
    engine.load_base_model(preload_target)
    is_ready = True
  else:
    if scheduled_worker:
      print("[WORKER] Scheduler mode enabled; deferring model materialization until acquire.")
    else:
      print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

  if os.getenv("OPEN_RL_DISABLE_WORKER_HEALTHZ") != "1":
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
