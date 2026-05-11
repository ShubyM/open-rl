# This file contains the clock cycle loop implementation for orchestrating training requests in Open-RL.

import asyncio
import os
import threading
import traceback
from dataclasses import replace

import uvicorn
from fastapi import FastAPI, HTTPException
from model_state import latest_training_state_alias, state_ref
from opentelemetry import context as otel_context
from opentelemetry import propagate
from store import get_store
from telemetry import get_tracer

tracer = get_tracer("openrl.trainer_worker")

engine = None


def get_engine():
  global engine
  if engine is None:
    from trainer import TrainerEngine

    engine = TrainerEngine()
  return engine


def _parse_datum(raw: dict):
  """Convert wire-format datum (with chunks) to our flat Datum type."""
  from trainer import Datum

  chunks = raw.get("model_input", {}).get("chunks", [])
  tokens: list[int] = []
  for chunk in chunks:
    tokens.extend(chunk.get("tokens", []))

  loss_inputs = raw.get("loss_fn_inputs", {})
  return Datum(model_input=tokens, loss_fn_inputs=loss_inputs)


async def ensure_adapter_ready(store, model_id: str) -> None:
  trainer_engine = get_engine()
  if trainer_engine.has_adapter(model_id):
    await asyncio.to_thread(trainer_engine.set_active_adapter, model_id)
    return

  state = await store.get_model_state(latest_training_state_alias(model_id))
  if state:
    restore_optimizer = bool(state.get("optimizer_ref") or state.get("checkpoint_metadata", {}).get("restore_optimizer", False))
    ref = state_ref(state)
    if ref:
      await asyncio.to_thread(trainer_engine.load_from_state, model_id, ref, restore_optimizer)
  await asyncio.to_thread(trainer_engine.set_active_adapter, model_id)


async def clock_cycle_loop() -> None:
  store = get_store()
  trainer_engine = get_engine()

  print("[WORKER] Training worker started.")

  while True:
    try:
      batch = await store.get_requests()
      if not batch:
        await asyncio.sleep(0.1)
        continue

      m_id = batch[0].get("model_id", "default")

      with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
        batch_span.set_attribute("batch_size", len(batch))
        batch_span.set_attribute("model_id", m_id)

        print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests for tenant: {m_id}")

        SKIP_ADAPTER_SWITCH = {"create_model", "create_model_from_state"}
        if not any(r.get("type") in SKIP_ADAPTER_SWITCH for r in batch):
          try:
            await ensure_adapter_ready(store, m_id)
          except Exception as e:
            print(f"Failed to set adapter {m_id}: {e}")
            for r in batch:
              await store.set_future(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})
            continue

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
                from trainer import LoraConfig

                raw_config = r.get("lora_config") or {}
                lora_config = LoraConfig(**{k: v for k, v in raw_config.items() if k in LoraConfig.model_fields})

                await asyncio.to_thread(trainer_engine.load_base_model, base_model)
                await asyncio.to_thread(trainer_engine.create_adapter, m_id, lora_config)

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
                result = await asyncio.to_thread(trainer_engine.load_from_state, m_id, state_path, restore_optimizer)
                result["type"] = "create_model_from_state"
                await store.publish_checkpoint(
                  m_id,
                  state_path,
                  {"restore_optimizer": restore_optimizer, "source": "create_model_from_state", "base_model": result.get("base_model")},
                )
                await store.set_future(req_id, result)

              case "forward_backward":
                raw_data = r["data"]
                loss_fn = r["loss_fn"]
                loss_config = r.get("loss_config")

                typed_data = [_parse_datum(item) for item in raw_data]

                result = await asyncio.to_thread(trainer_engine.forward_backward, typed_data, loss_fn, loss_config, m_id)
                result["type"] = "forward_backward"
                await store.set_future(req_id, result)

              case "optim_step":
                adam_params = r["adam_params"]
                result = await asyncio.to_thread(trainer_engine.optim_step, adam_params, m_id)
                result["type"] = "optim_step"
                await store.set_future(req_id, result)

              case "sample":
                prompt_tokens = r["prompt_tokens"]
                max_tokens = r["max_tokens"]
                num_samples = r["num_samples"]
                temperature = r.get("temperature", 0.0)
                prompt_logprobs = bool(r.get("prompt_logprobs", False))

                result = await asyncio.to_thread(
                  trainer_engine.generate,
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
                include_optimizer = bool(r.get("include_optimizer", False))
                kind = r.get("kind", "state")

                result = await asyncio.to_thread(trainer_engine.save_state, m_id, state_path, include_optimizer, kind)
                # SDK's save_state() returns SaveWeightsResponse which requires type="save_weights".
                result["type"] = "save_weights"
                if model_state := result.get("model_state"):
                  await store.publish_model_state(model_state["state_id"], model_state)
                  await store.publish_model_state_alias(latest_training_state_alias(m_id), model_state["state_id"])
                else:
                  await store.publish_checkpoint(
                    m_id,
                    result["path"],
                    {"restore_optimizer": include_optimizer, "source": "save_state", "delta_ref": result.get("delta_ref")},
                  )
                await store.set_future(req_id, result)

              case "load_weights":
                state_path = r["state_path"]
                restore_optimizer = bool(r.get("restore_optimizer", False))
                result = await asyncio.to_thread(trainer_engine.load_from_state, m_id, state_path, restore_optimizer)
                await store.publish_checkpoint(
                  m_id,
                  state_path,
                  {"restore_optimizer": restore_optimizer, "source": "load_weights", "base_model": result.get("base_model")},
                )
                await store.set_future(req_id, {"path": state_path, "type": "load_weights"})

              case "save_weights_for_sampler":
                published = await asyncio.to_thread(
                  trainer_engine.publish_adapter_for_inference,
                  m_id,
                  r.get("sampling_session_id"),
                  r.get("alias"),
                )
                state_ids = {state_id for state_id in (r.get("path"), r.get("sampling_session_id"), published.state_id) if state_id}
                for state_id in state_ids:
                  await store.publish_model_state(state_id, replace(published, state_id=state_id).to_dict())
                await store.set_future(
                  req_id,
                  {
                    "path": r.get("path"),
                    "sampling_session_id": r.get("sampling_session_id"),
                    "state_id": r.get("path") or r.get("sampling_session_id"),
                    "version": published.version,
                    "base_model": published.base_model,
                    "type": "save_weights_for_sampler",
                  },
                )

              case "save_weights":
                await asyncio.to_thread(trainer_engine.save_adapter, m_id, r.get("alias"))
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
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}\n")

  preload_target = os.getenv("BASE_MODEL")
  is_ready = False
  if preload_target:
    get_engine().load_base_model(preload_target)
    is_ready = True
  else:
    print("[WARNING] BASE_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True

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
