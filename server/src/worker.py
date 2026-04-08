import asyncio
import os
import traceback
from opentelemetry import propagate
from opentelemetry import context as otel_context
from opentelemetry import trace
from types import SimpleNamespace

# Import our new engine and type
from server.src.trainer2 import TrainerEngine, Datum
from server.src.store import get_store

tracer = trace.get_tracer(__name__)

async def clock_cycle_loop():
  store = get_store()
  engine = TrainerEngine()
  
  print("[WORKER] Training worker started.")
  
  while True:
    try:
      # Block until requests are available
      batch = await store.get_requests()
      if not batch:
        await asyncio.sleep(0.1)
        continue
        
      m_id = batch[0].get("model_id", "default")
      
      with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
        batch_span.set_attribute("batch_size", len(batch))
        batch_span.set_attribute("model_id", m_id)
        
        print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests for tenant: {m_id}")
        
        # Execute sequentially
        for r in batch:
          req_id = r["req_id"]
          req_type = r["type"]
          
          # Extract trace context if present
          carrier = r.get("trace_context", {})
          ctx = propagate.extract(carrier) if carrier else None
          token = otel_context.attach(ctx) if ctx else None
          
          try:
            match req_type:
              case "create_model":
                base_model = r["base_model"]
                lora_config = r.get("lora_config") or {}
                rank = lora_config.get("rank", 16)
                
                # Load base model (eagerly, or check if already loaded)
                await asyncio.to_thread(engine.load_base_model, base_model)
                
                # Create adapter
                config_obj = SimpleNamespace(
                  rank=rank,
                  train_attn=lora_config.get("train_attn", True),
                  train_mlp=lora_config.get("train_mlp", True),
                  train_unembed=lora_config.get("train_unembed", False)
                )
                await asyncio.to_thread(engine.create_adapter, m_id, config_obj)
                
                await store.set_future(req_id, {"model_id": m_id, "is_lora": True, "lora_rank": rank, "type": "create_model"})
                
              case "forward_backward":
                raw_data = r["data"]
                loss_fn = r["loss_fn"]
                loss_config = r.get("loss_config")
                
                # Convert raw data to our Datum objects
                typed_data = [Datum(**item) for item in raw_data]
                
                result = await asyncio.to_thread(engine.forward_backward, typed_data, loss_fn, loss_config, m_id)
                result["type"] = "forward_backward"
                await store.set_future(req_id, result)
                
              case "optim_step":
                adam_params = r["adam_params"]
                lr = adam_params.get("learning_rate", 1e-4)
                clip = adam_params.get("grad_clip_norm")
                
                result = await asyncio.to_thread(engine.optim_step, m_id, lr, clip)
                result["type"] = "optim_step"
                await store.set_future(req_id, result)
                
              case "save_weights_for_sampler" | "save_weights":
                await asyncio.to_thread(engine.save_adapter, m_id)
                await store.set_future(req_id, {"status": "ok", "type": req_type})
                
              case _:
                print(f"Warning: Unhandled request type: {req_type}")
                
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
      
      # Handle Redis reconnection logic
      import redis
      if isinstance(e, redis.exceptions.ConnectionError):
        print("[worker] Destroying RequestStore singleton to force Redis reconnection...")
        from server.src import store as store_mod
        store_mod._store_instance = None
        store = store_mod.get_store()
        
      await asyncio.sleep(1)

if __name__ == "__main__":
  asyncio.run(clock_cycle_loop())
