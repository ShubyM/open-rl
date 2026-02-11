import uuid
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .trainer.engine import engine
import traceback
from contextlib import asynccontextmanager
import logging

class FilterNoisyEndpoints(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "retrieve_future" not in msg and "session_heartbeat" not in msg

logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())

futures_store = {}
request_queue = asyncio.Queue()

async def clock_cycle_loop():
    while True:
        try:
            # Wait for at least one item
            req = await request_queue.get()
            batch = [req]
            
            # Briefly sleep to allow "pipelining" (grouping requests that arrive concurrently)
            await asyncio.sleep(0.05) 
            
            # Drain the queue of any other requests that arrived during the wait
            while not request_queue.empty():
                batch.append(request_queue.get_nowait())
                
            # Group by model_id
            models_to_reqs = {}
            for r in batch:
                m_id = r.get("model_id")
                if m_id not in models_to_reqs:
                    models_to_reqs[m_id] = []
                models_to_reqs[m_id].append(r)
                
            print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests across {len(models_to_reqs)} distinct model tenant(s).")
                
            for m_id, reqs in models_to_reqs.items():
                if len(reqs) == 0:
                    continue
                    
                print(f"  -> [TENSOR CORE] Hot-swapping to LoRA adapter: {m_id}")
                
                # Set active adapter
                try:
                    await asyncio.to_thread(engine.set_active_adapter, m_id)
                except Exception as e:
                    print(f"Failed to set adapter {m_id}: {e}")
                    for r in reqs:
                        set_future_result(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})
                    continue
                    
                print(f"     Executing {len(reqs)} operations for {m_id}...")
                # Execute sequentially
                for r in reqs:
                    req_id = r["req_id"]
                    req_type = r["type"]
                    try:
                        if req_type == "forward_backward":
                            data = r["data"]
                            loss_fn = r["loss_fn"]
                            loss_config = r["loss_config"]
                            result = await asyncio.to_thread(engine.forward_backward, data, loss_fn, loss_config, m_id)
                            result["type"] = "forward_backward"
                            set_future_result(req_id, result)
                        elif req_type == "optim_step":
                            adam_params = r["adam_params"]
                            result = await asyncio.to_thread(engine.optim_step, adam_params, m_id)
                            result["type"] = "optim_step"
                            set_future_result(req_id, result)
                        elif req_type == "asample":
                            prompt_tokens = r["prompt_tokens"]
                            max_tokens = r["max_tokens"]
                            num_samples = r["num_samples"]
                            result = await asyncio.to_thread(engine.generate, prompt_tokens, max_tokens, num_samples, m_id)
                            result["type"] = "sample"
                            set_future_result(req_id, result)
                    except Exception as e:
                        traceback.print_exc()
                        set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})
                        
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in clock cycle loop: {e}")
            traceback.print_exc()
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the clock cycle loop
    task = asyncio.create_task(clock_cycle_loop())
    yield
    # Cleanup if needed
    task.cancel()

app = FastAPI(title="Kube-RL Server MVP", lifespan=lifespan)

def set_future_result(req_id, result_data):
    futures_store[req_id] = result_data

@app.get("/api/v1/healthz")
async def health_check():
    return {"status": "ok"}

@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
    return {
        "supported_models": [
            {"model_name": "Qwen/Qwen2.5-0.5B"}
        ]
    }

@app.post("/api/v1/create_session")
async def create_session(req: dict):
    return {"session_id": "sess-real-123", "type": "create_session"}

@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(req: dict):
    return {"type": "session_heartbeat"}

@app.post("/api/v1/create_model")
async def create_model(req: dict):
    # We use req_id as model_id
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    base_model = req.get("base_model", "Qwen/Qwen2.5-0.5B")
    lora_config = req.get("lora_config", {})
    rank = lora_config.get("rank", 16)
    
    model_id = req_id 
    
    async def _load_model_task():
        try:
            await asyncio.to_thread(engine.load_model, base_model, rank, model_id)
            set_future_result(req_id, {
                "model_id": model_id,
                "is_lora": True,
                "lora_rank": rank,
                "type": "create_model"
            })
        except Exception as e:
            traceback.print_exc()
            set_future_result(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})

    asyncio.create_task(_load_model_task())
    return {"request_id": req_id}

@app.post("/api/v1/get_info")
async def get_info(req: dict):
    model_name = engine.base_model_name or "Qwen/Qwen2.5-0.5B"
    return {
        "model_data": {
            "arch": "qwen",
            "model_name": model_name,
            "tokenizer_id": model_name
        },
        "model_id": req.get("model_id", "model-live-123"), # Will be whatever client passed
        "is_lora": True,
        "lora_rank": 16,
        "model_name": model_name,
        "type": "get_info"
    }

@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    fwd_input = req.get("forward_backward_input", {})
    data = fwd_input.get("data", [])
    loss_fn = fwd_input.get("loss_fn", "cross_entropy")
    loss_config = fwd_input.get("loss_fn_config", {})
    model_id = req.get("model_id")
    
    # Push to queue instead of thread
    await request_queue.put({
        "req_id": req_id,
        "model_id": model_id,
        "type": "forward_backward",
        "data": data,
        "loss_fn": loss_fn,
        "loss_config": loss_config
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    adam_params = req.get("adam_params", {})
    model_id = req.get("model_id")
    
    await request_queue.put({
        "req_id": req_id,
        "model_id": model_id,
        "type": "optim_step",
        "adam_params": adam_params
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
    req_id = str(uuid.uuid4())
    model_id = req.get("model_id") # Client passes the TrainingClient's model_id
    futures_store[req_id] = {
        "path": None,
        "sampling_session_id": model_id, # Link the sampler to the SAME model_id!
        "type": "save_weights_for_sampler"
    }
    return {"request_id": req_id}

@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
    # If client manually creates a session, we'll assign a mock one, 
    # but the usual flow is save_weights -> uses the existing model_id
    return {"sampling_session_id": req.get("model_id", "samp-session-live-123"), "type": "create_sampling_session"}

@app.post("/api/v1/asample")
async def asample(req: dict):
    req_id = str(uuid.uuid4())
    futures_store[req_id] = {"status": "pending"}
    
    prompt = req.get("prompt", {}).get("chunks", [])[0].get("tokens", [])
    params = req.get("sampling_params", {})
    max_tokens = params.get("max_tokens", 20)
    num_samples = req.get("num_samples", 1)
    
    # Tinker API sends sampling_session_id during `asample`, which we mapped to model_id
    model_id = req.get("model_id") or req.get("sampling_session_id")
    
    await request_queue.put({
        "req_id": req_id,
        "model_id": model_id,
        "type": "asample",
        "prompt_tokens": prompt,
        "max_tokens": max_tokens,
        "num_samples": num_samples
    })
    
    return {"request_id": req_id}

@app.post("/api/v1/retrieve_future")
async def retrieve_future(req: dict):
    request_id = req.get("request_id")
    if request_id in futures_store:
        result = futures_store[request_id]
        if result.get("status") == "pending":
            return {
                "type": "try_again", 
                "request_id": request_id, 
                "queue_state": "active"
            }
        return result
    return {"type": "RequestFailedResponse", "error_message": "Future not found"}

@app.post("/api/v1/telemetry")
async def telemetry(req: dict):
    return {"status": "accepted"}

