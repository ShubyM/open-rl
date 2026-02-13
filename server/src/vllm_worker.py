import os
import uvicorn
from fastapi import FastAPI, Request

try:
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    AsyncLLMEngine = None

app = FastAPI(title="Kube-RL vLLM Subprocess")

engine = None

@app.on_event("startup")
async def startup():
    global engine
    
    mock_vllm = os.environ.get("MOCK_VLLM", "0") == "1"
    if mock_vllm or AsyncLLMEngine is None:
        print("[vLLM Subprocess] MOCK_VLLM=1 or vllm not installed, bypassing real engine init for local dev.")
        return

    # Use arguments suitable for the v2 architecture
    engine_args = AsyncEngineArgs(
        model=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        enable_lora=True,
        max_loras=4,
        max_lora_rank=16,
        max_model_len=8192, # Prevent KV cache OOM on massive context windows
        gpu_memory_utilization=0.60, # Leave room for other things if needed
        enforce_eager=True # Useful for small setups
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("[vLLM Subprocess] Engine initialized and ready to serve IPC requests.")

@app.get("/healthz")
async def healthz():
    return {"status": "ok", "mock": engine is None}

@app.post("/generate")
async def generate(req: Request):
    try:
        data = await req.json()
        
        request_id = data.get("request_id")
        prompt_token_ids = data.get("prompt_token_ids")
        max_tokens = data.get("max_tokens", 20)
        temperature = data.get("temperature", 1.0)
        
        lora_id = data.get("lora_id", None)
        lora_path = data.get("lora_path", None)
        
        if engine is None:
            # Mocking for local Mac dev
            import asyncio
            await asyncio.sleep(0.1)
            # return dummy tokens locally
            return {"sequences": [{"tokens": [0]*max_tokens, "logprobs": [-0.1]*max_tokens}]}

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1 # return logprobs for TITO RL
        )
        
        lora_request = None
        if lora_id and lora_path:
            lora_request = LoRARequest(lora_id, 1, lora_path)

        results_generator = engine.generate(
            prompt={"prompt_token_ids": prompt_token_ids},
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request
        )
        
        final_output = None
        async for request_output in results_generator:
            # vLLM streams back incremental states, we wait for the final one
            final_output = request_output
            
        # Extract the top completion
        output = final_output.outputs[0]
        generated_token_ids = output.token_ids
        
        # Reconstruct logprobs
        logprobs = []
        if output.logprobs:
            for idx, token_logprobs in enumerate(output.logprobs):
                # token_logprobs is a dict of {token_id: Logprob}
                token_id = generated_token_ids[idx]
                if token_logprobs and token_id in token_logprobs:
                   logprob = token_logprobs[token_id].logprob
                else:
                   logprob = -9999.0
                logprobs.append(logprob)

        return {"sequences": [{"tokens": generated_token_ids, "logprobs": logprobs, "stop_reason": output.finish_reason}]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return explicit 500 so upstream client logs it
        return {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
