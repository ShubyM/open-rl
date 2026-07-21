"""Isolation test script to verify collective_rpc across vLLM Engine APIs."""

import asyncio
import os


def test_worker_check(worker):
  """Function executed on worker processes via collective_rpc."""
  wt_engine = getattr(worker, "weight_transfer_engine", None)
  engine_name = type(wt_engine).__name__ if wt_engine else "None"
  return f"[Worker PID {os.getpid()}] WeightTransferEngine={engine_name}"


async def test_async_engine():
  print("\n--- Testing AsyncLLMEngine (vLLM V1) ---")
  try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
      model="Qwen/Qwen3-0.6B",
      trust_remote_code=True,
      dtype="bfloat16",
      max_model_len=2048,
      gpu_memory_utilization=0.3,
      enforce_eager=True,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print(f"Engine class: {type(engine).__name__}")
    print(f"Top-level hasattr(engine, 'collective_rpc'): {hasattr(engine, 'collective_rpc')}")

    # Inspect internal sub-objects
    if hasattr(engine, "engine"):
      print(f"hasattr(engine.engine, 'collective_rpc'): {hasattr(engine.engine, 'collective_rpc')}")
    if hasattr(engine, "engine_core"):
      print(f"hasattr(engine.engine_core, 'collective_rpc'): {hasattr(engine.engine_core, 'collective_rpc')}")
      print(f"hasattr(engine.engine_core, 'collective_rpc_async'): {hasattr(engine.engine_core, 'collective_rpc_async')}")

    # Test invoking collective_rpc via available paths
    if hasattr(engine, "collective_rpc"):
      res = await engine.collective_rpc("check_health")
      print(f"Result via engine.collective_rpc: {res}")
    elif hasattr(engine, "engine_core") and hasattr(engine.engine_core, "collective_rpc_async"):
      res = await engine.engine_core.collective_rpc_async("check_health")
      print(f"Result via engine.engine_core.collective_rpc_async: {res}")
    elif hasattr(engine, "engine") and hasattr(engine.engine, "collective_rpc"):
      res = engine.engine.collective_rpc("check_health")
      print(f"Result via engine.engine.collective_rpc: {res}")
    else:
      print("No direct collective_rpc method found on AsyncLLMEngine instance.")

  except Exception as e:
    print(f"Error testing AsyncLLMEngine: {e}")
    import traceback

    traceback.print_exc()


def test_sync_llm():
  print("\n--- Testing LLM (Synchronous vLLM Entrypoint) ---")
  try:
    from vllm import LLM

    llm = LLM(
      model="Qwen/Qwen3-0.6B",
      trust_remote_code=True,
      dtype="bfloat16",
      max_model_len=2048,
      gpu_memory_utilization=0.3,
      enforce_eager=True,
    )
    print(f"LLM class: {type(llm).__name__}")
    print(f"hasattr(llm, 'collective_rpc'): {hasattr(llm, 'collective_rpc')}")

    if hasattr(llm, "collective_rpc"):
      res = llm.collective_rpc(test_worker_check)
      print(f"Result via llm.collective_rpc: {res}")

  except Exception as e:
    print(f"Error testing LLM: {e}")
    import traceback

    traceback.print_exc()


if __name__ == "__main__":
  test_sync_llm()
  asyncio.run(test_async_engine())
