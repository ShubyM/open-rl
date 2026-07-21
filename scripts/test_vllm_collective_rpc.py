"""Isolation test script to inspect vLLM AsyncLLMEngine object structure and collective_rpc support."""

import asyncio
import os

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def main():
  print("=" * 60)
  print("vLLM Engine Architecture & collective_rpc Inspection Test")
  print("=" * 60)

  model_name = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
  print(f"Creating AsyncLLMEngine for model: {model_name}...")

  engine_args = AsyncEngineArgs(
    model=model_name,
    trust_remote_code=True,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.4,
    enforce_eager=True,
  )

  engine = AsyncLLMEngine.from_engine_args(engine_args)
  print(f"\n1. Top-level Engine Type: {type(engine)}")
  print(f"   hasattr(engine, 'collective_rpc'): {hasattr(engine, 'collective_rpc')}")

  # Check internal attributes
  for attr in ["engine", "engine_core", "model_executor", "driver_worker", "worker"]:
    val = getattr(engine, attr, None)
    if val is not None:
      print(f"\n2. Found attribute 'engine.{attr}': {type(val)}")
      print(f"   hasattr(engine.{attr}, 'collective_rpc'): {hasattr(val, 'collective_rpc')}")
      print(f"   hasattr(engine.{attr}, 'collective_rpc_async'): {hasattr(val, 'collective_rpc_async')}")
      print(f"   hasattr(engine.{attr}, 'receive_weights'): {hasattr(val, 'receive_weights')}")

  print("\n3. Testing execution of collective_rpc...")
  try:
    if hasattr(engine, "collective_rpc"):
      res = await engine.collective_rpc("check_health")
      print(f"   SUCCESS: engine.collective_rpc('check_health') -> {res}")
    elif hasattr(engine, "engine") and hasattr(engine.engine, "collective_rpc"):
      res = engine.engine.collective_rpc("check_health")
      print(f"   SUCCESS: engine.engine.collective_rpc('check_health') -> {res}")
    elif hasattr(engine, "engine_core") and hasattr(engine.engine_core, "collective_rpc"):
      res = engine.engine_core.collective_rpc("check_health")
      print(f"   SUCCESS: engine.engine_core.collective_rpc('check_health') -> {res}")
    else:
      print("   NO collective_rpc method exposed on top-level AsyncLLMEngine.")
  except Exception as e:
    print(f"   FAILED to execute collective_rpc: {e}")

  print("\nInspection test complete.")


if __name__ == "__main__":
  asyncio.run(main())
