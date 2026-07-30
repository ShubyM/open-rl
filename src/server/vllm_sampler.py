# This file contains the vLLM worker implementation for high-throughput inference in Open-RL.

import argparse
import asyncio
import hashlib
import os
import sys
import time
import traceback
from typing import Any

try:
  from vllm import SamplingParams
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine
  from vllm.lora.request import LoRARequest
  from vllm.sampling_params import RequestOutputKind

  VLLM_AVAILABLE = True
except ImportError:
  SamplingParams = None
  AsyncEngineArgs = None
  AsyncLLMEngine = None
  LoRARequest = None
  RequestOutputKind = None
  VLLM_AVAILABLE = False

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from accel_timeslicer.time_slicer import is_time_slicing_enabled

provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter for vLLM Worker")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")

tracer = trace.get_tracer("vllm.inference.worker")

engine: Any = None
CURRENT_LOADED_SAMPLER_WEIGHTS: str | None = None
reload_lock = asyncio.Lock()

# Weight reloads sleep and rebuild the engine; doing that under an in-flight
# decode corrupts its output. Generations register here and a reload drains
# them first (new arrivals queue on reload_lock, so the drain terminates).
ACTIVE_GENERATIONS = 0
generation_idle = asyncio.Condition()


async def begin_generation() -> None:
  global ACTIVE_GENERATIONS
  async with generation_idle:
    ACTIVE_GENERATIONS += 1


async def end_generation() -> None:
  global ACTIVE_GENERATIONS
  async with generation_idle:
    ACTIVE_GENERATIONS -= 1
    generation_idle.notify_all()


async def wait_for_generation_drain() -> None:
  async with generation_idle:
    await generation_idle.wait_for(lambda: ACTIVE_GENERATIONS == 0)


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def vllm_sleep_level() -> int:
  """Use weight-preserving sleep unless the deployment opts into discard/reload."""
  level = int(os.getenv("OPEN_RL_VLLM_SLEEP_LEVEL", "1"))
  if level not in (1, 2):
    raise ValueError(f"OPEN_RL_VLLM_SLEEP_LEVEL must be 1 or 2, got {level}")
  return level


def vllm_language_model_only() -> bool:
  # Both supported base families (Qwen3.5/3.6, Gemma 4) are multimodal
  # wrappers and sampling is text-only, so the vision tower is off by
  # default. This only disables multimodal inputs — the constructed graph
  # and its weight names are unchanged, so LoRA adapter key matching is
  # unaffected. Set 0 to sample with multimodal inputs enabled.
  return os.getenv("OPEN_RL_VLLM_LANGUAGE_MODEL_ONLY", "1") == "1"


def architecture_override_missing_for_fft() -> bool:
  """FFT reloads need the checkpoint's text-only weight names to match the graph.

  language_model_only only disables multimodal inputs - it does not change the
  constructed architecture or its weight names. Reloading a text-only FFT
  checkpoint into a multimodal graph silently skips most weights ("Following
  weights were not loaded from checkpoint"), so multimodal base models must set
  VLLM_ARCHITECTURE_OVERRIDE (e.g. Gemma4ForCausalLM) to build the text graph.
  """
  return is_fft_enabled() and vllm_language_model_only() and not os.getenv("VLLM_ARCHITECTURE_OVERRIDE")


async def publish_sampler_ready(store: Any, model_id: str, instance_id: str) -> None:
  if store.redis is not None:
    await store.redis.set(f"open_rl:sampler_ready:{model_id}", instance_id)


async def clear_sampler_ready(store: Any, model_id: str, instance_id: str) -> None:
  if store.redis is None:
    return
  key = f"open_rl:sampler_ready:{model_id}"
  value = await store.redis.get(key)
  if isinstance(value, bytes):
    value = value.decode("utf-8")
  if value == instance_id:
    await store.redis.delete(key)


time_slicer: Any = None
if is_fft_enabled() and is_time_slicing_enabled():
  from accel_timeslicer.time_slicer import time_slicer_client_from_env, workload_from_env
  from accel_timeslicer.workload import SAMPLER_TIME_SLICE_GROUP, workload_job_id

  time_slicer = time_slicer_client_from_env()


def build_engine_kwargs(model_name: str) -> dict:
  hf_overrides: dict = {}
  arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
  if arch_override:
    hf_overrides["architectures"] = [arch_override]

  engine_kwargs = {
    "model": model_name,
    "enable_sleep_mode": is_fft_enabled(),
    "enable_lora": not is_fft_enabled(),
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
    "language_model_only": vllm_language_model_only(),
    "enable_prefix_caching": False,
    "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
  }
  if attention_backend := os.getenv("OPEN_RL_VLLM_ATTENTION_BACKEND"):
    engine_kwargs["attention_backend"] = attention_backend
  if not is_fft_enabled():
    engine_kwargs["max_loras"] = 8
    engine_kwargs["max_lora_rank"] = 64
  if hf_overrides:
    engine_kwargs["hf_overrides"] = hf_overrides
  return engine_kwargs


def init_engine():
  global engine

  print("\n" + "=" * 50)
  print("        Open-RL vLLM Inference Engine (Queue Mode)")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  model_name = os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL")
  print(f"-> Hardware     : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> Model        : {model_name or 'Not Set'}\n")

  mock_vllm = os.getenv("MOCK_VLLM", "0") == "1"
  if mock_vllm or not VLLM_AVAILABLE:
    print("[vLLM Worker] MOCK_VLLM=1 or vllm not installed, bypassing real engine init for local dev.")
  elif not model_name:
    print("[vLLM Worker] Error: BASE_MODEL environment variable is required.")
    sys.exit(1)
  else:
    if architecture_override_missing_for_fft():
      print(
        "[vLLM Worker] WARNING: OPEN_RL_ENABLE_FFT is on but VLLM_ARCHITECTURE_OVERRIDE is unset. "
        "For multimodal base models (e.g. Gemma) the text-only FFT checkpoints will NOT match the "
        "multimodal graph's weight names and reloads will be silently skipped "
        "('Following weights were not loaded from checkpoint'). Set VLLM_ARCHITECTURE_OVERRIDE "
        "(e.g. Gemma4ForCausalLM) unless the base model is already text-only."
      )
    engine_args = AsyncEngineArgs(**build_engine_kwargs(model_name))
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print("[vLLM Worker] Engine initialized successfully.")


async def warmup_engine() -> None:
  """Run one throwaway generation so runtime-JIT kernels compile before the
  worker starts pulling requests.

  vLLM's own startup warmup does not cover every kernel shape: Qwen's
  GDN/gated-deltanet prefill+decode Triton kernels and the LoRA shrink/expand
  kernels compile lazily on first use (the JIT monitor logs them as
  "JIT compilation during inference"). On a cold Triton cache that build takes
  minutes, and a real request stalling through it reads as a sampler timeout
  client-side. Disable with OPEN_RL_VLLM_WARMUP=0.
  """
  if engine is None or os.getenv("OPEN_RL_VLLM_WARMUP", "1") != "1":
    return
  try:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    max_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    prompt_len = max(8, min(1024, max_len - 16))
    params = SamplingParams(n=1, temperature=0.0, max_tokens=8, output_kind=RequestOutputKind.FINAL_ONLY)
    started = time.monotonic()
    print(f"[vLLM Worker] Warming up ({prompt_len}-token prefill + 8 decode steps) to compile JIT kernels...")
    async for _ in engine.generate(prompt={"prompt_token_ids": [1] * prompt_len}, sampling_params=params, request_id=f"warmup-{os.getpid()}"):
      pass
    print(f"[vLLM Worker] Warmup finished in {time.monotonic() - started:.1f}s.")
  except Exception as exc:
    print(f"[vLLM Worker] Warmup generation failed (continuing): {exc}")


async def prepare_engine(weights_path: str | None, weights_revision: str | None = None) -> None:
  """Wake vLLM and load a changed full-model checkpoint in place."""
  global CURRENT_LOADED_SAMPLER_WEIGHTS

  if engine is None:
    init_engine()

  if engine is None:
    return

  sleeping = await engine.is_sleeping()
  sleep_level = vllm_sleep_level()
  if sleeping and sleep_level == 2 and not weights_path:
    raise RuntimeError("A checkpoint path is required to wake vLLM from sleep level 2")

  target_revision = weights_revision or weights_path
  weights_discarded = sleeping and sleep_level == 2
  if weights_path and (weights_discarded or target_revision != CURRENT_LOADED_SAMPLER_WEIGHTS):
    print(f"[vLLM Worker] Loading checkpoint path={weights_path} revision={CURRENT_LOADED_SAMPLER_WEIGHTS} -> {target_revision}")
    await wait_for_generation_drain()
    try:
      if not sleeping:
        await engine.sleep(level=sleep_level)
      await engine.wake_up(tags=["weights"])
      await engine.collective_rpc("reload_weights", kwargs={"weights_path": weights_path})
      await engine.wake_up(tags=["kv_cache"])
      # Cached KV blocks were computed under the previous weights. Prefix
      # caching is disabled in our engine args, so this is a no-op today, but
      # reloading weights without it would silently serve stale-cache garbage
      # if caching is ever enabled.
      await engine.reset_prefix_cache()
    except Exception:
      # A failed transition leaves the engine in an unknown half-woken state
      # with possibly partially swapped weights. Poison the loaded revision so
      # the next request forces a full reload instead of trusting it.
      CURRENT_LOADED_SAMPLER_WEIGHTS = None
      raise
    CURRENT_LOADED_SAMPLER_WEIGHTS = target_revision
    return

  if sleeping:
    print("[vLLM Worker] Waking engine for sampling...")
    await engine.wake_up()


async def run_generation_backend(
  request_id: str,
  prompt_token_ids: list[int],
  max_tokens: int,
  temperature: float,
  stop: list[int] | None,
  top_p: float,
  top_k: int,
  num_samples: int,
  lora_id: str | None,
  lora_path: str | None,
  include_prompt_logprobs: bool,
) -> dict[str, Any]:
  try:
    # The client cannot discover this server's context window, so multi-turn
    # rollouts can legitimately outgrow it mid-episode (tool outputs append
    # between the client's own budget checks). Translate "prompt won't fit"
    # into the length-stop the RL env already treats as graceful truncation,
    # instead of failing the request and killing the run.
    max_model_len = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
    prompt_len = len(prompt_token_ids or [])
    if prompt_len >= max_model_len or max_tokens <= 0:
      print(
        f"[vLLM Worker] Prompt of {prompt_len} tokens leaves no room in max_model_len={max_model_len} "
        f"(max_tokens={max_tokens}); returning empty length-stop truncation."
      )
      return {"sequences": [{"tokens": [], "logprobs": [], "stop_reason": "length"} for _ in range(num_samples)]}
    if prompt_len + max_tokens > max_model_len:
      max_tokens = max_model_len - prompt_len

    current_engine = engine
    if current_engine is None:
      # Mocking for local Mac dev
      await asyncio.sleep(0.1)
      # return dummy tokens locally
      return {"sequences": [{"tokens": [0] * max_tokens, "logprobs": [-0.1] * max_tokens, "stop_reason": "length"}]}

    prompt_logprobs_val = 1 if include_prompt_logprobs else None
    sampling_params = SamplingParams(
      n=num_samples,
      temperature=temperature,
      max_tokens=max_tokens,
      stop_token_ids=stop,
      top_p=top_p,
      top_k=top_k,
      logprobs=1,  # return logprobs for TITO RL
      prompt_logprobs=prompt_logprobs_val,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    lora_request = None
    if lora_id and lora_path:
      # vLLM natively relies on lora_int_id to track cached adapter weights.
      # Convert the sequence identifier UUID to a stable 32-bit positive integer hash.
      lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
      lora_request = LoRARequest(lora_id, lora_int_id, lora_path)

    results_generator = current_engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
    )

    final_output = None
    with tracer.start_as_current_span("vllm_generate_tokens") as span:
      span.set_attribute("vllm.prompt_len", len(prompt_token_ids) if prompt_token_ids else 0)
      span.set_attribute("vllm.max_tokens", max_tokens)
      if lora_id:
        span.set_attribute("vllm.lora_id", lora_id)
      async for request_output in results_generator:
        final_output = request_output

    outputs = final_output.outputs if final_output else []
    sequences_out = []
    for output in outputs:
      generated_token_ids = list(output.token_ids)
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
      sequences_out.append({"tokens": generated_token_ids, "logprobs": logprobs, "stop_reason": output.finish_reason})

    prompt_logprobs_out = None
    if final_output and final_output.prompt_logprobs:
      prompt_logprobs_out = []
      for idx, token_logprobs in enumerate(final_output.prompt_logprobs):
        if token_logprobs is None:
          prompt_logprobs_out.append(None)
        else:
          token_id = prompt_token_ids[idx]
          if token_id in token_logprobs:
            prompt_logprobs_out.append(token_logprobs[token_id].logprob)
          else:
            prompt_logprobs_out.append(None)

    res = {"sequences": sequences_out}
    if prompt_logprobs_out is not None:
      res["prompt_logprobs"] = prompt_logprobs_out
    return res
  except Exception as e:
    traceback.print_exc()
    return {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(e)}"}


async def process_sampling_request(req: dict, store: Any) -> None:
  global engine

  request_id = req["request_id"]
  trace_context = req.get("trace_context", {})

  parent_span = propagate.extract(trace_context)
  with tracer.start_as_current_span("process_sampling_request", context=parent_span):
    try:
      # 1. Load the exact full-model checkpoint and wake the engine.
      weights_path = req.get("weights_path")
      weights_revision = req.get("weights_revision") or weights_path
      generation_registered = False
      if is_fft_enabled():
        async with reload_lock:
          await prepare_engine(weights_path, weights_revision)
          # Register before releasing the lock so a reload for a different
          # revision cannot slip in between prepare and generate.
          await begin_generation()
          generation_registered = True

      try:
        # 2. Run inference
        prompt_token_ids = req.get("prompt_token_ids", [])
        max_tokens = req.get("max_tokens", 20)
        temperature = req.get("temperature", 1.0)
        stop = req.get("stop")
        top_p = req.get("top_p", 1.0)
        top_k = req.get("top_k", -1)
        num_samples = req.get("num_samples", 1)
        lora_id = req.get("lora_id")
        lora_path = req.get("lora_path")
        include_prompt_logprobs = req.get("include_prompt_logprobs", False)

        result = await run_generation_backend(
          request_id=request_id,
          prompt_token_ids=prompt_token_ids,
          max_tokens=max_tokens,
          temperature=temperature,
          stop=stop,
          top_p=top_p,
          top_k=top_k,
          num_samples=num_samples,
          lora_id=lora_id,
          lora_path=lora_path,
          include_prompt_logprobs=include_prompt_logprobs,
        )
      finally:
        if generation_registered:
          await end_generation()

      if result.get("type") != "RequestFailedResponse":
        result["type"] = "sample"

      await store.set_future(request_id, result)
    except Exception as exc:
      traceback.print_exc()
      await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(exc)}"})


async def run_sampling_worker(model_id: str) -> None:
  global engine
  from server.store import get_store

  store = get_store()
  instance_id = os.getenv("OPEN_RL_WORKER_INSTANCE_ID", "1")
  snapshot_registered = False
  workload = None
  if time_slicer is not None:
    workload = workload_from_env(os.getpid(), job_id=workload_job_id("sampler", model_id), group=SAMPLER_TIME_SLICE_GROUP)

  if time_slicer is not None:
    assert workload is not None
    try:
      print(f"[vLLM Worker] Registering workload {workload.key} for initialization lock...")
      await time_slicer.register(workload)
      snapshot_registered = True
      async with time_slicer.acquire(workload):
        print("[vLLM Worker] Initializing vLLM engine under parent lock...")
        init_engine()
        print("[vLLM Worker] Engine initialized successfully.")
        await warmup_engine()
        if engine is not None:
          print("[vLLM Worker] Sleeping engine after init to yield GPU memory...")
          await engine.sleep(level=vllm_sleep_level())
    except Exception as exc:
      print(f"[vLLM Worker] Failed to perform coordinated initialization: {exc}")
      traceback.print_exc()
      if engine is None:
        init_engine()
  else:
    init_engine()
    await warmup_engine()

  async def exit_gracefully() -> None:
    print(f"[vLLM Worker] Initiating immediate exit for model {model_id} sampler worker...")
    nonlocal snapshot_registered
    await clear_sampler_ready(store, model_id, instance_id)
    if snapshot_registered and time_slicer is not None:
      assert workload is not None
      try:
        await time_slicer.unregister(workload)
        snapshot_registered = False
      except Exception as exc:
        print(f"[vLLM Worker] Failed to unregister: {exc}")
    if time_slicer is not None:
      try:
        await time_slicer.close()
      except Exception:
        pass
    os._exit(0)

  if time_slicer is not None:
    import signal

    async def handle_shutdown():
      print(f"[vLLM Worker] Received termination signal, shutting down model {model_id} sampler worker...")
      await exit_gracefully()

    try:
      loop = asyncio.get_running_loop()
      for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(handle_shutdown()))
    except NotImplementedError:
      pass

  await publish_sampler_ready(store, model_id, instance_id)

  print(f"[vLLM Worker] Listening for sampling requests on queue for model: {model_id}...")
  try:
    while True:
      try:
        batch = await store.get_sampling_requests_for_model(model_id)
        if not batch:
          await asyncio.sleep(0.05)
          continue

        has_shutdown = False
        sampling_reqs = []
        for req in batch:
          if req.get("request_id") == "SHUTDOWN_SENTINEL":
            has_shutdown = True
          else:
            sampling_reqs.append(req)

        if sampling_reqs:
          if time_slicer is not None:
            assert workload is not None
            async with time_slicer.acquire(workload):
              tasks = [asyncio.create_task(process_sampling_request(req, store)) for req in sampling_reqs]
              await asyncio.gather(*tasks)
              if has_shutdown:
                await exit_gracefully()
              if engine is not None:
                print("[vLLM Worker] Exiting batch: sleeping engine to yield GPU memory...")
                await engine.sleep(level=vllm_sleep_level())
          else:
            tasks = [asyncio.create_task(process_sampling_request(req, store)) for req in sampling_reqs]
            await asyncio.gather(*tasks)

        if has_shutdown:
          print("[vLLM Worker] Shutdown sentinel popped from queue. Initiating clean exit...")
          await exit_gracefully()
      except asyncio.CancelledError:
        break
      except Exception as exc:
        print(f"Error in sampling worker loop: {exc}")
        traceback.print_exc()
        await asyncio.sleep(1)
  finally:
    await clear_sampler_ready(store, model_id, instance_id)
    if time_slicer is not None:
      assert workload is not None
      try:
        if snapshot_registered:
          await time_slicer.unregister(workload)
      finally:
        await time_slicer.close()
        os._exit(0)


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL vLLM Pull-Mode Sampler Worker")
  parser.add_argument("--model-id", type=str, required=True, help="The model ID of the RL job to process requests for")
  args = parser.parse_args()

  try:
    asyncio.run(run_sampling_worker(args.model_id))
  except KeyboardInterrupt:
    print("[vLLM Worker] Exiting via KeyboardInterrupt.")


if __name__ == "__main__":
  main()
