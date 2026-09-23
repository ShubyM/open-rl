"""Queue-driven vLLM sampler. One engine serves a model's queue; a request names
either a LoRA adapter (lora_id) or a whole-model weights version (weights_path)."""

import argparse
import asyncio
import hashlib
import os
import signal
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from itertools import groupby
from typing import Any

from opentelemetry import propagate, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from vllm import SamplingParams
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind

from accel_timeslicer.time_slicer import TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM, WorkloadRef, local_workload_name
from server.store import RedisStore, RequestStore, get_store
from server.vllm_options import gpu_memory_utilization, split_stop, text_only_engine_kwargs

tracer = trace.get_tracer("vllm.inference.worker")
SHUTDOWN_SENTINEL = "SHUTDOWN_SENTINEL"
READY_TTL_SECONDS = 3600


def failed_response(message: str) -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message}


def engine_kwargs_from_env(fft_enabled: bool) -> dict[str, Any]:
  model_name = os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL")
  if not model_name:
    raise ValueError("BASE_MODEL or VLLM_MODEL is required")
  engine_kwargs = {
    "model": model_name,
    "enable_sleep_mode": fft_enabled,
    "enable_lora": not fft_enabled,
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
    "gpu_memory_utilization": gpu_memory_utilization(),
    "enable_prefix_caching": False,
    "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    **text_only_engine_kwargs(),
  }
  architecture = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
  if architecture:
    engine_kwargs["hf_overrides"] = {"architectures": [architecture]}
  if fft_enabled:
    engine_kwargs["weight_transfer_config"] = WeightTransferConfig(backend="delta_snapshot")
  else:
    engine_kwargs["max_loras"] = int(os.getenv("VLLM_MAX_LORAS", "8"))
    engine_kwargs["max_lora_rank"] = int(os.getenv("VLLM_MAX_LORA_RANK", "64"))
  return engine_kwargs


def lora_request_for(request: dict[str, Any]) -> LoRARequest | None:
  """A LoRA request when the adapter exists on disk. The gateway sends the adapter
  directory once it exists; before the first save has finished a LoRA job samples
  from the base model, so a missing adapter is not an error."""
  lora_id, lora_path = request.get("lora_id"), request.get("lora_path")
  if not lora_id or not lora_path or not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
    return None
  lora_int_id = int(hashlib.md5(lora_id.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
  return LoRARequest(lora_id, lora_int_id, lora_path)


def sampled_logprobs(output: CompletionOutput) -> list[float]:
  """One logprob per generated token; -9999.0 where vLLM did not report the sampled token."""
  if not output.logprobs:
    return []
  return [step[token_id].logprob if step and token_id in step else -9999.0 for token_id, step in zip(output.token_ids, output.logprobs)]


def prompt_logprobs(output: RequestOutput | None, prompt_token_ids: list[int]) -> list[float | None] | None:
  """One logprob per prompt token; None where vLLM reports none (always the first token)."""
  if output is None or not output.prompt_logprobs:
    return None
  return [step[token_id].logprob if step and token_id in step else None for token_id, step in zip(prompt_token_ids, output.prompt_logprobs)]


class Sampler:
  """Consume one model's sampling queue with a single vLLM engine.

  An FFT sampler is one per job: requests carry a weights_path, consecutive
  groups sharing one are served together, and the engine is updated between
  groups, never while a request is generating. A failed update poisons the
  sampler until the process is restarted. A LoRA sampler is one per base
  model and shared by its jobs: each request names its adapter with lora_id.
  """

  def __init__(
    self,
    model_id: str,
    store: RequestStore,
    make_engine: Callable[[], AsyncLLMEngine],
    *,
    time_slicer: TimeSlicerClient | None = None,
    workload: WorkloadRef | None = None,
  ) -> None:
    if (time_slicer is None) != (workload is None):
      raise ValueError("time_slicer and workload must be provided together")
    self.model_id = model_id
    self.store = store
    self.make_engine = make_engine
    self.time_slicer = time_slicer
    self.workload = workload
    self.engine: AsyncLLMEngine | None = None
    self.weights_path: str | None = None
    self.update_failed = False
    self._registered = False
    self._ready_at: float | None = None
    self._closed = False

  @asynccontextmanager
  async def gpu(self):
    """Hold the GPU for the body. Under a time slicer that means acquiring the
    slot, waking the engine if it exists, and sleeping it again on the way out."""
    if self.time_slicer is None:
      yield
      return
    async with self.time_slicer.acquire(self.workload):
      if self.engine is not None:
        await self.engine.wake_up(tags=["weights", "kv_cache"])
      try:
        yield
      finally:
        if self.engine is not None:
          await self.engine.sleep(level=1)

  async def start(self) -> None:
    if self.time_slicer is not None:
      await self.time_slicer.register(self.workload)
      self._registered = True
    async with self.gpu():
      self.engine = self.make_engine()
    await self.mark_ready()

  async def mark_ready(self) -> None:
    """The gateway waits on this key. A shared LoRA sampler outlives the TTL, so the loop refreshes it."""
    if isinstance(self.store, RedisStore):
      await self.store.redis.set(f"open_rl:sampler_ready:{self.model_id}", "1", ex=READY_TTL_SECONDS)
      self._ready_at = time.monotonic()

  async def run(self) -> None:
    try:
      await self.start()
      while True:
        if self._ready_at is not None and time.monotonic() - self._ready_at > 60:
          await self.mark_ready()
        batch = await self.store.get_sampling_requests_for_model(self.model_id)
        if not batch:
          await asyncio.sleep(0.05)
          continue
        shutdown = any(req.get("request_id") == SHUTDOWN_SENTINEL for req in batch)
        requests = [req for req in batch if req.get("request_id") != SHUTDOWN_SENTINEL]
        if requests:
          async with self.gpu():
            await self.process_batch(requests)
        if shutdown:
          return
    finally:
      await self.close()

  async def process_batch(self, requests: list[dict[str, Any]]) -> None:
    for weights_path, group in groupby(requests, key=lambda req: req.get("weights_path")):
      batch = list(group)
      try:
        if self.update_failed:
          raise RuntimeError("Weight update failed; restart the sampler before serving")
        if weights_path and weights_path != self.weights_path:
          await self.update_weights(weights_path)
      except Exception as exc:
        for request in batch:
          await self.store.set_future(request["request_id"], failed_response(f"vLLM weight update failed: {exc}"))
        continue
      await asyncio.gather(*(self.process_request(req) for req in batch))

  async def process_request(self, request: dict[str, Any]) -> None:
    with tracer.start_as_current_span("process_sampling_request", context=propagate.extract(request.get("trace_context", {}))):
      try:
        result = await self.generate(request)
        result["type"] = "sample"
      except Exception as exc:
        result = failed_response(f"vLLM Worker Error: {exc}")
      await self.store.set_future(request["request_id"], result)

  async def update_weights(self, weights_path: str) -> None:
    # No finally/resume on failure: partially updated weights must not be served.
    try:
      await self.engine.pause_generation(mode="wait", clear_cache=True)
      await self.engine.start_weight_update()
      await self.engine.update_weights(WeightTransferUpdateRequest(update_info={"target_weights_path": weights_path}))
      await self.engine.finish_weight_update(weight_version=weights_path)
      await self.engine.reset_encoder_cache()
      await self.engine.resume_generation()
    except BaseException:
      self.update_failed = True
      raise
    self.weights_path = weights_path

  async def generate(self, request: dict[str, Any]) -> dict[str, Any]:
    request_id = request["request_id"]
    prompt_token_ids = request.get("prompt_token_ids", [])
    max_tokens = request.get("max_tokens", 20)
    lora_request = lora_request_for(request)
    stop_strings, stop_token_ids = split_stop(request.get("stop"))
    sampling_params = SamplingParams(
      n=request.get("num_samples", 1),
      temperature=request.get("temperature", 1.0),
      max_tokens=max_tokens,
      stop=stop_strings,
      stop_token_ids=stop_token_ids,
      top_p=request.get("top_p", 1.0),
      top_k=request.get("top_k", -1),
      logprobs=1,  # return logprobs for TITO RL
      prompt_logprobs=1 if request.get("include_prompt_logprobs", False) else None,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    results_generator = self.engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
    )

    final_output = None
    with tracer.start_as_current_span("vllm_generate_tokens") as span:
      span.set_attribute("vllm.prompt_len", len(prompt_token_ids) if prompt_token_ids else 0)
      span.set_attribute("vllm.max_tokens", max_tokens)
      if lora_request is not None:
        span.set_attribute("vllm.lora_id", lora_request.lora_name)
      async for request_output in results_generator:
        final_output = request_output

    sequences = [
      {"tokens": list(output.token_ids), "logprobs": sampled_logprobs(output), "stop_reason": output.finish_reason}
      for output in (final_output.outputs if final_output else [])
    ]
    res: dict[str, Any] = {"sequences": sequences}
    prompt_logprobs_out = prompt_logprobs(final_output, prompt_token_ids)
    if prompt_logprobs_out is not None:
      res["prompt_logprobs"] = prompt_logprobs_out
    return res

  async def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if self.engine is not None:
        self.engine.shutdown()
    finally:
      try:
        if self._ready_at is not None:
          self._ready_at = None
          await self.store.delete_values(f"open_rl:sampler_ready:{self.model_id}")
      finally:
        if self.time_slicer is not None:
          try:
            if self._registered:
              self._registered = False
              await self.time_slicer.unregister(self.workload)
          finally:
            await self.time_slicer.close()


async def run_sampling_worker(model_id: str) -> None:
  fft_enabled = os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"
  engine_kwargs = engine_kwargs_from_env(fft_enabled)
  sampler = Sampler(
    model_id,
    get_store(),
    lambda: AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs)),
    time_slicer=time_slicer_client_from_env() if fft_enabled else None,
    workload=workload_from_env(os.getpid(), name=local_workload_name("sampler", model_id), claim=SAMPLER_CLAIM) if fft_enabled else None,
  )
  loop = asyncio.get_running_loop()
  task = asyncio.current_task()
  assert task is not None
  installed_signals = []
  try:
    for sig in (signal.SIGTERM, signal.SIGINT):
      try:
        loop.add_signal_handler(sig, task.cancel)
        installed_signals.append(sig)
      except NotImplementedError:
        break
    await sampler.run()
  except asyncio.CancelledError:
    pass  # Sampler.run has completed cleanup before cancellation reaches here.
  finally:
    for sig in installed_signals:
      loop.remove_signal_handler(sig)


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL vLLM Pull-Mode Sampler Worker")
  parser.add_argument("--model-id", type=str, required=True, help="The model ID of the RL job to process requests for")
  args = parser.parse_args()
  provider = TracerProvider()
  trace.set_tracer_provider(provider)
  try:
    if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
      from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

      provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
    asyncio.run(run_sampling_worker(args.model_id))
  finally:
    provider.shutdown()


if __name__ == "__main__":
  main()
