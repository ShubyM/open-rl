"""A sampling pool replica with vLLM in this process. It serves its pool's stream itself, with no router, and when its
job shares a GPU with the trainer it takes a time-slicer turn around each batch, the way the trainer does.

  python -m server.vllm_sampler --pool <pool>
"""

import argparse
import asyncio
import logging
import os
import signal
import uuid
from pathlib import Path
from typing import Any

from accel_timeslicer.time_slicer import time_slicer_client_from_env, time_slicing_enabled, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM, local_workload_name
from server.lora_snapshots import snapshot_root
from server.sampling_dispatcher import Dispatcher, delta_applies, failed, holding_gpu, redis_client
from server.sampling_store import SamplingStore
from server.vllm_options import engine_args

logger = logging.getLogger(__name__)

# The readiness probe passes once vLLM is up, and asleep when it shares its GPU.
READY_FILE = os.getenv("OPEN_RL_SAMPLER_READY_FILE", "/tmp/open-rl-sampler-ready")


class Sampler:
  """A vLLM engine that answers a pool's records and swaps its weights between batches. `revision` is the weights
  it holds, "" for the base model. A failed update leaves partial weights behind, so the sampler refuses work until
  the process is restarted."""

  def __init__(self, engine: Any, base_model: str) -> None:
    self.engine, self.base_model = engine, base_model
    self.lora_ids: dict[str, int] = {}
    self.revision = ""
    self.update_failed = False

  async def wake(self) -> None:
    await self.engine.wake_up(tags=["weights", "kv_cache"])

  async def sleep(self) -> None:
    await self.engine.sleep(level=1)

  async def ensure_weights(self, revision: str) -> bool:
    """Whether the engine holds `revision` after this, loading it if it can."""
    if self.update_failed:
      return False
    if revision == self.revision:
      return True
    if not revision:
      return False  # base weights cannot come back in place
    try:
      if not delta_applies(revision, self.revision or "default"):
        logger.warning("vLLM holds %r, not a revision delta %r applies on", self.revision or "default", revision)
        return False
    except FileNotFoundError:
      return False  # the dispatcher fails the revision's records
    from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

    # No resume on failure: partially updated weights must not be served.
    try:
      await self.engine.pause_generation(mode="wait", clear_cache=True)
      await self.engine.start_weight_update()
      await self.engine.update_weights(WeightTransferUpdateRequest(update_info={"target_weights_path": revision}))
      await self.engine.finish_weight_update(weight_version=revision)
      await self.engine.reset_encoder_cache()
      await self.engine.resume_generation()
    except BaseException:
      self.update_failed = True
      raise
    self.revision = revision
    return True

  async def generate(self, r: dict[str, Any]) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    prompt = list(r["prompt_token_ids"])
    try:
      params = SamplingParams(
        **r["sampling_params"],
        logprobs=0,
        prompt_logprobs=0 if r["include_prompt_logprobs"] else None,
        output_kind=RequestOutputKind.FINAL_ONLY,
      )
    except (TypeError, ValueError) as exc:
      return failed(f"Sampler rejected the request: {exc}")
    request_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    final = None
    stream = self.engine.generate(prompt={"prompt_token_ids": prompt}, sampling_params=params, request_id=request_id, lora_request=self.adapter(r))
    async for output in stream:
      final = output
    sequences = []
    for choice in final.outputs:
      tokens = list(choice.token_ids)
      logprobs = [step[token].logprob for token, step in zip(tokens, choice.logprobs or [])]
      sequences.append({"tokens": tokens, "logprobs": logprobs, "stop_reason": choice.finish_reason})
    result: dict[str, Any] = {"type": "sample", "sequences": sequences}
    if r["include_prompt_logprobs"]:
      result["prompt_logprobs"] = [None if step is None else step[token].logprob for token, step in zip(prompt, final.prompt_logprobs)]
    return result

  def adapter(self, r: dict[str, Any]) -> Any:
    if r["model"] == self.base_model:
      return None
    from vllm.lora.request import LoRARequest

    lora_id = self.lora_ids.setdefault(r["model"], len(self.lora_ids) + 1)
    return LoRARequest(r["model"], lora_id, str(snapshot_root() / r["model"]))


def make_engine(sleep_mode: bool, is_lora: bool) -> Any:
  from vllm.config.weight_transfer import WeightTransferConfig
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine

  args = engine_args(os.environ["BASE_MODEL"], is_lora, os.environ, sleep_mode=sleep_mode)
  if transfer := args.get("weight_transfer_config"):
    args["weight_transfer_config"] = WeightTransferConfig(**transfer)
  return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=os.environ["BASE_MODEL"], **args))


async def run_sampling_worker(pool: str) -> None:
  """Boots vLLM, inside a turn when it shares its GPU, then serves the pool until the engine dies, a weight update
  breaks off, or the slicer cannot park this process. Each of those exits with an error, so the worker restarts."""
  time_slicer = time_slicer_client_from_env() if time_slicing_enabled() else None
  workload = workload_from_env(os.getpid(), name=local_workload_name("sampler", pool), claim=SAMPLER_CLAIM) if time_slicer else None
  store = SamplingStore(redis_client(os.environ["REDIS_URL"]))
  if time_slicer is not None:
    await time_slicer.register(workload)
  sampler = None
  try:
    async with holding_gpu(time_slicer, workload):
      is_lora = os.environ.get("OPEN_RL_FINE_TUNING_TYPE") == "lora"
      sampler = Sampler(make_engine(sleep_mode=time_slicer is not None, is_lora=is_lora), os.environ["BASE_MODEL"])
      if time_slicer is not None:
        await sampler.sleep()  # give the memory back before releasing the turn
    Path(READY_FILE).touch()
    dispatcher = Dispatcher(store, pool, sampler, int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256")), time_slicer, workload)
    task = asyncio.create_task(dispatcher.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    while not task.done():
      sampler_failed = sampler.engine.errored or sampler.update_failed
      time_slicer_failed = time_slicer is not None and time_slicer.faulted
      if sampler_failed or time_slicer_failed:
        task.cancel()
        raise RuntimeError(f"The sampler for pool {pool} cannot serve: {sampler_failed=} {time_slicer_failed=}")
      await asyncio.wait({task}, timeout=1)
  finally:
    # A faulted process still holds the GPU; the turn moves on once it exits and its connection closes.
    if time_slicer is not None and not time_slicer.faulted:
      await time_slicer.unregister(workload)
    if sampler is not None:
      sampler.engine.shutdown()
    await store.redis.aclose()


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL sampling pool replica with vLLM in this process")
  parser.add_argument("--pool", required=True, help="The sampling pool whose stream this replica serves")
  args = parser.parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  asyncio.run(run_sampling_worker(args.pool))


if __name__ == "__main__":
  main()
