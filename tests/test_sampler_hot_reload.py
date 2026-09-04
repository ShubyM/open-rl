import asyncio
import os
import unittest
from unittest.mock import patch

from accel_timeslicer.time_slicer import (
  NoOpTimeSlicer,
  SocketTimeSlicerClient,
  is_time_slicing_enabled,
  time_slicer_client_from_env,
)
from accel_timeslicer.workload import WorkloadRef
from server import vllm_sampler


class FakeEngine:
  def __init__(self, sleeping: bool = False):
    self.sleeping = sleeping
    self.sleep_calls: list[int] = []
    self.wake_calls: list[list[str] | None] = []
    self.rpc_calls: list[tuple[str, dict]] = []
    self.active_rpcs = 0
    self.max_concurrent_rpcs = 0

  async def is_sleeping(self) -> bool:
    return self.sleeping

  async def sleep(self, level: int = 1) -> None:
    self.sleep_calls.append(level)
    self.sleeping = True

  async def wake_up(self, tags: list[str] | None = None) -> None:
    self.wake_calls.append(tags)
    self.sleeping = False

  async def collective_rpc(self, method: str, kwargs: dict | None = None) -> None:
    self.active_rpcs += 1
    self.max_concurrent_rpcs = max(self.max_concurrent_rpcs, self.active_rpcs)
    await asyncio.sleep(0.01)
    self.rpc_calls.append((method, kwargs or {}))
    self.active_rpcs -= 1

  async def reset_prefix_cache(self) -> None:
    self.prefix_cache_resets = getattr(self, "prefix_cache_resets", 0) + 1


class NoOpTimeSlicerTest(unittest.IsolatedAsyncioTestCase):
  async def test_satisfies_client_protocol_methods(self) -> None:
    slicer = NoOpTimeSlicer()
    workload = WorkloadRef(job_id="sampler-model-a", group="samplers")

    self.assertEqual(await slicer.register(workload), {"ok": True})
    entered = False
    async with slicer.acquire(workload):
      entered = True
    self.assertTrue(entered)
    self.assertEqual(await slicer.unregister(workload), {"ok": True})
    self.assertIsNone(await slicer.close())

  async def test_satisfies_daemon_protocol_methods(self) -> None:
    slicer = NoOpTimeSlicer()
    workload = WorkloadRef(job_id="trainer-model-a", group="trainers")

    self.assertEqual(await slicer.register(workload), {"ok": True})
    self.assertEqual(await slicer.acquire(workload), {"ok": True})
    self.assertEqual(await slicer.release(workload), {"ok": True})
    self.assertEqual(await slicer.unregister(workload), {"ok": True})


class TimeSlicingEnvTest(unittest.TestCase):
  def test_time_slicing_defaults_on(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertTrue(is_time_slicing_enabled())
      self.assertIsInstance(time_slicer_client_from_env(), SocketTimeSlicerClient)

  def test_time_slicing_off_returns_noop_client(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TIME_SLICING": "off"}, clear=True):
      self.assertFalse(is_time_slicing_enabled())
      self.assertIsInstance(time_slicer_client_from_env(), NoOpTimeSlicer)

  def test_time_slicing_off_is_case_insensitive(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TIME_SLICING": "OFF"}, clear=True):
      self.assertFalse(is_time_slicing_enabled())

  def test_other_values_keep_time_slicing_on(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TIME_SLICING": "on"}, clear=True):
      self.assertTrue(is_time_slicing_enabled())


class EngineKwargsTest(unittest.TestCase):
  def test_default_kwargs_are_text_only_without_lora(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      kwargs = vllm_sampler.build_engine_kwargs("base-model")

    self.assertEqual(kwargs["model"], "base-model")
    self.assertFalse(kwargs["enable_sleep_mode"])
    # LoRA sampling lives in lora_sampler.py; this engine never loads adapters.
    self.assertFalse(kwargs["enable_lora"])
    self.assertNotIn("max_loras", kwargs)
    self.assertNotIn("language_model_only", kwargs)
    self.assertNotIn("hf_overrides", kwargs)

  def test_fft_mode_enables_sleep(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}, clear=True):
      kwargs = vllm_sampler.build_engine_kwargs("base-model")

    self.assertTrue(kwargs["enable_sleep_mode"])
    self.assertFalse(kwargs["enable_lora"])

  def test_attention_backend_env_passes_through(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_VLLM_ATTENTION_BACKEND": "FLASHINFER"}, clear=True):
      kwargs = vllm_sampler.build_engine_kwargs("base-model")

    self.assertEqual(kwargs["attention_backend"], "FLASHINFER")

  def test_architecture_override_populates_hf_overrides(self) -> None:
    env = {"VLLM_ARCHITECTURE_OVERRIDE": "Gemma4ForCausalLM"}
    with patch.dict(os.environ, env, clear=True):
      kwargs = vllm_sampler.build_engine_kwargs("base-model")

    self.assertEqual(kwargs["hf_overrides"], {"architectures": ["Gemma4ForCausalLM"]})


class ArchitectureOverrideWarningTest(unittest.TestCase):
  def test_fft_without_override_flags_reload_mismatch_risk(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}, clear=True):
      self.assertTrue(vllm_sampler.architecture_override_missing_for_fft())

  def test_override_set_is_fine(self) -> None:
    env = {"OPEN_RL_ENABLE_FFT": "true", "VLLM_ARCHITECTURE_OVERRIDE": "Gemma4ForCausalLM"}
    with patch.dict(os.environ, env, clear=True):
      self.assertFalse(vllm_sampler.architecture_override_missing_for_fft())

  def test_non_fft_never_warns(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      self.assertFalse(vllm_sampler.architecture_override_missing_for_fft())

  def test_multimodal_enabled_never_warns(self) -> None:
    env = {"OPEN_RL_ENABLE_FFT": "true", "VLLM_ENABLE_MULTIMODAL": "1"}
    with patch.dict(os.environ, env, clear=True):
      self.assertFalse(vllm_sampler.architecture_override_missing_for_fft())


# The reload_weights RPC is the full-checkpoint path; the delta strategy
# (upstream's default) ships a collective_rpc callable instead.
FULL_RELOAD_ENV = {"OPEN_RL_ENABLE_FFT": "true", "OPEN_RL_WEIGHT_SYNC_STRATEGY": "full"}


class ReloadGuardTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.engine = FakeEngine()
    self.old_engine = vllm_sampler.engine
    self.old_weights = vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS
    vllm_sampler.engine = self.engine
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = None
    # Module-level primitives may be bound to a previous test's event loop.
    vllm_sampler.reload_lock = asyncio.Lock()
    vllm_sampler.generation_idle = asyncio.Condition()
    vllm_sampler.ACTIVE_GENERATIONS = 0

  async def asyncTearDown(self) -> None:
    vllm_sampler.engine = self.old_engine
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = self.old_weights

  async def guarded_prepare(self, weights_path: str, weights_revision: str) -> None:
    """The reload guard exactly as process_sampling_request runs it."""
    async with vllm_sampler.reload_lock:
      await vllm_sampler.prepare_engine(weights_path, weights_revision)

  async def test_new_weights_path_reloads_once(self) -> None:
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await self.guarded_prepare("/tmp/w1", "rev-1")

    self.assertEqual(self.engine.rpc_calls, [("reload_weights", {"weights_path": "/tmp/w1"})])
    self.assertEqual(self.engine.wake_calls, [["weights"], ["kv_cache"]])
    # KV blocks cached under the old weights must be invalidated with them.
    self.assertEqual(getattr(self.engine, "prefix_cache_resets", 0), 1)
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "rev-1")

  async def test_same_revision_skips_reload_rpc(self) -> None:
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await self.guarded_prepare("/tmp/w1", "rev-1")
      await self.guarded_prepare("/tmp/w1", "rev-1")

    self.assertEqual(len(self.engine.rpc_calls), 1)

  async def test_revision_change_reloads_again(self) -> None:
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await self.guarded_prepare("/tmp/w1", "rev-1")
      await self.guarded_prepare("/tmp/w2", "rev-2")

    self.assertEqual(len(self.engine.rpc_calls), 2)
    self.assertEqual(self.engine.rpc_calls[1], ("reload_weights", {"weights_path": "/tmp/w2"}))
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "rev-2")

  async def test_concurrent_requests_reload_once_and_serialized(self) -> None:
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await asyncio.gather(
        self.guarded_prepare("/tmp/w1", "rev-1"),
        self.guarded_prepare("/tmp/w1", "rev-1"),
        self.guarded_prepare("/tmp/w1", "rev-1"),
      )

    self.assertEqual(len(self.engine.rpc_calls), 1)
    self.assertEqual(self.engine.max_concurrent_rpcs, 1)

  async def test_wakes_sleeping_engine_without_new_weights(self) -> None:
    self.engine.sleeping = True
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await vllm_sampler.prepare_engine(None, None)

    self.assertEqual(self.engine.rpc_calls, [])
    self.assertEqual(self.engine.wake_calls, [None])
    self.assertEqual(getattr(self.engine, "prefix_cache_resets", 0), 0)

  async def test_failed_reload_poisons_revision_and_retries_fully(self) -> None:
    fail_once = {"armed": True}
    original_rpc = self.engine.collective_rpc

    async def flaky_rpc(method, kwargs=None):
      if fail_once["armed"]:
        fail_once["armed"] = False
        raise RuntimeError("engine hiccup during reload")
      await original_rpc(method, kwargs)

    self.engine.collective_rpc = flaky_rpc
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      # Seed a previously loaded revision, then fail a reload to a new one.
      vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "rev-0"
      with self.assertRaisesRegex(RuntimeError, "engine hiccup"):
        await self.guarded_prepare("/tmp/w1", "rev-1")

      # The engine state is unknown: the old revision must not be trusted.
      self.assertIsNone(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS)

      # A follow-up request for the OLD revision must reload rather than skip.
      await self.guarded_prepare("/tmp/w0", "rev-0")

    self.assertEqual(self.engine.rpc_calls[-1], ("reload_weights", {"weights_path": "/tmp/w0"}))
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "rev-0")

  async def test_reload_drains_inflight_generations_first(self) -> None:
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await vllm_sampler.begin_generation()
      reload_task = asyncio.create_task(self.guarded_prepare("/tmp/w1", "rev-1"))
      await asyncio.sleep(0.05)

      # The reload must not sleep or swap weights under a live decode.
      self.assertFalse(reload_task.done())
      self.assertEqual(self.engine.sleep_calls, [])
      self.assertEqual(self.engine.rpc_calls, [])

      await vllm_sampler.end_generation()
      await asyncio.wait_for(reload_task, timeout=2.0)

    self.assertEqual(len(self.engine.rpc_calls), 1)
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "rev-1")

  async def test_same_revision_does_not_wait_for_drain(self) -> None:
    with patch.dict(os.environ, FULL_RELOAD_ENV, clear=True):
      await self.guarded_prepare("/tmp/w1", "rev-1")
      await vllm_sampler.begin_generation()
      try:
        await asyncio.wait_for(self.guarded_prepare("/tmp/w1", "rev-1"), timeout=1.0)
      finally:
        await vllm_sampler.end_generation()

    self.assertEqual(len(self.engine.rpc_calls), 1)

  async def test_sleep_level_two_requires_checkpoint_to_wake(self) -> None:
    self.engine.sleeping = True
    env = {"OPEN_RL_ENABLE_FFT": "true", "OPEN_RL_VLLM_SLEEP_LEVEL": "2"}
    with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(RuntimeError, "checkpoint path"):
      await vllm_sampler.prepare_engine(None, None)


class ContextWindowGuardTest(unittest.IsolatedAsyncioTestCase):
  """Oversized prompts become graceful length-stops, not run-killing errors."""

  async def _generate(self, prompt_len: int, max_tokens: int, num_samples: int = 1):
    return await vllm_sampler.run_generation_backend(
      request_id="req-1",
      prompt_token_ids=list(range(prompt_len)),
      max_tokens=max_tokens,
      temperature=1.0,
      stop=None,
      top_p=1.0,
      top_k=-1,
      num_samples=num_samples,
      lora_id=None,
      lora_path=None,
      include_prompt_logprobs=False,
    )

  async def test_prompt_at_or_over_model_len_returns_length_stop(self) -> None:
    with patch.dict(os.environ, {"VLLM_MAX_MODEL_LEN": "64"}, clear=True):
      result = await self._generate(prompt_len=64, max_tokens=16, num_samples=2)

    self.assertEqual(len(result["sequences"]), 2)
    for seq in result["sequences"]:
      self.assertEqual(seq["tokens"], [])
      self.assertEqual(seq["stop_reason"], "length")

  async def test_zero_max_tokens_returns_length_stop(self) -> None:
    with patch.dict(os.environ, {"VLLM_MAX_MODEL_LEN": "64"}, clear=True):
      result = await self._generate(prompt_len=8, max_tokens=0)

    self.assertEqual(result["sequences"][0]["stop_reason"], "length")

  async def test_overhanging_max_tokens_is_clamped_to_fit(self) -> None:
    # Engine is None in tests, so the mock branch echoes max_tokens as the
    # generated length: a 60-token prompt in a 64-token window leaves 4.
    with patch.dict(os.environ, {"VLLM_MAX_MODEL_LEN": "64"}, clear=True):
      result = await self._generate(prompt_len=60, max_tokens=16)

    self.assertEqual(len(result["sequences"][0]["tokens"]), 4)

  async def test_fitting_request_is_untouched(self) -> None:
    with patch.dict(os.environ, {"VLLM_MAX_MODEL_LEN": "64"}, clear=True):
      result = await self._generate(prompt_len=10, max_tokens=16)

    self.assertEqual(len(result["sequences"][0]["tokens"]), 16)


class SamplerReadyKeyTest(unittest.IsolatedAsyncioTestCase):
  class RedisStub:
    def __init__(self):
      self.values: dict[str, str] = {}

    async def set(self, key: str, value: str) -> None:
      self.values[key] = value

    async def get(self, key: str):
      return self.values.get(key)

    async def delete(self, key: str) -> None:
      self.values.pop(key, None)

  class StoreStub:
    def __init__(self, redis):
      self.redis = redis

  async def test_publish_and_clear_round_trip(self) -> None:
    redis = self.RedisStub()
    store = self.StoreStub(redis)

    await vllm_sampler.publish_sampler_ready(store, "model-a", "instance-1")
    self.assertEqual(redis.values, {"open_rl:sampler_ready:model-a": "instance-1"})

    await vllm_sampler.clear_sampler_ready(store, "model-a", "instance-1")
    self.assertEqual(redis.values, {})

  async def test_clear_leaves_key_owned_by_other_instance(self) -> None:
    redis = self.RedisStub()
    store = self.StoreStub(redis)
    redis.values["open_rl:sampler_ready:model-a"] = "instance-2"

    await vllm_sampler.clear_sampler_ready(store, "model-a", "instance-1")
    self.assertEqual(redis.values, {"open_rl:sampler_ready:model-a": "instance-2"})

  async def test_no_redis_is_a_noop(self) -> None:
    store = self.StoreStub(None)
    await vllm_sampler.publish_sampler_ready(store, "model-a", "instance-1")
    await vllm_sampler.clear_sampler_ready(store, "model-a", "instance-1")


if __name__ == "__main__":
  unittest.main()
