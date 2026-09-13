import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from server import gateway, trainer_config, vllm_sampler
from server.store import InMemoryStore
from training import distributed


class TrainerConfigTest(unittest.TestCase):
  def test_explicit_backend_overrides_legacy_flag_at_both_ends(self) -> None:
    cases = (
      ({}, "lora", False, True),
      ({"OPEN_RL_ENABLE_FFT": "true"}, "fft", True, False),
      ({"OPEN_RL_TRAINER_BACKEND": "lora", "OPEN_RL_ENABLE_FFT": "true"}, "lora", False, True),
      ({"OPEN_RL_TRAINER_BACKEND": "megatron"}, "megatron", True, True),
      ({"OPEN_RL_TRAINER_BACKEND": "automodel"}, "automodel", True, False),
      ({"OPEN_RL_TRAINER_BACKEND": "automodel", "OPEN_RL_AUTOMODEL_LORA_RANK": "32"}, "automodel", True, True),
    )
    for env, backend, dedicated, adapter in cases:
      with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
        self.assertEqual(trainer_config.trainer_backend(), backend)
        self.assertEqual(gateway.uses_dedicated_trainer(), dedicated)
        self.assertEqual(gateway.trainer_publishes_adapter(), adapter)
        kwargs = vllm_sampler.build_engine_kwargs("base-model")
        self.assertEqual(kwargs["enable_lora"], adapter)
        self.assertEqual(kwargs["enable_sleep_mode"], not adapter)

  def test_unknown_backend_fails_before_routing_requests(self) -> None:
    with (
      patch.dict(os.environ, {"OPEN_RL_TRAINER_BACKEND": "typo"}, clear=True),
      self.assertRaisesRegex(RuntimeError, "Unknown OPEN_RL_TRAINER_BACKEND"),
    ):
      gateway.gateway_launches_trainers()

  def test_external_automodel_does_not_launch_another_trainer(self) -> None:
    env = {"OPEN_RL_TRAINER_BACKEND": "automodel", "OPEN_RL_EXTERNAL_TRAINER": "1"}
    with patch.dict(os.environ, env, clear=True):
      self.assertFalse(gateway.gateway_launches_trainers())


class AutomodelSamplingRoutingTest(unittest.IsolatedAsyncioTestCase):
  async def test_deleting_a_model_only_stops_gateway_owned_trainers(self) -> None:
    for external in ("0", "1"):
      env = {"OPEN_RL_TRAINER_BACKEND": "automodel", "OPEN_RL_EXTERNAL_TRAINER": external}
      store = AsyncMock()
      with self.subTest(external=external), patch.dict(os.environ, env, clear=True), patch.object(gateway, "store", store):
        result = await gateway.delete_model({"model_id": "model-a"})
      self.assertEqual(result, {"status": "ok"})
      if external == "1":
        store.put_request.assert_not_awaited()
      else:
        store.put_request.assert_awaited_once()
      store.put_sampling_request.assert_awaited_once()
      store.delete_values.assert_awaited_once()

  async def test_adapter_requests_reach_external_sampler_without_legacy_flag(self) -> None:
    env = {
      "OPEN_RL_TRAINER_BACKEND": "automodel",
      "OPEN_RL_AUTOMODEL_LORA_RANK": "32",
      "SAMPLER_BASE_URL": "http://sampler:8000",
      "OPEN_RL_SNAPSHOT_DIR": "/snapshots",
    }
    store = InMemoryStore()
    with (
      patch.dict(os.environ, env, clear=True),
      patch.object(gateway, "store", store),
      patch.object(gateway, "_sample_via_external_server", new_callable=AsyncMock) as sample,
    ):
      result = await gateway.asample({"sampling_session_id": "tinker://model-a/sampler_weights/sampler-1"})
      await asyncio.gather(*gateway._external_sampler_tasks)

    sample.assert_awaited_once()
    request_id, request = sample.call_args.args
    self.assertEqual(request_id, result["request_id"])
    self.assertEqual(request["lora_path"], "/snapshots/model-a/sampler-1")
    self.assertIsNone(request["weights_path"])

  async def test_full_weights_with_external_sampler_fail_at_startup(self) -> None:
    env = {"OPEN_RL_TRAINER_BACKEND": "automodel", "SAMPLER_BASE_URL": "http://sampler:8000"}
    with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(RuntimeError, "LoRA adapters"):
      async with gateway.lifespan(gateway.app):
        pass


class SingleRankProcessGroupTest(unittest.TestCase):
  def test_automodel_can_initialize_a_one_rank_torchrun_group(self) -> None:
    env = {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"}
    with (
      patch.dict(os.environ, env, clear=True),
      patch.object(distributed.dist, "is_initialized", return_value=False),
      patch.object(distributed.dist, "init_process_group") as initialize,
      patch.object(distributed.torch.cuda, "is_available", return_value=False),
    ):
      distributed.initialize(require_process_group=True)
    initialize.assert_called_once()

  def test_ordinary_single_gpu_training_does_not_require_torchrun(self) -> None:
    with (
      patch.dict(os.environ, {}, clear=True),
      patch.object(distributed.dist, "is_initialized", return_value=False),
      patch.object(distributed.dist, "init_process_group") as initialize,
    ):
      distributed.initialize()
    initialize.assert_not_called()
