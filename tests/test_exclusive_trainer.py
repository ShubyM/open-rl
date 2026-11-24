import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server import training_requests_processor as trp
from server.store import InMemoryStore, RedisStore
from tests.test_queue import FakeRedis
from training import commands
from training.types import Datum, FFTConfig, TensorData


class PerModelStore(InMemoryStore):
  """The in-memory store, draining one model's queue non-destructively the way RedisStore does."""

  async def get_requests_for_model(self, model_id):
    async with self.active_tenants_cv:
      await self.active_tenants_cv.wait_for(lambda: bool(self.queues.get(model_id)))
      return list(self.queues[model_id].values())


def tiny_worker() -> trp.FFTTrainingWorker:
  torch.manual_seed(0)
  config = LlamaConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=64,
  )
  worker = trp.FFTTrainingWorker()
  worker.device = torch.device("cpu")
  # Already loaded, so create_model skips the hub download.
  worker.model = LlamaForCausalLM(config)
  worker.base_model_name = "tiny"
  return worker


class ExclusiveTrainerTest(unittest.IsolatedAsyncioTestCase):
  """An exclusive FFT trainer runs with no time slicer: it drains its own
  model's queue and saves while resident on the device."""

  async def test_a_full_step_runs_without_a_time_slicer(self) -> None:
    store = PerModelStore()
    row = [3, 5, 8, 13, 21, 34]
    datum = Datum(model_input=row[:-1], loss_fn_inputs={"target_tokens": TensorData(data=row[1:]), "weights": TensorData(data=[1.0] * 5)})
    with tempfile.TemporaryDirectory() as tmp:
      steps = [
        commands.CreateModel(
          request_id="create", model_id="run-a", base_model="tiny", fine_tuning_type="full", full_config=FFTConfig(cpu_offload=False)
        ),
        commands.ForwardBackward(request_id="fb", model_id="run-a", data=[datum]),
        commands.OptimStep(request_id="optim", model_id="run-a", adam_params={"learning_rate": 1e-3}),
        commands.SaveState(request_id="save", model_id="run-a", state_path=os.path.join(tmp, "state")),
      ]
      for step in steps:
        await store.put_request(commands.wire(step))

      with (
        patch.dict(os.environ, {"REDIS_URL": "redis://test", "OPEN_RL_TMP_DIR": tmp, "OPEN_RL_TIME_SLICING": "off"}),
        patch.object(trp, "time_slicer_client_from_env", side_effect=AssertionError("an exclusive trainer must not reach the slicer")),
      ):
        worker = tiny_worker()
        before = worker.model.lm_head.weight.detach().clone()
        processor = asyncio.create_task(trp.run_training_requests_processor(worker, "run-a", store=store))
        try:
          results = {step.request_id: await store.get_result(step.request_id, timeout=60) for step in steps}
        finally:
          processor.cancel()

      for request_id, result in results.items():
        self.assertIsNotNone(result, request_id)
        self.assertNotEqual(result.get("type"), "RequestFailedResponse", f"{request_id}: {result}")
      self.assertFalse(torch.equal(before, worker.model.lm_head.weight))
      self.assertTrue(os.listdir(os.path.join(tmp, "state")))

  async def test_crashing_mid_command_leaves_request_for_restarted_processor(self) -> None:
    class CrashingWorker:
      def optim_step(self, adam_params, model_id):
        raise asyncio.CancelledError("simulated worker crash before completion")

    class HealthyWorker:
      def optim_step(self, adam_params, model_id):
        return {"metrics": {"grad_norm:mean": 0.5}}

    for label, store in (("in_memory", PerModelStore()), ("redis_stream", RedisStore(FakeRedis()))):
      with self.subTest(store=label):
        step = commands.OptimStep(request_id="opt-crash", model_id="run-a", adam_params={"learning_rate": 1e-3})
        await store.put_request(commands.wire(step))
        with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
          crashed = trp.TrainingRequestsProcessor(store, CrashingWorker(), "run-a")
          with self.assertRaises(asyncio.CancelledError):
            await crashed.run_once()
          # Entry stayed in the queue and no future was written.
          self.assertEqual(await store.queued("run-a"), 1)
          self.assertEqual((await store.get_result("opt-crash", timeout=0.01))["type"], "try_again")

          restarted = trp.TrainingRequestsProcessor(store, HealthyWorker(), "run-a")
          await restarted.run_once()

        result = await store.get_result("opt-crash", timeout=5)
        self.assertEqual(result, {"metrics": {"grad_norm:mean": 0.5}, "type": "optim_step_completed"})
        self.assertEqual(await store.queued("run-a"), 0)


if __name__ == "__main__":
  unittest.main()
