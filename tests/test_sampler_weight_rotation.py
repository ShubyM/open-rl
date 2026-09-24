import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from server import training_requests_processor as trp
from server.store import InMemoryStore
from tests.test_fft_batch_failure import SlicerStub
from tests.test_training_requests_processor import tiny_llama
from training import commands
from training.trainer_worker import TrainerWorker


class SamplerWeightRotationTest(unittest.TestCase):
  def test_only_the_newest_versions_stay_on_the_volume(self) -> None:
    with (
      tempfile.TemporaryDirectory() as tmp,
      patch.dict(os.environ, {"REDIS_URL": "redis://test", "OPEN_RL_TMP_DIR": tmp, "OPEN_RL_WEIGHT_SYNC_STRATEGY": "full"}),
      patch.object(trp, "SAMPLER_VERSIONS_KEPT", 3),
    ):
      worker = TrainerWorker()
      worker.device = torch.device("cpu")
      worker.base, worker.tokenizer = tiny_llama()
      worker.base_name = "tiny"
      proc = trp.TrainingRequestsProcessor(InMemoryStore(), worker, "run-a", time_slicer=SlicerStub())
      asyncio.run(proc.dispatch_operation(commands.CreateModel(request_id="c", model_id="run-a", base_model="tiny", fine_tuning_type="full")))
      versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
      for step in range(1, 6):
        command = commands.SaveWeightsForSampler(request_id=f"r{step}", model_id="run-a", path=f"tinker://run-a/sampler_weights/sampler-{step}")
        asyncio.run(proc.dispatch_operation(command))
        time.sleep(0.01)
      self.assertEqual(sorted(os.listdir(versions)), ["sampler-3", "sampler-4", "sampler-5"])
      self.assertTrue(os.path.exists(os.path.join(versions, "sampler-5", "model.safetensors")))


if __name__ == "__main__":
  unittest.main()
