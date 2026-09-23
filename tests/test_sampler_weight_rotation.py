import asyncio
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from transformers import LlamaConfig, LlamaForCausalLM

from server import training_requests_processor as trp
from server.model_metadata import WeightSyncConfig
from server.store import InMemoryStore
from tests.test_fft_batch_failure import SlicerStub
from training import commands
from training.fft_trainer_worker import FFTTrainingWorker


class SamplerWeightRotationTest(unittest.TestCase):
  def test_only_the_newest_versions_stay_on_the_volume(self) -> None:
    with (
      tempfile.TemporaryDirectory() as tmp,
      patch.dict(os.environ, {"REDIS_URL": "redis://test", "OPEN_RL_TMP_DIR": tmp}),
      patch.object(trp, "SAMPLER_VERSIONS_KEPT", 3),
    ):
      model = LlamaForCausalLM(
        LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
      )
      worker = FFTTrainingWorker(
        model=model, device="cpu", base_model_name="tiny", cpu_offload=False, weight_sync_cfg=WeightSyncConfig(strategy="full")
      )
      proc = trp.TrainingRequestsProcessor(InMemoryStore(), worker, "run-a", time_slicer=SlicerStub())
      versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
      for step in range(1, 6):
        command = commands.SaveWeightsForSampler(request_id=f"r{step}", model_id="run-a", path=f"tinker://run-a/sampler_weights/sampler-{step}")
        asyncio.run(proc.dispatch_operation(command))
        time.sleep(0.01)
      self.assertEqual(sorted(os.listdir(versions)), ["sampler-3", "sampler-4", "sampler-5"])
      for version in os.listdir(versions):
        self.assertTrue(os.path.isfile(os.path.join(versions, version, "model.safetensors")))


if __name__ == "__main__":
  unittest.main()
