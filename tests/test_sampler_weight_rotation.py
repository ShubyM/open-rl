"""Only the newest sampler weight versions stay on the volume."""

import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from training import commands
from training import trainer_worker as tw


class RecordingTrainer(tw.Trainer):
  def __init__(self):
    model = torch.nn.Linear(1, 1)
    super().__init__("run-a", model, list(model.parameters()))
    self.saves = []

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict:
    os.makedirs(state_path, exist_ok=True)
    os.utime(state_path, (len(self.saves), len(self.saves)))
    self.saves.append(os.path.basename(state_path))
    return {"path": state_path}


class SamplerWeightRotationTest(unittest.TestCase):
  def test_only_the_newest_versions_stay_on_the_volume(self) -> None:
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}), patch.object(tw, "SAMPLER_VERSIONS_KEPT", 3):
      trainer = RecordingTrainer()
      versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
      for step in range(1, 6):
        command = commands.SaveWeightsForSampler(request_id=f"s{step}", model_id="run-a", path=f"tinker://run-a/sampler_weights/sampler-{step}")
        trainer.publish_sampler_weights(command)
        time.sleep(0.01)
      self.assertEqual(trainer.saves, [f"sampler-{s}" for s in range(1, 6)])
      self.assertEqual(sorted(os.listdir(versions)), ["sampler-3", "sampler-4", "sampler-5"])


if __name__ == "__main__":
  unittest.main()
