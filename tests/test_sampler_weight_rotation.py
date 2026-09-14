import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from server import training_requests_processor as trp
from server.store import InMemoryStore
from tests.test_fft_batch_failure import SlicerStub


class RecordingWorker:
  def __init__(self):
    self.saves = []

  def save_state(self, model_id, state_path, include_optimizer=False, kind="state", full=False):
    os.makedirs(state_path, exist_ok=True)
    self.saves.append((os.path.basename(state_path), full))
    return {"path": state_path}


class SamplerWeightRotationTest(unittest.TestCase):
  def test_every_nth_save_is_full_and_drops_the_deltas_before_it(self) -> None:
    with (
      tempfile.TemporaryDirectory() as tmp,
      patch.dict(os.environ, {"REDIS_URL": "redis://test", "OPEN_RL_TMP_DIR": tmp}),
      patch.object(trp, "SAMPLER_FULL_EVERY", 3),
    ):
      worker = RecordingWorker()
      proc = trp.FFTTrainingRequestsProcessor(InMemoryStore(), worker, "run-a", SlicerStub())
      versions = os.path.join(tmp, "sampler_full", "run-a", "sampler_weights")
      for step in range(1, 5):
        asyncio.run(proc.save_weights_for_sampler({"path": f"tinker://run-a/sampler_weights/sampler-{step}"}, "run-a"))
        if step == 2:
          self.assertEqual(sorted(os.listdir(versions)), ["sampler-1", "sampler-2"])
        if step == 3:
          # The full replaces the chain, so only it remains.
          self.assertEqual(os.listdir(versions), ["sampler-3"])
      self.assertEqual(worker.saves, [("sampler-1", False), ("sampler-2", False), ("sampler-3", True), ("sampler-4", False)])
      self.assertEqual(sorted(os.listdir(versions)), ["sampler-3", "sampler-4"])


if __name__ == "__main__":
  unittest.main()
