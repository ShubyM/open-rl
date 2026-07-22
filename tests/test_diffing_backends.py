import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

from training.fft_trainer_worker import FFTTrainingWorker


class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(16, 16, bias=False)
    self.fc2 = nn.Linear(16, 16, bias=False)


class TestUniversalStreamedDiffing(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

  def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def create_worker(self, *, with_gradients: bool) -> FFTTrainingWorker:
    torch.manual_seed(42)
    model = DummyModel().to(self.device)
    worker = FFTTrainingWorker()
    worker.cpu_offload = False
    worker.base_model_name = "dummy"
    worker.model = model
    worker.prepare_model_for_training()

    if with_gradients:
      model.fc1.weight.grad = torch.zeros_like(model.fc1.weight)
      model.fc1.weight.grad[0, 0] = 1.0
      model.fc1.weight.grad[3, 5] = -1.0
      model.fc2.weight.grad = torch.zeros_like(model.fc2.weight)
      model.fc2.weight.grad[2, 2] = 1.0

    return worker

  def test_streamed_diffing_optim_step_and_multi_save_idempotency(self):
    """Verifies optimizer-coupled updates can be saved repeatedly without mutation."""
    worker = self.create_worker(with_gradients=True)
    worker.optim_step({"learning_rate": 0.1, "beta1": 0.0, "beta2": 0.0, "eps": 1e-8, "weight_decay": 0.0})

    save_dir_1 = os.path.join(self.temp_dir, "save_optim_1")
    worker.save_state_delta("dummy", save_dir_1, kind="sampler")

    save_dir_2 = os.path.join(self.temp_dir, "save_optim_2")
    worker.save_state_delta("dummy", save_dir_2, kind="state")
    with open(os.path.join(save_dir_1, "metadata.json")) as f:
      meta_1 = json.load(f)
    with open(os.path.join(save_dir_2, "metadata.json")) as f:
      meta_2 = json.load(f)

    self.assertEqual(meta_1["changed_elements"], 3)
    self.assertEqual(meta_1["changed_elements"], meta_2["changed_elements"])
    self.assertEqual(meta_1["layer_names"], meta_2["layer_names"])

  def test_streamed_diffing_empty_delta_fallback(self):
    """Verifies save_state_delta before optim_step emits an exact O(1) empty delta."""
    worker = self.create_worker(with_gradients=False)
    save_dir = os.path.join(self.temp_dir, "save_empty")
    worker.save_state_delta("dummy", save_dir, kind="sampler")
    with open(os.path.join(save_dir, "metadata.json")) as f:
      meta = json.load(f)

    self.assertEqual(meta["changed_elements"], 0)
    self.assertEqual(meta["total_elements"], worker.total_model_elements)
    self.assertEqual(meta["layer_names"], worker.model_layer_names)


if __name__ == "__main__":
  unittest.main()
