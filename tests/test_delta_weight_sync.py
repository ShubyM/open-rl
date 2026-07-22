import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn
from safetensors.torch import load_file

from training.fft_trainer_worker import FFTTrainingWorker


class SimpleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(10, 10, bias=False)

  def forward(self, x):
    return self.fc(x)


class DeltaWeightSyncTest(unittest.TestCase):
  def setUp(self):
    self.test_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.test_dir, ignore_errors=True)

  def test_sparse_delta_encoding_and_lossless_overwrite(self):
    worker = FFTTrainingWorker()
    worker.cpu_offload = False
    worker.base_model_name = "test-simple-model"
    worker.model = SimpleModel()
    worker.prepare_model_for_training()

    # Run AdamW with a gradient that changes 2 out of 100 elements.
    orig_w0 = worker.model.fc.weight.data.clone()
    worker.model.fc.weight.grad = torch.zeros_like(worker.model.fc.weight)
    worker.model.fc.weight.grad[0, 2] = 1.0
    worker.model.fc.weight.grad[5, 7] = -1.0

    worker.optim_step({"learning_rate": 0.1, "beta1": 0.0, "beta2": 0.0, "eps": 1e-8, "weight_decay": 0.0})
    state_path = os.path.join(self.test_dir, "step_1")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    # 1. Verify metadata
    metadata_path = os.path.join(state_path, "metadata.json")
    self.assertTrue(os.path.exists(metadata_path))
    with open(metadata_path) as f:
      meta = json.load(f)
    self.assertEqual(meta["format"], "sparse_delta")
    self.assertEqual(meta["changed_elements"], 2)
    self.assertEqual(meta["total_elements"], 100)
    self.assertEqual(meta["density_pct"], 2.0)

    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    sparse_delta = load_file(delta_file)

    self.assertIn("delta.indices_flat", sparse_delta)
    self.assertIn("delta.values_flat", sparse_delta)
    self.assertEqual(sparse_delta["delta.indices_flat"].numel(), 2)
    self.assertEqual(sparse_delta["delta.indices_flat"].dtype, torch.int32)

    # 3. Verify Lossless Selective Overwrite reproduces exact target W1
    simulated_sampler_weight = orig_w0.clone()
    indices = sparse_delta["delta.indices_flat"]
    values = sparse_delta["delta.values_flat"]
    simulated_sampler_weight.view(-1)[indices.to(torch.int64)] = values

    self.assertTrue(
      torch.equal(simulated_sampler_weight, worker.model.fc.weight.data),
      "Lossless selective overwrite must produce bitwise identical tensors (0 ULP drift)",
    )

    # 4. Verify worker's CPU shadow was updated to W1 so next step diffs correctly
    self.assertTrue(
      torch.equal(worker.param_shadow[worker.model.fc.weight][1], worker.model.fc.weight.data.cpu()),
      "Worker shadow must be updated after delta save",
    )

  def test_dense_optimizer_update_uses_absolute_changed_tensor(self):
    worker = FFTTrainingWorker()
    worker.cpu_offload = False
    worker.base_model_name = "test-simple-model"
    worker.model = SimpleModel()
    worker.prepare_model_for_training()

    worker.model.fc.weight.grad = torch.ones_like(worker.model.fc.weight)
    worker.optim_step({"learning_rate": 0.1, "beta1": 0.0, "beta2": 0.0, "eps": 1e-8, "weight_decay": 0.0})

    state_path = os.path.join(self.test_dir, "dense_step")
    result = worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")
    with open(os.path.join(state_path, "metadata.json")) as f:
      metadata = json.load(f)
    update = load_file(os.path.join(state_path, "delta.safetensors"))

    self.assertEqual(result["format"], "absolute_tensors")
    self.assertEqual(metadata["format"], "absolute_tensors")
    self.assertEqual(metadata["changed_elements"], 100)
    self.assertLessEqual(metadata["dense_bytes"], metadata["sparse_bytes"])
    self.assertEqual(set(update), {"fc.weight"})
    self.assertTrue(torch.equal(update["fc.weight"], worker.model.fc.weight.detach().cpu()))

  def test_parameter_without_gradient_is_not_considered_for_sync(self):
    worker = FFTTrainingWorker()
    worker.cpu_offload = False
    worker.base_model_name = "test-simple-model"
    worker.model = SimpleModel()
    worker.prepare_model_for_training()

    worker.optim_step({"weight_decay": 0.0})
    state_path = os.path.join(self.test_dir, "no_gradient_step")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")
    with open(os.path.join(state_path, "metadata.json")) as f:
      metadata = json.load(f)

    self.assertEqual(metadata["layer_names"], [])
    self.assertEqual(metadata["changed_elements"], 0)

  def test_weight_sync_strategy_selection(self):
    worker = FFTTrainingWorker()
    self.assertEqual(worker.weight_sync_strategy, "delta")
    worker.set_weight_sync_strategy("full")
    self.assertEqual(worker.weight_sync_strategy, "full")
    with self.assertRaises(ValueError):
      worker.set_weight_sync_strategy("invalid_strategy")

  def test_save_state_delta_with_offloading(self):
    """Test that save_state_delta() succeeds cleanly when the model is offloaded."""
    worker = FFTTrainingWorker()
    worker.base_model_name = "test-offload-model"
    worker.model = SimpleModel()
    worker.prepare_model_for_training()

    worker.model.fc.weight.grad = torch.zeros_like(worker.model.fc.weight)
    worker.model.fc.weight.grad[1, 1] = 1.0
    worker.optim_step({"learning_rate": 0.1, "beta1": 0.0, "beta2": 0.0, "eps": 1e-8, "weight_decay": 0.0})
    expected_value = worker.model.fc.weight.detach()[1, 1].item()

    # The update payload must remain saveable after the live parameter is offloaded.
    worker.is_offloaded = True
    worker.model.fc.weight.data = torch.empty(0, dtype=worker.model.fc.weight.dtype, device="cpu")

    state_path = os.path.join(self.test_dir, "step_offload")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    # Verify that delta.safetensors was cleanly saved from offloaded CPU buffer
    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    sparse_delta = load_file(delta_file)
    self.assertIn("delta.indices_flat", sparse_delta)
    self.assertEqual(sparse_delta["delta.indices_flat"].numel(), 1)
    self.assertEqual(sparse_delta["delta.values_flat"][0].item(), expected_value)


if __name__ == "__main__":
  unittest.main()
