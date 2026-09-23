import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

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

    # Simulate an Adam update where 2 out of 100 elements change (2% sparsity)
    orig_w0 = worker.model.fc.weight.data.clone()
    worker.model.fc.weight.data[0, 2] = 42.0
    worker.model.fc.weight.data[5, 7] = -13.37

    worker.optim_step({})
    state_path = os.path.join(self.test_dir, "step_1")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    # 1. Verify metadata
    metadata_path = os.path.join(state_path, "metadata.json")
    self.assertTrue(os.path.exists(metadata_path))
    with open(metadata_path) as f:
      meta = json.load(f)
    self.assertEqual(meta["format"], "sparse_delta")
    self.assertEqual(meta["format_version"], 2)
    self.assertEqual(meta["layer_shapes"], [[10, 10]])
    self.assertEqual(meta["changed_elements"], 2)
    self.assertEqual(meta["total_elements"], 100)
    self.assertEqual(meta["density_pct"], 2.0)

    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    import safetensors.torch

    sparse_delta = safetensors.torch.load_file(delta_file)

    self.assertIn("0.indices", sparse_delta)
    self.assertIn("0.values", sparse_delta)
    self.assertEqual(sparse_delta["0.indices"].numel(), 2)
    self.assertEqual(sparse_delta["0.indices"].dtype, torch.int32)

    # 3. Verify Lossless Selective Overwrite reproduces exact target W1
    simulated_sampler_weight = orig_w0.clone()
    indices = sparse_delta["0.indices"]
    values = sparse_delta["0.values"]
    simulated_sampler_weight.view(-1)[indices.to(torch.int64)] = values

    self.assertTrue(
      torch.equal(simulated_sampler_weight, worker.model.fc.weight.data),
      "Lossless selective overwrite must produce bitwise identical tensors (0 ULP drift)",
    )

    # 4. Verify next step diffs against W1 (only newly changed elements emitted)
    worker.model.fc.weight.data[3, 4] = 99.0
    worker.optim_step({})
    state_path_2 = os.path.join(self.test_dir, "step_2")
    worker.save_state_delta(model_id="test-model", state_path=state_path_2, kind="sampler")
    with open(os.path.join(state_path_2, "metadata.json")) as f:
      meta_2 = json.load(f)
    self.assertEqual(meta_2["changed_elements"], 1)

  def test_weight_sync_strategy_selection(self):
    worker = FFTTrainingWorker()
    self.assertEqual(worker.weight_sync_cfg.strategy, "delta")
    worker.set_weight_sync_strategy("full")
    self.assertEqual(worker.weight_sync_cfg.strategy, "full")
    with self.assertRaises(ValueError):
      worker.set_weight_sync_strategy("invalid_strategy")

  def test_save_state_delta_with_offloading(self):
    """Test that save_state_delta() succeeds cleanly after sleep() offloads the model to CPU."""
    worker = FFTTrainingWorker()
    worker.base_model_name = "test-offload-model"
    worker.model = SimpleModel().to(worker.device)
    worker.prepare_model_for_training()

    with torch.no_grad():
      worker.model.fc.weight[1, 1] = 77.7
    worker.optim_step({})
    worker.sleep()

    state_path = os.path.join(self.test_dir, "step_offload")
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    # Verify that delta.safetensors was cleanly saved from offloaded CPU parameters
    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    import safetensors.torch

    sparse_delta = safetensors.torch.load_file(delta_file)
    self.assertIn("0.indices", sparse_delta)
    self.assertEqual(sparse_delta["0.indices"].numel(), 1)
    self.assertAlmostEqual(sparse_delta["0.values"][0].item(), 77.7, places=4)


if __name__ == "__main__":
  unittest.main()
