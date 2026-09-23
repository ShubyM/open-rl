import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

from training.fft_trainer_worker import FFTConfig, FFTTrainingWorker, SparseDelta


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

    # 4. The baseline advanced to W1, so a step that changes nothing publishes nothing.
    worker.optim_step({})
    self.assertEqual(worker.pending_delta.changed_elements, 0, "Baseline must advance after a delta is taken")

  def test_weight_sync_strategy_selection(self):
    worker = FFTTrainingWorker()
    self.assertEqual(worker.weight_sync_strategy, "delta")
    self.assertEqual(FFTConfig(weight_sync_strategy="full").weight_sync_strategy, "full")
    with self.assertRaises(ValueError):
      FFTConfig(weight_sync_strategy="invalid_strategy")

  def test_save_state_delta_with_offloading(self):
    """save_state_delta publishes the pending delta while the worker is off the GPU, and refuses while it is on."""
    worker = FFTTrainingWorker()
    worker.base_model_name = "test-offload-model"
    worker.model = SimpleModel()
    worker.prepare_model_for_training()

    # What optim_step left behind before the lease was released.
    worker.pending_delta = SparseDelta(
      total_elements=100,
      names=["fc.weight"],
      shapes=[[10, 10]],
      indices=[torch.tensor([1 * 10 + 1], dtype=torch.int32)],
      values=[torch.tensor([77.7], dtype=torch.float32)],
    )

    state_path = os.path.join(self.test_dir, "step_offload")
    with self.assertRaises(RuntimeError):
      worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    worker.mirror.offloaded = True
    worker.save_state_delta(model_id="test-model", state_path=state_path, kind="sampler")

    delta_file = os.path.join(state_path, "delta.safetensors")
    self.assertTrue(os.path.exists(delta_file))
    import safetensors.torch

    sparse_delta = safetensors.torch.load_file(delta_file)
    self.assertIn("0.indices", sparse_delta)
    self.assertEqual(sparse_delta["0.indices"].numel(), 1)
    self.assertAlmostEqual(sparse_delta["0.values"][0].item(), 77.7, places=4)
    with open(os.path.join(state_path, "metadata.json")) as f:
      meta = json.load(f)
    self.assertEqual(meta["layer_shapes"], [[10, 10]])
    self.assertEqual(meta["changed_elements"], 1)


if __name__ == "__main__":
  unittest.main()
