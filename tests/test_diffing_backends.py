"""Export format compatibility and the separation from GPU offload storage."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import safetensors.torch
import torch

from server.model_metadata import WeightSyncConfig
from tests.test_delta_weight_sync import TwoWeights, apply_export, model_weights
from training.fft_trainer_worker import FFTTrainingWorker


class SamplerExportFormatTest(unittest.TestCase):
  def test_native_coordinates_preserve_shapes_biases_and_mixed_dtypes(self):
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(3, 2).to(torch.bfloat16)
    model.norm = torch.nn.LayerNorm(2)
    worker = FFTTrainingWorker(model=model, device="cpu", cpu_offload=False, weight_sync_cfg=WeightSyncConfig())
    sampler = model_weights(model)
    with torch.no_grad():
      model.q_proj.weight[1, 2] = 5.0
      model.q_proj.bias[1] = 7.0
      model.norm.weight[0] = 1.125

    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      path = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/native")
      with open(os.path.join(path, "metadata.json")) as file:
        metadata = json.load(file)
      delta = safetensors.torch.load_file(os.path.join(path, "delta.safetensors"))
      self.assertEqual(metadata["format_version"], 2)
      self.assertEqual(metadata["layer_names"], ["q_proj.weight", "q_proj.bias", "norm.weight"])
      self.assertEqual(metadata["layer_shapes"], [[2, 3], [2], [2]])
      self.assertEqual([delta[f"{i}.indices"].tolist() for i in range(3)], [[5], [1], [0]])
      self.assertEqual([delta[f"{i}.values"].dtype for i in range(3)], [torch.bfloat16, torch.bfloat16, torch.float32])
      apply_export(path, sampler)
    for name, param in model.named_parameters():
      torch.testing.assert_close(sampler[name], param.detach(), rtol=0, atol=0)

  @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA to exercise physical offload")
  def test_sleep_wake_preserves_unexported_updates_and_optimizer_identity(self):
    model = TwoWeights().cuda()
    worker = FFTTrainingWorker(model=model, device="cuda", cpu_offload=True, base_model_name="tiny", weight_sync_cfg=WeightSyncConfig())
    sampler = model_weights(model)
    model.first.grad = torch.tensor([1.0, 0.0], device="cuda")
    worker.optim_step({"learning_rate": 0.1})
    optimizer = worker.optimizer
    params = list(worker.params)
    model.second.grad = torch.tensor([0.0, 1.0], device="cuda")
    pending_grad = model.second.grad
    expected_grad = pending_grad.clone()
    worker.sleep()
    worker.wake_up()
    self.assertIs(worker.optimizer, optimizer)
    self.assertTrue(all(before is after for before, after in zip(params, worker.params, strict=True)))
    self.assertIs(model.second.grad, pending_grad)
    torch.testing.assert_close(model.second.grad, expected_grad)
    worker.optim_step({"learning_rate": 0.1})
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      path = worker.save_for_sampler("job", None, "tinker://job/sampler_weights/after-sleep")
      apply_export(path, sampler)
    for name, param in model.named_parameters():
      torch.testing.assert_close(sampler[name], param.detach().cpu(), rtol=0, atol=0)


if __name__ == "__main__":
  unittest.main()
