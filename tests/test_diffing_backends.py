"""Export format compatibility and the separation from GPU offload storage."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from server.model_metadata import WeightSyncConfig
from tests.test_delta_weight_sync import TwoWeights, apply_export, model_weights
from training.fft_trainer_worker import FFTTrainingWorker
from training.weight_export import remap_hf_to_vllm_fused


class SamplerExportFormatTest(unittest.TestCase):
  def test_fused_names_preserve_weight_and_bias_offsets(self):
    model = TwoWeights()
    model.config = SimpleNamespace(hidden_size=4, num_attention_heads=2, num_key_value_heads=1, head_dim=2, intermediate_size=8)
    names = [
      "model.layers.0.self_attn.q_proj.weight",
      "model.layers.0.self_attn.k_proj.weight",
      "model.layers.0.self_attn.v_proj.bias",
      "model.layers.0.mlp.up_proj.weight",
    ]
    indices = [torch.tensor([1], dtype=torch.int64) for _ in names]
    mapped, offsets = remap_hf_to_vllm_fused(model, names, indices)
    self.assertEqual(mapped[0], "model.layers.0.self_attn.qkv_proj.weight")
    self.assertEqual(mapped[1], mapped[0])
    self.assertEqual(mapped[2], "model.layers.0.self_attn.qkv_proj.bias")
    self.assertEqual(mapped[3], "model.layers.0.mlp.gate_up_proj.weight")
    self.assertEqual([value.item() for value in offsets], [1, 17, 7, 33])
    self.assertTrue(all(value.dtype == torch.int64 for value in offsets))
    self.assertEqual([value.item() for value in indices], [1, 1, 1, 1])

  @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA to exercise physical offload")
  def test_sleep_wake_preserves_unexported_updates_and_optimizer_identity(self):
    model = TwoWeights().cuda()
    worker = FFTTrainingWorker(
      model=model, device="cuda", cpu_offload=True, base_model_name="tiny", weight_sync_cfg=WeightSyncConfig(delta_format="native")
    )
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
