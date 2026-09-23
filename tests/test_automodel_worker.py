"""Automodel worker checks that run on CPU without nemo-automodel."""

import os
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server.training_requests_processor import build_worker
from training import automodel_worker
from training.automodel_worker import AutomodelTrainingWorker
from training.trainer_worker import BaseTrainerWorker


class ClipGradientsTest(unittest.TestCase):
  """The per-tensor norm stack equals torch's clip_grad_norm_ on plain tensors."""

  def make_worker(self):
    torch.manual_seed(1)
    params = [torch.nn.Parameter(torch.randn(3, 4)), torch.nn.Parameter(torch.randn(5))]
    for param in params:
      param.grad = torch.randn_like(param)
    worker = AutomodelTrainingWorker()
    worker.trainable_params = params
    return worker, params

  def test_clips_like_torch(self) -> None:
    worker, params = self.make_worker()
    reference = [torch.nn.Parameter(param.detach().clone()) for param in params]
    for ref, param in zip(reference, params, strict=True):
      ref.grad = param.grad.clone()

    total = worker.clip_gradients(0.5)
    expected_total = torch.nn.utils.clip_grad_norm_(reference, 0.5)

    self.assertAlmostEqual(total, expected_total.item(), places=5)
    for ref, param in zip(reference, params, strict=True):
      torch.testing.assert_close(param.grad, ref.grad)

  def test_no_clipping_under_the_threshold(self) -> None:
    worker, params = self.make_worker()
    before = [param.grad.clone() for param in params]
    worker.clip_gradients(float("inf"))
    for grad, param in zip(before, params, strict=True):
      torch.testing.assert_close(param.grad, grad)


class BuildWorkerTest(unittest.TestCase):
  def test_backend_env_selects_automodel(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TRAINER_BACKEND": "automodel"}):
      self.assertIsInstance(build_worker(is_lora=True), AutomodelTrainingWorker)

  def test_an_unsharded_worker_is_one_shard(self) -> None:
    # No mesh until load_base_model, so the base loop runs every datum.
    worker = AutomodelTrainingWorker()
    self.assertEqual((worker.shard_rank(), worker.shard_count()), (0, 1))


class ChunkedLogprobsTest(unittest.TestCase):
  """Chunked projection from hidden states matches the full-logits path, values and gradients."""

  def test_matches_full_logits(self) -> None:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
    model = LlamaForCausalLM(config)
    input_ids = torch.randint(0, 64, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 5:] = 0
    targets = torch.randint(0, 64, (2, 7))

    def logprobs_and_grad(compute):
      model.zero_grad()
      logprobs = compute(model, input_ids, attention_mask, targets)
      logprobs[attention_mask.bool()].sum().backward()
      return logprobs.detach(), model.model.embed_tokens.weight.grad.clone()

    expected, expected_grad = logprobs_and_grad(BaseTrainerWorker().compute_target_logprobs)
    with patch.object(automodel_worker, "LOGPROB_CHUNK", 3):
      actual, actual_grad = logprobs_and_grad(AutomodelTrainingWorker().compute_target_logprobs)

    mask = attention_mask.bool()
    torch.testing.assert_close(actual[mask], expected[mask])
    torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
  unittest.main()
