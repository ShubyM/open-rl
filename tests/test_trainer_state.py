"""CPU reference tests for training state, independent of the worker runtime."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from server.model_metadata import WeightSyncConfig
from training.fft_trainer_worker import FFTTrainingWorker
from training.hf_operations import build_optimizer, forward_backward, optim_step
from training.lora_trainer_worker import LoraTrainingWorker
from training.types import Datum
from training.types import LoraConfig as WorkerLoraConfig


def tiny_model() -> LlamaForCausalLM:
  torch.manual_seed(1)
  return LlamaForCausalLM(
    LlamaConfig(
      vocab_size=32,
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=2,
      num_key_value_heads=2,
      max_position_embeddings=32,
    )
  )


def training_data() -> list[Datum]:
  return [
    Datum(model_input=[1, 2, 3], loss_fn_inputs={"target_tokens": {"data": [2, 3, 4]}}),
    Datum(model_input=[5, 6], loss_fn_inputs={"target_tokens": {"data": [6, 7]}, "weights": {"data": [0.5, 1.0]}}),
  ]


def fft_worker() -> FFTTrainingWorker:
  return FFTTrainingWorker(
    model=tiny_model(),
    tokenizer=PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 32)}}, unk_token="<unk>"))
    ),
    device="cpu",
    base_model_name="tiny-llama",
    cpu_offload=False,
    weight_sync_cfg=WeightSyncConfig(strategy="full"),
  )


class TrainerStateTest(unittest.TestCase):
  def test_accumulated_requests_match_one_batch_and_default_adamw(self) -> None:
    model = tiny_model()
    reference = copy.deepcopy(model)
    data = training_data()

    for datum in data:
      forward_backward(model, [datum], "cross_entropy")
    for datum in data:
      inputs = torch.tensor([datum.model_input])
      targets = torch.tensor([datum.loss_fn_inputs["target_tokens"].data])
      logits = reference(inputs, use_cache=False).logits
      logprobs = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
      weights = torch.tensor(datum.loss_fn_inputs["weights"].data) if "weights" in datum.loss_fn_inputs else torch.ones_like(logprobs)
      (-logprobs * weights).sum().backward()

    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
      torch.testing.assert_close(actual.grad, expected.grad)

    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-12, weight_decay=0.0)
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), float("inf"))
    reference_optimizer.step()
    optimizer = build_optimizer(list(model.parameters()), {})
    result = optim_step(optimizer, {})

    self.assertAlmostEqual(result["grad_norm:mean"], expected_norm.item())
    self.assertTrue(all(param.grad is None for param in model.parameters()))
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
      torch.testing.assert_close(actual, expected)

  def test_lora_adapters_preserve_each_others_pending_gradients_and_optimizer(self) -> None:
    worker = LoraTrainingWorker(base_model=tiny_model(), device="cpu", base_model_name="tiny-llama")
    config = WorkerLoraConfig(rank=2, lora_alpha=2, lora_dropout=0.0, seed=1)
    worker.create_adapter("a", config)
    worker.create_adapter("b", config)

    worker.forward_backward(training_data(), "cross_entropy", model_id="a")
    a_grads = [param.grad.clone() for param in worker.adapters["a"].params]
    worker.forward_backward(training_data(), "cross_entropy", model_id="b")
    b_weights = [param.detach().clone() for param in worker.adapters["b"].params]
    b_grads = [param.grad.clone() for param in worker.adapters["b"].params]
    for param, saved in zip(worker.adapters["a"].params, a_grads, strict=True):
      torch.testing.assert_close(param.grad, saved)

    worker.optim_step({"learning_rate": 1e-2}, "a")
    self.assertIsNone(worker.adapters["b"].optimizer)
    for param, weight, grad in zip(worker.adapters["b"].params, b_weights, b_grads, strict=True):
      torch.testing.assert_close(param, weight)
      torch.testing.assert_close(param.grad, grad)

    worker.optim_step({"learning_rate": 2e-2}, "b")
    self.assertIsNot(worker.adapters["a"].optimizer, worker.adapters["b"].optimizer)
    self.assertTrue(any(not torch.equal(param, before) for param, before in zip(worker.adapters["b"].params, b_weights, strict=True)))

  def test_learning_rate_changes_preserve_optimizer_moments(self) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = build_optimizer(list(model.parameters()), {"learning_rate": 0.01})
    model(torch.ones(1, 2)).sum().backward()
    optim_step(optimizer, {"learning_rate": 0.01})
    model(torch.ones(1, 2)).sum().backward()
    optim_step(optimizer, {"learning_rate": 0.02})
    self.assertEqual(optimizer.param_groups[0]["lr"], 0.02)
    self.assertEqual(optimizer.state[model.weight]["step"].item(), 2)

  def test_fft_reload_uses_fresh_parameters_and_discards_the_old_optimizer(self) -> None:
    worker = fft_worker()
    worker.forward_backward(training_data(), "cross_entropy")
    worker.optim_step({})
    previous_params = worker.params
    old_params = {id(param) for param in previous_params}

    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, "checkpoint")
      worker.save_state("m", path)
      worker.load_from_state("m", path)

    self.assertIsNone(worker.optimizer)
    new_params = {id(param) for param in worker.params}
    self.assertTrue(old_params.isdisjoint(new_params))
    worker.forward_backward(training_data(), "cross_entropy")
    worker.optim_step({})
    optimized_params = {id(param) for group in worker.optimizer.param_groups for param in group["params"]}
    self.assertEqual(optimized_params, new_params)

  def test_fft_optimizer_resume_matches_the_uninterrupted_next_update(self) -> None:
    worker = fft_worker()
    worker.forward_backward(training_data(), "cross_entropy")
    worker.optim_step({"learning_rate": 0.003, "beta1": 0.7, "beta2": 0.8, "eps": 1e-8, "weight_decay": 0.1})

    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, "checkpoint")
      worker.save_state("m", path, include_optimizer=True)
      worker.forward_backward(training_data(), "cross_entropy")
      worker.optim_step({"learning_rate": 0.002})
      uninterrupted = [param.detach().clone() for param in worker.params]

      worker.load_from_state("m", path, restore_optimizer=True)
      worker.forward_backward(training_data(), "cross_entropy")
      worker.optim_step({"learning_rate": 0.002})

    for param, expected in zip(worker.params, uninterrupted, strict=True):
      torch.testing.assert_close(param, expected)

  def test_optimizer_restore_requires_optimizer_state_for_both_backends(self) -> None:
    lora = LoraTrainingWorker(base_model=tiny_model(), device="cpu", base_model_name="tiny-llama")
    lora.create_adapter("m", WorkerLoraConfig(rank=2, seed=1))
    for worker in (fft_worker(), lora):
      with self.subTest(backend=type(worker).__name__), tempfile.TemporaryDirectory() as path:
        worker.save_state("m", path, include_optimizer=False)
        with self.assertRaisesRegex(ValueError, "has no optimizer state"):
          worker.load_from_state("m", path, restore_optimizer=True)

  def test_lora_worker_save_and_restore_select_the_named_adapter(self) -> None:
    worker = LoraTrainingWorker(base_model=tiny_model(), device="cpu", base_model_name="tiny-llama")
    config = WorkerLoraConfig(rank=2, lora_dropout=0.0, seed=1)

    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      worker.create_adapter("a", config)
      worker.create_adapter("b", config)
      worker.forward_backward(training_data(), "cross_entropy", model_id="a")
      worker.optim_step({"learning_rate": 0.01}, "a")
      expected = worker.forward_backward(training_data(), "cross_entropy", model_id="a", forward_only=True)
      previous = worker.adapters["a"]

      worker.forward_backward(training_data(), "cross_entropy", model_id="b")
      b_grads = [param.grad.clone() for param in worker.adapters["b"].params]
      path = os.path.join(directory, "checkpoint")
      worker.save_state("a", path, include_optimizer=True)

      # Advance the uninterrupted trajectory, then restore and repeat that step.
      worker.forward_backward(training_data(), "cross_entropy", model_id="a")
      worker.optim_step({"learning_rate": 0.01}, "a")
      expected_next = [param.detach().clone() for param in previous.params]
      worker.load_from_state("a", path, restore_optimizer=True)

      restored = worker.adapters["a"]
      self.assertIsNot(restored, previous)
      self.assertTrue({id(param) for param in previous.params}.isdisjoint({id(param) for param in restored.params}))
      self.assertEqual(restored.optimizer.param_groups[0]["lr"], 0.01)
      optimized_params = {id(param) for group in restored.optimizer.param_groups for param in group["params"]}
      self.assertEqual(optimized_params, {id(param) for param in restored.params})
      actual = worker.forward_backward(training_data(), "cross_entropy", model_id="a", forward_only=True)
      self.assertAlmostEqual(actual["metrics"]["loss:sum"], expected["metrics"]["loss:sum"], places=5)
      worker.forward_backward(training_data(), "cross_entropy", model_id="a")
      worker.optim_step({"learning_rate": 0.01}, "a")
      for param, uninterrupted in zip(restored.params, expected_next, strict=True):
        torch.testing.assert_close(param, uninterrupted)
      for param, saved in zip(worker.adapters["b"].params, b_grads, strict=True):
        torch.testing.assert_close(param.grad, saved)


if __name__ == "__main__":
  unittest.main()
