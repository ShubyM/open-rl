"""CPU reference tests for training state, independent of the worker runtime."""

import copy
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraTrainingWorker
from training.trainer import Trainer
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


class TrainerStateTest(unittest.TestCase):
  def test_accumulated_requests_match_one_batch_and_default_adamw(self) -> None:
    model = tiny_model()
    reference = copy.deepcopy(model)
    trainer = Trainer(model, list(model.parameters()))
    data = training_data()

    for datum in data:
      trainer.forward_backward([datum], "cross_entropy")
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
    result = trainer.optim_step({})

    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], expected_norm.item())
    self.assertTrue(all(param.grad is None for param in trainer.params))
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
      torch.testing.assert_close(actual, expected)

  def test_lora_trainers_preserve_each_others_pending_gradients_and_optimizer(self) -> None:
    config = LoraConfig(task_type="CAUSAL_LM", r=2, lora_alpha=2, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(tiny_model(), config, adapter_name="a")
    model.add_adapter("b", config)
    trainers = {}
    for name in ("a", "b"):
      model.set_adapter(name)
      trainers[name] = Trainer(model, [param for param in model.parameters() if param.requires_grad])

    model.set_adapter("a")
    trainers["a"].forward_backward(training_data(), "cross_entropy")
    a_grads = [param.grad.clone() for param in trainers["a"].params]
    model.set_adapter("b")
    trainers["b"].forward_backward(training_data(), "cross_entropy")
    b_weights = [param.detach().clone() for param in trainers["b"].params]
    b_grads = [param.grad.clone() for param in trainers["b"].params]
    for param, saved in zip(trainers["a"].params, a_grads, strict=True):
      torch.testing.assert_close(param.grad, saved)

    model.set_adapter("a")
    trainers["a"].optim_step({"learning_rate": 1e-2})
    self.assertIsNone(trainers["b"].optimizer)
    for param, weight, grad in zip(trainers["b"].params, b_weights, b_grads, strict=True):
      torch.testing.assert_close(param, weight)
      torch.testing.assert_close(param.grad, grad)

    model.set_adapter("b")
    trainers["b"].optim_step({"learning_rate": 2e-2})
    self.assertIsNot(trainers["a"].optimizer, trainers["b"].optimizer)
    self.assertTrue(any(not torch.equal(param, before) for param, before in zip(trainers["b"].params, b_weights, strict=True)))

  def test_learning_rate_changes_preserve_optimizer_moments(self) -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    trainer = Trainer(model, list(model.parameters()))
    model(torch.ones(1, 2)).sum().backward()
    trainer.optim_step({"learning_rate": 0.01})
    optimizer = trainer.optimizer
    model(torch.ones(1, 2)).sum().backward()
    trainer.optim_step({"learning_rate": 0.02})
    self.assertIs(trainer.optimizer, optimizer)
    self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.02)
    self.assertEqual(trainer.optimizer.state[model.weight]["step"].item(), 2)

  def test_fft_reload_uses_fresh_parameters_and_discards_the_old_optimizer(self) -> None:
    worker = FFTTrainingWorker()
    worker.device = torch.device("cpu")
    worker.model = tiny_model()
    worker.base_model_name = "tiny-llama"
    worker.tokenizer = PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 32)}}, unk_token="<unk>"))
    )
    worker.cpu_offload = False
    worker.set_weight_sync_strategy("full")
    worker.prepare_model_for_training()
    worker.forward_backward(training_data(), "cross_entropy")
    worker.optim_step({})
    previous = worker.trainer
    old_params = {id(param) for param in previous.params}

    with tempfile.TemporaryDirectory() as directory:
      path = os.path.join(directory, "checkpoint")
      worker.save_state("m", path)
      worker.load_from_state("m", path)

    self.assertIsNot(worker.trainer, previous)
    self.assertIsNone(worker.trainer.optimizer)
    new_params = {id(param) for param in worker.trainer.params}
    self.assertTrue(old_params.isdisjoint(new_params))
    worker.forward_backward(training_data(), "cross_entropy")
    worker.optim_step({})
    optimized_params = {id(param) for group in worker.trainer.optimizer.param_groups for param in group["params"]}
    self.assertEqual(optimized_params, new_params)

  def test_lora_worker_save_and_restore_select_the_named_trainer(self) -> None:
    worker = LoraTrainingWorker()
    worker.device = torch.device("cpu")
    worker.base_model = tiny_model()
    worker.base_model_name = "tiny-llama"
    config = WorkerLoraConfig(rank=2, lora_dropout=0.0, seed=1)

    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      worker.create_adapter("a", config)
      worker.create_adapter("b", config)
      worker.forward_backward(training_data(), "cross_entropy", model_id="a")
      worker.optim_step({"learning_rate": 0.01}, "a")
      expected = worker.forward_backward(training_data(), "cross_entropy", model_id="a", forward_only=True)
      previous = worker.trainers["a"]

      worker.forward_backward(training_data(), "cross_entropy", model_id="b")
      b_grads = [param.grad.clone() for param in worker.trainers["b"].params]
      path = os.path.join(directory, "checkpoint")
      worker.save_state("a", path, include_optimizer=True)

      # Advance the uninterrupted trajectory, then restore and repeat that step.
      worker.forward_backward(training_data(), "cross_entropy", model_id="a")
      worker.optim_step({"learning_rate": 0.01}, "a")
      expected_next = [param.detach().clone() for param in previous.params]
      worker.load_from_state("a", path, restore_optimizer=True)

      restored = worker.trainers["a"]
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
      for param, saved in zip(worker.trainers["b"].params, b_grads, strict=True):
        torch.testing.assert_close(param.grad, saved)


if __name__ == "__main__":
  unittest.main()
