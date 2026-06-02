import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests._server_fixture import SERVER_DIR


def _load_trainer_module():
  stubs = {
    "peft": types.SimpleNamespace(
      LoraConfig=object,
      PeftModelForCausalLM=object,
      get_peft_model=lambda *_args, **_kwargs: None,
    ),
    "transformers": types.SimpleNamespace(
      AutoModelForCausalLM=object,
      AutoTokenizer=object,
      PreTrainedModel=object,
      PreTrainedTokenizerBase=object,
    ),
  }
  spec = importlib.util.spec_from_file_location("trainer_under_test", Path(SERVER_DIR) / "trainer.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(module)
  return module


trainer_module = _load_trainer_module()
TrainerEngine = trainer_module.TrainerEngine


class _PeftModelStub:
  def __init__(self, adapter_params):
    self.adapter_params = adapter_params
    self.active_adapter = None

  def set_adapter(self, adapter_id):
    self.active_adapter = adapter_id
    for param in self.parameters():
      param.requires_grad_(False)
    for param in self.adapter_params[adapter_id]:
      param.requires_grad_(True)

  def parameters(self):
    for params in self.adapter_params.values():
      yield from params

  def save_pretrained(self, *_args, **_kwargs):
    return None


class _TokenizerStub:
  pad_token_id = 0


class _LogitModelStub:
  def __init__(self, vocab_size: int = 17):
    self.vocab_size = vocab_size
    self.calls = []

  def train(self):
    return None

  def __call__(self, input_tensor, attention_mask=None, **_kwargs):
    if attention_mask is not None:
      self.calls.append((input_tensor.detach().clone(), attention_mask.detach().clone()))
    vocab = torch.arange(self.vocab_size, dtype=torch.float32, device=input_tensor.device).view(1, 1, -1)
    positions = torch.arange(input_tensor.shape[1], dtype=torch.float32, device=input_tensor.device).view(1, -1, 1)
    logits = torch.cos(input_tensor.float().unsqueeze(-1) * 0.11 + positions * 0.07 + vocab * 0.13)
    logits.requires_grad_()
    return types.SimpleNamespace(logits=logits)


def _datum(model_input, target_tokens, *, weights=None, logprobs=None, advantages=None):
  loss_fn_inputs = {"target_tokens": trainer_module.TensorData(data=target_tokens)}
  if weights is not None:
    loss_fn_inputs["weights"] = trainer_module.TensorData(data=weights)
  if logprobs is not None:
    loss_fn_inputs["logprobs"] = trainer_module.TensorData(data=logprobs)
  if advantages is not None:
    loss_fn_inputs["advantages"] = trainer_module.TensorData(data=advantages)
  return trainer_module.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


class TestTrainerOptimizerCorrectness(unittest.TestCase):
  def test_optim_step_only_updates_active_adapter_params(self) -> None:
    active_param = torch.nn.Parameter(torch.tensor([1.0]))
    other_param = torch.nn.Parameter(torch.tensor([1.0]))
    active_param.grad = torch.tensor([1.0])
    other_param.grad = torch.tensor([10.0])

    engine = TrainerEngine()
    engine.peft_model = _PeftModelStub(
      {
        "adapter-a": [active_param],
        "adapter-b": [other_param],
      }
    )
    engine.adapter_states = {
      "adapter-a": {"trainable_params": trainer_module.active_adapter_parameters(engine.peft_model, "adapter-a"), "optimizer": None}
    }
    engine.save_adapter = lambda *_args, **_kwargs: None

    result = engine.optim_step(
      {
        "learning_rate": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
        "eps": 1e-8,
        "weight_decay": 0.0,
      },
      "adapter-a",
    )

    self.assertEqual(engine.peft_model.active_adapter, "adapter-a")
    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(active_param.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(other_param.detach(), torch.tensor([1.0])))
    if active_param.grad is not None:
      self.assertTrue(torch.allclose(active_param.grad, torch.zeros_like(active_param.grad)))
    self.assertIsNotNone(other_param.grad)

  def test_optim_step_updates_full_model_params_without_adapter(self) -> None:
    full_param = torch.nn.Parameter(torch.tensor([1.0]))
    full_param.grad = torch.tensor([1.0])

    engine = TrainerEngine()
    engine.base_model = types.SimpleNamespace()
    engine.peft_model = None
    engine.full_model_id = "full-run"
    engine.adapter_states = {"full-run": {"trainable_params": [full_param], "optimizer": None, "training_mode": "full"}}
    engine.save_adapter = lambda *_args, **_kwargs: self.fail("full fine-tuning optim_step must not save a LoRA adapter")

    result = engine.optim_step(
      {
        "learning_rate": 0.1,
        "beta1": 0.0,
        "beta2": 0.0,
        "eps": 1e-8,
        "weight_decay": 0.0,
      },
      "full-run",
    )

    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(full_param.detach(), torch.tensor([1.0])))
    if full_param.grad is not None:
      self.assertTrue(torch.allclose(full_param.grad, torch.zeros_like(full_param.grad)))


class TestTrainerPaddedBatchingMath(unittest.TestCase):
  def _engine(self) -> TrainerEngine:
    engine = TrainerEngine()
    engine.device = torch.device("cpu")
    engine.tokenizer = _TokenizerStub()
    engine.peft_model = _LogitModelStub()
    return engine

  def _data(self):
    return [
      _datum(
        [3, 4, 5, 6],
        [1, 2, 3, 4],
        weights=[1.0, 0.5, 0.25, 2.0],
        logprobs=[-0.1, -0.2, -0.3, -0.4],
        advantages=[1.0, -0.5, 2.0, 0.25],
      ),
      _datum(
        [7, 8],
        [2, 3],
        logprobs=[-0.7, -0.8],
        advantages=[0.75, 1.25],
      ),
      _datum(
        [9, 10, 11],
        [5, 6, 7, 8],
        weights=[0.2, 0.4, 0.6, 0.8],
        logprobs=[-0.9, -1.0, -1.1, -1.2],
        advantages=[-1.0, 0.3, 0.9, 1.7],
      ),
    ]

  def test_padded_batch_logprobs_and_losses_match_per_example_math(self) -> None:
    engine = self._engine()
    data = self._data()

    batch_logprobs, batch_weights, batch_aux, batch_lengths = engine._get_logprobs_batch(data)
    single_results = [engine._get_logprobs_batch([datum]) for datum in data]

    for row, (single_logprobs, single_weights, single_aux, single_lengths) in enumerate(single_results):
      length = batch_lengths[row]
      self.assertEqual(length, single_lengths[0])
      torch.testing.assert_close(batch_logprobs[row, :length], single_logprobs[0, :length])
      torch.testing.assert_close(batch_weights[row, :length], single_weights[0, :length])
      torch.testing.assert_close(batch_weights[row, length:], torch.zeros_like(batch_weights[row, length:]))
      for key in ("logprobs", "advantages"):
        torch.testing.assert_close(batch_aux[key][row, :length], single_aux[key][0, :length])
        torch.testing.assert_close(batch_aux[key][row, length:], torch.zeros_like(batch_aux[key][row, length:]))

    def single_sum(fn):
      losses = [fn(logprobs, weights, aux).sum() for logprobs, weights, aux, _lengths in single_results]
      return torch.stack(losses).sum()

    torch.testing.assert_close(
      engine._cross_entropy_loss(batch_logprobs, batch_weights).sum(),
      single_sum(lambda logprobs, weights, _aux: engine._cross_entropy_loss(logprobs, weights)),
    )
    torch.testing.assert_close(
      engine._importance_sampling_loss(batch_logprobs, batch_weights, batch_aux).sum(),
      single_sum(lambda logprobs, weights, aux: engine._importance_sampling_loss(logprobs, weights, aux)),
    )
    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      engine._ppo_loss(batch_logprobs, batch_weights, batch_aux, ppo_config).sum(),
      single_sum(lambda logprobs, weights, aux: engine._ppo_loss(logprobs, weights, aux, ppo_config)),
    )

  def test_token_budget_batches_preserve_examples(self) -> None:
    engine = self._engine()
    data = self._data()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = engine._make_training_batches(data)

    seen = [idx for batch in batches for idx, _datum in batch]
    self.assertCountEqual(seen, range(len(data)))
    for batch in batches:
      padded_tokens = max(len(datum.model_input) for _idx, datum in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_forward_backward_padded_batches_preserve_client_output_shape(self) -> None:
    engine = self._engine()
    data = self._data()

    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      result = engine.forward_backward(data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(engine.peft_model.calls), 0)
    self.assertTrue(any(call[0].shape[0] > 1 for call in engine.peft_model.calls))
    for datum, output in zip(data, result["loss_fn_outputs"], strict=True):
      logprobs = output["logprobs"]
      self.assertEqual(logprobs["shape"], [min(len(datum.model_input), len(datum.loss_fn_inputs["target_tokens"].data))])

  def test_forward_backward_supports_full_model_without_peft_adapter(self) -> None:
    engine = self._engine()
    engine.base_model = engine.peft_model
    engine.peft_model = None
    engine.full_model_id = "full-run"
    engine.adapter_states = {"full-run": {"trainable_params": [], "optimizer": None, "training_mode": "full"}}

    result = engine.forward_backward(self._data()[:1], "cross_entropy", model_id="full-run")

    self.assertEqual(len(result["loss_fn_outputs"]), 1)
    self.assertGreater(result["metrics"]["loss:sum"], 0.0)


if __name__ == "__main__":
  unittest.main()
