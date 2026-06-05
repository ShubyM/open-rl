import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch

import torch

from tests._server_fixture import SERVER_DIR


def _load_trainer_modules():
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
  old_path = list(sys.path)
  sys.path.insert(0, str(SERVER_DIR))
  with patch.dict(sys.modules, stubs):
    for module_name in list(sys.modules):
      if module_name == "training" or module_name.startswith("training."):
        del sys.modules[module_name]
    trainer_worker = importlib.import_module("training.trainer_worker")
    lora_trainer_worker = importlib.import_module("training.lora_trainer_worker")
    losses = importlib.import_module("training.losses")
  sys.path = old_path
  return trainer_worker, lora_trainer_worker, losses


trainer_worker_module, lora_trainer_worker_module, losses_module = _load_trainer_modules()
BaseTrainerWorker = trainer_worker_module.BaseTrainerWorker
LoraTrainingWorker = lora_trainer_worker_module.LoraTrainingWorker


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
  loss_fn_inputs = {"target_tokens": trainer_worker_module.TensorData(data=target_tokens)}
  if weights is not None:
    loss_fn_inputs["weights"] = trainer_worker_module.TensorData(data=weights)
  if logprobs is not None:
    loss_fn_inputs["logprobs"] = trainer_worker_module.TensorData(data=logprobs)
  if advantages is not None:
    loss_fn_inputs["advantages"] = trainer_worker_module.TensorData(data=advantages)
  return trainer_worker_module.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


class TestTrainerOptimizerCorrectness(unittest.TestCase):
  def test_optim_step_only_updates_active_adapter_params(self) -> None:
    active_param = torch.nn.Parameter(torch.tensor([1.0]))
    other_param = torch.nn.Parameter(torch.tensor([1.0]))
    active_param.grad = torch.tensor([1.0])
    other_param.grad = torch.tensor([10.0])

    engine = LoraTrainingWorker()
    engine.peft_model = _PeftModelStub(
      {
        "adapter-a": [active_param],
        "adapter-b": [other_param],
      }
    )
    engine.adapter_states = {
      "adapter-a": {"trainable_params": lora_trainer_worker_module.active_adapter_parameters(engine.peft_model, "adapter-a"), "optimizer": None}
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


class TestTrainerPaddedBatchingMath(unittest.TestCase):
  def _worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = torch.device("cpu")
    worker.tokenizer = _TokenizerStub()
    return worker

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

  def training_tensors(self, worker, model, data):
    input_ids, attention_mask, input_lengths = worker.pad_model_inputs(data)
    target_token_ids, weights, lengths = worker.pad_targets_and_weights(data, input_lengths)
    logprobs = worker.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)
    old_logprobs = worker.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in data], lengths, torch.float32)
    advantages = worker.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in data], lengths, torch.float32)
    return logprobs, weights, old_logprobs, advantages, lengths

  def test_padded_batch_logprobs_and_losses_match_per_example_math(self) -> None:
    worker = self._worker()
    model = _LogitModelStub()
    data = self._data()

    batch_logprobs, batch_weights, batch_old_logprobs, batch_advantages, batch_lengths = self.training_tensors(worker, model, data)
    single_results = [self.training_tensors(worker, model, [datum]) for datum in data]

    for row, (single_logprobs, single_weights, single_old_logprobs, single_advantages, single_lengths) in enumerate(single_results):
      length = batch_lengths[row]
      self.assertEqual(length, single_lengths[0])
      torch.testing.assert_close(batch_logprobs[row, :length], single_logprobs[0, :length])
      torch.testing.assert_close(batch_weights[row, :length], single_weights[0, :length])
      torch.testing.assert_close(batch_weights[row, length:], torch.zeros_like(batch_weights[row, length:]))
      torch.testing.assert_close(batch_old_logprobs[row, :length], single_old_logprobs[0, :length])
      torch.testing.assert_close(batch_old_logprobs[row, length:], torch.zeros_like(batch_old_logprobs[row, length:]))
      torch.testing.assert_close(batch_advantages[row, :length], single_advantages[0, :length])
      torch.testing.assert_close(batch_advantages[row, length:], torch.zeros_like(batch_advantages[row, length:]))

    def single_sum(fn):
      losses = [fn(logprobs, weights, old_logprobs, advantages).sum() for logprobs, weights, old_logprobs, advantages, _lengths in single_results]
      return torch.stack(losses).sum()

    torch.testing.assert_close(
      losses_module.cross_entropy_loss(batch_logprobs, batch_weights).sum(),
      single_sum(lambda logprobs, weights, _old_logprobs, _advantages: losses_module.cross_entropy_loss(logprobs, weights)),
    )
    torch.testing.assert_close(
      losses_module.importance_sampling_loss(
        batch_logprobs,
        batch_weights,
        batch_old_logprobs,
        batch_advantages,
      ).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.importance_sampling_loss(
          logprobs,
          weights,
          old_logprobs,
          advantages,
        )
      ),
    )
    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      losses_module.ppo_loss(
        batch_logprobs,
        batch_weights,
        batch_old_logprobs,
        batch_advantages,
        ppo_config,
      ).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.ppo_loss(
          logprobs,
          weights,
          old_logprobs,
          advantages,
          ppo_config,
        )
      ),
    )

  def test_token_budget_batches_preserve_examples(self) -> None:
    engine = self._worker()
    data = self._data()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = engine.make_training_batches(data)

    seen = [idx for batch in batches for idx, _datum in batch]
    self.assertCountEqual(seen, range(len(data)))
    for batch in batches:
      padded_tokens = max(len(datum.model_input) for _idx, datum in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_forward_backward_padded_batches_preserve_client_output_shape(self) -> None:
    worker = self._worker()
    model = _LogitModelStub()
    data = self._data()

    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      result = worker.forward_backward(model, data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(model.calls), 0)
    self.assertTrue(any(call[0].shape[0] > 1 for call in model.calls))
    for datum, output in zip(data, result["loss_fn_outputs"], strict=True):
      logprobs = output["logprobs"]
      self.assertEqual(logprobs["shape"], [min(len(datum.model_input), len(datum.loss_fn_inputs["target_tokens"].data))])


if __name__ == "__main__":
  unittest.main()
