import json
import os
import tempfile
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

from server.model_metadata import WeightSyncConfig
from training import losses as losses_module
from training import trainer_worker as trainer_worker_module
from training.fft_trainer_worker import FFTTrainer, trainable_model_parameters
from training.lora_trainer_worker import LoraConfig, LoraTrainer, LoraTrainingWorker, active_adapter_parameters
from training.trainer_worker import Datum, TensorData, Trainer


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

  def train(self):
    return None

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


class _FullModelStub:
  def __init__(self, params):
    self.params = params

  def train(self):
    return None

  def parameters(self):
    yield from self.params


class TinyHFModel(torch.nn.Module):
  """A real nn.Module so FFTTrainer can enumerate named_parameters, with a
  save_pretrained that records rather than writes."""

  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(2, 2, bias=False)
    self.saved: list[str] = []

  def save_pretrained(self, path):
    self.saved.append(path)


class TwoParamModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.trainable = torch.nn.Parameter(torch.tensor([1.0]))
    self.frozen = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)


def _datum(model_input, target_tokens, *, weights=None, logprobs=None, advantages=None):
  loss_fn_inputs = {"target_tokens": TensorData(data=target_tokens)}
  if weights is not None:
    loss_fn_inputs["weights"] = TensorData(data=weights)
  if logprobs is not None:
    loss_fn_inputs["logprobs"] = TensorData(data=logprobs)
  if advantages is not None:
    loss_fn_inputs["advantages"] = TensorData(data=advantages)
  return Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


def logit_trainer(model, tokenizer=None):
  """A base Trainer whose model is a test stub. The dummy param keeps the
  optimizer paths honest; the padding/logprob math never touches it."""
  trainer = Trainer("job", model, [torch.nn.Parameter(torch.zeros(1))], tokenizer=tokenizer or _TokenizerStub())
  trainer.input_device = torch.device("cpu")
  return trainer


class TestLoraTargetModules(unittest.TestCase):
  def test_targets_survive_peft_wrapping(self) -> None:
    # The first adapter wraps the targeted Linears (PEFT moves each under
    # base_layer). A second job with a different config must still resolve
    # them, or the shared LoRA runtime refuses every job after the first.
    block = torch.nn.Module()
    block.q_proj = torch.nn.Linear(4, 4)
    block.gate_proj = torch.nn.Linear(4, 4)
    model = torch.nn.Module()
    model.layer = block
    worker = LoraTrainingWorker()
    worker.base_model = model

    attn_only = LoraConfig(train_attn=True, train_mlp=False, train_unembed=False)
    self.assertEqual(worker.target_lora_modules(attn_only), ["layer.q_proj"])

    wrapped = torch.nn.Module()
    wrapped.base_layer = block.q_proj
    block.q_proj = wrapped
    attn_and_mlp = LoraConfig(train_attn=True, train_mlp=True, train_unembed=False)
    self.assertEqual(worker.target_lora_modules(attn_and_mlp), ["layer.q_proj", "layer.gate_proj"])


class TestTrainerCheckpoints(unittest.TestCase):
  def test_save_adapter_selects_the_adapter_it_saves(self) -> None:
    adapter_a_param = torch.nn.Parameter(torch.tensor([1.0]))
    adapter_b_param = torch.nn.Parameter(torch.tensor([1.0]))
    peft = _PeftModelStub({"adapter-a": [adapter_a_param], "adapter-b": [adapter_b_param]})
    peft.set_adapter("adapter-b")
    trainer = LoraTrainer("adapter-a", peft, [adapter_a_param], None, "base")

    with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp_dir}):
      trainer.save_adapter()
      self.assertTrue(os.path.exists(os.path.join(tmp_dir, "peft", "adapter-a", "metadata.json")))

    self.assertEqual(peft.active_adapter, "adapter-a")

  def test_lora_save_state_writes_the_optimizer_next_to_the_adapter(self) -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    peft = _PeftModelStub({"job-a": [param]})
    trainer = LoraTrainer("job-a", peft, [param], None, "base")
    trainer.optimizer = torch.optim.AdamW([param], lr=0.1)

    with tempfile.TemporaryDirectory() as tmp_dir:
      state_dir = os.path.join(tmp_dir, "step-5")
      trainer.save_state(state_dir, include_optimizer=True)
      self.assertTrue(os.path.exists(os.path.join(state_dir, "optimizer.pt")))
      with open(os.path.join(state_dir, "metadata.json")) as f:
        self.assertTrue(json.load(f)["has_optimizer"])

  def test_fft_save_state_under_delta_writes_a_delta_whatever_was_asked(self) -> None:
    model = TinyHFModel()
    trainer = FFTTrainer(
      "job-a", model, trainable_model_parameters(model), None, "base", cpu_offload=False, weight_sync_cfg=WeightSyncConfig(strategy="delta")
    )
    with patch.object(trainer, "save_state_delta", return_value={"path": "delta"}) as delta:
      self.assertEqual(trainer.save_state("/tmp/x", include_optimizer=True), {"path": "delta"})
    delta.assert_called_once()

  def test_fft_save_state_skips_the_optimizer_until_fft_resume_exists(self) -> None:
    model = TinyHFModel()
    params = trainable_model_parameters(model)
    trainer = FFTTrainer("job-a", model, params, None, "base", cpu_offload=False, weight_sync_cfg=WeightSyncConfig(strategy="full"))
    trainer.optimizer = torch.optim.AdamW(params, lr=0.1)
    with tempfile.TemporaryDirectory() as tmp_dir:
      state_dir = os.path.join(tmp_dir, "step-5")
      trainer.save_state(state_dir, include_optimizer=True)
      self.assertFalse(os.path.exists(os.path.join(state_dir, "optimizer.pt")))
      with open(os.path.join(state_dir, "metadata.json")) as f:
        self.assertFalse(json.load(f)["has_optimizer"])
      self.assertEqual(model.saved, [state_dir])


class TestOptimStepIsolation(unittest.TestCase):
  def test_lora_optim_step_only_updates_active_adapter_params(self) -> None:
    active_param = torch.nn.Parameter(torch.tensor([1.0]))
    other_param = torch.nn.Parameter(torch.tensor([1.0]))
    active_param.grad = torch.tensor([1.0])
    other_param.grad = torch.tensor([10.0])

    peft = _PeftModelStub({"adapter-a": [active_param], "adapter-b": [other_param]})
    params = active_adapter_parameters(peft, "adapter-a")
    trainer = LoraTrainer("adapter-a", peft, params, None, "base")
    trainer.save_adapter = lambda *_args, **_kwargs: None

    result = trainer.optim_step({"learning_rate": 0.1, "beta1": 0.0, "beta2": 0.0, "eps": 1e-8, "weight_decay": 0.0})

    self.assertEqual(peft.active_adapter, "adapter-a")
    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(active_param.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(other_param.detach(), torch.tensor([1.0])))
    if active_param.grad is not None:
      self.assertTrue(torch.allclose(active_param.grad, torch.zeros_like(active_param.grad)))
    self.assertIsNotNone(other_param.grad)

  def test_fft_optim_step_updates_full_model_trainable_params(self) -> None:
    model = TwoParamModel()
    model.trainable.grad = torch.tensor([1.0])
    model.frozen.grad = torch.tensor([10.0])
    params = trainable_model_parameters(model)
    trainer = FFTTrainer("job", model, params, None, "base", cpu_offload=False, weight_sync_cfg=WeightSyncConfig(strategy="full"))

    result = trainer.optim_step({"learning_rate": 0.1, "beta1": 0.0, "beta2": 0.0, "eps": 1e-8, "weight_decay": 0.0})

    self.assertAlmostEqual(result["metrics"]["grad_norm:mean"], 1.0)
    self.assertFalse(torch.allclose(model.trainable.detach(), torch.tensor([1.0])))
    self.assertTrue(torch.allclose(model.frozen.detach(), torch.tensor([1.0])))
    if model.trainable.grad is not None:
      self.assertTrue(torch.allclose(model.trainable.grad, torch.zeros_like(model.trainable.grad)))
    self.assertIsNotNone(model.frozen.grad)


class TestTrainerPaddedBatchingMath(unittest.TestCase):
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

  def training_tensors(self, trainer, data):
    input_ids, attention_mask, input_lengths = trainer.pad_model_inputs(data)
    target_token_ids, weights, lengths = trainer.pad_targets_and_weights(data, input_lengths)
    logprobs = trainer.compute_target_logprobs(input_ids, attention_mask, target_token_ids)
    old_logprobs = trainer.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in data], lengths, torch.float32)
    advantages = trainer.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in data], lengths, torch.float32)
    return logprobs, weights, old_logprobs, advantages, lengths

  def test_padded_batch_logprobs_and_losses_match_per_example_math(self) -> None:
    trainer = logit_trainer(_LogitModelStub())
    data = self._data()

    batch_logprobs, batch_weights, batch_old_logprobs, batch_advantages, batch_lengths = self.training_tensors(trainer, data)
    single_results = [self.training_tensors(trainer, [datum]) for datum in data]

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
      losses_module.importance_sampling_loss(batch_logprobs, batch_weights, batch_old_logprobs, batch_advantages).sum(),
      single_sum(
        lambda logprobs, weights, old_logprobs, advantages: losses_module.importance_sampling_loss(logprobs, weights, old_logprobs, advantages)
      ),
    )
    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      losses_module.ppo_loss(batch_logprobs, batch_weights, batch_old_logprobs, batch_advantages, ppo_config).sum(),
      single_sum(lambda logprobs, weights, old_logprobs, advantages: losses_module.ppo_loss(logprobs, weights, old_logprobs, advantages, ppo_config)),
    )

  def test_token_budget_batches_preserve_examples(self) -> None:
    trainer = logit_trainer(_LogitModelStub())
    data = self._data()
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = trainer.make_training_batches(data)

    seen = [idx for batch in batches for idx, _datum in batch]
    self.assertCountEqual(seen, range(len(data)))
    for batch in batches:
      padded_tokens = max(len(datum.model_input) for _idx, datum in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_forward_backward_padded_batches_preserve_client_output_shape(self) -> None:
    model = _LogitModelStub()
    trainer = logit_trainer(model)
    data = self._data()

    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      result = trainer.forward_backward(data, "cross_entropy")

    self.assertEqual(len(result["loss_fn_outputs"]), len(data))
    self.assertGreater(len(model.calls), 0)
    self.assertTrue(any(call[0].shape[0] > 1 for call in model.calls))
    for datum, output in zip(data, result["loss_fn_outputs"], strict=True):
      logprobs = output["logprobs"]
      self.assertEqual(logprobs["shape"], [min(len(datum.model_input), len(datum.loss_fn_inputs["target_tokens"].data))])


class TestZeroAdvantageBackward(unittest.TestCase):
  """A policy-gradient batch without effective advantages carries no gradient
  and skips its backward; a KL penalty brings the gradient back."""

  def test_zero_effective_advantages_skip_policy_backward(self) -> None:
    for loss_fn in ("importance_sampling", "ppo"):
      with self.subTest(loss_fn=loss_fn):
        parameter = torch.nn.Parameter(torch.tensor(0.25))
        trainer = Trainer("job", _FullModelStub([parameter]), [parameter], tokenizer=_TokenizerStub())
        data = [_datum([3, 4], [1, 2], weights=[1.0, 0.0], logprobs=[-0.1, -0.2], advantages=[0.0, 2.0])]

        with patch.object(
          trainer,
          "compute_target_logprobs",
          side_effect=lambda _inputs, _mask, targets, parameter=parameter: parameter.expand_as(targets),
        ):
          result = trainer.forward_backward(data, loss_fn)

        self.assertIsNone(parameter.grad)
        self.assertEqual(result["metrics"], {"loss:mean": 0.0, "loss:sum": 0.0})
        self.assertEqual(result["loss_fn_outputs"][0]["logprobs"]["shape"], [2])

  def test_ppo_kl_penalty_keeps_backward_for_zero_advantages(self) -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    trainer = Trainer("job", _FullModelStub([parameter]), [parameter], tokenizer=_TokenizerStub())
    data = [_datum([3], [1], weights=[1.0], logprobs=[-0.1], advantages=[0.0])]

    with patch.object(
      trainer,
      "compute_target_logprobs",
      side_effect=lambda _inputs, _mask, targets: parameter.expand_as(targets),
    ):
      trainer.forward_backward(data, "ppo", {"kl_coeff": 0.1})

    self.assertIsNotNone(parameter.grad)
    self.assertNotEqual(parameter.grad.item(), 0.0)


class TestDataParallelForwardBackward(unittest.TestCase):
  """Datum sharding must reproduce single-process gradients under FSDP's per-backward averaging."""

  PLACEHOLDER = {"logprobs": {"data": [], "dtype": "float32", "shape": [0]}}

  def _data(self):
    return [
      _datum([3, 4, 5], [1, 2, 3], weights=[1.0, 0.5, 0.25]),
      _datum([7, 8], [2, 3], weights=[2.0, 0.75]),
      _datum([9], [4], weights=[1.5]),
    ]

  def _run_forward_backward(self, data, *, loss_fn="cross_entropy", rank=None, world=2, captured=None):
    """Run one rank of a fake two-rank group, or a single process when rank is None."""
    parameter = torch.nn.Parameter(torch.tensor(0.25))
    trainer = Trainer("dp", _FullModelStub([parameter]), [parameter], tokenizer=_TokenizerStub())
    trainer.input_device = torch.device("cpu")
    captured = {} if captured is None else captured

    def gather(part, _group):
      captured["part"] = part
      return [part, {idx: self.PLACEHOLDER for idx in range(len(data)) if idx not in part}]

    def reduce_sum(value, _group):
      captured["total"] = value
      return value

    fakes = {
      "group_rank": lambda _group: rank,
      "group_size": lambda _group: world,
      "all_reduce_max": lambda _passes, _group: 2,
      "all_reduce_sum": reduce_sum,
      "all_gather_object": gather,
    }
    with ExitStack() as stack:
      if rank is not None:
        stack.enter_context(patch.object(trainer, "data_parallel_group", return_value=object()))
        stack.enter_context(patch.object(trainer, "data_parallel_loss_scale", return_value=float(world)))
        for name, fake in fakes.items():
          stack.enter_context(patch.object(trainer_worker_module, name, fake))
      compute_calls = stack.enter_context(
        patch.object(
          trainer,
          "compute_target_logprobs",
          side_effect=lambda _inputs, _mask, targets, parameter=parameter: parameter.expand_as(targets),
        )
      )
      result = trainer.forward_backward(data, loss_fn)
    return result, parameter, compute_calls

  def test_sharded_gradients_average_to_single_process_gradient(self) -> None:
    data = self._data()
    reference, reference_param, _calls = self._run_forward_backward(data)

    rank_grads, rank_totals, rank_parts = [], [], {}
    for rank in (0, 1):
      captured = {}
      _result, parameter, _calls = self._run_forward_backward(data, rank=rank, captured=captured)
      rank_grads.append(parameter.grad)
      rank_totals.append(captured["total"])
      rank_parts.update(captured["part"])

    # FSDP averages each backward across ranks; the group-size loss scaling
    # must make that average equal the single-process gradient sum.
    torch.testing.assert_close((rank_grads[0] + rank_grads[1]) / 2, reference_param.grad)
    self.assertAlmostEqual(sum(rank_totals), reference["metrics"]["loss:sum"], places=5)
    self.assertEqual(sorted(rank_parts), list(range(len(data))))

  def test_short_rank_pads_with_zero_scaled_filler_passes(self) -> None:
    data = self._data()
    result, parameter, compute_calls = self._run_forward_backward(data, rank=1)

    # Rank 1 owns one datum but must run two passes; the filler pass leaves
    # gradients and reported loss untouched.
    self.assertEqual(compute_calls.call_count, 2)
    weights_rank1 = 2.0 + 0.75
    torch.testing.assert_close(parameter.grad, torch.tensor(2 * -weights_rank1))
    self.assertAlmostEqual(result["metrics"]["loss:sum"], 0.25 * -weights_rank1, places=5)
    self.assertEqual(result["loss_fn_outputs"][1]["logprobs"]["shape"], [2])

  def test_distributed_ranks_never_skip_zero_advantage_backward(self) -> None:
    data = [_datum([3, 4], [1, 2], weights=[1.0, 0.0], logprobs=[-0.1, -0.2], advantages=[0.0, 2.0])]
    result, parameter, _calls = self._run_forward_backward(data, loss_fn="importance_sampling", rank=0)

    # Single-process mode skips this backward entirely; a distributed rank
    # must still run it so the group's collective counts stay aligned.
    self.assertIsNotNone(parameter.grad)
    torch.testing.assert_close(parameter.grad, torch.tensor(0.0))
    self.assertEqual(result["metrics"]["loss:sum"], 0.0)

  def test_empty_batch_on_a_distributed_rank_returns_nothing(self) -> None:
    result, parameter, compute_calls = self._run_forward_backward([], rank=0)

    self.assertEqual(result["loss_fn_outputs"], [])
    self.assertEqual(result["metrics"]["loss:sum"], 0.0)
    self.assertIsNone(parameter.grad)
    self.assertEqual(compute_calls.call_count, 0)


if __name__ == "__main__":
  unittest.main()
