"""Trainer worker behavior on a real two-layer model with real peft, on CPU.

Every test here runs the actual forward, backward and optimizer code. Nothing
about the model or the adapter library is stubbed, so a change in how peft
isolates adapters or how the worker saves state shows up as a failure.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from training import losses
from training.fft_trainer_worker import FFTTrainingWorker
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import BaseTrainerWorker, Datum, TensorData

VOCAB_SIZE = 64
CPU = torch.device("cpu")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else CPU
ADAM = {"learning_rate": 0.05, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8}


def tiny_model(seed: int = 0) -> LlamaForCausalLM:
  torch.manual_seed(seed)
  config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=64,
    tie_word_embeddings=False,
  )
  return LlamaForCausalLM(config)


def tiny_tokenizer() -> PreTrainedTokenizerFast:
  vocab = {f"t{i}": i for i in range(VOCAB_SIZE)}
  vocab["<pad>"] = 0
  return PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="t1")), pad_token="<pad>")


def datum(model_input: list[int], target_tokens: list[int], **loss_inputs: list[float]) -> Datum:
  inputs = {"target_tokens": TensorData(data=target_tokens)}
  inputs.update({name: TensorData(data=values) for name, values in loss_inputs.items()})
  return Datum(model_input=model_input, loss_fn_inputs=inputs)


BATCH = [datum([3, 4, 5, 6], [4, 5, 6, 7]), datum([7, 8], [8, 9]), datum([9, 10, 11], [10, 11, 12])]


def lora_worker() -> LoraTrainingWorker:
  worker = LoraTrainingWorker()
  worker.device = CPU
  worker.base_model = tiny_model()
  worker.base_model_name = "tiny"
  worker.tokenizer = tiny_tokenizer()
  return worker


def loader_dtype() -> torch.dtype:
  """load_from_state picks bf16 whenever the CUDA build supports it, even for a CPU worker."""
  return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32


def fft_worker(device: torch.device = CPU, cpu_offload: bool = False) -> FFTTrainingWorker:
  worker = FFTTrainingWorker()
  worker.device = device
  worker.cpu_offload = cpu_offload
  worker.set_weight_sync_strategy("full")
  worker.model = tiny_model().to(device=device, dtype=loader_dtype())
  worker.base_model_name = "tiny"
  worker.tokenizer = tiny_tokenizer()
  worker.prepare_model_for_training()
  return worker


def named_params(model: torch.nn.Module, contains: str = "", excludes: str | None = None) -> dict[str, torch.Tensor]:
  return {
    name: param.detach().cpu().clone() for name, param in model.named_parameters() if contains in name and (excludes is None or excludes not in name)
  }


def assert_params_equal(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> None:
  assert before.keys() == after.keys()
  for name in before:
    torch.testing.assert_close(after[name], before[name], rtol=0, atol=0, msg=name)


def assert_params_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> None:
  assert before.keys() == after.keys()
  assert any(not torch.equal(before[name], after[name]) for name in before)


def assert_optimizer_state_equal(expected: torch.optim.Optimizer, actual: torch.optim.Optimizer) -> None:
  expected_state, actual_state = expected.state_dict(), actual.state_dict()
  assert expected_state["param_groups"] == actual_state["param_groups"]
  assert expected_state["state"].keys() == actual_state["state"].keys()
  for index, moments in expected_state["state"].items():
    for key, value in moments.items():
      torch.testing.assert_close(actual_state["state"][index][key].cpu(), value.cpu(), rtol=0, atol=0, msg=f"param {index} {key}")


class TmpDirCase(unittest.TestCase):
  """A fresh OPEN_RL_TMP_DIR per test, so adapter saves land somewhere disposable."""

  def setUp(self) -> None:
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.tmp_dir = tmp.name
    env = patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp_dir})
    env.start()
    self.addCleanup(env.stop)


class TestLoraWorker(TmpDirCase):
  def test_optim_step_updates_only_the_active_adapter(self):
    worker = lora_worker()
    worker.create_adapter("adapter-a", LoraConfig(rank=2, seed=1, lora_dropout=0.0))
    worker.create_adapter("adapter-b", LoraConfig(rank=2, seed=2, lora_dropout=0.0))
    a_before = named_params(worker.peft_model, ".adapter-a.")
    b_before = named_params(worker.peft_model, ".adapter-b.")
    base_before = named_params(worker.peft_model, excludes="lora_")

    worker.forward_backward(BATCH, "cross_entropy", None, "adapter-a")
    worker.optim_step(ADAM, "adapter-a")

    assert_params_changed(a_before, named_params(worker.peft_model, ".adapter-a."))
    assert_params_equal(b_before, named_params(worker.peft_model, ".adapter-b."))
    assert_params_equal(base_before, named_params(worker.peft_model, excludes="lora_"))

  def test_interleaving_a_second_adapter_does_not_change_the_first_ones_trajectory(self):
    solo = lora_worker()
    solo.create_adapter("adapter-a", LoraConfig(rank=2, seed=1, lora_dropout=0.0))
    for _ in range(3):
      solo.forward_backward(BATCH, "cross_entropy", None, "adapter-a")
      solo.optim_step(ADAM, "adapter-a")

    shared = lora_worker()
    shared.create_adapter("adapter-a", LoraConfig(rank=2, seed=1, lora_dropout=0.0))
    shared.create_adapter("adapter-b", LoraConfig(rank=2, seed=2, lora_dropout=0.0))
    for _ in range(3):
      shared.forward_backward(BATCH, "cross_entropy", None, "adapter-a")
      shared.forward_backward(BATCH[::-1], "cross_entropy", None, "adapter-b")
      shared.optim_step(ADAM, "adapter-b")
      shared.optim_step(ADAM, "adapter-a")

    assert_params_equal(named_params(solo.peft_model, ".adapter-a."), named_params(shared.peft_model, ".adapter-a."))

  def test_save_adapter_writes_only_the_named_adapter(self):
    worker = lora_worker()
    worker.create_adapter("adapter-a", LoraConfig(rank=2, seed=1))
    worker.create_adapter("adapter-b", LoraConfig(rank=2, seed=2))
    worker.peft_model.set_adapter("adapter-b")

    worker.save_adapter("adapter-a", alias="final")

    save_dir = os.path.join(self.tmp_dir, "peft", "adapter-a")
    saved = load_file(os.path.join(save_dir, "adapter-a", "adapter_model.safetensors"))
    self.assertTrue(saved)
    self.assertTrue(all("lora_" in name for name in saved))
    self.assertFalse(any("adapter-b" in name for name in saved))
    with open(os.path.join(save_dir, "metadata.json")) as f:
      self.assertEqual(json.load(f)["alias"], "final")


class TestFFTWorker(TmpDirCase):
  def test_optim_step_leaves_frozen_params_alone(self):
    worker = fft_worker()
    worker.model.lm_head.weight.requires_grad_(False)
    worker.trainable_params = [param for param in worker.model.parameters() if param.requires_grad]
    frozen_before = named_params(worker.model, "lm_head")
    trainable_before = named_params(worker.model, excludes="lm_head")

    worker.forward_backward(BATCH, "cross_entropy")
    worker.optim_step(ADAM)

    assert_params_equal(frozen_before, named_params(worker.model, "lm_head"))
    assert_params_changed(trainable_before, named_params(worker.model, excludes="lm_head"))

  def test_save_state_and_load_from_state_round_trip_restores_weights_and_optimizer(self):
    state_dir = os.path.join(self.tmp_dir, "state")
    original = fft_worker(device=DEVICE)
    original.forward_backward(BATCH, "cross_entropy")
    original.optim_step(ADAM)
    original.save_state("model-a", state_dir, include_optimizer=True)

    restored = FFTTrainingWorker()
    restored.device = DEVICE
    restored.cpu_offload = False
    restored.set_weight_sync_strategy("full")
    restored.load_from_state("model-a", state_dir, restore_optimizer=True)

    assert_params_equal(named_params(original.model), named_params(restored.model))
    assert_optimizer_state_equal(original.optimizer, restored.optimizer)

  @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
  def test_sleep_and_wake_between_any_two_ops_is_transparent(self):
    def run(suspend_between_ops: bool) -> dict[str, torch.Tensor]:
      worker = fft_worker(device=torch.device("cuda"), cpu_offload=True)
      for _ in range(2):
        worker.forward_backward(BATCH, "cross_entropy")
        if suspend_between_ops:
          worker.sleep()
          worker.wake_up()
        worker.optim_step(ADAM)
        if suspend_between_ops:
          worker.sleep()
          worker.wake_up()
      return named_params(worker.model)

    assert_params_equal(run(suspend_between_ops=False), run(suspend_between_ops=True))


class CosineLogitModel:
  """Deterministic stand-in for a causal LM: logits are a fixed function of token and position."""

  def __init__(self, vocab_size: int = 17):
    self.vocab_size = vocab_size
    self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

  def train(self):
    return None

  def __call__(self, input_tensor, attention_mask=None, **_kwargs):
    if attention_mask is not None:
      self.calls.append((input_tensor.detach().clone(), attention_mask.detach().clone()))
    vocab = torch.arange(self.vocab_size, dtype=torch.float32).view(1, 1, -1)
    positions = torch.arange(input_tensor.shape[1], dtype=torch.float32).view(1, -1, 1)
    logits = torch.cos(input_tensor.float().unsqueeze(-1) * 0.11 + positions * 0.07 + vocab * 0.13)
    logits.requires_grad_()
    return type("Output", (), {"logits": logits})()


RL_BATCH = [
  datum([3, 4, 5, 6], [1, 2, 3, 4], weights=[1.0, 0.5, 0.25, 2.0], logprobs=[-0.1, -0.2, -0.3, -0.4], advantages=[1.0, -0.5, 2.0, 0.25]),
  datum([7, 8], [2, 3], logprobs=[-0.7, -0.8], advantages=[0.75, 1.25]),
  datum([9, 10, 11], [5, 6, 7, 8], weights=[0.2, 0.4, 0.6, 0.8], logprobs=[-0.9, -1.0, -1.1, -1.2], advantages=[-1.0, 0.3, 0.9, 1.7]),
]


class TestPaddedBatchingMath(unittest.TestCase):
  """Batching several sequences with padding must not change any per-example number."""

  def worker(self) -> BaseTrainerWorker:
    worker = BaseTrainerWorker()
    worker.device = CPU
    worker.tokenizer = tiny_tokenizer()
    return worker

  def training_tensors(self, worker, model, data):
    input_ids, attention_mask, input_lengths = worker.pad_model_inputs(data)
    target_token_ids, weights, lengths = worker.pad_targets_and_weights(data, input_lengths)
    logprobs = worker.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)
    old_logprobs = worker.pad_sequences([d.loss_fn_inputs["logprobs"].data for d in data], lengths, torch.float32)
    advantages = worker.pad_sequences([d.loss_fn_inputs["advantages"].data for d in data], lengths, torch.float32)
    return logprobs, weights, old_logprobs, advantages, lengths

  def test_batched_logprobs_and_losses_match_per_example_math(self):
    worker = self.worker()
    model = CosineLogitModel()

    batched = self.training_tensors(worker, model, RL_BATCH)
    singles = [self.training_tensors(worker, model, [d]) for d in RL_BATCH]

    batch_logprobs, batch_weights, batch_old, batch_adv, batch_lengths = batched
    for row, (logprobs, weights, old, adv, lengths) in enumerate(singles):
      n = batch_lengths[row]
      self.assertEqual(n, lengths[0])
      torch.testing.assert_close(batch_logprobs[row, :n], logprobs[0, :n])
      torch.testing.assert_close(batch_weights[row, :n], weights[0, :n])
      torch.testing.assert_close(batch_weights[row, n:], torch.zeros_like(batch_weights[row, n:]))
      torch.testing.assert_close(batch_old[row, :n], old[0, :n])
      torch.testing.assert_close(batch_adv[row, :n], adv[0, :n])

    def summed_over_singles(loss_fn):
      return torch.stack([loss_fn(lp, w, old, adv).sum() for lp, w, old, adv, _ in singles]).sum()

    ppo_config = {"clip_range": 0.2, "kl_coeff": 0.03}
    torch.testing.assert_close(
      losses.cross_entropy_loss(batch_logprobs, batch_weights).sum(),
      summed_over_singles(lambda lp, w, _old, _adv: losses.cross_entropy_loss(lp, w)),
    )
    torch.testing.assert_close(
      losses.importance_sampling_loss(batch_logprobs, batch_weights, batch_old, batch_adv).sum(),
      summed_over_singles(losses.importance_sampling_loss),
    )
    torch.testing.assert_close(
      losses.ppo_loss(batch_logprobs, batch_weights, batch_old, batch_adv, ppo_config).sum(),
      summed_over_singles(lambda lp, w, old, adv: losses.ppo_loss(lp, w, old, adv, ppo_config)),
    )

  def test_token_budget_batches_cover_every_example_once(self):
    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "6"}):
      batches = self.worker().make_training_batches(RL_BATCH)

    seen = sorted(idx for batch in batches for idx, _ in batch)
    self.assertEqual(seen, list(range(len(RL_BATCH))))
    for batch in batches:
      padded_tokens = max(len(d.model_input) for _, d in batch) * len(batch)
      self.assertTrue(len(batch) == 1 or padded_tokens <= 6)

  def test_forward_backward_returns_one_output_per_input_in_input_order(self):
    model = CosineLogitModel()

    with patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "12"}):
      result = self.worker().forward_backward(model, RL_BATCH, "cross_entropy")

    self.assertTrue(any(call[0].shape[0] > 1 for call in model.calls), "expected at least one multi-row batch")
    self.assertEqual(len(result["loss_fn_outputs"]), len(RL_BATCH))
    for d, output in zip(RL_BATCH, result["loss_fn_outputs"], strict=True):
      self.assertEqual(output["logprobs"]["shape"], [min(len(d.model_input), len(d.loss_fn_inputs["target_tokens"].data))])


if __name__ == "__main__":
  unittest.main()
