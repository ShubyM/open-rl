"""Automodel worker checks that run on CPU without nemo-automodel."""

import os
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from server.training_requests_processor import build_worker
from training import automodel_worker
from training.automodel_worker import AutomodelTrainingWorker, datum_inputs, lora_target_patterns, model_inputs, split_rows
from training.trainer_worker import BaseTrainerWorker
from training.types import Datum, LoraConfig, TensorData


def datum(model_input, target_tokens, **inputs) -> Datum:
  loss_fn_inputs = {"target_tokens": TensorData(data=target_tokens), **{key: TensorData(data=values) for key, values in inputs.items()}}
  return Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


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


class LoraTargetsTest(unittest.TestCase):
  """The adapter wraps what the client's LoraConfig asks for, the way the HF LoRA worker does."""

  def test_targets_follow_the_config(self) -> None:
    attn = lora_target_patterns(LoraConfig(train_attn=True, train_mlp=False), tied_embeddings=False)
    self.assertIn("model.*.layers.*.q_proj", attn)
    self.assertIn("model.*.layers.*.in_proj_qkv", attn)
    self.assertFalse(any(pattern.endswith(("gate_proj", "up_proj", "down_proj", "lm_head")) for pattern in attn))
    both = lora_target_patterns(LoraConfig(), tied_embeddings=False)
    self.assertIn("model.*.layers.*.gate_proj", both)
    self.assertNotIn("lm_head", both)

  def test_unembed_is_wrapped_only_when_the_head_is_untied(self) -> None:
    self.assertIn("lm_head", lora_target_patterns(LoraConfig(train_unembed=True), tied_embeddings=False))
    self.assertNotIn("lm_head", lora_target_patterns(LoraConfig(train_unembed=True), tied_embeddings=True))
    with self.assertRaises(ValueError):
      lora_target_patterns(LoraConfig(train_attn=False, train_mlp=False, train_unembed=True), tied_embeddings=True)


class DatumInputsTest(unittest.TestCase):
  """Automodel keys every per-token input to the input length; our datums do not always line up."""

  def test_targets_past_the_input_are_dropped(self) -> None:
    ids, inputs, scored = datum_inputs(datum([9, 10, 11], [5, 6, 7, 8], weights=[0.2, 0.4, 0.6, 0.8], logprobs=[-1.0, -2.0, -3.0, -4.0]))
    self.assertEqual((ids, scored), ([9, 10, 11], 3))
    self.assertEqual(inputs["target_tokens"], [5, 6, 7])
    self.assertEqual(inputs["weights"], [0.2, 0.4, 0.6])
    self.assertEqual(inputs["logprobs"], [-1.0, -2.0, -3.0])

  def test_input_past_the_targets_carries_no_loss(self) -> None:
    ids, inputs, scored = datum_inputs(datum([3, 4, 5, 6], [1, 2]))
    self.assertEqual(scored, 2)
    self.assertEqual(inputs["target_tokens"], [1, 2, 0, 0])
    self.assertEqual(inputs["weights"], [1.0, 1.0, 0.0, 0.0])
    self.assertNotIn("advantages", inputs)

  def test_all_datums_of_a_pass_carry_the_same_keys(self) -> None:
    # collate_datums refuses mixed keys, so weights is always present.
    _, inputs, _ = datum_inputs(datum([1, 2], [2, 3]))
    self.assertEqual(set(inputs), {"target_tokens", "weights"})


class BatchShapeTest(unittest.TestCase):
  def test_an_all_ones_mask_is_dropped_so_sdpa_can_use_flash(self) -> None:
    batch = {
      "input_ids": torch.ones(2, 3, dtype=torch.long),
      "attention_mask": torch.ones(2, 3, dtype=torch.long),
      "labels": torch.zeros(2, 3),
      "weights": torch.ones(2, 3),
    }
    self.assertEqual(set(model_inputs(batch)), {"input_ids"})
    batch["attention_mask"][1, 2] = 0
    self.assertEqual(set(model_inputs(batch)), {"input_ids", "attention_mask"})

  def test_packed_keys_reach_the_model_and_ours_do_not(self) -> None:
    batch = {
      "input_ids": torch.ones(1, 5, dtype=torch.long),
      "position_ids": torch.arange(5)[None],
      "seq_lens": torch.tensor([[2, 3]]),
      "qkv_format": "thd",
      "labels": None,
      "weights": None,
    }
    self.assertEqual(set(model_inputs(batch)), {"input_ids", "position_ids", "seq_lens", "qkv_format"})

  def test_rows_split_the_same_way_padded_and_packed(self) -> None:
    logprobs = torch.tensor([[-1.0, -2.0, -3.0, 0.0], [-4.0, -5.0, 0.0, 0.0]])
    padded = split_rows(logprobs, None, [3, 2])
    packed = split_rows(torch.tensor([[-1.0, -2.0, -3.0, 0.0, -4.0, -5.0, 0.0]]), [4, 3], [3, 2])
    self.assertEqual(padded, [[-1.0, -2.0, -3.0], [-4.0, -5.0]])
    self.assertEqual(packed, padded)

  def test_infinite_logprobs_are_clamped_for_the_client(self) -> None:
    self.assertEqual(split_rows(torch.tensor([[float("-inf"), -1e5]]), None, [2]), [[-9999.0, -9999.0]])

  def test_packed_passes_are_bounded_by_total_tokens(self) -> None:
    data = [datum([1] * n, [1] * n) for n in (5, 2, 4, 3)]
    worker = AutomodelTrainingWorker()
    with patch.object(automodel_worker, "AUTOMODEL_PACKED", True), patch.dict(os.environ, {"OPEN_RL_TRAIN_TOKEN_BUDGET": "7"}):
      batches = worker.make_training_batches(data)
    self.assertCountEqual([idx for batch in batches for idx, _ in batch], range(4))
    for batch in batches:
      self.assertLessEqual(sum(len(item.model_input) for _, item in batch), 7)
    self.assertTrue(any(len(batch) > 1 for batch in batches))


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
      logprobs = compute()
      logprobs[attention_mask.bool()].sum().backward()
      return logprobs.detach(), model.model.embed_tokens.weight.grad.clone()

    expected, expected_grad = logprobs_and_grad(lambda: BaseTrainerWorker().compute_target_logprobs(model, input_ids, attention_mask, targets))
    with patch.object(automodel_worker, "LOGPROB_CHUNK", 3):
      actual, actual_grad = logprobs_and_grad(
        lambda: AutomodelTrainingWorker().compute_target_logprobs(model, targets, input_ids=input_ids, attention_mask=attention_mask)
      )

    mask = attention_mask.bool()
    torch.testing.assert_close(actual[mask], expected[mask])
    torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
  unittest.main()
