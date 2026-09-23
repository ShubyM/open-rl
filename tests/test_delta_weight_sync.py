"""Sampler deltas are explicit, ordered exports of current model weights."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import safetensors.torch
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from server.model_metadata import WeightSyncConfig
from training.fft_trainer_worker import FFTTrainingWorker


class TwoWeights(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.first = torch.nn.Parameter(torch.zeros(2))
    self.second = torch.nn.Parameter(torch.zeros(2))


def apply_export(path: str, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  with open(os.path.join(path, "metadata.json")) as file:
    metadata = json.load(file)
  delta = safetensors.torch.load_file(os.path.join(path, "delta.safetensors"))
  indices = delta["delta.indices_flat"].split(delta["delta.layer_lengths"].tolist())
  values = delta["delta.values_flat"].split(delta["delta.layer_lengths"].tolist())
  for name, index, value in zip(metadata["layer_names"], indices, values, strict=True):
    weights[name].view(-1)[index] = value
  return weights


def model_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
  return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


class DeltaWeightSyncTest(unittest.TestCase):
  def setUp(self):
    self.directory = tempfile.TemporaryDirectory()
    self.addCleanup(self.directory.cleanup)
    self.env = patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.directory.name})
    self.env.start()
    self.addCleanup(self.env.stop)

  def worker(self, model=None, *, strategy="delta"):
    return FFTTrainingWorker(
      model=TwoWeights() if model is None else model,
      device="cpu",
      base_model_name="tiny-model",
      cpu_offload=False,
      weight_sync_cfg=WeightSyncConfig(strategy=strategy, delta_format="native"),
    )

  def export(self, worker, version):
    return worker.save_for_sampler("job", None, f"tinker://job/sampler_weights/{version}")

  def assert_weights_equal(self, actual, model):
    expected = model_weights(model)
    self.assertEqual(actual.keys(), expected.keys())
    for name in expected:
      torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

  def test_export_covers_multiple_optimizer_steps_without_implicit_publication(self):
    worker = self.worker()
    sampler = model_weights(worker.model)
    worker.model.first.grad = torch.tensor([1.0, 0.0])
    worker.optim_step({"learning_rate": 0.1})
    worker.model.second.grad = torch.tensor([0.0, -1.0])
    worker.optim_step({"learning_rate": 0.1})
    self.assertEqual(os.listdir(self.directory.name), [])

    path = self.export(worker, "first")
    self.assert_weights_equal(apply_export(path, sampler), worker.model)
    delta = safetensors.torch.load_file(os.path.join(path, "delta.safetensors"))
    self.assertEqual(delta["delta.indices_flat"].dtype, torch.int64)
    self.assertEqual(delta["delta.indices_flat"].numel(), 2)
    with open(os.path.join(path, "delta.safetensors"), "rb") as file:
      exported = file.read()
    worker.model.first.grad = torch.ones(2)
    worker.optim_step({"learning_rate": 0.1})
    with open(os.path.join(path, "delta.safetensors"), "rb") as file:
      self.assertEqual(file.read(), exported)

  def test_later_export_replaces_a_coordinate_that_returns_to_its_initial_value(self):
    worker = self.worker()
    sampler = model_weights(worker.model)
    with torch.no_grad():
      worker.model.first[0] = 42.0
    apply_export(self.export(worker, "first"), sampler)
    with torch.no_grad():
      worker.model.first[0] = 0.0
      worker.model.second[1] = 13.0
    self.assert_weights_equal(apply_export(self.export(worker, "second"), sampler), worker.model)

  def test_failed_artifact_write_does_not_consume_changes(self):
    for target in ("safetensors.torch.save_file", "json.dump"):
      with self.subTest(target=target):
        worker = self.worker()
        sampler = model_weights(worker.model)
        with torch.no_grad():
          worker.model.first[0] = 7.0
        with patch(f"training.weight_export.{target}", side_effect=OSError("disk full")), self.assertRaisesRegex(OSError, "disk full"):
          self.export(worker, "failed")
        with torch.no_grad():
          worker.model.second[0] = 9.0
        self.assert_weights_equal(apply_export(self.export(worker, "retry"), sampler), worker.model)

  def test_full_checkpoint_does_not_advance_sampler_baseline(self):
    model = LlamaForCausalLM(
      LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    )
    worker = self.worker(model)
    sampler = model_weights(model)
    params = list(model.parameters())
    with torch.no_grad():
      params[0].view(-1)[0] += 0.25
    state_path = os.path.join(self.directory.name, "checkpoint")
    worker.save_state("job", state_path)
    self.assertTrue(os.path.isfile(os.path.join(state_path, "config.json")))
    self.assertFalse(os.path.exists(os.path.join(state_path, "delta.safetensors")))
    with torch.no_grad():
      params[1].view(-1)[0] += 0.5
    self.assert_weights_equal(apply_export(self.export(worker, "after-checkpoint"), sampler), model)

  def test_unchanged_successive_exports_are_empty_and_preserve_current_sampler(self):
    worker = self.worker()
    sampler = model_weights(worker.model)
    with torch.no_grad():
      worker.model.first[0] = 5.0
    apply_export(self.export(worker, "first"), sampler)
    path = self.export(worker, "second")
    delta = safetensors.torch.load_file(os.path.join(path, "delta.safetensors"))
    self.assertEqual(delta["delta.indices_flat"].numel(), 0)
    self.assert_weights_equal(apply_export(path, sampler), worker.model)

  def test_full_export_uses_hf_checkpoint_format(self):
    model = LlamaForCausalLM(
      LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    )
    path = self.export(self.worker(model, strategy="full"), "full")
    restored = LlamaForCausalLM.from_pretrained(path)
    self.assert_weights_equal(model_weights(restored), model)
    self.assertFalse(os.path.exists(os.path.join(path, "delta.safetensors")))

  def test_first_export_after_checkpoint_restore_replaces_all_sampler_weights(self):
    model = LlamaForCausalLM(
      LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    )
    tokenizer = PreTrainedTokenizerFast(
      tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0, **{str(index): index for index in range(1, 16)}}, unk_token="<unk>"))
    )
    worker = FFTTrainingWorker(
      model=model,
      tokenizer=tokenizer,
      device="cpu",
      base_model_name="tiny",
      cpu_offload=False,
      weight_sync_cfg=WeightSyncConfig(delta_format="native"),
    )
    sampler = model_weights(model)
    with torch.no_grad():
      next(model.parameters()).view(-1)[0] += 0.5
    checkpoint = os.path.join(self.directory.name, "checkpoint")
    worker.save_state("job", checkpoint)
    worker.load_from_state("job", checkpoint)
    path = self.export(worker, "restored")
    self.assert_weights_equal(apply_export(path, sampler), worker.model)
    with open(os.path.join(path, "metadata.json")) as file:
      metadata = json.load(file)
    self.assertEqual(metadata["changed_elements"], metadata["total_elements"])

  def test_weight_sync_strategy_selection(self):
    worker = self.worker()
    self.assertEqual(worker.weight_sync_cfg.strategy, "delta")
    worker.set_weight_sync_strategy("full")
    self.assertEqual(worker.weight_sync_cfg.strategy, "full")
    with self.assertRaises(ValueError):
      worker.set_weight_sync_strategy("invalid_strategy")


if __name__ == "__main__":
  unittest.main()
