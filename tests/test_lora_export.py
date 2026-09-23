"""LoRA training stays in memory until an explicit sampler export succeeds."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor
from training import commands
from training.lora_trainer_worker import LoraTrainingWorker
from training.types import Datum, LoraConfig


def tiny_worker() -> LoraTrainingWorker:
  torch.manual_seed(7)
  model = LlamaForCausalLM(
    LlamaConfig(
      vocab_size=16,
      hidden_size=16,
      intermediate_size=32,
      num_hidden_layers=1,
      num_attention_heads=2,
      num_key_value_heads=2,
      max_position_embeddings=32,
    )
  )
  return LoraTrainingWorker(base_model=model, device="cpu", base_model_name="tiny-lora")


def train_step(worker: LoraTrainingWorker, model_id: str) -> None:
  data = [Datum(model_input=[1, 2, 3], loss_fn_inputs={"target_tokens": {"data": [2, 3, 4]}})]
  worker.forward_backward(data, "cross_entropy", model_id=model_id)
  worker.optim_step({"learning_rate": 0.01}, model_id)


class LoraExportTest(unittest.IsolatedAsyncioTestCase):
  def assert_export_matches(self, worker: LoraTrainingWorker, adapter_id: str, path: Path) -> None:
    exported = load_file(str(path))
    expected = get_peft_model_state_dict(worker.peft_model, adapter_name=adapter_id)
    self.assertEqual(exported.keys(), expected.keys())
    for name, tensor in expected.items():
      torch.testing.assert_close(exported[name], tensor, rtol=0, atol=0)

  def test_training_changes_exported_weights_only_when_explicitly_saved(self):
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      worker = tiny_worker()
      worker.create_model("tiny-lora", "a", LoraConfig(rank=2, lora_dropout=0.0, seed=1))
      self.assertEqual(list(Path(directory).iterdir()), [])
      train_step(worker, "a")
      self.assertEqual(list(Path(directory).iterdir()), [])

      self.assertIsNone(worker.save_for_sampler("a", None, "tinker://a/sampler_weights/sampler-0"))
      path = Path(directory) / "peft" / "a" / "a" / "adapter_model.safetensors"
      self.assert_export_matches(worker, "a", path)
      first_export = path.read_bytes()

      train_step(worker, "a")
      self.assertEqual(path.read_bytes(), first_export)
      worker.save_for_sampler("a", None, "tinker://a/sampler_weights/sampler-1")
      self.assertNotEqual(path.read_bytes(), first_export)
      self.assert_export_matches(worker, "a", path)

  def test_export_selects_the_named_adapter_and_preserves_alias(self):
    with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"OPEN_RL_TMP_DIR": directory}):
      worker = tiny_worker()
      config = LoraConfig(rank=2, lora_dropout=0.0, seed=1)
      worker.create_adapter("a", config)
      worker.create_adapter("b", config)
      train_step(worker, "a")
      train_step(worker, "b")
      self.assertEqual(worker.peft_model.active_adapter, "b")

      worker.save_for_sampler("a", "evaluation", "tinker://a/sampler_weights/evaluation")
      parent = Path(directory) / "peft" / "a"
      self.assert_export_matches(worker, "a", parent / "a" / "adapter_model.safetensors")
      self.assertTrue((parent / "a" / "adapter_config.json").exists())
      self.assertFalse((Path(directory) / "peft" / "b").exists())
      self.assertFalse((parent / "optimizer.pt").exists())
      metadata = json.loads((parent / "metadata.json").read_text())
      self.assertEqual(metadata["alias"], "evaluation")
      self.assertEqual(metadata["model_id"], "a")
      self.assertFalse(metadata["has_optimizer"])

  async def test_export_write_failure_fails_the_request(self):
    with tempfile.TemporaryDirectory() as directory:
      unusable_root = Path(directory) / "file"
      unusable_root.write_text("not a directory")
      worker = tiny_worker()
      worker.create_adapter("a", LoraConfig(rank=2, seed=1))
      store = InMemoryStore()
      processor = TrainingRequestsProcessor(store, worker)
      command = commands.SaveWeightsForSampler(request_id="export", model_id="a", sampling_session_id="tinker://a/sampler_weights/sampler-0")
      with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": str(unusable_root)}):
        await processor.process_request(commands.wire(command))

      result = store.futures_store["export"]
      self.assertEqual(result["type"], "RequestFailedResponse")
      self.assertIn("Not a directory", result["error_message"])


if __name__ == "__main__":
  unittest.main()
