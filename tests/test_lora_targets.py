from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker


class DummyTiedModel(torch.nn.Module):
  def __init__(self, tie_word_embeddings: bool):
    super().__init__()
    self.q_proj = torch.nn.Linear(4, 4)
    self.gate_proj = torch.nn.Linear(4, 4)
    self.lm_head = torch.nn.Linear(4, 8)
    self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)


def make_worker(tie_word_embeddings: bool) -> LoraTrainingWorker:
  worker = LoraTrainingWorker()
  worker.base_model = DummyTiedModel(tie_word_embeddings)
  worker.base_model_name = "dummy"
  return worker


class LoraTargetModulesTest(unittest.TestCase):
  def test_train_unembed_targets_lm_head_when_untied(self):
    worker = make_worker(tie_word_embeddings=False)
    targets = worker.target_lora_modules(LoraConfig(train_attn=True, train_mlp=False, train_unembed=True))
    self.assertIn("lm_head", targets)

  def test_train_unembed_ignored_when_tied(self):
    # vLLM cannot load lm_head adapter weights for tied-embeddings models, so
    # the worker must log and skip lm_head instead of producing an unservable adapter.
    worker = make_worker(tie_word_embeddings=True)
    targets = worker.target_lora_modules(LoraConfig(train_attn=True, train_mlp=True, train_unembed=True))
    self.assertNotIn("lm_head", targets)
    self.assertIn("q_proj", targets)
    self.assertIn("gate_proj", targets)

  def test_unembed_only_on_tied_model_raises(self):
    worker = make_worker(tie_word_embeddings=True)
    with self.assertRaises(ValueError):
      worker.target_lora_modules(LoraConfig(train_attn=False, train_mlp=False, train_unembed=True))


if __name__ == "__main__":
  unittest.main()
