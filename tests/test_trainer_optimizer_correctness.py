import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests._server_fixture import SERVER_DIR


def _load_trainer_module():
  sys.path.insert(0, str(SERVER_DIR))
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


if __name__ == "__main__":
  unittest.main()
