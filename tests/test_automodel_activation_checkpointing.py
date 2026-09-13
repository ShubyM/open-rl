import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from training import automodel_worker


@dataclass
class FSDPConfig:
  activation_checkpointing: bool = False
  tp_plan: object = None


@dataclass(frozen=True)
class DistributedSetup:
  mesh_context: object
  strategy_config: FSDPConfig
  activation_checkpointing: bool


def make_config(model_type, **kwargs):
  text = SimpleNamespace(model_type=model_type, **kwargs)
  return SimpleNamespace(get_text_config=lambda: text, output_hidden_states=False)


def make_model(config, *, native):
  model = torch.nn.Module()
  model.config = config
  model.model = torch.nn.Module()
  model.model.norm = torch.nn.Identity()
  layers = {str(i): torch.nn.Linear(2, 2) for i in range(6)}
  model.model.layers = torch.nn.ModuleDict(layers) if native else torch.nn.ModuleList(layers.values())
  model.gradient_checkpointing_enable = Mock()
  return model


class ActivationCheckpointingTests(unittest.TestCase):
  def test_policy_is_set_before_loading_and_refreshes_when_reusing_mesh(self):
    for group_size in (0, 4):
      with self.subTest(group_size=group_size):
        self.check_policy_switch(group_size)

  def check_policy_switch(self, group_size):
    configs = {
      "native": make_config("qwen3_5_text"),
      "hf": make_config("llama"),
      "shared": make_config("gemma4_text", num_kv_shared_layers=2, head_dim=256, global_head_dim=512),
    }
    worker = automodel_worker.AutomodelTrainingWorker.__new__(automodel_worker.AutomodelTrainingWorker)
    worker.model = None
    worker.base_model_name = None
    worker.distributed_setup = None
    worker.device = torch.device("cpu")
    worker.cp_size = worker.tp_size = 1
    worker.is_lora = False
    worker.peft_config = None
    worker.release_model = Mock()
    worker.load_hf_config = Mock(side_effect=configs.__getitem__)
    mesh = object()
    build_mesh = Mock(return_value=SimpleNamespace(device_mesh=mesh))
    modules = {
      "nemo_automodel.components.distributed.config": SimpleNamespace(DistributedSetup=DistributedSetup, FSDP2Config=FSDPConfig),
      "nemo_automodel.components.distributed.mesh": SimpleNamespace(MeshContext=SimpleNamespace(build=build_mesh), ParallelismSizes=SimpleNamespace),
    }
    policies = []

    def load(name, *, distributed_setup, **kwargs):
      expected = group_size > 0 and name != "native"
      self.assertEqual(distributed_setup.activation_checkpointing, expected)
      self.assertEqual(distributed_setup.strategy_config.activation_checkpointing, expected)
      self.assertIs(distributed_setup.mesh_context.device_mesh, mesh)
      self.assertEqual(kwargs["attn_implementation"], "ffpa" if name == "shared" else "sdpa")
      if name == "native":
        self.assertEqual(kwargs["backend"], {"attn": "sdpa"})
        self.assertEqual(kwargs["num_nextn_predict_layers"], 0)
      else:
        self.assertNotIn("backend", kwargs)
        self.assertNotIn("num_nextn_predict_layers", kwargs)
      if name == "shared":
        self.assertFalse(kwargs["use_sdpa_patching"])
      policies.append(distributed_setup)
      return make_model(configs[name], native=name == "native")

    with (
      patch.dict("sys.modules", modules),
      patch.object(automodel_worker, "RECOMPUTE_NUM_LAYERS", group_size),
      patch.object(automodel_worker, "AUTOMODEL_ATTN", "auto"),
      patch.object(automodel_worker, "require_automodel", return_value=SimpleNamespace(from_pretrained=load)),
      patch.object(automodel_worker, "AutoTokenizer", SimpleNamespace(from_pretrained=Mock(return_value=object()))),
      patch.object(automodel_worker.torch.cuda, "set_device"),
      patch.object(automodel_worker.dist, "is_initialized", return_value=True),
      patch.object(automodel_worker.dist, "get_world_size", return_value=1),
    ):
      for name in ("native", "hf", "shared", "native"):
        worker.load_base_model(name)
        worker.model.gradient_checkpointing_enable.assert_not_called()
        layers = worker.model.model.layers
        groups = list(layers.values()) if isinstance(layers, torch.nn.ModuleDict) else list(layers)
        self.assertEqual(len(groups), 2 if name == "native" and group_size > 0 else 6)
        self.assertFalse(worker.model.config.output_hidden_states)
        self.assertEqual(len(worker.model.model.norm._forward_hooks), 0)
        if name == "shared":
          self.assertEqual(worker.forward_kwargs["kernel_options"]["fwd_BLOCK_M"], 64)
        else:
          self.assertEqual(worker.forward_kwargs, {})
    build_mesh.assert_called_once()
    # Updating policy must replace the frozen setup without changing prior loads.
    self.assertFalse(policies[0].activation_checkpointing)
    self.assertEqual(policies[1].activation_checkpointing, group_size > 0)


if __name__ == "__main__":
  unittest.main()
