"""Unit tests for DeltaSnapshotWeightTransferEngine."""

import json
import logging
import os
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import safetensors
import torch
from safetensors.torch import save_file


@dataclass
class StubWeightTransferInitInfo:
  pass


@dataclass
class StubWeightTransferUpdateInfo:
  is_checkpoint_format: bool = True


class StubWeightTransferEngine:
  def __init__(self, *args, model=None, vllm_config=None, **kwargs) -> None:
    self.model = model
    self.model_config = vllm_config

  @classmethod
  def parse_init_info(cls, init_dict):
    return cls.init_info_cls(**init_dict)

  @classmethod
  def parse_update_info(cls, update_dict):
    return cls.update_info_cls(**update_dict)


class StubWeightTransferEngineFactory:
  @classmethod
  def register_engine(cls, name, engine_cls) -> None:
    pass


def unavailable_weight_utility(*args, **kwargs):
  raise RuntimeError("Test must patch the vLLM weight utility before use.")


vllm_module = types.ModuleType("vllm")
distributed_module = types.ModuleType("vllm.distributed")
weight_transfer_module = types.ModuleType("vllm.distributed.weight_transfer")
base_module = types.ModuleType("vllm.distributed.weight_transfer.base")
base_module.WeightTransferEngine = StubWeightTransferEngine
base_module.WeightTransferInitInfo = StubWeightTransferInitInfo
base_module.WeightTransferUpdateInfo = StubWeightTransferUpdateInfo
factory_module = types.ModuleType("vllm.distributed.weight_transfer.factory")
factory_module.WeightTransferEngineFactory = StubWeightTransferEngineFactory
logger_module = types.ModuleType("vllm.logger")
logger_module.init_logger = logging.getLogger
model_executor_module = types.ModuleType("vllm.model_executor")
model_loader_module = types.ModuleType("vllm.model_executor.model_loader")
weight_utils_module = types.ModuleType("vllm.model_executor.model_loader.weight_utils")
weight_utils_module.download_weights_from_hf = unavailable_weight_utility
weight_utils_module.safetensors_weights_iterator = unavailable_weight_utility

sys.modules.update(
  {
    "vllm": vllm_module,
    "vllm.distributed": distributed_module,
    "vllm.distributed.weight_transfer": weight_transfer_module,
    "vllm.distributed.weight_transfer.base": base_module,
    "vllm.distributed.weight_transfer.factory": factory_module,
    "vllm.logger": logger_module,
    "vllm.model_executor": model_executor_module,
    "vllm.model_executor.model_loader": model_loader_module,
    "vllm.model_executor.model_loader.weight_utils": weight_utils_module,
  }
)

from src.server import delta_weight_transfer_engine as delta_module
from src.server.delta_weight_transfer_engine import (
  DeltaSnapshotInitInfo,
  DeltaSnapshotUpdateInfo,
  DeltaSnapshotWeightTransferEngine,
)


class RecordingModel:
  def __init__(self) -> None:
    self.load_calls: list[list[tuple[str, torch.Tensor]]] = []

  def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
    self.load_calls.append(list(weights))


class DeltaSnapshotWeightTransferEngineTest(unittest.TestCase):
  def test_delta_snapshot_weight_transfer_engine_contract(self):
    """Test that DeltaSnapshotWeightTransferEngine satisfies the vLLM contract."""
    engine = DeltaSnapshotWeightTransferEngine(
      config=None,
      parallel_config=None,  # type: ignore
    )

    init_info = engine.parse_init_info({"model_name_or_path": "Qwen/Qwen3-8B"})
    self.assertIsInstance(init_info, DeltaSnapshotInitInfo)
    self.assertEqual(init_info.model_name_or_path, "Qwen/Qwen3-8B")

    update_info = engine.parse_update_info({"target_weights_path": "/path/to/weights"})
    self.assertIsInstance(update_info, DeltaSnapshotUpdateInfo)
    self.assertEqual(update_info.target_weights_path, "/path/to/weights")
    self.assertTrue(update_info.is_checkpoint_format)

  def test_receive_weights_loads_tensors_into_callback(self):
    """Test that receive_weights parses safetensors and passes to load_weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
      dummy_weights = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
      }
      file_path = os.path.join(tmpdir, "delta.safetensors")
      save_file(dummy_weights, file_path)

      model = RecordingModel()
      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        parallel_config=None,  # type: ignore
        model=model,
      )
      update_info = DeltaSnapshotUpdateInfo(target_weights_path=file_path)

      engine.receive_weights(update_info)

      self.assertEqual(engine.current_weights_path, file_path)
      loaded_tensors = model.load_calls[0]
      self.assertEqual(len(loaded_tensors), 2)
      loaded_names = {k for k, _ in loaded_tensors}
      self.assertIn("model.layers.0.self_attn.q_proj.weight", loaded_names)

  def test_noop_patch_detection_skips_gpu_reload(self):
    """Test that applying an identical patch is identified as a no-op and skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
      weights_v1 = {
        "layer.0.weight": torch.ones(4, 4),
        "layer.1.weight": torch.zeros(4, 4),
      }
      file_v1 = os.path.join(tmpdir, "delta1.safetensors")
      save_file(weights_v1, file_v1)

      model = RecordingModel()
      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        parallel_config=None,  # type: ignore
        model=model,
      )

      # First update: 2 new/changed tensors
      engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=file_v1))
      self.assertEqual(len(model.load_calls), 1)
      self.assertEqual(len(model.load_calls[0]), 2)

      # Second update: identical weights (NO-OP patch)
      file_v2 = os.path.join(tmpdir, "delta2.safetensors")
      save_file(weights_v1, file_v2)

      engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=file_v2))
      # Must detect complete no-op and skip calling model.load_weights.
      self.assertEqual(len(model.load_calls), 1)
      self.assertEqual(engine.current_weights_path, file_v2)

  def test_selective_layer_filtering_skips_noop_tensors(self):
    """Test that only genuinely modified tensors are passed to load_weights."""
    with tempfile.TemporaryDirectory() as tmpdir:
      weights_v1 = {
        "layer.0.weight": torch.ones(4, 4),
        "layer.1.weight": torch.zeros(4, 4),
      }
      file_v1 = os.path.join(tmpdir, "delta1.safetensors")
      save_file(weights_v1, file_v1)

      model = RecordingModel()
      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        parallel_config=None,  # type: ignore
        model=model,
      )
      engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=file_v1))

      # Update 2: layer.0.weight is unchanged (no-op), layer.1.weight is modified
      weights_v2 = {
        "layer.0.weight": torch.ones(4, 4),
        "layer.1.weight": torch.full((4, 4), 2.5),
      }
      file_v2 = os.path.join(tmpdir, "delta2.safetensors")
      save_file(weights_v2, file_v2)

      engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=file_v2))

      # Only layer.1.weight should be passed to callback
      loaded_calls = model.load_calls[-1]
      self.assertEqual(len(loaded_calls), 1)
      self.assertEqual(loaded_calls[0][0], "layer.1.weight")
      self.assertTrue(torch.equal(loaded_calls[0][1], torch.full((4, 4), 2.5)))

  def test_receive_weights_sparse_delta_patching(self):
    """Test that receive_weights parses sparse_delta metadata, applies indices to CPU snapshot, and passes full reconstructed layer tensor."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create metadata specifying sparse_delta format and layer_names
      with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
        json.dump({"format": "sparse_delta", "changed_elements": 1, "layer_names": ["layer.0.weight"]}, f)

      # Create sparse coordinate arrays in 1D flat packed format
      sparse_dict = {
        "delta.indices_flat": torch.tensor([5], dtype=torch.int32),
        "delta.values_flat": torch.tensor([99.0], dtype=torch.float32),
        "delta.layer_lengths": torch.tensor([1], dtype=torch.int64),
      }
      save_file(sparse_dict, os.path.join(tmpdir, "delta.safetensors"))

      # Mock active vLLM model holding initial weights (all zeros)
      class DummyModel(RecordingModel):
        def __init__(self) -> None:
          super().__init__()
          self.weight = torch.nn.Parameter(torch.zeros(4, 4))

        def named_parameters(self):
          return [("layer.0.weight", self.weight)]

        def named_buffers(self):
          return []

      dummy_model = DummyModel()
      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        parallel_config=None,  # type: ignore
        model=dummy_model,
      )
      engine.cpu_snapshot = {"layer.0.weight": dummy_model.weight.detach().clone()}
      engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=tmpdir))

      # Assert mock_loader received only the full reconstructed 2D layer tensor, NOT .indices / .values
      loaded_calls = dummy_model.load_calls[0]
      self.assertEqual(len(loaded_calls), 1)
      self.assertEqual(loaded_calls[0][0], "layer.0.weight")
      self.assertEqual(loaded_calls[0][1].shape, (4, 4))
      self.assertEqual(loaded_calls[0][1].view(-1)[5].item(), 99.0)

  def test_receive_weights_absolute_changed_tensor_patching(self):
    """Test that adaptive dense updates replace the CPU snapshot and load only changed tensors."""
    with tempfile.TemporaryDirectory() as tmpdir:
      with open(os.path.join(tmpdir, "metadata.json"), "w") as f:
        json.dump({"format": "absolute_tensors", "layer_names": ["layer.0.weight"]}, f)
      updated_weight = torch.full((4, 4), 3.5)
      save_file({"layer.0.weight": updated_weight}, os.path.join(tmpdir, "delta.safetensors"))

      model = RecordingModel()
      engine = DeltaSnapshotWeightTransferEngine(
        config=None,
        parallel_config=None,  # type: ignore
        model=model,
      )
      engine.cpu_snapshot = {"layer.0.weight": torch.zeros(4, 4)}

      update_info = DeltaSnapshotUpdateInfo(target_weights_path=tmpdir)
      engine.receive_weights(update_info)
      engine.receive_weights(update_info)

      self.assertEqual(len(model.load_calls), 1)
      self.assertEqual(model.load_calls[0][0][0], "layer.0.weight")
      self.assertTrue(torch.equal(model.load_calls[0][0][1], updated_weight))
      self.assertTrue(torch.equal(engine.cpu_snapshot["layer.0.weight"], updated_weight))

  def test_receive_weights_base_model_directory_loading(self):
    """Test that receive_weights directly populates CPU snapshot from base_model_path when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
      base_dir = os.path.join(tmpdir, "base_model")
      os.makedirs(base_dir)
      base_weights = {"layer.0.weight": torch.zeros(4, 4)}
      save_file(base_weights, os.path.join(base_dir, "model.safetensors"))

      step_dir = os.path.join(tmpdir, "step_1")
      os.makedirs(step_dir)
      with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump({"format": "sparse_delta", "changed_elements": 1, "layer_names": ["layer.0.weight"]}, f)

      sparse_dict = {
        "delta.indices_flat": torch.tensor([2], dtype=torch.int32),
        "delta.values_flat": torch.tensor([42.0], dtype=torch.float32),
        "delta.layer_lengths": torch.tensor([1], dtype=torch.int64),
      }
      save_file(sparse_dict, os.path.join(step_dir, "delta.safetensors"))

      def fake_iterator(hf_weights_files, use_tqdm_on_load=False):
        for p in hf_weights_files:
          if os.path.exists(p):
            with safetensors.safe_open(p, framework="pt", device="cpu") as f:
              for key in list(f.keys()):
                yield key, f.get_tensor(key)

      with (
        patch.object(delta_module, "download_weights_from_hf", side_effect=lambda model, **_: model),
        patch.object(delta_module, "safetensors_weights_iterator", side_effect=fake_iterator),
      ):
        model = RecordingModel()
        engine = DeltaSnapshotWeightTransferEngine(
          config=None,
          parallel_config=None,  # type: ignore
          model=model,
        )

        engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=step_dir, base_model_path=base_dir))

        loaded_calls = model.load_calls[0]
        self.assertEqual(len(loaded_calls), 1)
        self.assertEqual(loaded_calls[0][0], "layer.0.weight")
        self.assertEqual(loaded_calls[0][1].shape, (4, 4))
        self.assertEqual(loaded_calls[0][1].view(-1)[2].item(), 42.0)

  def test_receive_weights_hf_cache_and_env_loading(self):
    """Test that ensure_cpu_snapshot resolves HF model IDs from OPEN_RL_BASE_MODEL."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Mock HF hub cache structure: ~/.cache/huggingface/hub/models--Qwen--Test-4B/snapshots/commit123/
      hf_folder = os.path.join(tmpdir, "models--Qwen--Test-4B", "snapshots", "commit123")
      os.makedirs(hf_folder)
      save_file({"layer.0.weight": torch.zeros(3, 3)}, os.path.join(hf_folder, "model-00001-of-00001.safetensors"))

      step_dir = os.path.join(tmpdir, "step_1")
      os.makedirs(step_dir)
      with open(os.path.join(step_dir, "metadata.json"), "w") as f:
        json.dump({"format": "sparse_delta", "changed_elements": 1, "layer_names": ["layer.0.weight"]}, f)

      sparse_dict = {
        "delta.indices_flat": torch.tensor([4], dtype=torch.int32),
        "delta.values_flat": torch.tensor([88.0], dtype=torch.float32),
        "delta.layer_lengths": torch.tensor([1], dtype=torch.int64),
      }
      save_file(sparse_dict, os.path.join(step_dir, "delta.safetensors"))

      def fake_iterator(hf_weights_files, use_tqdm_on_load=False):
        for p in hf_weights_files:
          if os.path.exists(p):
            with safetensors.safe_open(p, framework="pt", device="cpu") as f:
              for key in list(f.keys()):
                yield key, f.get_tensor(key)

      with (
        patch.dict(os.environ, {"OPEN_RL_BASE_MODEL": "Qwen/Test-4B"}),
        patch.object(delta_module, "download_weights_from_hf", return_value=hf_folder),
        patch.object(delta_module, "safetensors_weights_iterator", side_effect=fake_iterator),
      ):
        model = RecordingModel()
        engine = DeltaSnapshotWeightTransferEngine(
          config=None,
          parallel_config=None,  # type: ignore
          model=model,
        )

        engine.receive_weights(DeltaSnapshotUpdateInfo(target_weights_path=step_dir))

        loaded_calls = model.load_calls[0]
        self.assertEqual(len(loaded_calls), 1)
        self.assertEqual(loaded_calls[0][0], "layer.0.weight")
        self.assertEqual(loaded_calls[0][1].shape, (3, 3))
        self.assertEqual(loaded_calls[0][1].view(-1)[4].item(), 88.0)


if __name__ == "__main__":
  unittest.main()
