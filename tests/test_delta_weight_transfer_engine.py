"""File validation and integration with the installed vLLM weight loader."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from vllm.model_executor.models.utils import WeightsMapper

from server.delta_weight_transfer_engine import DeltaSnapshotWeightTransferEngine, read_sparse_patches, read_weight_metadata


def write_delta(path, names=None, shapes=None, tensors=None, **metadata):
  path = Path(path)
  path.mkdir(exist_ok=True)
  info = {
    "format": "sparse_delta",
    "format_version": 2,
    "layer_names": names if names is not None else ["q_proj.weight"],
    "layer_shapes": shapes if shapes is not None else [[4, 2]],
  }
  info.update(metadata)
  (path / "metadata.json").write_text(json.dumps(info))
  save_file(
    tensors if tensors is not None else {"0.indices": torch.tensor([1, 6], dtype=torch.int32), "0.values": torch.tensor([7.0, 9.0])},
    path / "delta.safetensors",
  )
  return info


class PatchFileTest(unittest.TestCase):
  def test_mixed_dtypes_preserved(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      metadata = write_delta(
        path,
        names=["norm", "projection"],
        shapes=[[2], [2]],
        tensors={
          "0.indices": torch.tensor([0], dtype=torch.int32),
          "0.values": torch.tensor([3.0], dtype=torch.float32),
          "1.indices": torch.tensor([1], dtype=torch.int32),
          "1.values": torch.tensor([4.0], dtype=torch.bfloat16),
        },
      )
      patches = read_sparse_patches(path, metadata)
      self.assertEqual([p.values.dtype for p in patches], [torch.float32, torch.bfloat16])
      self.assertEqual(patches[1].indices.dtype, torch.int32)

  def test_rejects_legacy_and_malformed_payloads(self):
    for change in ({"format_version": 1}, {"layer_shapes": []}, {"layer_shapes": [[-1, 2]]}, {"layer_names": [""]}):
      with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        metadata = write_delta(path)
        metadata.update(change)
        with self.assertRaises(ValueError):
          read_sparse_patches(path, metadata)

  def test_missing_values_is_not_silent_noop(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      metadata = write_delta(path, tensors={"0.indices": torch.tensor([1], dtype=torch.int32)})
      with self.assertRaises(ValueError):
        read_sparse_patches(path, metadata)

  def test_empty_patch_file(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      write_delta(path, names=[], shapes=[], tensors={})
      self.assertEqual(read_sparse_patches(path, read_weight_metadata(path)), [])


class PackedModel(torch.nn.Module):
  """Native loader with packing and rank slicing, exercised by upstream helper."""

  def __init__(self, rank=0, world_size=1):
    super().__init__()
    self.rank = rank
    self.rows = 4 // world_size
    self.qkv = torch.nn.Parameter(torch.zeros(3 * self.rows, 2))
    self.bias = torch.nn.Parameter(torch.zeros(3 * self.rows))
    self.fail_after_write = False

  def load_weights(self, weights):
    loaded = set()
    for name, value in weights:
      projection, kind = name.split(".")
      offset = {"q_proj": 0, "k_proj": 1, "v_proj": 2}[projection] * self.rows
      destination = self.bias if kind == "bias" else self.qkv
      destination.data[offset : offset + self.rows].copy_(value[self.rank * self.rows : (self.rank + 1) * self.rows])
      loaded.add(name)
      if self.fail_after_write:
        raise RuntimeError("injected loader failure")
    return loaded


class WeightTransferEngineTest(unittest.TestCase):
  def make_engine(self, model):
    config = SimpleNamespace(parallel_config=SimpleNamespace(), model_config=SimpleNamespace())
    return DeltaSnapshotWeightTransferEngine(None, config, torch.device("cpu"), model)

  def apply(self, engine, directory):
    engine.start_weight_update()
    with patch("torch.accelerator.synchronize"):
      engine.update_weights({"target_weights_path": directory})
    engine.finish_weight_update()

  def test_native_packing_and_tp_shards_match_dense_loader(self):
    for world_size in (1, 2):
      for rank in range(world_size):
        with self.subTest(world_size=world_size, rank=rank), tempfile.TemporaryDirectory() as directory:
          model = PackedModel(rank, world_size)
          expected = PackedModel(rank, world_size)
          names = ["q_proj.weight", "k_proj.bias", "v_proj.weight"]
          shapes = [[4, 2], [4], [4, 2]]
          tensors = {}
          dense = []
          for i, (name, shape) in enumerate(zip(names, shapes, strict=True)):
            indices = torch.tensor([0, 3], dtype=torch.int32)
            values = torch.tensor([float(i + 1), float(i + 4)])
            tensors[f"{i}.indices"] = indices
            tensors[f"{i}.values"] = values
            value = torch.zeros(shape)
            value.view(-1)[indices.long()] = values
            dense.append((name, value))
          write_delta(directory, names, shapes, tensors)
          engine = self.make_engine(model)
          pointers = (model.qkv.data_ptr(), model.bias.data_ptr())
          self.apply(engine, directory)
          expected.load_weights(dense)
          self.assertTrue(torch.equal(model.qkv, expected.qkv))
          self.assertTrue(torch.equal(model.bias, expected.bias))
          self.assertEqual(pointers, (model.qkv.data_ptr(), model.bias.data_ptr()))

  def test_invalid_indices_and_nan_fail_before_any_writes(self):
    # Include the wrapped negative index from the large Gemma embedding failure.
    cases = [([-1], [1.0]), ([-2001493954], [1.0]), ([8], [1.0]), ([2**31], [1.0]), ([1, 1], [1.0, 2.0]), ([1], [float("nan")])]
    for indices, values in cases:
      with self.subTest(indices=indices, values=values), tempfile.TemporaryDirectory() as directory:
        model = PackedModel()
        write_delta(
          directory,
          names=["q_proj.weight", "k_proj.weight"],
          shapes=[[4, 2], [4, 2]],
          tensors={
            "0.indices": torch.tensor([0], dtype=torch.int32),
            "0.values": torch.tensor([7.0]),
            "1.indices": torch.tensor(indices, dtype=torch.int64),
            "1.values": torch.tensor(values),
          },
        )
        engine = self.make_engine(model)
        with patch.object(model, "load_weights", wraps=model.load_weights) as load, self.assertRaises(ValueError):
          self.apply(engine, directory)
        load.assert_not_called()
        self.assertEqual(torch.count_nonzero(model.qkv), 0)

  def test_checkpoint_name_mapping_belongs_to_model_loader_for_sparse_and_full(self):
    class MappedModel(PackedModel):
      hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": ""})

      def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(4, 2))

      def load_weights(self, weights):
        loaded = set()
        for name, value in self.hf_to_vllm_mapper.apply(weights):
          if name == "embed_tokens.weight":
            self.embedding.data.copy_(value)
            loaded.add(name)
          else:
            loaded.update(super().load_weights([(name, value)]))
        return loaded

    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory)
      names = ["model.language_model.embed_tokens.weight", "model.language_model.k_proj.weight"]
      write_delta(
        path / "sparse",
        names=names,
        shapes=[[4, 2], [4, 2]],
        tensors={
          "0.indices": torch.tensor([3], dtype=torch.int32),
          "0.values": torch.tensor([7.0]),
          "1.indices": torch.tensor([5], dtype=torch.int32),
          "1.values": torch.tensor([42.0]),
        },
      )
      model = MappedModel()
      self.apply(self.make_engine(model), str(path / "sparse"))
      self.assertEqual(model.embedding.view(-1)[3].item(), 7.0)
      self.assertEqual(model.qkv[4:8].reshape(-1)[5].item(), 42.0)
      self.assertEqual(torch.count_nonzero(model.qkv), 1)
      self.assertEqual(torch.count_nonzero(model.embedding), 1)

      dense = MappedModel()
      full = {names[0]: model.embedding.detach().contiguous(), names[1]: model.qkv[4:8].detach().contiguous()}
      save_file(full, path / "full.safetensors")
      with (
        patch("server.delta_weight_transfer_engine.initialize_layerwise_reload"),
        patch("server.delta_weight_transfer_engine.finalize_layerwise_reload"),
      ):
        self.apply(self.make_engine(dense), str(path / "full.safetensors"))
      self.assertTrue(torch.equal(model.embedding, dense.embedding))
      self.assertTrue(torch.equal(model.qkv, dense.qkv))

  def test_loader_failure_propagates(self):
    with tempfile.TemporaryDirectory() as directory:
      model = PackedModel()
      model.fail_after_write = True
      engine = self.make_engine(model)
      write_delta(directory)
      with self.assertRaisesRegex(RuntimeError, "injected loader failure"):
        self.apply(engine, directory)

  def test_repeated_and_empty_updates_preserve_unchanged_values(self):
    with tempfile.TemporaryDirectory() as directory:
      model = PackedModel()
      engine = self.make_engine(model)
      write_delta(directory)
      self.apply(engine, directory)
      baseline = model.qkv.detach().clone()
      self.apply(engine, directory)
      write_delta(directory, names=[], shapes=[], tensors={})
      self.apply(engine, directory)
      self.assertTrue(torch.equal(model.qkv, baseline))

  def test_dense_checkpoint_uses_reload_lifecycle(self):
    with tempfile.TemporaryDirectory() as directory:
      save_file({"q_proj.weight": torch.ones(4, 2)}, Path(directory) / "model.safetensors")
      engine = self.make_engine(PackedModel())
      with (
        patch("server.delta_weight_transfer_engine.initialize_layerwise_reload") as start,
        patch("server.delta_weight_transfer_engine.finalize_layerwise_reload") as finish,
      ):
        self.apply(engine, directory)
        start.assert_called_once_with(engine.model)
        finish.assert_called_once_with(engine.model, engine.model_config)
      self.assertTrue(torch.equal(engine.model.qkv[:4], torch.ones(4, 2)))
