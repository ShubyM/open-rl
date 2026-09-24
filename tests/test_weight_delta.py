import json
import os
import tempfile
import types
import unittest

import torch
import torch.nn as nn
from safetensors.torch import load_file

from training.weight_delta import SparseWeightDelta, fuse_for_vllm


class SimpleModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(10, 10, bias=False)


def params(model: nn.Module) -> dict[str, torch.Tensor]:
  return dict(model.named_parameters())


class SparseWeightDeltaTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.tmp.cleanup()

  def test_sparse_delta_encoding_and_lossless_overwrite(self):
    model = SimpleModel()
    delta = SparseWeightDelta(params(model))
    original = model.fc.weight.data.clone()

    # An Adam update where 2 of 100 elements change.
    model.fc.weight.data[0, 2] = 42.0
    model.fc.weight.data[5, 7] = -13.37
    self.assertEqual(delta.record(params(model)), 2)
    path = os.path.join(self.tmp.name, "step_1")
    delta.write(path, base_model="test-simple-model", model_id="test-model")

    with open(os.path.join(path, "metadata.json")) as f:
      meta = json.load(f)
    self.assertEqual(meta["format"], "sparse_delta")
    self.assertEqual(meta["changed_elements"], 2)
    self.assertEqual(meta["total_elements"], 100)
    self.assertEqual(meta["density_pct"], 2.0)

    sparse = load_file(os.path.join(path, "delta.safetensors"))
    self.assertEqual(sparse["delta.indices_flat"].numel(), 2)
    # int64: Gemma 4's per-layer embedding table exceeds 2**31 elements.
    self.assertEqual(sparse["delta.indices_flat"].dtype, torch.int64)

    # Selective overwrite reproduces the new weights bit for bit.
    sampler_weight = original.clone()
    sampler_weight.view(-1)[sparse["delta.indices_flat"]] = sparse["delta.values_flat"]
    self.assertTrue(torch.equal(sampler_weight, model.fc.weight.data))

    # The host copy now holds the new weights, so the next step diffs against them.
    self.assertTrue(torch.equal(delta.previous["fc.weight"], model.fc.weight.data.cpu()))

  def test_the_delta_before_the_first_step_is_empty_but_names_every_layer(self):
    model = SimpleModel()
    delta = SparseWeightDelta(params(model))
    path = os.path.join(self.tmp.name, "empty")
    delta.write(path, base_model="dummy", model_id="dummy")
    with open(os.path.join(path, "metadata.json")) as f:
      meta = json.load(f)
    self.assertEqual(meta["changed_elements"], 0)
    self.assertEqual(meta["total_elements"], 100)
    self.assertEqual(meta["layer_names"], ["fc.weight"])
    self.assertEqual(load_file(os.path.join(path, "delta.safetensors"))["delta.indices_flat"].numel(), 0)

  def test_several_writes_read_one_record_without_consuming_it(self):
    model = SimpleModel()
    delta = SparseWeightDelta(params(model))
    with torch.no_grad():
      model.fc.weight[0, 0] += 1.5
      model.fc.weight[3, 5] -= 0.75
      model.fc.weight[2, 2] += 2.0
    delta.record(params(model))

    metas = []
    for name in ("sampler", "state"):
      path = os.path.join(self.tmp.name, name)
      delta.write(path, base_model="dummy", model_id="dummy", kind=name)
      with open(os.path.join(path, "metadata.json")) as f:
        metas.append(json.load(f))
    self.assertEqual(metas[0]["changed_elements"], 3)
    self.assertEqual(metas[0]["changed_elements"], metas[1]["changed_elements"])
    self.assertEqual(metas[0]["layer_names"], metas[1]["layer_names"])

  def test_fuse_for_vllm_offsets_indices_into_the_fused_tensors(self):
    config = types.SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1, head_dim=4, intermediate_size=6)
    names = [
      f"model.layers.0.{suffix}.weight" for suffix in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj", "self_attn.o_proj")
    ]
    indices = [torch.tensor([1]) for _ in names]

    fused_names, fused_indices = fuse_for_vllm(names, indices, config)

    q_numel, k_numel, gate_numel = 2 * 4 * 8, 1 * 4 * 8, 6 * 8
    self.assertEqual(
      fused_names,
      ["model.layers.0.self_attn.qkv_proj.weight"] * 3 + ["model.layers.0.mlp.gate_up_proj.weight", "model.layers.0.self_attn.o_proj.weight"],
    )
    self.assertEqual([int(index) for index in fused_indices], [1, 1 + q_numel, 1 + q_numel + k_numel, 1 + gate_numel, 1])


if __name__ == "__main__":
  unittest.main()
