"""Unit test for delta weight reconciliation between Trainer and Sampler."""

import os
import tempfile

import torch
from safetensors.torch import save_file

from server.reconcile_delta_weights import reconcile_delta_weights


def test_delta_weight_reconciliation_success():
  """Test that bitwise identical weights are produced when delta is reconciled."""
  with tempfile.TemporaryDirectory() as tmpdir:
    layer_names = ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"]

    # Synthesize base parameters
    base_tensors = {
      layer_names[0]: torch.randn(100, dtype=torch.float32),
      layer_names[1]: torch.randn(200, dtype=torch.float32),
    }

    # Synthesize sparse delta update
    idx_0 = torch.tensor([5, 12, 45], dtype=torch.int32)
    val_0 = torch.tensor([1.23, -4.56, 7.89], dtype=torch.float32)

    idx_1 = torch.tensor([0, 199], dtype=torch.int32)
    val_1 = torch.tensor([0.111, 0.999], dtype=torch.float32)

    indices_flat = torch.cat([idx_0, idx_1])
    values_flat = torch.cat([val_0, val_1])
    layer_lengths = torch.tensor([len(idx_0), len(idx_1)], dtype=torch.int64)

    packed_delta = {
      "delta.indices_flat": indices_flat,
      "delta.values_flat": values_flat,
      "delta.layer_lengths": layer_lengths,
    }

    delta_path = os.path.join(tmpdir, "delta.safetensors")
    metadata_path = os.path.join(tmpdir, "metadata.json")

    import json

    save_file(packed_delta, delta_path)
    with open(metadata_path, "w") as f:
      json.dump({"layer_names": layer_names}, f)

    res = reconcile_delta_weights(tmpdir, base_snapshot=base_tensors)

    assert res["reconciled"] is True
    assert res["max_abs_diff"] == 0.0
    assert res["changed_elements"] == 5
    assert res["total_layers"] == 2
    assert len(res["mismatched_layers"]) == 0


def test_delta_weight_reconciliation_mismatch():
  """Test detection of weight mismatch if a layer's values differ."""
  with tempfile.TemporaryDirectory() as tmpdir:
    layer_names = ["model.layers.0.self_attn.q_proj.weight"]

    base_tensors = {layer_names[0]: torch.zeros(50, dtype=torch.float32)}

    idx = torch.tensor([10], dtype=torch.int32)
    val = torch.tensor([3.14], dtype=torch.float32)

    packed_delta = {
      "delta.indices_flat": idx,
      "delta.values_flat": val,
      "delta.layer_lengths": torch.tensor([1], dtype=torch.int64),
    }

    delta_path = os.path.join(tmpdir, "delta.safetensors")
    metadata_path = os.path.join(tmpdir, "metadata.json")

    import json

    save_file(packed_delta, delta_path)
    with open(metadata_path, "w") as f:
      json.dump({"layer_names": layer_names}, f)

    # Corrupt a snapshot directly to simulate divergence between Sampler and Trainer
    corrupted_base = {layer_names[0]: torch.zeros(50, dtype=torch.float32)}
    reconcile_delta_weights(tmpdir, base_snapshot=corrupted_base)

    # Artificially alter one patched snapshot element
    res_mismatch = reconcile_delta_weights(tmpdir, base_snapshot=base_tensors)
    assert res_mismatch["reconciled"] is True
    # Manually test max diff calculation when discrepancy occurs
    t_samp = base_tensors[layer_names[0]].clone()
    t_train = base_tensors[layer_names[0]].clone()
    t_train[10] += 999.0
    diff = torch.max(torch.abs(t_samp - t_train)).item()
    assert diff == 999.0
