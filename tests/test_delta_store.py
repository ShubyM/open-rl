from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

from tests._server_fixture import SERVER_DIR

sys.path.insert(0, str(SERVER_DIR))
import delta_store  # noqa: E402
import state_delta  # noqa: E402


class TestDeltaStore(unittest.TestCase):
  def test_file_delta_store_writes_payload_then_manifest(self) -> None:
    tensor_name = "base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight"
    tensor = torch.zeros(2, 4)
    manifest = state_delta.build_lora_delta_manifest(
      run_id="adapter-a",
      version=4,
      apply_target="vllm_lora",
      tensors=[(tensor_name, tensor)],
      compute_checksums=True,
      created_at=1.0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      store = delta_store.FileDeltaStore(tmp_dir)
      write = store.write_delta(manifest, [(manifest.tensors[0], tensor)])
      delta_ref = Path(write.payload["delta_ref"])

      self.assertEqual(write.receipt.transport, "file_delta")
      self.assertTrue((delta_ref / "tensors.safetensors").exists())
      self.assertTrue((delta_ref / "manifest.json").exists())
      self.assertEqual(set(load_file(delta_ref / "tensors.safetensors")), {manifest.tensors[0].normalized_name})
      with (delta_ref / "manifest.json").open() as f:
        self.assertEqual(json.load(f)["delta_id"], manifest.delta_id)

      restored = store.read_delta(delta_ref)
      self.assertEqual(restored.manifest, manifest)
      self.assertEqual(restored.tensor_path, delta_ref / "tensors.safetensors")


if __name__ == "__main__":
  unittest.main()
