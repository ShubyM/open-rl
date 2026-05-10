from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from tests._server_fixture import SERVER_DIR


def _load_state_delta_module():
  spec = importlib.util.spec_from_file_location("state_delta_under_test", Path(SERVER_DIR) / "state_delta.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


state_delta = _load_state_delta_module()


class TestStateDelta(unittest.TestCase):
  def test_build_lora_delta_manifest_normalizes_names_and_hashes_config(self) -> None:
    manifest = state_delta.build_lora_delta_manifest(
      run_id="adapter-a",
      version=3,
      base_ref="adapter-a:2",
      apply_target="vllm_lora",
      adapter_config={"r": 2, "peft_type": "LORA"},
      tensors=[
        ("base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight", torch.zeros(2, 4)),
        ("base_model.model.layers.0.self_attn.q_proj.lora_B.adapter-a.weight", torch.ones(4, 2)),
      ],
      compute_checksums=True,
      created_at=1.0,
    )

    self.assertEqual(manifest.schema_version, 1)
    self.assertEqual(manifest.base_ref, "adapter-a:2")
    self.assertEqual(manifest.training_mode, "lora")
    self.assertEqual(manifest.apply_target, "vllm_lora")
    self.assertIsNotNone(manifest.adapter_config_hash)
    self.assertEqual(
      [entry.normalized_name for entry in manifest.tensors],
      [
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight",
      ],
    )
    self.assertEqual([entry.role for entry in manifest.tensors], ["lora_a", "lora_b"])
    self.assertTrue(all(entry.checksum for entry in manifest.tensors))

  def test_manifest_round_trips_as_dict(self) -> None:
    manifest = state_delta.build_lora_delta_manifest(
      run_id="adapter-a",
      version=1,
      apply_target="trainer_lora",
      tensors=[("base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight", torch.zeros(2, 4))],
      created_at=1.0,
    )

    restored = state_delta.StateDeltaManifest.from_dict(manifest.to_dict())

    self.assertEqual(restored, manifest)

  def test_validate_manifest_rejects_duplicate_normalized_names(self) -> None:
    manifest = state_delta.build_lora_delta_manifest(
      run_id="adapter-a",
      version=1,
      apply_target="vllm_lora",
      tensors=[("base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight", torch.zeros(2, 4))],
      created_at=1.0,
    )
    duplicate = replace(manifest, tensors=(manifest.tensors[0], replace(manifest.tensors[0], storage_key="other-key")))

    with self.assertRaisesRegex(ValueError, "duplicate tensor entries"):
      state_delta.validate_delta_manifest(duplicate)

  def test_validate_for_apply_rejects_stale_or_wrong_base(self) -> None:
    manifest = state_delta.build_lora_delta_manifest(
      run_id="adapter-a",
      version=2,
      base_ref="adapter-a:1",
      apply_target="vllm_lora",
      tensors=[("base_model.model.layers.0.self_attn.q_proj.lora_A.adapter-a.weight", torch.zeros(2, 4))],
      created_at=1.0,
    )

    state_delta.validate_for_apply(manifest, state_delta.ReceiverState(ref="adapter-a:1", version=1))
    with self.assertRaisesRegex(ValueError, "delta does not apply"):
      state_delta.validate_for_apply(manifest, state_delta.ReceiverState(ref="adapter-a:1", version=2))
    with self.assertRaisesRegex(ValueError, "delta does not apply"):
      state_delta.validate_for_apply(manifest, state_delta.ReceiverState(ref="adapter-a:0", version=1))


if __name__ == "__main__":
  unittest.main()
