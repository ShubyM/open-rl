from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from tests._server_fixture import SERVER_DIR

sys.path.insert(0, str(SERVER_DIR))
import checkpoint_store  # noqa: E402


class TestCheckpointStore(unittest.TestCase):
  def test_file_checkpoint_metadata_round_trips_training_and_inference_refs(self) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
      store = checkpoint_store.FileCheckpointStore(Path(tmp_dir) / "checkpoints")
      checkpoint_dir = store.resolve("adapter-a-step-1")
      metadata = checkpoint_store.lora_checkpoint_metadata(
        base_model="Qwen/Qwen3-0.6B",
        model_id="adapter-a",
        checkpoint_dir=checkpoint_dir,
        kind="weights",
        state_id="tinker://adapter-a/sampler_weights/current",
        version=7,
        adapter_name="adapter-a",
        inference_backend="vllm",
        optimizer_ref=str(checkpoint_dir / "optimizer.pt"),
        state_delta_ref=str(checkpoint_dir / "delta" / "adapter-a" / "1"),
      )

      metadata_path = store.write_metadata(checkpoint_dir, metadata)
      restored = store.read_metadata(checkpoint_dir)

      self.assertEqual(Path(metadata_path), checkpoint_dir / "metadata.json")
      self.assertEqual(restored.base_model, "Qwen/Qwen3-0.6B")
      self.assertEqual(restored.state_id, "tinker://adapter-a/sampler_weights/current")
      self.assertEqual(restored.version, 7)
      self.assertEqual(restored.targets, ("trainer", "inference"))
      self.assertEqual(restored.adapter_ref, str(checkpoint_dir))
      self.assertEqual(restored.optimizer_ref, str(checkpoint_dir / "optimizer.pt"))
      self.assertEqual(restored.state_delta_ref, str(checkpoint_dir / "delta" / "adapter-a" / "1"))
      self.assertEqual(restored.adapter_name, "adapter-a")
      self.assertEqual(restored.inference_backend, "vllm")

      model_state = restored.to_model_state(str(checkpoint_dir))
      self.assertEqual(model_state.state_id, "tinker://adapter-a/sampler_weights/current")
      self.assertEqual(model_state.base_model, "Qwen/Qwen3-0.6B")
      self.assertEqual(model_state.training_mode, "lora")
      self.assertEqual(model_state.version, 7)
      self.assertEqual(model_state.adapter_ref, str(checkpoint_dir))

  def test_checkpoint_metadata_reads_open_rl_checkpoint_baseline_shape(self) -> None:
    metadata = checkpoint_store.CheckpointMetadata.from_dict(
      {
        "base_model": "Qwen/Qwen3-0.6B",
        "created_at": "2026-05-09T00:00:00",
        "has_optimizer": True,
        "kind": "weights",
        "model_id": "adapter-a",
        "timestamp": 123.0,
      }
    )

    self.assertEqual(metadata.base_model, "Qwen/Qwen3-0.6B")
    self.assertEqual(metadata.model_id, "adapter-a")
    self.assertEqual(metadata.created_at, 123.0)
    self.assertTrue(metadata.has_optimizer)


if __name__ == "__main__":
  unittest.main()
