"""CPU regressions for collective publication and Automodel checkpoint resume."""

import json
import multiprocessing as mp
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from training import adapter_snapshot
from training.automodel_worker import AutomodelTrainingWorker


class AdapterModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.base = torch.nn.Parameter(torch.tensor([10.0]), requires_grad=False)
    self.lora_A = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    self.lora_B = torch.nn.Parameter(torch.tensor([3.0, 4.0]))


def checkpoint_worker(*, lora=True):
  worker = AutomodelTrainingWorker.__new__(AutomodelTrainingWorker)
  worker.model = AdapterModel()
  worker.base_model_name = "test/base"
  worker.is_lora = lora
  worker.optimizer = None
  worker.trainable_params = [p for p in worker.model.parameters() if p.requires_grad]
  worker.cp_context = None
  worker.checkpointer = None
  worker.peft_config = None
  worker.tokenizer = None
  return worker


def write_adapter_config(root, **overrides):
  config = {"base_model_name_or_path": "test/base", "peft_type": "LORA", "r": 2, "lora_alpha": 16, **overrides}
  (Path(root) / "adapter_config.json").write_text(json.dumps(config))


def run_checkpoint_rank(rank, root):
  os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
  dist.init_process_group("gloo", init_method=f"file://{root}/rendezvous", rank=rank, world_size=2)
  try:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    worker = checkpoint_worker()

    def write_weights(path):
      Path(path).mkdir(parents=True, exist_ok=True)
      (Path(path) / f"rank-{rank}").write_text(str(rank))
      dist.barrier()

    worker.write_weights = write_weights
    worker.save_state("test", f"{root}/state")
    with patch.object(adapter_snapshot.paths, "snapshot_root", return_value=f"{root}/snapshots"):
      worker.write_adapter("test", alias="final", session_label="sampler-1")

    mesh = init_device_mesh("cpu", (2,))
    for name in ("lora_A", "lora_B"):
      parameter = getattr(worker.model, name)
      setattr(worker.model, name, torch.nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Shard(0)])))
    if rank == 0:
      save_file({"lora_A": torch.tensor([5.0, 6.0]), "lora_B": torch.tensor([7.0, 8.0])}, f"{root}/adapter_model.safetensors")
    dist.barrier()
    worker.load_adapter_weights(root)
    torch.testing.assert_close(worker.model.lora_A.full_tensor(), torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(worker.model.lora_B.full_tensor(), torch.tensor([7.0, 8.0]))
  finally:
    dist.destroy_process_group()


class AutomodelCheckpointTest(unittest.TestCase):
  def test_all_ranks_publish_into_one_checkpoint_and_snapshot(self):
    with tempfile.TemporaryDirectory() as root:
      ctx = mp.get_context("spawn")
      processes = [ctx.Process(target=run_checkpoint_rank, args=(rank, root)) for rank in range(2)]
      try:
        for process in processes:
          process.start()
        for process in processes:
          process.join(timeout=45)
          self.assertEqual(process.exitcode, 0, "Checkpoint rank failed or hung")
      finally:
        for process in processes:
          if process.is_alive():
            process.terminate()
            process.join(timeout=5)
      for relative in ("state", "snapshots/test/sampler-1", "snapshots/test/final"):
        for rank in range(2):
          self.assertEqual((Path(root) / relative / f"rank-{rank}").read_text(), str(rank))
      self.assertFalse(list(Path(root).rglob("*staging*")))

  def test_non_primary_snapshot_rank_keeps_shared_staging_files(self):
    with tempfile.TemporaryDirectory() as root:
      staged = Path(root) / ".staging" / "adapter"
      staged.mkdir(parents=True)
      weights = staged / "adapter_model.safetensors"
      weights.write_bytes(b"weights")
      with (
        patch.object(adapter_snapshot, "is_primary", return_value=False),
        patch.object(adapter_snapshot, "broadcast_object", return_value=str(staged.parent)),
        patch.object(adapter_snapshot.paths, "snapshot_root", return_value=root),
      ):
        adapter_snapshot.publish("test", lambda _: str(staged), session_label="sampler-1")
      self.assertTrue(weights.exists())

  def test_failed_checkpoint_publication_restores_previous_save(self):
    worker = checkpoint_worker()
    with tempfile.TemporaryDirectory() as root:
      target = Path(root) / "state"
      target.mkdir()
      (target / "old").write_text("previous")
      worker.write_weights = lambda path: (Path(path) / "new").write_text("replacement")
      rename = os.rename

      def fail_publication(source, destination):
        if str(source).endswith(".previous") or str(source) == str(target):
          return rename(source, destination)
        raise OSError("Simulated rename failure")

      with patch("training.automodel_worker.os.rename", side_effect=fail_publication), self.assertRaisesRegex(OSError, "Simulated"):
        worker.save_state("test", str(target))
      self.assertEqual((target / "old").read_text(), "previous")
      self.assertEqual(list(Path(root).iterdir()), [target])

  def test_adapter_restore_maps_native_keys_and_preserves_frozen_weights(self):
    worker = checkpoint_worker()
    worker.model.state_dict_adapter = SimpleNamespace(
      from_hf=lambda state, device_mesh: {key.replace("hf_A", "lora_A").replace("hf_B", "lora_B"): value for key, value in state.items()}
    )
    with tempfile.TemporaryDirectory() as root:
      save_file(
        {"base_model.model.hf_A": torch.tensor([5.0, 6.0]), "base_model.model.hf_B": torch.tensor([7.0, 8.0])},
        f"{root}/adapter_model.safetensors",
      )
      worker.load_adapter_weights(root)
    torch.testing.assert_close(worker.model.lora_A, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(worker.model.lora_B, torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(worker.model.base, torch.tensor([10.0]))

  def test_adapter_restore_rejects_missing_unexpected_or_wrong_shape_tensors(self):
    worker = checkpoint_worker()
    a, b = torch.zeros(2), torch.ones(2)
    states = (
      {"lora_A": a},
      {"lora_A": a, "lora_B": b, "other": a.clone()},
      {"lora_A": torch.zeros(3), "lora_B": b},
    )
    with tempfile.TemporaryDirectory() as root:
      for state in states:
        with self.subTest(keys=list(state)):
          save_file(state, f"{root}/adapter_model.safetensors")
          with self.assertRaises(ValueError):
            worker.load_adapter_weights(root)
          torch.testing.assert_close(worker.model.lora_A, torch.tensor([1.0, 2.0]))

  def test_adapter_restore_uses_canonical_keys_for_checkpoint_wrapped_layers(self):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    worker = checkpoint_worker()
    layer = worker.model
    layer.register_buffer("positions", torch.arange(2))
    worker.model = torch.nn.ModuleDict({"layer": checkpoint_wrapper(layer)})
    self.assertIn("layer._checkpoint_wrapped_module.lora_A", dict(worker.model.named_parameters()))
    with tempfile.TemporaryDirectory() as root:
      save_file({"layer.lora_A": torch.tensor([5.0, 6.0]), "layer.lora_B": torch.tensor([7.0, 8.0])}, f"{root}/adapter_model.safetensors")
      worker.load_adapter_weights(root)
    torch.testing.assert_close(layer.lora_A, torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(layer.lora_B, torch.tensor([7.0, 8.0]))
    torch.testing.assert_close(layer.base, torch.tensor([10.0]))
    torch.testing.assert_close(layer.positions, torch.arange(2))

  def test_adapter_config_rejects_rank_scaling_and_dropout_drift(self):
    worker = checkpoint_worker()
    worker.build_peft_config = lambda: SimpleNamespace(dim=2, alpha=16, use_dora=False, dropout=0.0, dropout_position="post")
    with tempfile.TemporaryDirectory() as root:
      write_adapter_config(root)
      worker.validate_adapter_config(root)
      for overrides in ({"r": 4}, {"lora_alpha": 32}, {"use_rslora": True}, {"lora_dropout": 0.1}):
        with self.subTest(overrides=overrides):
          write_adapter_config(root, **overrides)
          with self.assertRaises(ValueError):
            worker.validate_adapter_config(root)

  def test_weights_only_resume_replaces_model_and_discards_optimizer(self):
    worker = checkpoint_worker()
    old_model = worker.model
    worker.optimizer = torch.optim.AdamW(worker.trainable_params)

    def load_base_model(base_model):
      self.assertIsNone(worker.model)
      self.assertIsNone(worker.optimizer)
      worker.model = AdapterModel()

    worker.load_base_model = load_base_model
    worker.validate_adapter_config = Mock()
    with tempfile.TemporaryDirectory() as root:
      write_adapter_config(root)
      save_file({"lora_A": torch.zeros(2), "lora_B": torch.ones(2)}, f"{root}/adapter_model.safetensors")
      worker.load_from_state("test", root)
    self.assertIsNot(worker.model, old_model)
    self.assertIsNone(worker.optimizer)
    self.assertEqual(worker.base_model_name, "test/base")
    self.assertIs(worker.trainable_params[0], worker.model.lora_A)

  def test_optimizer_resume_matches_uninterrupted_next_step(self):
    worker = checkpoint_worker(lora=False)
    worker.prepare_model_for_training()
    worker.optimizer = torch.optim.AdamW(worker.trainable_params, lr=0.03, betas=(0.8, 0.9), weight_decay=0.2, foreach=False)

    def step(trainer):
      sum(param.square().sum() for param in trainer.trainable_params).backward()
      trainer.optimizer.step()
      trainer.optimizer.zero_grad(set_to_none=True)

    step(worker)
    with tempfile.TemporaryDirectory() as root:
      worker.write_weights = lambda path: save_file(worker.model.state_dict(), f"{path}/model.safetensors")
      worker.save_state("test", f"{root}/state", include_optimizer=True)
      restored = checkpoint_worker(lora=False)

      def load_base_model(path):
        from safetensors.torch import load_file

        restored.model = AdapterModel()
        restored.model.load_state_dict(load_file(f"{path}/model.safetensors"))

      restored.load_base_model = load_base_model
      restored.load_from_state("test", f"{root}/state", restore_optimizer=True)
      step(worker)
      step(restored)
      for expected, actual in zip(worker.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(actual, expected)
      self.assertEqual(tuple(restored.optimizer.param_groups[0]["betas"]), (0.8, 0.9))

  def test_incompatible_training_mode_or_missing_optimizer_fails_before_replacing_model(self):
    worker = checkpoint_worker(lora=False)
    original = worker.model
    with tempfile.TemporaryDirectory() as root:
      metadata = Path(root) / "metadata.json"
      metadata.write_text(json.dumps({"base_model": "test/base", "lora": True}))
      with self.assertRaisesRegex(ValueError, "training mode"):
        worker.load_from_state("test", root)
      self.assertIs(worker.model, original)
      metadata.write_text(json.dumps({"base_model": "test/base", "lora": False, "has_optimizer": True}))
      with self.assertRaisesRegex(FileNotFoundError, "optimizer"):
        worker.load_from_state("test", root, restore_optimizer=True)
      self.assertIs(worker.model, original)


if __name__ == "__main__":
  unittest.main()
