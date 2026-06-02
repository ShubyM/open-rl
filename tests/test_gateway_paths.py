import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._server_fixture import SERVER_DIR

sys.path.insert(0, str(SERVER_DIR))
redis_stub = types.ModuleType("redis")
redis_asyncio_stub = types.ModuleType("redis.asyncio")
redis_stub.asyncio = redis_asyncio_stub
sys.modules.setdefault("redis", redis_stub)
sys.modules.setdefault("redis.asyncio", redis_asyncio_stub)

try:
  import gateway  # noqa: E402
  from store import InMemoryStore  # noqa: E402

  HAS_SERVER_DEPS = True
except ModuleNotFoundError:
  gateway = None
  InMemoryStore = None
  HAS_SERVER_DEPS = False


@unittest.skipUnless(HAS_SERVER_DEPS, "server dependencies are not installed")
class GatewayWeightsPathTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.previous_tmp_dir = gateway.TMP_DIR
    self.previous_store = gateway.store
    gateway.TMP_DIR = self.tmp_dir.name
    gateway.store = InMemoryStore()

  def tearDown(self) -> None:
    gateway.TMP_DIR = self.previous_tmp_dir
    gateway.store = self.previous_store
    self.tmp_dir.cleanup()

  async def test_save_weights_scopes_same_alias_by_model(self) -> None:
    await gateway.save_weights({"model_id": "run-a", "path": "same"})
    await gateway.save_weights({"model_id": "run-b", "path": "same"})

    run_a = (await gateway.store.get_requests_for_model("run-a"))[0]
    run_b = (await gateway.store.get_requests_for_model("run-b"))[0]

    self.assertEqual(run_a["state_path"], os.path.join(self.tmp_dir.name, "checkpoints", "run-a", "weights", "same"))
    self.assertEqual(run_b["state_path"], os.path.join(self.tmp_dir.name, "checkpoints", "run-b", "weights", "same"))
    self.assertEqual(run_a["response_path"], "tinker://run-a/weights/same")
    self.assertEqual(run_b["response_path"], "tinker://run-b/weights/same")

  async def test_load_weights_resolves_tinker_path_to_source_model(self) -> None:
    await gateway.load_weights({"model_id": "new-run", "path": "tinker://old-run/weights/final", "optimizer": True})

    request = (await gateway.store.get_requests_for_model("new-run"))[0]

    self.assertEqual(request["state_path"], os.path.join(self.tmp_dir.name, "checkpoints", "old-run", "weights", "final"))
    self.assertTrue(request["restore_optimizer"])

  async def test_unnamed_sampler_snapshot_uses_generated_tinker_path(self) -> None:
    await gateway.save_weights_for_sampler({"model_id": "run-a", "sampling_session_seq_id": 7})

    request = (await gateway.store.get_requests_for_model("run-a"))[0]

    self.assertEqual(request["path"], "tinker://run-a/sampler_weights/sampler-7")
    self.assertEqual(request["sampling_session_id"], "tinker://run-a/sampler_weights/sampler-7")

  async def test_weights_info_reads_model_scoped_metadata(self) -> None:
    checkpoint = Path(self.tmp_dir.name) / "checkpoints" / "run-a" / "weights" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(
      json.dumps(
        {
          "base_model": "Qwen/test",
          "lora_rank": 8,
          "train_attn": True,
          "train_mlp": False,
          "train_unembed": False,
          "training_mode": "lora",
        }
      )
    )

    response = await gateway.weights_info({"tinker_path": "tinker://run-a/weights/final"})

    self.assertEqual(
      response,
      {
        "base_model": "Qwen/test",
        "is_lora": True,
        "lora_rank": 8,
        "train_unembed": False,
        "train_mlp": False,
        "train_attn": True,
      },
    )

  async def test_weights_info_reports_full_checkpoint_without_full_mode_lie(self) -> None:
    checkpoint = Path(self.tmp_dir.name) / "checkpoints" / "run-a" / "weights" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"base_model": "Qwen/test", "training_mode": "full"}))

    response = await gateway.weights_info({"tinker_path": "tinker://run-a/weights/final"})

    self.assertEqual(
      response,
      {
        "base_model": "Qwen/test",
        "is_lora": False,
        "lora_rank": None,
        "train_unembed": None,
        "train_mlp": None,
        "train_attn": None,
      },
    )

  async def test_weights_info_lies_for_full_checkpoint_in_full_mode(self) -> None:
    checkpoint = Path(self.tmp_dir.name) / "checkpoints" / "run-a" / "weights" / "final"
    checkpoint.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(json.dumps({"base_model": "Qwen/test", "training_mode": "full"}))

    with patch.dict(os.environ, {"OPEN_RL_TRAINING_MODE": "full"}):
      response = await gateway.weights_info({"tinker_path": "tinker://run-a/weights/final"})

    self.assertEqual(
      response,
      {
        "base_model": "Qwen/test",
        "is_lora": True,
        "lora_rank": 16,
        "train_unembed": True,
        "train_mlp": True,
        "train_attn": True,
      },
    )


if __name__ == "__main__":
  unittest.main()
