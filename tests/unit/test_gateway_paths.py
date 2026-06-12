import asyncio
import os
import tempfile
import unittest
from unittest.mock import patch

from server import gateway
from server.store import InMemoryStore


class GetInfoTest(unittest.TestCase):
  def setUp(self) -> None:
    patcher = patch.object(gateway, "store", InMemoryStore())
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_get_info_answers_per_model_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      created = asyncio.run(gateway.create_model({"base_model": "Qwen/Qwen3-0.6B"}))
      model_id = created["request_id"]
      info = asyncio.run(gateway.get_info({"model_id": model_id}))

    self.assertEqual(info["model_name"], "Qwen/Qwen3-0.6B")
    self.assertEqual(info["model_data"]["tokenizer_id"], "Qwen/Qwen3-0.6B")
    self.assertEqual(info["model_id"], model_id)

  def test_get_info_distinguishes_models(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      id_a = asyncio.run(gateway.create_model({"base_model": "model-a"}))["request_id"]
      id_b = asyncio.run(gateway.create_model({"base_model": "model-b"}))["request_id"]

      self.assertEqual(asyncio.run(gateway.get_info({"model_id": id_a}))["model_name"], "model-a")
      self.assertEqual(asyncio.run(gateway.get_info({"model_id": id_b}))["model_name"], "model-b")

  def test_get_info_falls_back_to_base_model_env_for_unknown_model(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "no-such-model"}))
    self.assertEqual(info["model_name"], "env-model")

  def test_get_info_404s_without_mapping_or_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = asyncio.run(gateway.get_info({"model_id": "no-such-model"}))
    self.assertEqual(response.status_code, 404)


class GatewayPathTest(unittest.TestCase):
  def test_checkpoint_state_paths_are_model_scoped(self) -> None:
    old_tmp_dir = gateway.TMP_DIR
    with tempfile.TemporaryDirectory() as tmp_dir:
      gateway.TMP_DIR = tmp_dir
      self.addCleanup(setattr, gateway, "TMP_DIR", old_tmp_dir)

      self.assertEqual(
        gateway.checkpoint_state_path("job-a", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-a", "weights", "final"),
      )
      self.assertEqual(
        gateway.checkpoint_state_path("job-b", "final"),
        os.path.join(tmp_dir, "checkpoints", "job-b", "weights", "final"),
      )

  def test_checkpoint_state_paths_accept_explicit_output_directories(self) -> None:
    self.assertEqual(gateway.checkpoint_state_path("job-a", "/mnt/checkpoints/final"), "/mnt/checkpoints/final")


if __name__ == "__main__":
  unittest.main()
