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

  def test_get_info_uses_base_model_env(self) -> None:
    with patch.dict(os.environ, {"BASE_MODEL": "env-model"}, clear=True):
      info = asyncio.run(gateway.get_info({"model_id": "model-a"}))

    self.assertEqual(info["model_name"], "env-model")
    self.assertEqual(info["model_data"]["tokenizer_id"], "env-model")
    self.assertEqual(info["model_id"], "model-a")

  def test_get_info_404s_without_base_model_env(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      response = asyncio.run(gateway.get_info({"model_id": "model-a"}))
    self.assertEqual(response.status_code, 404)

  def test_create_model_requires_base_model_payload(self) -> None:
    response = asyncio.run(gateway.create_model({}))
    self.assertEqual(response.status_code, 400)

  def test_create_model_accepts_base_model_payload(self) -> None:
    created = asyncio.run(gateway.create_model({"base_model": "my-model"}))
    model_id = created["request_id"]
    queued = asyncio.run(gateway.store.get_requests())
    self.assertEqual(queued[0]["model_id"], model_id)
    self.assertEqual(queued[0]["payload"]["base_model"], "my-model")


class SessionTrackingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    for patcher in (
      patch.object(gateway, "store", self.store),
      patch.object(gateway, "get_store", lambda: self.store),
    ):
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_create_session_mints_unique_tracked_sessions(self) -> None:
    first = asyncio.run(gateway.create_session({}))
    second = asyncio.run(gateway.create_session({}))

    self.assertNotEqual(first["session_id"], second["session_id"])
    sessions = asyncio.run(self.store.list_sessions())
    self.assertIn(first["session_id"], sessions)
    self.assertIn(second["session_id"], sessions)

  def test_session_heartbeat_upserts_even_unknown_sessions(self) -> None:
    response = asyncio.run(gateway.session_heartbeat({"session_id": "sess-legacy"}))

    self.assertEqual(response, {"type": "session_heartbeat"})
    self.assertIn("sess-legacy", asyncio.run(self.store.list_sessions()))

  def test_session_heartbeat_tolerates_missing_session_id(self) -> None:
    asyncio.run(gateway.session_heartbeat({}))
    self.assertEqual(asyncio.run(self.store.list_sessions()), {})

  def test_create_model_registers_model_under_its_session(self) -> None:
    created = asyncio.run(gateway.create_model({"base_model": "my-model", "session_id": "sess-a"}))
    model_id = created["request_id"]

    self.assertEqual(asyncio.run(self.store.get_session_models("sess-a")), [model_id])
    self.assertIn("sess-a", asyncio.run(self.store.list_sessions()))


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
