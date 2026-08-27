import asyncio
import json
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
    self.assertEqual(queued[0]["payload"], {})
    meta = json.loads(gateway.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "my-model")


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


class ClaimReconcilerTest(unittest.IsolatedAsyncioTestCase):
  class _K8sManager:
    def __init__(self) -> None:
      self.calls = 0

    def reconcile_managed_claims(self) -> list[str]:
      self.calls += 1
      return ["claim-idle"]

  class _LocalManager:
    """Stands in for LocalWorkerManager, which provisions no DRA claims."""

  async def test_reconciler_runs_on_its_interval(self) -> None:
    manager = self._K8sManager()
    task = asyncio.create_task(gateway.run_claim_reconciler(manager, interval=0.01))
    await asyncio.sleep(0.1)
    task.cancel()

    self.assertGreater(manager.calls, 1, "reconcile loop should fire repeatedly, not once")

  async def test_reconciler_survives_a_failing_pass(self) -> None:
    manager = self._K8sManager()

    def boom() -> list[str]:
      manager.calls += 1
      raise RuntimeError("API server unavailable")

    manager.reconcile_managed_claims = boom
    task = asyncio.create_task(gateway.run_claim_reconciler(manager, interval=0.01))
    await asyncio.sleep(0.1)
    task.cancel()

    # A transient API error must not silently kill the only thing reclaiming GPUs.
    self.assertGreater(manager.calls, 1)

  async def test_reconciler_starts_only_for_claim_provisioning_managers(self) -> None:
    self.assertIsNone(gateway.start_claim_reconciler(None))
    self.assertIsNone(gateway.start_claim_reconciler(self._LocalManager()))

    task = gateway.start_claim_reconciler(self._K8sManager())
    self.assertIsNotNone(task)
    task.cancel()

  async def test_reconciler_can_be_disabled(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_CLAIM_RECONCILE_INTERVAL_SECONDS": "0"}):
      self.assertIsNone(gateway.start_claim_reconciler(self._K8sManager()))


class SessionRegistryTest(unittest.TestCase):
  def test_expiring_last_session_releases_its_models(self) -> None:
    registry = gateway.SessionRegistry(ttl_seconds=-1.0)
    registry.open("sess-a")
    registry.bind("sess-a", "model-1")

    self.assertEqual(registry.expire(), {"model-1"})
    self.assertEqual(registry.expire(), set())

  def test_live_session_holds_a_shared_model(self) -> None:
    registry = gateway.SessionRegistry(ttl_seconds=3600)
    registry.open("sess-a")
    registry.open("sess-b")
    registry.bind("sess-a", "model-1")
    registry.bind("sess-b", "model-1")
    registry._sessions["sess-a"]["last_seen"] -= 7200

    self.assertEqual(registry.expire(), set())

  def test_heartbeat_recreates_a_forgotten_session(self) -> None:
    registry = gateway.SessionRegistry(ttl_seconds=3600)
    registry.touch("sess-after-restart")
    self.assertIn("sess-after-restart", registry._sessions)

  def test_bind_without_a_known_session_is_ignored(self) -> None:
    registry = gateway.SessionRegistry()
    registry.bind(None, "model-1")
    registry.bind("sess-missing", "model-1")
    self.assertEqual(registry.live_models(), set())


class SessionTeardownTest(unittest.TestCase):
  def setUp(self) -> None:
    self.registry = gateway.SessionRegistry(ttl_seconds=3600)
    patcher = patch.object(gateway, "session_registry", self.registry)
    patcher.start()
    self.addCleanup(patcher.stop)

  def _age_out(self, session_id: str) -> None:
    self.registry._sessions[session_id]["last_seen"] -= 7200

  def test_shared_lora_worker_survives_until_the_last_adapter_leaves(self) -> None:
    info = patch.object(gateway, "get_model_target_info", side_effect=lambda m: (None, "qwen-base", True))
    info.start()
    self.addCleanup(info.stop)

    self.registry.open("sess-a")
    self.registry.bind("sess-a", "lora-job-1")
    self.registry.open("sess-b")
    self.registry.bind("sess-b", "lora-job-2")

    self._age_out("sess-a")
    self.assertEqual(gateway._workers_to_teardown(), [])

    self._age_out("sess-b")
    self.assertEqual(gateway._workers_to_teardown(), ["lora-job-2"])

  def test_fft_model_tears_down_as_soon_as_its_session_dies(self) -> None:
    info = patch.object(gateway, "get_model_target_info", side_effect=lambda m: (None, m, False))
    info.start()
    self.addCleanup(info.stop)

    self.registry.open("sess-a")
    self.registry.bind("sess-a", "fft-job-1")
    self._age_out("sess-a")

    self.assertEqual(gateway._workers_to_teardown(), ["fft-job-1"])


if __name__ == "__main__":
  unittest.main()
