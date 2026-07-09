import unittest

from server.session_reaper import SessionReaper
from server.store import InMemoryStore


class _FakeWorkerManager:
  def __init__(self):
    self.trainer_shutdowns: list[str] = []
    self.shutdowns: list[str] = []

  def launch(self, model_id: str, base_model: str | None = None) -> None:
    pass

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> None:
    pass

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> None:
    pass

  def shutdown(self, model_id: str) -> None:
    self.shutdowns.append(model_id)

  def shutdown_trainer(self, model_id: str) -> None:
    self.trainer_shutdowns.append(model_id)

  def shutdown_all(self) -> None:
    pass


class SessionReaperTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.store = InMemoryStore()
    self.manager = _FakeWorkerManager()
    self.reaper = SessionReaper(self.store, self.manager, idle_timeout_sec=100.0, poll_interval_sec=1.0, teardown_grace_sec=30.0)

  async def _drain_queue(self) -> list[dict]:
    requests = []
    while self.store.active_tenants:
      requests.extend(await self.store.get_requests())
    return requests

  async def test_fresh_session_is_left_alone(self) -> None:
    await self.store.touch_session("sess-a")
    await self.store.add_session_model("sess-a", "model-a")
    last_seen = self.store.session_last_seen["sess-a"]

    await self.reaper.run_once(now=last_seen + 50)

    self.assertEqual(self.manager.trainer_shutdowns, [])
    self.assertIn("sess-a", await self.store.list_sessions())
    self.assertEqual(await self._drain_queue(), [])

  async def test_idle_session_gets_sentinel_then_forced_teardown(self) -> None:
    await self.store.touch_session("sess-a")
    await self.store.add_session_model("sess-a", "model-a")
    await self.store.add_session_model("sess-a", "model-b")
    last_seen = self.store.session_last_seen["sess-a"]

    # First pass past the idle timeout: graceful sentinels only.
    await self.reaper.run_once(now=last_seen + 101)
    queued = await self._drain_queue()
    self.assertEqual({req["model_id"] for req in queued}, {"model-a", "model-b"})
    self.assertTrue(all(req["request_id"] == "SHUTDOWN_SENTINEL" for req in queued))
    self.assertEqual(self.manager.trainer_shutdowns, [])
    self.assertIn("sess-a", await self.store.list_sessions())

    # Within the grace period: nothing further happens.
    await self.reaper.run_once(now=last_seen + 110)
    self.assertEqual(self.manager.trainer_shutdowns, [])

    # After the grace period: force trainer shutdown and forget the session.
    await self.reaper.run_once(now=last_seen + 101 + 31)
    self.assertEqual(sorted(self.manager.trainer_shutdowns), ["model-a", "model-b"])
    self.assertEqual(self.manager.shutdowns, [])  # samplers untouched
    self.assertNotIn("sess-a", await self.store.list_sessions())
    self.assertEqual(await self.store.get_session_models("sess-a"), [])

  async def test_heartbeat_during_grace_cancels_forced_teardown(self) -> None:
    await self.store.touch_session("sess-a")
    await self.store.add_session_model("sess-a", "model-a")
    last_seen = self.store.session_last_seen["sess-a"]

    await self.reaper.run_once(now=last_seen + 101)
    await self._drain_queue()

    await self.store.touch_session("sess-a")
    fresh = self.store.session_last_seen["sess-a"]
    await self.reaper.run_once(now=fresh + 1)
    # Even once the session goes idle again, the old sentinel timestamp must
    # not carry over: the next reap starts from the graceful phase.
    await self.reaper.run_once(now=fresh + 101)
    self.assertEqual(self.manager.trainer_shutdowns, [])
    self.assertEqual(len(await self._drain_queue()), 1)

  async def test_idle_session_without_models_is_deleted_immediately(self) -> None:
    await self.store.touch_session("sess-empty")
    last_seen = self.store.session_last_seen["sess-empty"]

    await self.reaper.run_once(now=last_seen + 101)

    self.assertNotIn("sess-empty", await self.store.list_sessions())
    self.assertEqual(self.manager.trainer_shutdowns, [])
    self.assertEqual(await self._drain_queue(), [])


class InMemorySessionStoreTest(unittest.IsolatedAsyncioTestCase):
  async def test_session_roundtrip(self) -> None:
    store = InMemoryStore()
    await store.touch_session("sess-a")
    await store.add_session_model("sess-a", "model-b")
    await store.add_session_model("sess-a", "model-a")
    await store.add_session_model("sess-a", "model-a")

    self.assertEqual(list(await store.list_sessions()), ["sess-a"])
    self.assertEqual(await store.get_session_models("sess-a"), ["model-a", "model-b"])

    await store.delete_session("sess-a")
    self.assertEqual(await store.list_sessions(), {})
    self.assertEqual(await store.get_session_models("sess-a"), [])


if __name__ == "__main__":
  unittest.main()
