import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from server import gateway
from server.protocol import ClientSession, CreateSessionRequest, SessionHeartbeatRequest
from server.store import InMemoryStore, RedisStore


class GatewaySessionLifecycleTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    self.old_store = gateway.store
    gateway.store = self.store
    self.env_patch = patch.dict(os.environ, {"OPEN_RL_SESSION_TTL_SECONDS": "10"})
    self.env_patch.start()

  def tearDown(self) -> None:
    self.env_patch.stop()
    gateway.store = self.old_store

  async def create_session(self, **overrides):
    request = CreateSessionRequest(
      tags=overrides.get("tags", []),
      user_metadata=overrides.get("user_metadata"),
      sdk_version=overrides.get("sdk_version"),
      project_id=overrides.get("project_id"),
    )
    return await gateway.create_session(request)

  async def test_create_session_returns_unique_tinker_responses(self) -> None:
    first = await self.create_session()
    second = await self.create_session()

    self.assertEqual(first["type"], "create_session")
    self.assertEqual(second["type"], "create_session")
    self.assertTrue(first["session_id"].startswith("sess-"))
    self.assertNotEqual(first["session_id"], second["session_id"])

  async def test_create_session_persists_sdk_metadata(self) -> None:
    response = await self.create_session(
      tags=["test", "gpu"],
      user_metadata={"job": "smoke", "attempt": 2},
      sdk_version="0.18.2",
      project_id="project-a",
    )

    raw_session = await self.store.get_value(gateway.session_key(response["session_id"]))
    session = ClientSession.model_validate_json(raw_session)
    self.assertEqual(session.tags, ["test", "gpu"])
    self.assertEqual(session.user_metadata, {"job": "smoke", "attempt": 2})
    self.assertEqual(session.sdk_version, "0.18.2")
    self.assertEqual(session.project_id, "project-a")

  async def test_heartbeat_updates_session_and_refreshes_ttl(self) -> None:
    with patch("server.store.time.monotonic", return_value=100):
      created = await self.create_session()
    key = gateway.session_key(created["session_id"])
    self.assertEqual(self.store.kv_expirations[key], 110)

    with (
      patch("server.store.time.monotonic", return_value=105),
      patch("server.gateway.time.time", return_value=1234),
    ):
      response = await gateway.session_heartbeat(SessionHeartbeatRequest(session_id=created["session_id"]))

    self.assertEqual(response, {"type": "session_heartbeat"})
    self.assertEqual(self.store.kv_expirations[key], 115)
    session = ClientSession.model_validate_json(self.store.kv_store[key])
    self.assertEqual(session.last_heartbeat, 1234)

  async def test_unknown_and_expired_sessions_return_not_found(self) -> None:
    unknown = await gateway.session_heartbeat(SessionHeartbeatRequest(session_id="sess-unknown"))
    self.assertEqual(unknown.status_code, 404)
    self.assertEqual(json.loads(unknown.body), {"error": "session not found or expired"})

    with patch("server.store.time.monotonic", return_value=100):
      created = await self.create_session()
    with patch("server.store.time.monotonic", return_value=110):
      expired = await gateway.session_heartbeat(SessionHeartbeatRequest(session_id=created["session_id"]))

    self.assertEqual(expired.status_code, 404)
    self.assertEqual(json.loads(expired.body), {"error": "session not found or expired"})

  async def test_missing_and_blank_session_ids_are_rejected(self) -> None:
    transport = httpx.ASGITransport(app=gateway.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
      missing = await client.post("/api/v1/session_heartbeat", json={})
      blank = await client.post("/api/v1/session_heartbeat", json={"session_id": ""})

    self.assertEqual(missing.status_code, 422)
    self.assertEqual(blank.status_code, 422)

  async def test_non_positive_session_ttl_is_rejected_clearly(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_SESSION_TTL_SECONDS": "0"}):
      response = await self.create_session()

    self.assertEqual(response.status_code, 500)
    self.assertEqual(json.loads(response.body), {"error": "OPEN_RL_SESSION_TTL_SECONDS must be a positive integer"})


class InMemoryStoreExpirationTest(unittest.IsolatedAsyncioTestCase):
  async def test_values_expire_and_can_be_replaced_without_a_ttl(self) -> None:
    store = InMemoryStore()
    with patch("server.store.time.monotonic", return_value=10):
      await store.set_value("key", "old", expires_seconds=5)
    with patch("server.store.time.monotonic", return_value=15):
      self.assertIsNone(await store.get_value("key"))

    await store.set_value("key", "new")
    with patch("server.store.time.monotonic", return_value=1000):
      self.assertEqual(await store.get_value("key"), "new")


class RedisStoreValueTest(unittest.IsolatedAsyncioTestCase):
  async def test_set_and_get_use_native_redis_expiration_semantics(self) -> None:
    store = RedisStore("redis://unused")
    redis = AsyncMock()
    redis.get.return_value = "payload"
    store.redis = redis

    await store.set_value("open_rl:session:sess-a", "payload", expires_seconds=17)
    value = await store.get_value("open_rl:session:sess-a")

    redis.set.assert_awaited_once_with("open_rl:session:sess-a", "payload", ex=17)
    redis.get.assert_awaited_once_with("open_rl:session:sess-a")
    self.assertEqual(value, "payload")


if __name__ == "__main__":
  unittest.main()
