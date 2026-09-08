import os
import unittest
from unittest.mock import patch

from fastapi import Request

from server import gateway
from server.session_registry import SessionRegistry
from server.store import InMemoryStore


class RuntimeManager:
  def __init__(self):
    self.ensured = []
    self.released = []

  def ensure(self, model_id, role):
    self.ensured.append((model_id, role))

  def release_owner(self, owner):
    self.released.append(owner)
    return {"test-base"} if owner == "test-base" else set()


class SessionLifecycleTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.store = InMemoryStore()
    self.manager = RuntimeManager()
    self.registry = SessionRegistry(self.store)
    self.enterContext(patch.object(gateway, "store", self.store))
    self.enterContext(patch.object(gateway, "get_store", return_value=self.store))
    self.enterContext(patch("server.store.get_store", return_value=self.store))
    self.enterContext(patch.object(gateway, "session_registry", self.registry))
    self.enterContext(patch.object(gateway, "worker_manager", self.manager))
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "vllm", "OPEN_RL_ENABLE_FFT": "true"}))

  async def expire(self, session_id):
    # What the store does on its own once the heartbeats stop.
    await self.store.delete_values(f"open_rl:session:{session_id}")

  async def reap(self):
    for owner in await self.registry.abandoned():
      await gateway.teardown_owner(owner)
      await self.registry.forget(owner)

  async def test_shared_lora_owner_outlives_the_session_that_created_it(self):
    training = (await gateway.create_session({}))["session_id"]
    adapter = (await gateway.create_model({"base_model": "test-base", "session_id": training}))["request_id"]
    fft_request = Request({"type": "http", "headers": [(b"x-open-rl-fine-tuning-type", b"full")]})
    fft_model = (await gateway.create_model({"base_model": "fft-base", "session_id": training}, request=fft_request))["request_id"]
    await self.store.set_value("open_rl:sampler_ready:test-base", "1")
    sampling = (await gateway.create_session({}))["session_id"]
    await gateway.create_sampling_session({"model_path": f"tinker://{adapter}/sampler_weights/test", "session_id": sampling})
    self.assertEqual(self.manager.ensured, [(adapter, "trainer"), (fft_model, "trainer"), (adapter, "sampler")])

    await self.expire(training)
    await self.reap()
    self.assertEqual(self.manager.released, [fft_model.lower()])

    await self.expire(sampling)
    await self.reap()
    self.assertEqual(self.manager.released, [fft_model.lower(), "test-base"])
    self.assertIsNone(await self.store.get_value("open_rl:sampler_ready:test-base"))
    self.assertEqual(await self.registry.abandoned(), [])

  async def test_an_owner_stays_abandoned_until_forgotten(self):
    await self.registry.attach("a", "base")
    await self.expire("a")
    self.assertEqual(await self.registry.abandoned(), ["base"])
    self.assertEqual(await self.registry.abandoned(), ["base"])
    await self.registry.forget("base")
    self.assertEqual(await self.registry.abandoned(), [])

  async def test_a_session_attaching_during_teardown_keeps_the_owner(self):
    await self.registry.attach("a", "base")
    await self.expire("a")
    self.assertEqual(await self.registry.abandoned(), ["base"])
    await self.registry.attach("b", "base")
    await self.registry.forget("base")
    self.assertEqual(await self.registry.abandoned(), [])
    await self.expire("b")
    self.assertEqual(await self.registry.abandoned(), ["base"])

  async def test_a_heartbeat_for_an_unknown_session_opens_it(self):
    await self.registry.heartbeat("after-a-wiped-store")
    self.assertTrue(await self.registry.live("after-a-wiped-store"))
