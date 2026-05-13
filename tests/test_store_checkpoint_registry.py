from __future__ import annotations

import asyncio
import sys
import types
import unittest

from tests._server_fixture import SERVER_DIR

sys.modules.setdefault("redis", types.SimpleNamespace(asyncio=types.SimpleNamespace(from_url=lambda *_args, **_kwargs: None)))
sys.modules.setdefault("redis.asyncio", types.SimpleNamespace(from_url=lambda *_args, **_kwargs: None))
sys.path.insert(0, str(SERVER_DIR))
import store  # noqa: E402
from model_state import latest_training_state_alias  # noqa: E402


class FakeRedis:
  def __init__(self):
    self.hashes = {}

  async def hset(self, name, key, value):
    self.hashes.setdefault(name, {})[key] = value

  async def hget(self, name, key):
    return self.hashes.get(name, {}).get(key)


class TestCheckpointRegistry(unittest.TestCase):
  def test_in_memory_store_tracks_latest_checkpoint_by_model(self) -> None:
    async def run() -> None:
      request_store = store.InMemoryStore()
      self.assertIsNone(await request_store.latest_checkpoint("adapter-a"))

      await request_store.publish_checkpoint("adapter-a", "/tmp/open-rl/checkpoints/a-1", {"delta_ref": "delta-a-1"})
      await request_store.publish_checkpoint("adapter-b", "/tmp/open-rl/checkpoints/b-1", {"delta_ref": "delta-b-1"})
      await request_store.publish_checkpoint("adapter-a", "/tmp/open-rl/checkpoints/a-2", {"delta_ref": "delta-a-2"})

      self.assertEqual(
        await request_store.latest_checkpoint("adapter-a"),
        {"checkpoint_ref": "/tmp/open-rl/checkpoints/a-2", "metadata": {"delta_ref": "delta-a-2"}},
      )
      self.assertEqual(
        await request_store.latest_checkpoint("adapter-b"),
        {"checkpoint_ref": "/tmp/open-rl/checkpoints/b-1", "metadata": {"delta_ref": "delta-b-1"}},
      )
      self.assertEqual(
        (await request_store.get_model_state(latest_training_state_alias("adapter-a")))["delta_ref"],
        "delta-a-2",
      )

    asyncio.run(run())

  def test_in_memory_store_tracks_model_states_by_state_id(self) -> None:
    async def run() -> None:
      request_store = store.InMemoryStore()
      state = {
        "state_id": "tinker://adapter-a/sampler_weights/current",
        "model_id": "adapter-a",
        "base_model": "Qwen/Qwen3-0.6B",
        "training_mode": "lora",
        "version": 2,
        "adapter_ref": "/tmp/open-rl/checkpoints/a-2",
      }

      self.assertIsNone(await request_store.get_model_state(state["state_id"]))
      await request_store.publish_model_state(state["state_id"], state)

      self.assertEqual(await request_store.get_model_state(state["state_id"]), state)

    asyncio.run(run())

  def test_redis_store_tracks_latest_checkpoint_by_model(self) -> None:
    async def run() -> None:
      request_store = store.RedisStore("redis://unused")
      request_store.redis = FakeRedis()

      await request_store.publish_checkpoint("adapter-a", "/tmp/open-rl/checkpoints/a-1", {"restore_optimizer": True})

      self.assertEqual(
        await request_store.latest_checkpoint("adapter-a"),
        {"checkpoint_ref": "/tmp/open-rl/checkpoints/a-1", "metadata": {"restore_optimizer": True}},
      )
      self.assertIsNone(await request_store.latest_checkpoint("adapter-b"))

    asyncio.run(run())

  def test_redis_store_tracks_model_states_by_state_id(self) -> None:
    async def run() -> None:
      request_store = store.RedisStore("redis://unused")
      request_store.redis = FakeRedis()
      state = {
        "state_id": "tinker://adapter-a/sampler_weights/current",
        "model_id": "adapter-a",
        "base_model": "Qwen/Qwen3-0.6B",
        "training_mode": "lora",
        "version": 2,
        "adapter_ref": "/tmp/open-rl/checkpoints/a-2",
      }

      await request_store.publish_model_state(state["state_id"], state)

      self.assertEqual(await request_store.get_model_state(state["state_id"]), state)

    asyncio.run(run())


if __name__ == "__main__":
  unittest.main()
