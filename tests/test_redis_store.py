"""Tests for StateStore (in-memory & Redis), StreamQueue live-Redis integration, and model metadata decoding."""

import asyncio
import os
import shutil
import socket
import subprocess
import time
import unittest

import redis as sync_redis
import redis.asyncio as redis

from server.queue import StreamQueue, train_queue
from server.store import InMemoryStateStore, RedisStateStore

TEST_REDIS_URL = os.getenv("OPEN_RL_TEST_REDIS_URL")
REDIS_SERVER = shutil.which("redis-server")


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


@unittest.skipUnless(TEST_REDIS_URL or REDIS_SERVER, "needs OPEN_RL_TEST_REDIS_URL or redis-server on PATH")
class RedisStoreIntegrationTest(unittest.IsolatedAsyncioTestCase):
  server: subprocess.Popen | None = None
  redis_url: str

  @classmethod
  def setUpClass(cls) -> None:
    if TEST_REDIS_URL:
      cls.redis_url = TEST_REDIS_URL
      return
    port = free_port()
    cls.redis_url = f"redis://127.0.0.1:{port}"
    cls.server = subprocess.Popen(
      ["redis-server", "--port", str(port), "--save", ""],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10
    while True:
      try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
          return
      except OSError:
        if time.monotonic() > deadline:
          raise RuntimeError("redis-server did not come up") from None
        time.sleep(0.05)

  @classmethod
  def tearDownClass(cls) -> None:
    if cls.server is not None:
      cls.server.terminate()
      cls.server.wait(timeout=10)

  def setUp(self) -> None:
    client = redis.from_url(self.redis_url, decode_responses=True)
    self.queue = StreamQueue(client)
    self.state = RedisStateStore(client, sync_redis.Redis.from_url(self.redis_url, decode_responses=True))

  async def asyncSetUp(self) -> None:
    await self.queue.redis.flushdb()

  async def asyncTearDown(self) -> None:
    await self.queue.redis.aclose()
    self.state.sync_redis.close()

  async def test_stream_queue_futures_and_round_robin_on_live_redis(self) -> None:
    await self.queue.set_result("req-1", {"type": "sample", "ok": True})
    self.assertEqual(await self.queue.get_result("req-1", timeout=1.0), {"type": "sample", "ok": True})

    qa, qb = train_queue("tenant-a"), train_queue("tenant-b")
    await self.queue.enqueue(qa, {"model_id": "tenant-a", "request_id": "a0"}, group="base-1")
    await self.queue.enqueue(qb, {"model_id": "tenant-b", "request_id": "b0"}, group="base-1")
    served = []
    for _ in range(2):
      batch = await self.queue.next_batch("base-1", timeout_ms=1000)
      served.append(batch[0]["model_id"])
      await self.queue.complete_record(train_queue(batch[0]["model_id"]), batch[0], {"type": "ok"})
    self.assertEqual(served, ["tenant-a", "tenant-b"])

  async def test_values_expire_sets_and_scan_values(self) -> None:
    await self.state.set_value("k", "v", ttl_seconds=60)
    self.assertEqual(await self.state.get_value("k"), "v")
    await self.state.set_value("k", "v", ttl_seconds=0.2)
    await asyncio.sleep(0.5)
    self.assertIsNone(await self.state.get_value("k"))

    await self.state.add_to_set("s", "a")
    await self.state.add_to_set("s", "b")
    await self.state.remove_from_set("s", "a")
    self.assertEqual(await self.state.set_members("s"), {"b"})
    self.assertEqual(await self.state.set_members("missing"), set())

    await self.state.set_value("open_rl:model_meta:a", "1")
    await self.state.set_value("open_rl:model_meta:b", "2")
    await self.state.set_value("open_rl:session:a", "4")
    self.assertEqual(await self.state.scan_values("open_rl:model_meta:"), {"open_rl:model_meta:a": "1", "open_rl:model_meta:b": "2"})
    self.assertEqual(await self.state.scan_values("open_rl:missing:"), {})


class InMemoryStateStoreTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.state = InMemoryStateStore()

  async def test_values_expire_and_sets_hold_members(self) -> None:
    await self.state.set_value("k", "v", ttl_seconds=60)
    self.assertEqual(await self.state.get_value("k"), "v")
    await self.state.set_value("k", "v", ttl_seconds=0.2)
    await asyncio.sleep(0.5)
    self.assertIsNone(await self.state.get_value("k"))

    await self.state.add_to_set("s", "a")
    await self.state.add_to_set("s", "b")
    await self.state.remove_from_set("s", "a")
    self.assertEqual(await self.state.set_members("s"), {"b"})
    self.assertEqual(await self.state.set_members("missing"), set())

  async def test_scan_values_returns_live_values_under_a_prefix(self) -> None:
    await self.state.set_value("open_rl:model_meta:a", "1")
    await self.state.set_value("open_rl:model_meta:b", "2")
    await self.state.set_value("open_rl:model_meta:gone", "3", ttl_seconds=0.2)
    await self.state.set_value("open_rl:session:a", "4")
    await asyncio.sleep(0.5)
    self.assertEqual(await self.state.scan_values("open_rl:model_meta:"), {"open_rl:model_meta:a": "1", "open_rl:model_meta:b": "2"})
    self.assertEqual(await self.state.scan_values("open_rl:missing:"), {})


class MetadataTest(unittest.TestCase):
  def test_legacy_records_and_corruption_are_distinct(self):
    from server.model_metadata import decode_model_metadata

    self.assertIsNone(decode_model_metadata(None))
    old = decode_model_metadata('{"base_model": null, "fine_tuning_type": "restored", "lora_config": null}')
    self.assertEqual(old.fine_tuning_type, "lora")
    self.assertEqual(old.lora_config.rank, 16)
    for raw in ("bad json", "[]", "{}", '{"base_model": "base", "fine_tuning_type": "invalid"}'):
      with self.subTest(raw=raw), self.assertRaises(ValueError):
        decode_model_metadata(raw)


if __name__ == "__main__":
  unittest.main()
