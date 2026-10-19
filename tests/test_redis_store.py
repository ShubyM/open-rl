# Integration tests for the Redis-backed stores. They need a real Redis: either
# set OPEN_RL_TEST_REDIS_URL, or have redis-server on PATH (a throwaway instance
# is started on a free port). Skipped entirely otherwise, so `make test` stays
# green on machines without Redis.

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
import unittest
from unittest.mock import patch

import redis as sync_redis
import redis.asyncio as redis

from server.store import InMemoryStateStore, RedisStateStore, RedisStore

TEST_REDIS_URL = os.getenv("OPEN_RL_TEST_REDIS_URL")
REDIS_SERVER = shutil.which("redis-server")


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


class SampleStoreContract:
  async def test_step_revisions_skip_decoding_and_change_for_corrections_and_expiry(self) -> None:
    await self.state.put_steps("history", [{"step": 1, "value": 1}, {"step": 2, "value": 2}], 10, None)
    revision, rows = await self.state.read_steps("history")
    self.assertEqual([row["step"] for row in rows], [1, 2])
    await self.state.put_steps("history", [], 10, None)
    with patch("server.store.json.loads", side_effect=AssertionError("unchanged rows must not be decoded")):
      self.assertEqual(await self.state.read_steps("history", ("older", revision)), (revision, None))
    await self.state.put_steps("history", [{"step": 1, "value": 10}], 10, None)
    corrected, rows = await self.state.read_steps("history", (revision,))
    self.assertNotEqual(corrected, revision)
    self.assertEqual(rows[0]["value"], 10)
    self.assertEqual(json.loads(await self.state.get_value("latest:history"))["step"], 2)
    await self.state.delete_values("revision:history")
    self.assertEqual(await self.state.read_steps("history", (corrected,)), (None, rows))
    await self.state.put_steps("history", [{"step": 2, "value": 20}], 10, 0)
    self.assertEqual(await self.state.read_steps("history", (corrected,)), (None, []))
    await self.state.put_steps("history", [{"step": 1, "value": 100}], 10, None)
    recreated, rows = await self.state.read_steps("history", (revision, corrected))
    self.assertNotIn(recreated, (revision, corrected))
    self.assertEqual(rows, [{"step": 1, "value": 100}])
    await self.state.delete_values("history")
    self.assertEqual(await self.state.read_steps("history", (recreated,)), (None, []))

  async def test_sample_lists_keep_the_newest(self) -> None:
    for step in range(3):
      await self.state.append_sample("samples", {"step": step}, limit=2, ttl_seconds=60)
    self.assertEqual(await self.state.read_samples("samples"), [{"step": 1}, {"step": 2}])
    await self.state.delete_values("samples")
    self.assertEqual(await self.state.read_samples("samples"), [])

  async def test_prefix_scan_and_atomic_counter(self) -> None:
    await self.state.set_value("model:live", "metadata")
    await self.state.set_value("model:expired", "old", ttl_seconds=0.01)
    await self.state.set_value("other", "unrelated")
    counts = await asyncio.gather(*(self.state.increment("steps", ttl_seconds=60) for _ in range(40)))
    await asyncio.sleep(0.02)
    self.assertEqual(await self.state.scan_values("model:"), {"model:live": "metadata"})
    self.assertEqual(sorted(counts), list(range(1, 41)))


@unittest.skipUnless(TEST_REDIS_URL or REDIS_SERVER, "needs OPEN_RL_TEST_REDIS_URL or redis-server on PATH")
class RedisFutureTest(SampleStoreContract, unittest.IsolatedAsyncioTestCase):
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
    self.store = RedisStore(redis.from_url(self.redis_url, decode_responses=True))
    self.state = RedisStateStore(self.store.redis, sync_redis.Redis.from_url(self.redis_url, decode_responses=True))

  async def asyncSetUp(self) -> None:
    await self.store.redis.flushdb()

  async def test_samples_with_no_ttl_stop_expiring(self) -> None:
    await self.state.append_sample("ops", {"n": 1}, 10, 60)
    self.assertGreater(await self.store.redis.ttl("ops"), 0)
    await self.state.append_sample("ops", {"n": 2}, 10, None)
    self.assertEqual(await self.store.redis.ttl("ops"), -1)
    self.assertEqual([s["n"] for s in await self.state.read_samples("ops")], [1, 2])

  async def test_concurrent_replayed_batches_retain_distinct_steps_and_latest_precision(self) -> None:
    await self.state.put_steps("steps", [{"step": step, "value": step} for step in range(4)], 4, 60)
    latest = {"step": 2**53 - 1, "value": 0.12345678901234567}
    await asyncio.gather(
      *(self.state.put_steps("steps", rows, 4, 60) for rows in ([{"step": 3, "value": 30}] * 10, [latest], [{"step": 0, "value": 0}]))
    )
    _, rows = await self.state.read_steps("steps")
    self.assertEqual([row["step"] for row in rows], [1, 2, 3, latest["step"]])
    self.assertEqual(json.loads(await self.state.get_value("latest:steps")), latest)

  async def test_thousand_row_batches_and_matching_history_export_expiry(self) -> None:
    await self.state.put_steps("batch", [{"step": step} for step in range(1000)], 2000, None)
    _, rows = await self.state.read_steps("batch")
    self.assertEqual(len(rows), 1000)
    for key in ("batch", "latest:batch", "revision:batch"):
      self.assertEqual(await self.store.redis.ttl(key), -1)
    for ttl in (0, 0.0001, 0.01):
      await self.state.put_steps("batch", [{"step": 1}], 10, ttl)
      await asyncio.sleep(0.02)
      self.assertEqual(await self.state.read_steps("batch"), (None, []))
      self.assertIsNone(await self.state.get_value("latest:batch"))
      self.assertIsNone(await self.state.get_value("revision:batch"))

  async def asyncTearDown(self) -> None:
    await self.store.redis.aclose()
    self.state.sync_redis.close()

  async def test_get_future_returns_already_resolved_result(self) -> None:
    await self.store.set_future("req-1", {"type": "sample", "ok": True})
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample", "ok": True})

  async def test_get_future_wakes_on_resolution(self) -> None:
    async def resolve_later() -> None:
      await asyncio.sleep(0.2)
      await self.store.set_future("req-1", {"type": "sample"})

    resolver = asyncio.create_task(resolve_later())
    started = time.monotonic()
    result = await self.store.get_future("req-1", timeout=10.0)
    await resolver

    self.assertEqual(result, {"type": "sample"})
    # Woken by the publish, not by grinding through the whole long-poll window.
    self.assertLess(time.monotonic() - started, 5.0)

  async def test_result_survives_repeated_and_concurrent_reads(self) -> None:
    waiters = [asyncio.create_task(self.store.get_future("req-1", timeout=10.0)) for _ in range(5)]
    await asyncio.sleep(0.2)
    await self.store.set_future("req-1", {"type": "sample"})

    for result in await asyncio.gather(*waiters):
      self.assertEqual(result, {"type": "sample"})
    self.assertEqual(await self.store.get_future("req-1", timeout=1.0), {"type": "sample"})

  async def test_unresolved_future_times_out_with_try_again(self) -> None:
    result = await self.store.get_future("req-never", timeout=0.3)
    self.assertEqual(result["type"], "try_again")

  async def test_pending_markers_are_not_stored(self) -> None:
    await self.store.set_future("req-1", {"status": "pending"})
    self.assertEqual((await self.store.get_future("req-1", timeout=0.3))["type"], "try_again")

  async def test_get_requests_rotates_between_tenants(self) -> None:
    for i in range(3):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")
      await self.store.put_request({"model_id": "tenant-b", "request_id": f"b{i}"}, active_set_id="base-1")

    served = []
    for _ in range(2):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])

    self.assertEqual(served, ["tenant-a", "tenant-b"])

  async def test_busy_tenant_does_not_starve_its_peer(self) -> None:
    await self.store.put_request({"model_id": "busy", "request_id": "b0"}, active_set_id="base-1")
    await self.store.put_request({"model_id": "quiet", "request_id": "q0"}, active_set_id="base-1")

    # The busy tenant keeps enqueueing after each turn. Without rotation it holds
    # the head of the active list forever and 'quiet' is never served.
    served = []
    for i in range(4):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])
      await self.store.put_request({"model_id": "busy", "request_id": f"b{i + 1}"}, active_set_id="base-1")

    self.assertIn("quiet", served)

  async def test_get_requests_drains_only_the_depth_present_on_entry(self) -> None:
    for i in range(2):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")

    batch = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual([r["request_id"] for r in batch], ["a0", "a1"])

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


class InMemoryStoreTest(SampleStoreContract, unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    from server.store import InMemoryStore

    self.store = InMemoryStore()
    self.state = InMemoryStateStore()

  async def test_get_requests_rotates_between_tenants(self) -> None:
    for i in range(3):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")
      await self.store.put_request({"model_id": "tenant-b", "request_id": f"b{i}"}, active_set_id="base-1")

    served = []
    for _ in range(2):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])

    self.assertEqual(served, ["tenant-a", "tenant-b"])

  async def test_busy_tenant_does_not_starve_its_peer(self) -> None:
    await self.store.put_request({"model_id": "busy", "request_id": "b0"}, active_set_id="base-1")
    await self.store.put_request({"model_id": "quiet", "request_id": "q0"}, active_set_id="base-1")

    served = []
    for i in range(4):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])
      await self.store.put_request({"model_id": "busy", "request_id": f"b{i + 1}"}, active_set_id="base-1")

    self.assertIn("quiet", served)

  async def test_get_requests_drains_only_the_depth_present_on_entry(self) -> None:
    for i in range(2):
      await self.store.put_request({"model_id": "tenant-a", "request_id": f"a{i}"}, active_set_id="base-1")

    batch = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual([r["request_id"] for r in batch], ["a0", "a1"])

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


if __name__ == "__main__":
  unittest.main()


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
