# Integration tests for the Redis-backed stores. They need a real Redis: either
# set OPEN_RL_TEST_REDIS_URL, or have redis-server on PATH (a throwaway instance
# is started on a free port). Skipped entirely otherwise, so `make test` stays
# green on machines without Redis.

import asyncio
import os
import shutil
import socket
import subprocess
import time
import unittest
from unittest import mock

import redis as sync_redis
import redis.asyncio as redis

from server import telemetry
from server.dashboard import facts, fleet
from server.store import InMemoryStateStore, RedisStateStore, RedisStore

TEST_REDIS_URL = os.getenv("OPEN_RL_TEST_REDIS_URL")
REDIS_SERVER = shutil.which("redis-server")


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def workload(name: str, placed_at: str | None = "1970-01-01T00:01:40Z", released_at: str | None = None) -> dict:
  status = {"podName": f"orw-{name}", "nodeName": "n1", "gpuUUIDs": ["GPU-1"], "placedAt": placed_at, "releasedAt": released_at}
  return {"metadata": {"name": name}, "spec": {"role": "trainer", "modelID": "m1", "ownerID": "m1"}, "status": status}


class SampleStoreContract:
  async def test_placements_are_upserted_released_and_rewritten_only_on_change(self) -> None:
    async def sweep(*workloads, now=1000.0):
      with mock.patch.object(telemetry.time, "time", return_value=now):
        await telemetry.record_placements(self.state, list(workloads))
      return await self.state.read_samples(telemetry.key("placements", "m1"))

    self.assertEqual(await sweep(workload("unplaced", placed_at=None)), [])
    (placed,) = await sweep(workload("a"))
    self.assertEqual(
      placed,
      {"workload": "a", "role": "trainer", "model_id": "m1", "owner": "m1", "pod": "orw-a", "node": "n1", "gpu_uuids": ["GPU-1"]}
      | {"placed_at": 100.0, "released_at": None},
    )
    self.assertEqual(await self.state.set_members(telemetry.OPEN_PLACEMENTS), {"m1"})
    with mock.patch.object(self.state, "replace_samples", wraps=self.state.replace_samples) as rewrite:
      self.assertEqual(await sweep(workload("a")), [placed])
    rewrite.assert_not_awaited()

    released, moved = await sweep(workload("a", released_at="1970-01-01T00:02:30Z"), workload("b"))
    self.assertEqual((released["released_at"], moved["workload"], moved["released_at"]), (150.0, "b", None))
    # A Workload placed again starts a new record and releases the old one; one that is gone is released at the sweep.
    records = await sweep(workload("b", placed_at="1970-01-01T00:03:00Z"), now=2000.0)
    self.assertEqual([(r["workload"], r["placed_at"], r["released_at"]) for r in records], [("a", 100, 150), ("b", 100, 2000), ("b", 180, None)])
    records = await sweep(now=3000.0)
    self.assertEqual([r["released_at"] for r in records], [150, 2000, 3000])
    self.assertEqual(await self.state.set_members(telemetry.OPEN_PLACEMENTS), set())
    self.assertEqual(await sweep(now=4000.0), records)

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

  async def asyncTearDown(self) -> None:
    await self.store.redis.aclose()
    self.state.sync_redis.close()

  async def test_a_runs_request_counters_expire_together(self) -> None:
    async def count(event: str) -> None:
      telemetry.count(event, "r")
      await asyncio.gather(*list(telemetry.writes))

    run = {"run_id": "r", "runtime_id": "r", "lifecycle": "ended", "created_at": 0}
    with mock.patch.object(telemetry, "TTL_SECONDS", 1), mock.patch.object(telemetry, "get_state_store", return_value=self.state):
      await count("enqueued")  # the API server
      await asyncio.sleep(0.6)
      await count("popped")  # the trainer, later; the run's counters must not lose the API server's
    await asyncio.sleep(0.6)
    fleet.cache.clear()
    with mock.patch.object(facts, "get_state_store", return_value=self.state), mock.patch.object(fleet, "get_state_store", return_value=self.state):
      ctx = await facts.context({"observed_at": time.time(), "workers": [], "placements": [], "runs": [run]}, run)
    self.assertEqual(ctx["ledger"], {"enqueued": 1, "popped": 1})

  async def test_counters_recorded_before_the_hash_still_read(self) -> None:
    for event, value in (("enqueued", "3"), ("popped", "2")):
      await self.state.set_value(telemetry.key("ledger", "old", event), value)
    run = {"run_id": "old", "runtime_id": "old", "lifecycle": "ended", "created_at": 0}
    fleet.cache.clear()
    with mock.patch.object(facts, "get_state_store", return_value=self.state), mock.patch.object(fleet, "get_state_store", return_value=self.state):
      ctx = await facts.context({"observed_at": time.time(), "workers": [], "placements": [], "runs": [run]}, run)
    self.assertEqual(ctx["ledger"], {"enqueued": 3, "popped": 2})

  async def test_get_result_returns_already_resolved_result(self) -> None:
    await self.store.set_result("req-1", {"type": "sample", "ok": True})
    self.assertEqual(await self.store.get_result("req-1", timeout=1.0), {"type": "sample", "ok": True})

  async def test_get_result_wakes_on_resolution(self) -> None:
    async def resolve_later() -> None:
      await asyncio.sleep(0.2)
      await self.store.set_result("req-1", {"type": "sample"})

    resolver = asyncio.create_task(resolve_later())
    started = time.monotonic()
    result = await self.store.get_result("req-1", timeout=10.0)
    await resolver

    self.assertEqual(result, {"type": "sample"})
    # Woken by the publish, not by grinding through the whole long-poll window.
    self.assertLess(time.monotonic() - started, 5.0)

  async def test_result_survives_repeated_and_concurrent_reads(self) -> None:
    waiters = [asyncio.create_task(self.store.get_result("req-1", timeout=10.0)) for _ in range(5)]
    await asyncio.sleep(0.2)
    await self.store.set_result("req-1", {"type": "sample"})

    for result in await asyncio.gather(*waiters):
      self.assertEqual(result, {"type": "sample"})
    self.assertEqual(await self.store.get_result("req-1", timeout=1.0), {"type": "sample"})

  async def test_unresolved_future_times_out_with_try_again(self) -> None:
    result = await self.store.get_result("req-never", timeout=0.3)
    self.assertEqual(result["type"], "try_again")

  async def test_pending_markers_are_not_stored(self) -> None:
    await self.store.set_result("req-1", {"status": "pending"})
    self.assertEqual((await self.store.get_result("req-1", timeout=0.3))["type"], "try_again")

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

  async def test_sampling_queue_put_and_get(self) -> None:
    req1 = {"model_id": "base-m1", "request_id": "r1"}
    req2 = {"model_id": "base-m1", "request_id": "r2"}
    await self.store.put_sampling_request(req1)
    await self.store.put_sampling_request(req2)

    batch = await self.store.get_sampling_requests_for_model("base-m1")
    self.assertEqual(len(batch), 2)
    self.assertEqual(batch[0]["request_id"], "r1")
    self.assertEqual(batch[1]["request_id"], "r2")

    empty_batch = await self.store.get_sampling_requests_for_model("base-m1")
    self.assertEqual(empty_batch, [])

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
