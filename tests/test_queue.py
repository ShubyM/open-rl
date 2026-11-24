"""Tests for the unified Redis Streams queue (training, round-robin, sampling, and futures)."""

from __future__ import annotations

import asyncio
import time
import unittest
import uuid
from typing import Any
from unittest.mock import patch

from server.queue import (
  PENDING_TTL_SECONDS,
  RESULT_TTL_SECONDS,
  StreamQueue,
  future_key,
  idempotency_key,
  queue_key,
  sample_queue,
  stream_key,
  train_queue,
)
from tests.fake_redis import FakeRedis
from tests.redis_server import RedisTestCase


class _QueueContractMixin:
  """Shared contract executed on FakeRedis (CPU) and live redis-server (when configured)."""

  redis: Any
  queue: StreamQueue
  pool: str

  def sample_record(self, revision: str = "", key: str | None = None, job: str = "job-a") -> dict[str, Any]:
    return {
      "request_id": str(uuid.uuid4()),
      "pool": self.pool,
      "job": job,
      "revision": revision,
      "model": "base",
      "idempotency_key": key,
      "prompt_token_ids": [1, 2, 3],
    }

  async def pending_map(self) -> dict[str, dict[str, Any]]:
    records = await self.queue.pending(sample_queue(self.pool), compact=True)
    return {record["request_id"]: record for record in records}

  async def test_read_batch_crash_recovery_and_cursor_drain(self) -> None:
    queue_name = train_queue(f"model-{uuid.uuid4().hex[:8]}")
    await self.queue.enqueue(queue_name, {"model_id": queue_name, "request_id": "t-1", "op": "optim_step"})
    await self.queue.enqueue(queue_name, {"model_id": queue_name, "op": "shutdown_workers"})

    first = await self.queue.read_batch(queue_name)
    self.assertEqual([record.get("op") for record in first], ["optim_step", "shutdown_workers"])
    self.assertEqual(await self.queue.queued(queue_name), 2)

    # Cursor reads return only entries appended after the current batch.
    last_entry_id = first[-1]["entry_id"]
    self.assertEqual(await self.queue.pending(queue_name, after=last_entry_id), [])
    await self.queue.enqueue(queue_name, {"model_id": queue_name, "request_id": "t-3", "op": "optim_step"})
    drained = await self.queue.pending(queue_name, after=last_entry_id)
    self.assertEqual([record["request_id"] for record in drained], ["t-3"])

    # A restarted worker sees the uncompleted entries and finishes them atomically.
    restarted = await self.queue.read_batch(queue_name)
    self.assertEqual([record["entry_id"] for record in restarted[:2]], [record["entry_id"] for record in first])
    self.assertTrue(await self.queue.complete_record(queue_name, restarted[0], {"type": "optim_step", "ok": True}))
    self.assertFalse(await self.queue.complete_record(queue_name, restarted[0], {"type": "optim_step", "ok": False}))
    self.assertTrue(await self.queue.ack_record(queue_name, restarted[1]))
    self.assertTrue(await self.queue.complete_record(queue_name, restarted[2], {"type": "optim_step", "ok": True}))
    self.assertEqual(await self.queue.queued(queue_name), 0)
    self.assertEqual(await self.queue.get_result("t-1", timeout=1.0), {"type": "optim_step", "ok": True})

  async def test_round_robin_fairness_wake_and_empty_cleanup(self) -> None:
    group = f"base-{uuid.uuid4().hex[:6]}"
    busy_queue = train_queue("busy")
    quiet_queue = train_queue("quiet")
    await self.queue.enqueue(busy_queue, {"model_id": "busy", "request_id": "b0"}, group=group)
    await self.queue.enqueue(quiet_queue, {"model_id": "quiet", "request_id": "q0"}, group=group)

    served = []
    for i in range(4):
      batch = await self.queue.next_batch(group, timeout_ms=500)
      served.append(batch[0]["model_id"])
      for record in batch:
        await self.queue.complete_record(train_queue(record["model_id"]), record, {"type": "ok"})
      await self.queue.enqueue(busy_queue, {"model_id": "busy", "request_id": f"b{i + 1}"}, group=group)
    self.assertEqual(served[:2], ["busy", "quiet"])

    # Drain the remaining busy entry, then verify next_batch cleans empty streams and wakes on a new enqueue.
    for record in await self.queue.next_batch(group, timeout_ms=500):
      await self.queue.complete_record(busy_queue, record, {"type": "ok"})

    waiter = asyncio.create_task(self.queue.next_batch(group, timeout_ms=5000))
    await asyncio.sleep(0.05)
    started = time.monotonic()
    await self.queue.enqueue(quiet_queue, {"model_id": "quiet", "request_id": "q-wake"}, group=group)
    woken = await waiter
    self.assertLess(time.monotonic() - started, 1.0)
    self.assertEqual([record["request_id"] for record in woken], ["q-wake"])
    self.assertFalse(await self.redis.exists(queue_key(busy_queue)))

  async def test_xread_block_future_wakeup_retries_and_timeout(self) -> None:
    waiters = [asyncio.create_task(self.queue.get_result("req-wake", timeout=5.0)) for _ in range(3)]
    await asyncio.sleep(0.05)
    started = time.monotonic()
    await self.queue.set_result("req-wake", {"type": "forward_backward", "loss": 0.5})
    results = await asyncio.gather(*waiters)
    self.assertEqual(results, [{"type": "forward_backward", "loss": 0.5}] * 3)
    self.assertLess(time.monotonic() - started, 1.0)
    self.assertEqual(await self.queue.get_result("req-wake", timeout=0.5), {"type": "forward_backward", "loss": 0.5})

    await self.queue.set_result("req-pend", {"status": "pending"})
    timed_out = await self.queue.get_result("req-pend", timeout=0.05)
    self.assertEqual(timed_out["type"], "try_again")

  async def test_idempotency_dedup_and_ttl_narrowing(self) -> None:
    first = self.sample_record(key="sess:7")
    ids = [await self.queue.accept(first), await self.queue.accept(self.sample_record(key="sess:7"))]
    pending = await self.pending_map()
    self.assertEqual(ids, [first["request_id"], first["request_id"]])
    self.assertEqual(list(pending), [first["request_id"]])

    idem_key = idempotency_key(sample_queue(self.pool), "sess:7")
    self.assertAlmostEqual(await self.redis.ttl(idem_key), PENDING_TTL_SECONDS, delta=60)
    await self.queue.complete_sample(pending[first["request_id"]], {"type": "sample"})
    self.assertEqual(await self.queue.accept(self.sample_record(key="sess:7")), first["request_id"])
    self.assertAlmostEqual(await self.redis.ttl(idem_key), RESULT_TTL_SECONDS, delta=60)

  async def test_concurrent_complete_saves_exactly_one_result(self) -> None:
    record = self.sample_record()
    await self.queue.accept(record)
    [pending] = (await self.pending_map()).values()
    await asyncio.gather(*(self.redis.ping() for _ in range(5)))
    won = await asyncio.gather(*(self.queue.complete_sample(pending, {"type": "sample", "attempt": i}) for i in range(5)))

    self.assertEqual(won.count(True), 1)
    self.assertEqual(len(await self.redis.xrange(future_key(record["request_id"]))), 1)
    self.assertEqual(await self.queue.get_result(record["request_id"], timeout=0.5), {"type": "sample", "attempt": won.index(True)})
    self.assertEqual(await self.pending_map(), {})

  async def test_delete_if_empty_and_sampling_sweep_helpers(self) -> None:
    pool_queue = sample_queue(self.pool)
    r1 = self.sample_record(revision="/ckpt/r1", job="job-a")
    r2 = self.sample_record(revision="/ckpt/r1", job="job-b")
    r3 = self.sample_record(revision="/ckpt/r2", job="job-a")
    for record in (r1, r2, r3):
      await self.queue.accept(record)

    self.assertFalse(await self.queue.delete_if_empty(pool_queue))
    self.assertEqual(await self.queue.pinned_revisions(self.pool), {"/ckpt/r1", "/ckpt/r2"})
    self.assertEqual(await self.queue.pending_jobs(self.pool), {"job-a", "job-b"})

    deleted_result = {"type": "RequestFailedResponse", "error_message": "deleted"}
    self.assertEqual(await self.queue.fail_job(self.pool, "job-a", deleted_result), 2)
    self.assertEqual(await self.queue.pinned_revisions(self.pool), {"/ckpt/r1"})
    self.assertEqual(await self.queue.answered([r1["request_id"], r2["request_id"], r3["request_id"]]), {r1["request_id"], r3["request_id"]})

    expired_result = {"type": "RequestFailedResponse", "error_message": "expired"}
    self.assertEqual(await self.queue.fail_older_than(-60, expired_result), 1)
    self.assertTrue(await self.redis.exists(stream_key(self.pool)))
    self.assertTrue(await self.queue.delete_if_empty(pool_queue))
    self.assertFalse(await self.redis.exists(stream_key(self.pool)))


class StreamQueueCpuTest(_QueueContractMixin, unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.redis = FakeRedis()
    self.queue = StreamQueue(self.redis)
    self.pool = f"sampler-{uuid.uuid4().hex[:8]}"

  async def test_delete_if_empty_aborts_on_concurrent_enqueue(self) -> None:
    queue_name = train_queue("race-model")
    entry_id = await self.redis.xadd(queue_key(queue_name), {"record": '{"request_id":"r0"}'})
    await self.redis.xdel(queue_key(queue_name), entry_id)

    deleted, _ = await asyncio.gather(
      self.queue.delete_if_empty(queue_name),
      self.queue.enqueue(queue_name, {"request_id": "r1"}),
    )
    self.assertFalse(deleted)
    self.assertEqual([record["request_id"] for record in await self.queue.pending(queue_name)], ["r1"])

  async def test_key_layout_compact_prompt_and_telemetry(self) -> None:
    self.assertEqual(queue_key(train_queue("m1")), "open_rl:q:train:m1")
    self.assertEqual(stream_key("p1"), "open_rl:q:sample:p1")
    self.assertEqual(idempotency_key(sample_queue("p1"), "k1"), "open_rl:q:sample:p1:idem:k1")
    self.assertEqual(future_key("req-1"), "open_rl:f:req-1")

    pool_queue = sample_queue(self.pool)
    waiter = asyncio.create_task(self.queue.wait_for_new(pool_queue, after="0-0", timeout_ms=5000))
    await asyncio.sleep(0.05)
    record = self.sample_record(job="job-tel")
    await self.queue.accept(record)
    await waiter

    [compact] = await self.queue.pending(pool_queue, compact=True)
    [plain] = await self.queue.pending(pool_queue, compact=False)
    self.assertEqual(type(compact["prompt_token_ids"]).__name__, "array")
    self.assertEqual(list(compact["prompt_token_ids"]), [1, 2, 3])
    self.assertIsInstance(plain["prompt_token_ids"], list)

    with patch("server.queue.sample_answered") as answered:
      self.assertTrue(await self.queue.complete_sample(compact, {"type": "sample"}, started_at=123.0))
      self.assertFalse(await self.queue.complete_sample(compact, {"type": "sample"}, started_at=124.0))
    answered.assert_called_once_with("job-tel", self.pool, record["request_id"], {"type": "sample"}, 123.0)


class LiveRedisQueueTest(_QueueContractMixin, RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.queue = StreamQueue(self.redis)
    self.pool = f"sampler-{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
  unittest.main()
