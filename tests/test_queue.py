"""Tests for the unified pure-Streams StreamQueue (training, round-robin, sampling, and futures)."""

from __future__ import annotations

import asyncio
import os
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
  get_queue,
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
    return {r["request_id"]: r for r in await self.queue.pending(sample_queue(self.pool), compact=True)}

  async def test_read_batch_crash_recovery_and_cursor_drain(self) -> None:
    q = train_queue(f"model-{uuid.uuid4().hex[:8]}")
    await self.queue.enqueue(q, {"model_id": q, "request_id": "t-1", "op": "optim_step"})
    await self.queue.enqueue(q, {"model_id": q, "op": "shutdown_workers"})

    first = await self.queue.read_batch(q)
    self.assertEqual([r.get("op") for r in first], ["optim_step", "shutdown_workers"])
    self.assertEqual(await self.queue.queued(q), 2)

    # Cursor read returns only entries appended after the current batch.
    self.assertEqual(await self.queue.pending(q, after=first[-1]["entry_id"]), [])
    await self.queue.enqueue(q, {"model_id": q, "request_id": "t-3", "op": "optim_step"})
    self.assertEqual([r["request_id"] for r in await self.queue.pending(q, after=first[-1]["entry_id"])], ["t-3"])

    # Restarted worker sees the exact same uncompleted entries and finishes them atomically.
    restarted = await self.queue.read_batch(q)
    self.assertEqual([r["entry_id"] for r in restarted[:2]], [r["entry_id"] for r in first])
    self.assertTrue(await self.queue.complete_record(q, restarted[0], {"type": "optim_step", "ok": True}))
    self.assertFalse(await self.queue.complete_record(q, restarted[0], {"type": "optim_step", "ok": False}))
    self.assertTrue(await self.queue.ack_record(q, restarted[1]))
    self.assertTrue(await self.queue.complete_record(q, restarted[2], {"type": "optim_step", "ok": True}))
    self.assertEqual(await self.queue.queued(q), 0)
    self.assertEqual(await self.queue.get_result("t-1", timeout=1.0), {"type": "optim_step", "ok": True})

  async def test_round_robin_fairness_wake_and_empty_cleanup(self) -> None:
    group = f"base-{uuid.uuid4().hex[:6]}"
    qa, qb = train_queue("busy"), train_queue("quiet")
    await self.queue.enqueue(qa, {"model_id": "busy", "request_id": "b0"}, group=group)
    await self.queue.enqueue(qb, {"model_id": "quiet", "request_id": "q0"}, group=group)

    served = []
    for i in range(4):
      batch = await self.queue.next_batch(group, timeout_ms=500)
      served.append(batch[0]["model_id"])
      for rec in batch:
        await self.queue.complete_record(train_queue(rec["model_id"]), rec, {"type": "ok"})
      await self.queue.enqueue(qa, {"model_id": "busy", "request_id": f"b{i + 1}"}, group=group)
    self.assertEqual(served[:2], ["busy", "quiet"])

    # Drain remaining busy entry, then verify next_batch cleans empty streams and wakes on new enqueue.
    for rec in await self.queue.next_batch(group, timeout_ms=500):
      await self.queue.complete_record(qa, rec, {"type": "ok"})

    waiter = asyncio.create_task(self.queue.next_batch(group, timeout_ms=5000))
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await self.queue.enqueue(qb, {"model_id": "quiet", "request_id": "q-wake"}, group=group)
    woken = await waiter
    self.assertLess(time.monotonic() - t0, 1.0)
    self.assertEqual([r["request_id"] for r in woken], ["q-wake"])
    self.assertFalse(await self.redis.exists(queue_key(qa)))

  async def test_xread_block_future_wakeup_retries_and_timeout(self) -> None:
    waiters = [asyncio.create_task(self.queue.get_result("req-wake", timeout=5.0)) for _ in range(3)]
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await self.queue.set_result("req-wake", {"type": "forward_backward", "loss": 0.5})
    self.assertEqual(await asyncio.gather(*waiters), [{"type": "forward_backward", "loss": 0.5}] * 3)
    self.assertLess(time.monotonic() - t0, 1.0)
    self.assertEqual(await self.queue.get_result("req-wake", timeout=0.5), {"type": "forward_backward", "loss": 0.5})
    await self.queue.set_result("req-pend", {"status": "pending"})
    self.assertEqual((await self.queue.get_result("req-pend", timeout=0.05))["type"], "try_again")

  async def test_idempotency_dedup_and_ttl_narrowing(self) -> None:
    first = self.sample_record(key="sess:7")
    ids = [await self.queue.accept(first), await self.queue.accept(self.sample_record(key="sess:7"))]
    pending = await self.pending_map()
    self.assertEqual((ids, list(pending)), ([first["request_id"]] * 2, [first["request_id"]]))
    ikey = idempotency_key(sample_queue(self.pool), "sess:7")
    self.assertAlmostEqual(await self.redis.ttl(ikey), PENDING_TTL_SECONDS, delta=60)
    await self.queue.complete_sample(pending[first["request_id"]], {"type": "sample"})
    self.assertEqual(await self.queue.accept(self.sample_record(key="sess:7")), first["request_id"])
    self.assertAlmostEqual(await self.redis.ttl(ikey), RESULT_TTL_SECONDS, delta=60)

  async def test_concurrent_complete_saves_exactly_one_result(self) -> None:
    r = self.sample_record()
    await self.queue.accept(r)
    [pending] = (await self.pending_map()).values()
    await asyncio.gather(*(self.redis.ping() for _ in range(5)))
    won = await asyncio.gather(*(self.queue.complete_sample(pending, {"type": "sample", "attempt": i}) for i in range(5)))
    self.assertEqual((won.count(True), len(await self.redis.xrange(future_key(r["request_id"])))), (1, 1))
    self.assertEqual(await self.queue.get_result(r["request_id"], timeout=0.5), {"type": "sample", "attempt": won.index(True)})
    self.assertEqual(await self.pending_map(), {})

  async def test_delete_if_empty_and_sampling_sweep_helpers(self) -> None:
    q = sample_queue(self.pool)
    r1 = self.sample_record(revision="/ckpt/r1", job="job-a")
    r2 = self.sample_record(revision="/ckpt/r1", job="job-b")
    r3 = self.sample_record(revision="/ckpt/r2", job="job-a")
    for r in (r1, r2, r3):
      await self.queue.accept(r)
    self.assertFalse(await self.queue.delete_if_empty(q))
    self.assertEqual(await self.queue.pinned_revisions(self.pool), {"/ckpt/r1", "/ckpt/r2"})
    self.assertEqual(await self.queue.pending_jobs(self.pool), {"job-a", "job-b"})
    self.assertEqual(await self.queue.fail_job(self.pool, "job-a", {"type": "RequestFailedResponse", "error_message": "deleted"}), 2)
    self.assertEqual(await self.queue.pinned_revisions(self.pool), {"/ckpt/r1"})
    self.assertEqual(await self.queue.answered([r1["request_id"], r2["request_id"], r3["request_id"]]), {r1["request_id"], r3["request_id"]})
    self.assertEqual(await self.queue.fail_older_than(-60, {"type": "RequestFailedResponse", "error_message": "expired"}), 1)
    self.assertTrue(await self.redis.exists(stream_key(self.pool)))
    self.assertTrue(await self.queue.delete_if_empty(q))
    self.assertFalse(await self.redis.exists(stream_key(self.pool)))


class StreamQueueCpuTest(_QueueContractMixin, unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.redis = FakeRedis()
    self.queue = StreamQueue(self.redis)
    self.pool = f"sampler-{uuid.uuid4().hex[:8]}"

  async def test_delete_if_empty_aborts_on_concurrent_enqueue(self) -> None:
    q = train_queue("race-model")
    eid = await self.redis.xadd(queue_key(q), {"record": '{"request_id":"r0"}'})
    await self.redis.xdel(queue_key(q), eid)
    deleted, _ = await asyncio.gather(self.queue.delete_if_empty(q), self.queue.enqueue(q, {"request_id": "r1"}))
    self.assertFalse(deleted)
    self.assertEqual([r["request_id"] for r in await self.queue.pending(q)], ["r1"])

  async def test_key_layout_compact_prompt_telemetry_and_get_queue(self) -> None:
    self.assertEqual(queue_key(train_queue("m1")), "open_rl:q:train:m1")
    self.assertEqual(stream_key("p1"), "open_rl:q:sample:p1")
    self.assertEqual(idempotency_key(sample_queue("p1"), "k1"), "open_rl:q:sample:p1:idem:k1")
    self.assertEqual(future_key("req-1"), "open_rl:f:req-1")

    q = sample_queue(self.pool)
    waiter = asyncio.create_task(self.queue.wait_for_new(q, after="0-0", timeout_ms=5000))
    await asyncio.sleep(0.05)
    r = self.sample_record(job="job-tel")
    await self.queue.accept(r)
    await waiter
    [compact] = await self.queue.pending(q, compact=True)
    [plain] = await self.queue.pending(q, compact=False)
    self.assertEqual((type(compact["prompt_token_ids"]).__name__, list(compact["prompt_token_ids"])), ("array", [1, 2, 3]))
    self.assertIsInstance(plain["prompt_token_ids"], list)

    with patch("server.queue.sample_answered") as answered:
      self.assertTrue(await self.queue.complete_sample(compact, {"type": "sample"}, started_at=123.0))
      self.assertFalse(await self.queue.complete_sample(compact, {"type": "sample"}, started_at=124.0))
    answered.assert_called_once_with("job-tel", self.pool, r["request_id"], {"type": "sample"}, 123.0)

    with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL is required"):
      get_queue()


class LiveRedisQueueTest(_QueueContractMixin, RedisTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.queue = StreamQueue(self.redis)
    self.pool = f"sampler-{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
  unittest.main()
