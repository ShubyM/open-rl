"""Tests for the unified Redis Streams queue and XREAD BLOCK future layer."""

from __future__ import annotations

import asyncio
import fnmatch
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
from server.store import RedisStore
from tests.redis_server import RedisTestCase

_INF = (1 << 62, 1 << 62)


def _parse_id(raw: str, *, is_max: bool = False) -> tuple[tuple[int, int], bool]:
  if raw == "-":
    return (0, 0), False
  if raw == "+":
    return _INF, False
  exclusive = raw.startswith("(")
  s = raw[1:] if exclusive else raw
  if "-" in s:
    ms, seq = s.split("-", 1)
    return (int(ms), int(seq)), exclusive
  return (int(s), _INF[1] if is_max else 0), exclusive


class _FakePipeline:
  """Supports both `async with redis.pipeline()` and `redis.transaction(fn, *watches)`."""

  def __init__(self, db: FakeRedis, *, tx_mode: bool = False):
    self._db, self._tx_mode = db, tx_mode
    self._in_multi = not tx_mode
    self._ops: list[tuple[str, tuple, dict]] = []

  async def __aenter__(self) -> _FakePipeline:
    return self

  async def __aexit__(self, *args) -> None:
    pass

  def multi(self) -> None:
    self._in_multi = True

  async def execute(self) -> list[Any]:
    out = [getattr(self._db, f"_sync_{name}")(*args, **kwargs) for name, args, kwargs in self._ops]
    self._ops.clear()
    return out

  def __getattr__(self, name: str):
    if self._in_multi:

      def _queue(*args, **kwargs):
        self._ops.append((name, args, kwargs))
        return self

      return _queue
    return getattr(self._db, name)


class FakeRedis:
  """In-memory async Redis Streams + WATCH/MULTI harness for CPU unit tests without redis-server."""

  def __init__(self):
    self.strings: dict[str, str] = {}
    self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
    self.lists: dict[str, list[str]] = {}
    self.sets: dict[str, set[str]] = {}
    self.expiries: dict[str, float] = {}
    self._versions: dict[str, int] = {}
    self._events: dict[str, asyncio.Event] = {}
    self._last_ms = 0
    self._last_seq = 0

  def _touch(self, key: str) -> None:
    self._versions[key] = self._versions.get(key, 0) + 1

  def _notify(self, key: str) -> None:
    if ev := self._events.pop(key, None):
      ev.set()

  def _evict_if_expired(self, key: str) -> None:
    if key in self.expiries and time.monotonic() >= self.expiries[key]:
      for store in (self.strings, self.streams, self.lists, self.sets, self.expiries):
        store.pop(key, None)

  def _exists(self, key: str) -> bool:
    self._evict_if_expired(key)
    return key in self.strings or key in self.streams or key in self.lists or key in self.sets

  def _next_id(self) -> str:
    ms = int(time.time() * 1000)
    if ms <= self._last_ms:
      self._last_seq += 1
    else:
      self._last_ms, self._last_seq = ms, 0
    return f"{self._last_ms}-{self._last_seq}"

  def _sync_set(self, key: str, value: str, ex: float | None = None, px: int | None = None) -> bool:
    self.strings[key] = value
    self.expiries.pop(key, None)
    if ex is not None:
      self.expiries[key] = time.monotonic() + ex
    elif px is not None:
      self.expiries[key] = time.monotonic() + px / 1000.0
    self._touch(key)
    return True

  def _sync_get(self, key: str) -> str | None:
    self._evict_if_expired(key)
    return self.strings.get(key)

  def _sync_delete(self, *keys: str) -> int:
    deleted = 0
    for key in keys:
      existed = self._exists(key)
      for store in (self.strings, self.streams, self.lists, self.sets, self.expiries):
        store.pop(key, None)
      if existed:
        deleted += 1
        self._touch(key)
    return deleted

  def _sync_exists(self, key: str) -> int:
    return 1 if self._exists(key) else 0

  def _sync_expire(self, key: str, seconds: float) -> bool:
    if not self._exists(key):
      return False
    self.expiries[key] = time.monotonic() + seconds
    self._touch(key)
    return True

  def _sync_ttl(self, key: str) -> int:
    if not self._exists(key):
      return -2
    if key not in self.expiries:
      return -1
    return max(0, int(round(self.expiries[key] - time.monotonic())))

  def _sync_xadd(self, key: str, fields: dict[str, str], id: str = "*", maxlen: int | None = None) -> str:
    self._evict_if_expired(key)
    eid = self._next_id() if id == "*" else id
    stream = self.streams.setdefault(key, [])
    stream.append((eid, dict(fields)))
    if maxlen is not None and len(stream) > maxlen:
      del stream[:-maxlen]
    self._touch(key)
    self._notify(key)
    return eid

  def _sync_xrange(self, key: str, min: str = "-", max: str = "+", count: int | None = None) -> list[tuple[str, dict[str, str]]]:
    self._evict_if_expired(key)
    lo, lo_ex = _parse_id(min, is_max=False)
    hi, hi_ex = _parse_id(max, is_max=True)
    out = []
    for eid, fields in self.streams.get(key, ()):
      k, _ = _parse_id(eid)
      if (k <= lo if lo_ex else k < lo) or (k >= hi if hi_ex else k > hi):
        continue
      out.append((eid, dict(fields)))
      if count is not None and len(out) >= count:
        break
    return out

  def _sync_xdel(self, key: str, *ids: str) -> int:
    self._evict_if_expired(key)
    if key not in self.streams:
      return 0
    drop = set(ids)
    before = len(self.streams[key])
    self.streams[key] = [(eid, f) for eid, f in self.streams[key] if eid not in drop]
    removed = before - len(self.streams[key])
    if removed:
      self._touch(key)
    return removed

  def _sync_xlen(self, key: str) -> int:
    self._evict_if_expired(key)
    return len(self.streams.get(key, ()))

  def _sync_sadd(self, key: str, *members: str) -> int:
    s = self.sets.setdefault(key, set())
    before = len(s)
    s.update(members)
    if added := len(s) - before:
      self._touch(key)
    return added

  def _sync_srem(self, key: str, *members: str) -> int:
    s = self.sets.get(key)
    if not s:
      return 0
    before = len(s)
    s.difference_update(members)
    if removed := before - len(s):
      self._touch(key)
    return removed

  def _sync_smembers(self, key: str) -> set[str]:
    return set(self.sets.get(key, ()))

  def _sync_rpush(self, key: str, *values: str) -> int:
    lst = self.lists.setdefault(key, [])
    lst.extend(values)
    self._touch(key)
    self._notify(key)
    return len(lst)

  def _sync_lpush(self, key: str, *values: str) -> int:
    lst = self.lists.setdefault(key, [])
    for v in values:
      lst.insert(0, v)
    self._touch(key)
    self._notify(key)
    return len(lst)

  def _sync_lpop(self, key: str) -> str | None:
    lst = self.lists.get(key)
    if not lst:
      return None
    val = lst.pop(0)
    self._touch(key)
    return val

  def _sync_lindex(self, key: str, index: int) -> str | None:
    lst = self.lists.get(key, [])
    try:
      return lst[index]
    except IndexError:
      return None

  def _sync_lrange(self, key: str, start: int, end: int) -> list[str]:
    lst = self.lists.get(key, [])
    stop = None if end == -1 else (len(lst) + end + 1 if end < 0 else end + 1)
    return lst[start:stop]

  def _sync_lrem(self, key: str, count: int, value: str) -> int:
    lst = self.lists.get(key)
    if not lst:
      return 0
    before = len(lst)
    if count == 0:
      self.lists[key] = [x for x in lst if x != value]
    else:
      rem, out = abs(count), []
      for x in lst:
        if x == value and rem > 0:
          rem -= 1
        else:
          out.append(x)
      self.lists[key] = out
    removed = before - len(self.lists[key])
    if removed:
      self._touch(key)
    return removed

  def _sync_lmove(self, src: str, dst: str, wherefrom: str, whereto: str) -> str | None:
    slst = self.lists.get(src)
    if not slst:
      return None
    item = slst.pop(0) if wherefrom.upper() == "LEFT" else slst.pop()
    dlst = self.lists.setdefault(dst, [])
    if whereto.upper() == "LEFT":
      dlst.insert(0, item)
    else:
      dlst.append(item)
    self._touch(src)
    self._touch(dst)
    self._notify(dst)
    return item

  def __getattr__(self, name: str):
    if name.startswith("_") or (fn := getattr(self, f"_sync_{name}", None)) is None:
      raise AttributeError(name)

    async def _call(*args, **kwargs):
      return fn(*args, **kwargs)

    return _call

  async def xread(self, streams: dict[str, str], count: int | None = None, block: int | None = None) -> list[list[Any]]:
    deadline = None if block in (None, 0) else time.monotonic() + block / 1000.0
    while True:
      found = []
      for key, after in streams.items():
        if items := self._sync_xrange(key, min=f"({after}", count=count):
          found.append([key, items])
      if found or block is None:
        return found
      remaining = None if deadline is None else deadline - time.monotonic()
      if remaining is not None and remaining <= 0:
        return []
      events = [self._events.setdefault(k, asyncio.Event()) for k in streams]
      waits = [asyncio.create_task(ev.wait()) for ev in events]
      try:
        done, pending = await asyncio.wait(waits, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
          task.cancel()
        if not done:
          return []
      except BaseException:
        for task in waits:
          task.cancel()
        raise

  async def scan_iter(self, match: str = "*", _type: str | None = None, count: int | None = None):
    pools: dict[str, set[str]] = {"stream": set(self.streams), "string": set(self.strings), "list": set(self.lists), "set": set(self.sets)}
    keys = pools[_type] if _type in pools else set().union(*pools.values())
    for key in list(keys):
      if self._exists(key) and fnmatch.fnmatchcase(key, match):
        yield key

  async def brpoplpush(self, src: str, dst: str, timeout: float = 0) -> str | None:
    deadline = None if timeout == 0 else time.monotonic() + timeout
    while True:
      if (item := self._sync_lmove(src, dst, "RIGHT", "LEFT")) is not None:
        return item
      remaining = None if deadline is None else deadline - time.monotonic()
      if remaining is not None and remaining <= 0:
        return None
      ev = self._events.setdefault(src, asyncio.Event())
      try:
        await asyncio.wait_for(ev.wait(), timeout=remaining)
      except TimeoutError:
        return None

  def pipeline(self, transaction: bool = True) -> _FakePipeline:
    return _FakePipeline(self, tx_mode=False)

  async def transaction(self, func, *watches: str, value_from_callable: bool = False):
    while True:
      snap = {k: self._versions.get(k, 0) for k in watches}
      pipe = _FakePipeline(self, tx_mode=True)
      result = await func(pipe)
      if not pipe._in_multi:
        return result if value_from_callable else []
      # Yield once so overlapping coroutines actually contend on WATCHed keys.
      await asyncio.sleep(0)
      if any(self._versions.get(k, 0) != v for k, v in snap.items()):
        continue
      exec_res = await pipe.execute()
      return result if value_from_callable else exec_res

  async def ping(self) -> bool:
    return True

  async def flushdb(self) -> None:
    for d in (self.strings, self.streams, self.lists, self.sets, self.expiries, self._versions):
      d.clear()

  async def aclose(self) -> None:
    pass


class _QueueContractMixin:
  """Shared assertions run against FakeRedis (always on CPU) and real Redis (when available)."""

  redis: Any
  queue: StreamQueue
  store: RedisStore
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

  async def test_training_non_destructive_read_survives_crash_until_complete(self) -> None:
    model_id = f"model-{uuid.uuid4().hex[:8]}"
    await self.store.put_request({"model_id": model_id, "request_id": "t-1", "op": "optim_step"})
    await self.store.put_request({"model_id": model_id, "request_id": "t-2", "op": "shutdown_workers"})

    # Worker 1 reads batch and crashes before completing t-1.
    first_read = await self.store.get_requests_for_model(model_id)
    self.assertEqual([r["request_id"] for r in first_read], ["t-1", "t-2"])
    self.assertEqual(await self.store.queued(model_id), 2)

    # Worker 2 restarts, sees the same pending entries, completes t-1 and acks t-2.
    second_read = await self.store.get_requests_for_model(model_id)
    self.assertEqual([r["entry_id"] for r in second_read], [r["entry_id"] for r in first_read])
    self.assertTrue(await self.store.complete_request(second_read[0], {"type": "optim_step", "ok": True}))
    self.assertFalse(await self.store.complete_request(second_read[0], {"type": "optim_step", "ok": False}))
    self.assertTrue(await self.store.ack_request(second_read[1]))
    self.assertEqual(await self.store.queued(model_id), 0)
    self.assertEqual(await self.store.get_result("t-1", timeout=1.0), {"type": "optim_step", "ok": True})

  async def test_xread_block_future_wakes_immediately_and_survives_retries(self) -> None:
    waiters = [asyncio.create_task(self.queue.get_result("req-wake", timeout=5.0)) for _ in range(3)]
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await self.queue.set_result("req-wake", {"type": "forward_backward", "loss": 0.5})
    results = await asyncio.gather(*waiters)
    self.assertLess(time.monotonic() - t0, 1.0)
    self.assertEqual(results, [{"type": "forward_backward", "loss": 0.5}] * 3)
    # Non-destructive: retried retrieve_future still reads the same result.
    self.assertEqual(await self.queue.get_result("req-wake", timeout=0.5), {"type": "forward_backward", "loss": 0.5})
    # Pending markers are ignored; unresolved future returns try_again.
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
    entries = await self.redis.xrange(future_key(r["request_id"]))
    self.assertEqual(won.count(True), 1)
    self.assertEqual(len(entries), 1)
    self.assertEqual(await self.queue.get_result(r["request_id"], timeout=0.5), {"type": "sample", "attempt": won.index(True)})
    self.assertEqual(await self.pending_map(), {})

  async def test_delete_if_empty_preserves_non_empty_stream(self) -> None:
    q = sample_queue(self.pool)
    r = self.sample_record()
    await self.queue.accept(r)
    self.assertFalse(await self.queue.delete_if_empty(q))
    [pending] = (await self.pending_map()).values()
    await self.queue.complete_sample(pending, {"type": "sample"})
    self.assertTrue(await self.redis.exists(stream_key(self.pool)))
    self.assertTrue(await self.queue.delete_if_empty(q))
    self.assertFalse(await self.redis.exists(stream_key(self.pool)))

  async def test_fail_job_fail_older_than_and_pinned_revisions(self) -> None:
    r1 = self.sample_record(revision="/ckpt/r1", job="job-a")
    r2 = self.sample_record(revision="/ckpt/r1", job="job-b")
    r3 = self.sample_record(revision="/ckpt/r2", job="job-a")
    for r in (r1, r2, r3):
      await self.queue.accept(r)
    self.assertEqual(await self.queue.pinned_revisions(self.pool), {"/ckpt/r1", "/ckpt/r2"})
    self.assertEqual(await self.queue.pending_jobs(self.pool), {"job-a", "job-b"})
    self.assertEqual(await self.queue.fail_job(self.pool, "job-a", {"type": "RequestFailedResponse", "error_message": "deleted"}), 2)
    self.assertEqual(await self.queue.pinned_revisions(self.pool), {"/ckpt/r1"})
    self.assertEqual(await self.queue.answered([r1["request_id"], r2["request_id"], r3["request_id"]]), {r1["request_id"], r3["request_id"]})
    # Negative age sweeps remaining pending entry regardless of clock.
    self.assertEqual(await self.queue.fail_older_than(-60, {"type": "RequestFailedResponse", "error_message": "expired"}), 1)
    self.assertEqual(await self.pending_map(), {})

  async def test_active_set_round_robin_fairness_and_pending_after(self) -> None:
    await self.store.put_request({"model_id": "busy", "request_id": "b0"}, active_set_id="base-1")
    await self.store.put_request({"model_id": "quiet", "request_id": "q0"}, active_set_id="base-1")
    served = []
    for i in range(4):
      batch = await self.store.get_requests(active_set_id="base-1")
      served.append(batch[0]["model_id"])
      for rec in batch:
        await self.store.complete_request(rec, {"type": "ok"})
      await self.store.put_request({"model_id": "busy", "request_id": f"b{i + 1}"}, active_set_id="base-1")
    self.assertEqual(served[:2], ["busy", "quiet"])

    # pending_for_model(after=...) returns only entries appended after the first batch.
    mid = "drain-model"
    await self.store.put_request({"model_id": mid, "request_id": "d0"})
    first = await self.store.get_requests_for_model(mid)
    self.assertEqual(await self.store.pending_for_model(mid, after=first[-1]["entry_id"]), [])
    await self.store.put_request({"model_id": mid, "request_id": "d1"})
    more = await self.store.pending_for_model(mid, after=first[-1]["entry_id"])
    self.assertEqual([r["request_id"] for r in more], ["d1"])


class StreamQueueCpuTest(_QueueContractMixin, unittest.IsolatedAsyncioTestCase):
  """Runs the full StreamQueue + RedisStore contract on CPU via FakeRedis."""

  async def asyncSetUp(self) -> None:
    self.redis = FakeRedis()
    self.queue = StreamQueue(self.redis)
    self.store = RedisStore(self.redis)
    self.pool = f"sampler-{uuid.uuid4().hex[:8]}"

  async def test_delete_if_empty_aborts_when_entry_added_during_watch(self) -> None:
    q = train_queue("race-model")
    await self.redis.xadd(queue_key(q), {"record": '{"request_id":"r0"}'})
    await self.redis.xdel(queue_key(q), (await self.redis.xrange(queue_key(q)))[0][0])

    async def inject_during_watch():
      await self.queue.enqueue(q, {"request_id": "r1"})

    # Trigger concurrent enqueue while delete_if_empty yields before executing MULTI.
    deleted, _ = await asyncio.gather(self.queue.delete_if_empty(q), inject_during_watch())
    self.assertFalse(deleted)
    self.assertEqual([r["request_id"] for r in await self.queue.pending(q)], ["r1"])

  async def test_key_helpers_match_unified_layout(self) -> None:
    self.assertEqual(queue_key(train_queue("m1")), "open_rl:q:train:m1")
    self.assertEqual(stream_key("p1"), "open_rl:q:sample:p1")
    self.assertEqual(idempotency_key(sample_queue("p1"), "k1"), "open_rl:q:sample:p1:idem:k1")
    self.assertEqual(future_key("req-1"), "open_rl:f:req-1")

  async def test_complete_sample_emits_telemetry(self) -> None:
    r = self.sample_record(job="job-tel")
    await self.queue.accept(r)
    [pending] = (await self.pending_map()).values()
    with patch("server.telemetry.sample_answered") as answered:
      self.assertTrue(await self.queue.complete_sample(pending, {"type": "sample"}, started_at=123.0))
      self.assertFalse(await self.queue.complete_sample(pending, {"type": "sample"}, started_at=124.0))
    answered.assert_called_once_with("job-tel", self.pool, r["request_id"], {"type": "sample"}, 123.0)

  async def test_wait_for_new_wakes_on_enqueue_and_compact_prompt_uses_array(self) -> None:
    q = sample_queue(self.pool)
    waiter = asyncio.create_task(self.queue.wait_for_new(q, after="0-0", timeout_ms=5000))
    await asyncio.sleep(0.05)
    t0 = time.monotonic()
    await self.queue.accept(self.sample_record())
    await waiter
    self.assertLess(time.monotonic() - t0, 1.0)
    [compact] = await self.queue.pending(q, compact=True)
    [plain] = await self.queue.pending(q, compact=False)
    self.assertEqual((type(compact["prompt_token_ids"]).__name__, list(compact["prompt_token_ids"])), ("array", [1, 2, 3]))
    self.assertIsInstance(plain["prompt_token_ids"], list)

  async def test_active_set_cleans_up_empty_tenant_stream(self) -> None:
    await self.store.put_request({"model_id": "done-tenant", "request_id": "d0"}, active_set_id="base-1")
    await self.store.put_request({"model_id": "live-tenant", "request_id": "l0"}, active_set_id="base-1")
    batch = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual(batch[0]["model_id"], "done-tenant")
    await self.store.complete_request(batch[0], {"type": "ok"})
    # Next two rounds serve live-tenant and prune the empty done-tenant stream.
    second = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual(second[0]["model_id"], "live-tenant")
    third = await self.store.get_requests(active_set_id="base-1")
    self.assertEqual(third[0]["model_id"], "live-tenant")
    self.assertFalse(await self.redis.exists(queue_key(train_queue("done-tenant"))))


class LiveRedisQueueTest(_QueueContractMixin, RedisTestCase):
  """Runs the exact same contract against a real redis-server when available."""

  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.queue = StreamQueue(self.redis)
    self.store = RedisStore(self.redis)
    self.pool = f"sampler-{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
  unittest.main()
