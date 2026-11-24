"""In-memory async Redis harness (Streams, KV, Sets, TTL, WATCH/MULTI) for CPU tests without redis-server."""

from __future__ import annotations

import asyncio
import fnmatch
import time
from typing import Any

_MAX_STREAM_ID = (1 << 62, 1 << 62)


def _parse_id(raw: str, *, is_max: bool = False) -> tuple[tuple[int, int], bool]:
  if raw == "-":
    return (0, 0), False
  if raw == "+":
    return _MAX_STREAM_ID, False
  exclusive = raw.startswith("(")
  value = raw[1:] if exclusive else raw
  if "-" in value:
    ms, seq = value.split("-", 1)
    return (int(ms), int(seq)), exclusive
  seq_bound = _MAX_STREAM_ID[1] if is_max else 0
  return (int(value), seq_bound), exclusive


class _FakePipeline:
  """Supports both `async with redis.pipeline()` and `redis.transaction(fn, *watches)`."""

  def __init__(self, db: FakeRedis, *, tx_mode: bool = False):
    self._db = db
    self._in_multi = not tx_mode
    self._ops: list[tuple[str, tuple, dict]] = []

  async def __aenter__(self) -> _FakePipeline:
    return self

  async def __aexit__(self, *args) -> None:
    pass

  def multi(self) -> None:
    self._in_multi = True

  async def execute(self) -> list[Any]:
    results = []
    for name, args, kwargs in self._ops:
      fn = getattr(self._db, f"_sync_{name}")
      results.append(fn(*args, **kwargs))
    self._ops.clear()
    return results

  def __getattr__(self, name: str):
    if self._in_multi:

      def _queue(*args, **kwargs):
        self._ops.append((name, args, kwargs))
        return self

      return _queue
    return getattr(self._db, name)


class FakeRedis:
  """Async Redis Streams, KV, Set, and WATCH/MULTI test double."""

  def __init__(self):
    self.strings: dict[str, str] = {}
    self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
    self.sets: dict[str, set[str]] = {}
    self.expiries: dict[str, float] = {}
    self._versions: dict[str, int] = {}
    self._events: dict[str, asyncio.Event] = {}
    self._last_ms = 0
    self._last_seq = 0

  def _touch(self, key: str) -> None:
    self._versions[key] = self._versions.get(key, 0) + 1

  def _notify(self, key: str) -> None:
    event = self._events.pop(key, None)
    if event is not None:
      event.set()

  def _evict_if_expired(self, key: str) -> None:
    if key in self.expiries and time.monotonic() >= self.expiries[key]:
      self.strings.pop(key, None)
      self.streams.pop(key, None)
      self.sets.pop(key, None)
      self.expiries.pop(key, None)

  def _exists(self, key: str) -> bool:
    self._evict_if_expired(key)
    return key in self.strings or key in self.streams or key in self.sets

  def _next_id(self) -> str:
    ms = int(time.time() * 1000)
    if ms <= self._last_ms:
      self._last_seq += 1
    else:
      self._last_ms = ms
      self._last_seq = 0
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
      self.strings.pop(key, None)
      self.streams.pop(key, None)
      self.sets.pop(key, None)
      self.expiries.pop(key, None)
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
    entry_id = self._next_id() if id == "*" else id
    stream = self.streams.setdefault(key, [])
    stream.append((entry_id, dict(fields)))
    if maxlen is not None and len(stream) > maxlen:
      del stream[:-maxlen]
    self._touch(key)
    self._notify(key)
    return entry_id

  def _sync_xrange(self, key: str, min: str = "-", max: str = "+", count: int | None = None) -> list[tuple[str, dict[str, str]]]:
    self._evict_if_expired(key)
    low, low_exclusive = _parse_id(min, is_max=False)
    high, high_exclusive = _parse_id(max, is_max=True)
    result = []
    for entry_id, fields in self.streams.get(key, ()):
      parsed, _ = _parse_id(entry_id)
      if low_exclusive and parsed <= low:
        continue
      if not low_exclusive and parsed < low:
        continue
      if high_exclusive and parsed >= high:
        continue
      if not high_exclusive and parsed > high:
        continue
      result.append((entry_id, dict(fields)))
      if count is not None and len(result) >= count:
        break
    return result

  def _sync_xdel(self, key: str, *ids: str) -> int:
    self._evict_if_expired(key)
    if key not in self.streams:
      return 0
    drop = set(ids)
    before = len(self.streams[key])
    self.streams[key] = [(entry_id, fields) for entry_id, fields in self.streams[key] if entry_id not in drop]
    removed = before - len(self.streams[key])
    if removed:
      self._touch(key)
    return removed

  def _sync_xlen(self, key: str) -> int:
    self._evict_if_expired(key)
    return len(self.streams.get(key, ()))

  def _sync_sadd(self, key: str, *members: str) -> int:
    bucket = self.sets.setdefault(key, set())
    before = len(bucket)
    bucket.update(members)
    added = len(bucket) - before
    if added:
      self._touch(key)
    return added

  def _sync_srem(self, key: str, *members: str) -> int:
    bucket = self.sets.get(key)
    if not bucket:
      return 0
    before = len(bucket)
    bucket.difference_update(members)
    removed = before - len(bucket)
    if removed:
      self._touch(key)
    return removed

  def _sync_smembers(self, key: str) -> set[str]:
    return set(self.sets.get(key, ()))

  def __getattr__(self, name: str):
    if name.startswith("_"):
      raise AttributeError(name)
    fn = getattr(self, f"_sync_{name}", None)
    if fn is None:
      raise AttributeError(name)

    async def _call(*args, **kwargs):
      return fn(*args, **kwargs)

    return _call

  async def xread(self, streams: dict[str, str], count: int | None = None, block: int | None = None) -> list[list[Any]]:
    deadline = None if block in (None, 0) else time.monotonic() + block / 1000.0
    while True:
      found = []
      for key, after in streams.items():
        items = self._sync_xrange(key, min=f"({after}", count=count)
        if items:
          found.append([key, items])
      if found or block is None:
        return found
      remaining = None if deadline is None else deadline - time.monotonic()
      if remaining is not None and remaining <= 0:
        return []
      waits = [asyncio.create_task(self._events.setdefault(key, asyncio.Event()).wait()) for key in streams]
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
    by_type = {
      "stream": set(self.streams),
      "string": set(self.strings),
      "set": set(self.sets),
    }
    keys = by_type[_type] if _type in by_type else set().union(*by_type.values())
    for key in list(keys):
      if self._exists(key) and fnmatch.fnmatchcase(key, match):
        yield key

  def pipeline(self, transaction: bool = True) -> _FakePipeline:
    return _FakePipeline(self, tx_mode=False)

  async def transaction(self, func, *watches: str, value_from_callable: bool = False):
    while True:
      snapshot = {key: self._versions.get(key, 0) for key in watches}
      pipe = _FakePipeline(self, tx_mode=True)
      result = await func(pipe)
      if not pipe._in_multi:
        return result if value_from_callable else []
      # Yield once so concurrent coroutines overlap and exercise WATCH conflict retries.
      await asyncio.sleep(0)
      if any(self._versions.get(key, 0) != version for key, version in snapshot.items()):
        continue
      exec_results = await pipe.execute()
      return result if value_from_callable else exec_results

  async def ping(self) -> bool:
    return True

  async def flushdb(self) -> None:
    self.strings.clear()
    self.streams.clear()
    self.sets.clear()
    self.expiries.clear()
    self._versions.clear()

  async def aclose(self) -> None:
    pass
