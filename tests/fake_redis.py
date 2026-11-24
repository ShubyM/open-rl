"""In-memory async Redis harness (Streams, KV, Sets, TTL, WATCH/MULTI) for CPU tests without redis-server."""

from __future__ import annotations

import asyncio
import fnmatch
import time
from typing import Any

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
    self._db, self._in_multi = db, not tx_mode
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
  """Faithful async Redis Streams + KV + Set + WATCH/MULTI test double (zero Redis Lists)."""

  def __init__(self):
    self.strings: dict[str, str] = {}
    self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
    self.sets: dict[str, set[str]] = {}
    self.expiries: dict[str, float] = {}
    self._versions: dict[str, int] = {}
    self._events: dict[str, asyncio.Event] = {}
    self._last_ms = self._last_seq = 0

  def _touch(self, key: str) -> None:
    self._versions[key] = self._versions.get(key, 0) + 1

  def _notify(self, key: str) -> None:
    if ev := self._events.pop(key, None):
      ev.set()

  def _evict(self, key: str) -> None:
    if key in self.expiries and time.monotonic() >= self.expiries[key]:
      for store in (self.strings, self.streams, self.sets, self.expiries):
        store.pop(key, None)

  def _exists(self, key: str) -> bool:
    self._evict(key)
    return key in self.strings or key in self.streams or key in self.sets

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
    self._evict(key)
    return self.strings.get(key)

  def _sync_delete(self, *keys: str) -> int:
    deleted = 0
    for key in keys:
      existed = self._exists(key)
      for store in (self.strings, self.streams, self.sets, self.expiries):
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
    return -1 if key not in self.expiries else max(0, int(round(self.expiries[key] - time.monotonic())))

  def _sync_xadd(self, key: str, fields: dict[str, str], id: str = "*", maxlen: int | None = None) -> str:
    self._evict(key)
    eid = self._next_id() if id == "*" else id
    stream = self.streams.setdefault(key, [])
    stream.append((eid, dict(fields)))
    if maxlen is not None and len(stream) > maxlen:
      del stream[:-maxlen]
    self._touch(key)
    self._notify(key)
    return eid

  def _sync_xrange(self, key: str, min: str = "-", max: str = "+", count: int | None = None) -> list[tuple[str, dict[str, str]]]:
    self._evict(key)
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
    self._evict(key)
    if key not in self.streams:
      return 0
    drop, before = set(ids), len(self.streams[key])
    self.streams[key] = [(eid, f) for eid, f in self.streams[key] if eid not in drop]
    if removed := before - len(self.streams[key]):
      self._touch(key)
    return removed

  def _sync_xlen(self, key: str) -> int:
    self._evict(key)
    return len(self.streams.get(key, ()))

  def _sync_sadd(self, key: str, *members: str) -> int:
    s = self.sets.setdefault(key, set())
    before = len(s)
    s.update(members)
    if added := len(s) - before:
      self._touch(key)
    return added

  def _sync_srem(self, key: str, *members: str) -> int:
    if not (s := self.sets.get(key)):
      return 0
    before = len(s)
    s.difference_update(members)
    if removed := before - len(s):
      self._touch(key)
    return removed

  def _sync_smembers(self, key: str) -> set[str]:
    return set(self.sets.get(key, ()))

  def __getattr__(self, name: str):
    if name.startswith("_") or (fn := getattr(self, f"_sync_{name}", None)) is None:
      raise AttributeError(name)

    async def _call(*args, **kwargs):
      return fn(*args, **kwargs)

    return _call

  async def xread(self, streams: dict[str, str], count: int | None = None, block: int | None = None) -> list[list[Any]]:
    deadline = None if block in (None, 0) else time.monotonic() + block / 1000.0
    while True:
      found = [[k, items] for k, after in streams.items() if (items := self._sync_xrange(k, min=f"({after}", count=count))]
      if found or block is None:
        return found
      remaining = None if deadline is None else deadline - time.monotonic()
      if remaining is not None and remaining <= 0:
        return []
      waits = [asyncio.create_task(self._events.setdefault(k, asyncio.Event()).wait()) for k in streams]
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
    pools: dict[str, set[str]] = {"stream": set(self.streams), "string": set(self.strings), "set": set(self.sets)}
    keys = pools[_type] if _type in pools else set().union(*pools.values())
    for key in list(keys):
      if self._exists(key) and fnmatch.fnmatchcase(key, match):
        yield key

  def pipeline(self, transaction: bool = True) -> _FakePipeline:
    return _FakePipeline(self, tx_mode=False)

  async def transaction(self, func, *watches: str, value_from_callable: bool = False):
    while True:
      snap = {k: self._versions.get(k, 0) for k in watches}
      pipe = _FakePipeline(self, tx_mode=True)
      result = await func(pipe)
      if not pipe._in_multi:
        return result if value_from_callable else []
      await asyncio.sleep(0)
      if any(self._versions.get(k, 0) != v for k, v in snap.items()):
        continue
      exec_res = await pipe.execute()
      return result if value_from_callable else exec_res

  async def ping(self) -> bool:
    return True

  async def flushdb(self) -> None:
    for d in (self.strings, self.streams, self.sets, self.expiries, self._versions):
      d.clear()

  async def aclose(self) -> None:
    pass
