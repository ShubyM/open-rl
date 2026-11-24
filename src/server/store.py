"""Request transport and generic state backends, sharing Redis connections."""

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import redis as sync_redis
import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from server.queue import StreamQueue, train_queue


def _id_key(entry_id: str) -> tuple[int, int]:
  ms, _, seq = entry_id.partition("-")
  return int(ms), int(seq or 0)


class RequestStore(ABC):
  """Request queues and results, keyed by request ID."""

  @abstractmethod
  async def put_request(self, req_data: dict[str, Any], active_set_id: str | None = None) -> None: ...
  @abstractmethod
  async def get_requests(self, active_set_id: str | None = None) -> list[dict[str, Any]]: ...
  @abstractmethod
  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]: ...
  @abstractmethod
  async def set_result(self, req_id: str, result: dict[str, Any]) -> None: ...
  @abstractmethod
  async def get_result(self, req_id: str, timeout: float) -> dict[str, Any] | None: ...
  @abstractmethod
  async def queued(self, model_id: str) -> int: ...

  async def pending_for_model(self, model_id: str, *, after: str = "0-0") -> list[dict[str, Any]]:
    return await self.get_requests_for_model(model_id) if await self.queued(model_id) > 0 else []

  async def complete_request(self, record: dict[str, Any], result: dict[str, Any]) -> bool:
    if req_id := record.get("request_id"):
      await self.set_result(req_id, result)
    return True

  async def ack_request(self, record: dict[str, Any]) -> bool:
    return True


class InMemoryStore(RequestStore):
  def __init__(self):
    self.queues: dict[str, dict[str, dict[str, Any]]] = {}
    self.active_tenants: dict[str, list[str]] = {}
    self.futures_store: dict[str, dict[str, Any]] = {}
    self.active_tenants_cv, self.futures_cv, self._seq = asyncio.Condition(), asyncio.Condition(), 0

  async def put_request(self, req_data: dict[str, Any], active_set_id: str | None = None) -> None:
    model_id, set_key = req_data.get("model_id", "default"), active_set_id or "default"
    async with self.active_tenants_cv:
      self._seq += 1
      eid = f"{int(time.time() * 1000)}-{self._seq}"
      self.queues.setdefault(model_id, {})[eid] = {**req_data, "entry_id": eid}
      tenants = self.active_tenants.setdefault(set_key, [])
      if model_id not in tenants:
        tenants.append(model_id)
      self.active_tenants_cv.notify_all()

  async def get_requests(self, active_set_id: str | None = None) -> list[dict[str, Any]]:
    async with self.active_tenants_cv:
      while True:
        tenants = self.active_tenants.get(active_set_id) if active_set_id else next((t for t in self.active_tenants.values() if t), None)
        if not tenants:
          await self.active_tenants_cv.wait()
          continue
        if entries := self.queues.get(tenants[0]):
          tenants.append(tenants.pop(0))
          return list(entries.values())
        tenants.pop(0)

  async def queued(self, model_id: str) -> int:
    return len(self.queues.get(model_id, ()))

  async def pending_for_model(self, model_id: str, *, after: str = "0-0") -> list[dict[str, Any]]:
    cutoff = _id_key(after)
    return [r for eid, r in self.queues.get(model_id, {}).items() if _id_key(eid) > cutoff]

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Per-model full fine-tuning workers require REDIS_URL; in-memory queues cannot be shared across processes")

  async def complete_request(self, record: dict[str, Any], result: dict[str, Any]) -> bool:
    if result.get("status") == "pending":
      return False
    if (eid := record.get("entry_id")) is not None:
      entries = self.queues.get(record.get("model_id", "default"))
      if not entries or entries.pop(eid, None) is None:
        return False
    if req_id := record.get("request_id"):
      await self.set_result(req_id, result)
    return True

  async def ack_request(self, record: dict[str, Any]) -> bool:
    if (eid := record.get("entry_id")) is None:
      return True
    return self.queues.get(record.get("model_id", "default"), {}).pop(eid, None) is not None

  async def set_result(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return
    async with self.futures_cv:
      self.futures_store[req_id] = result
      self.futures_cv.notify_all()

  async def get_result(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    async with self.futures_cv:
      try:
        await asyncio.wait_for(self.futures_cv.wait_for(lambda: req_id in self.futures_store), timeout=timeout)
      except TimeoutError:
        return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
      return self.futures_store[req_id]


class RedisStore(RequestStore):
  """Training request store over the unified StreamQueue."""

  def __init__(self, client: redis.Redis):
    self.redis, self.queue = client, StreamQueue(client)
    self.active_list, self.active_set = "open_rl:active_tenants", "open_rl:active_tenants_set"

  def _active_keys(self, active_set_id: str | None) -> tuple[str, str]:
    return (f"{self.active_set}:{active_set_id}", f"{self.active_list}:{active_set_id}") if active_set_id else (self.active_set, self.active_list)

  async def _mark_active(self, active_set: str, active_list: str, model_id: str) -> None:
    if await self.redis.sadd(active_set, model_id) == 1:
      await self.redis.rpush(active_list, model_id)

  async def put_request(self, req_data: dict[str, Any], active_set_id: str | None = None) -> None:
    model_id = req_data.get("model_id", "default")
    await self.queue.enqueue(train_queue(model_id), req_data)
    await self._mark_active(*self._active_keys(active_set_id), model_id)

  async def get_requests(self, active_set_id: str | None = None) -> list[dict[str, Any]]:
    active_set, active_list = self._active_keys(active_set_id)
    while True:
      raw = await self.redis.lindex(active_list, 0)
      if not raw:
        try:
          raw = await self.redis.brpoplpush(active_list, active_list, timeout=5)
        except RedisTimeoutError:
          return []
        if not raw:
          return []
      model_id = raw.decode() if isinstance(raw, bytes) else str(raw)
      q = train_queue(model_id)
      if not (batch := await self.queue.pending(q)):
        await self.redis.lrem(active_list, 0, model_id)
        await self.redis.srem(active_set, model_id)
        if await self.queue.queued(q) > 0:
          await self._mark_active(active_set, active_list, model_id)
        else:
          await self.queue.delete_if_empty(q)
        continue
      await self.redis.lmove(active_list, active_list, "LEFT", "RIGHT")
      return batch

  async def queued(self, model_id: str) -> int:
    return await self.queue.queued(train_queue(model_id))

  async def pending_for_model(self, model_id: str, *, after: str = "0-0") -> list[dict[str, Any]]:
    return await self.queue.pending(train_queue(model_id), after=after)

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    q = train_queue(model_id)
    if batch := await self.queue.pending(q):
      return batch
    try:
      await self.queue.wait_for_new(q, after="0-0", timeout_ms=5000)
    except RedisTimeoutError:
      return []
    return await self.queue.pending(q)

  async def complete_request(self, record: dict[str, Any], result: dict[str, Any]) -> bool:
    req_id, eid, q = record.get("request_id"), record.get("entry_id"), train_queue(record.get("model_id", "default"))
    if eid is None:
      if req_id:
        await self.queue.set_result(req_id, result)
      return True
    return await self.queue.complete(q, eid, req_id, result) if req_id else await self.queue.ack(q, eid)

  async def ack_request(self, record: dict[str, Any]) -> bool:
    return True if (eid := record.get("entry_id")) is None else await self.queue.ack(train_queue(record.get("model_id", "default")), eid)

  async def set_result(self, req_id: str, result: dict[str, Any]) -> None:
    await self.queue.set_result(req_id, result)

  async def get_result(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    return await self.queue.get_result(req_id, timeout)


@dataclass
class Writes:
  """Writes applied together with a shared TTL: list appends (capped to `limit`), counters, hash counts, and values."""

  appends: dict[str, tuple[list[str], int]] = field(default_factory=dict)
  counts: Counter[str] = field(default_factory=Counter)
  fields: dict[str, Counter[str]] = field(default_factory=dict)
  values: dict[str, str] = field(default_factory=dict)


class StateStore(ABC):
  """Application state stored as values with expiry and sets of members."""

  @abstractmethod
  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None: ...
  @abstractmethod
  async def get_value(self, key: str) -> str | None: ...
  @abstractmethod
  def get_value_sync(self, key: str) -> str | None: ...
  @abstractmethod
  async def delete_values(self, *keys: str) -> None: ...
  @abstractmethod
  async def add_to_set(self, key: str, member: str) -> None: ...
  @abstractmethod
  async def remove_from_set(self, key: str, member: str) -> None: ...
  @abstractmethod
  async def set_members(self, key: str) -> set[str]: ...
  @abstractmethod
  async def scan_values(self, prefix: str) -> dict[str, str]: ...
  @abstractmethod
  def push_recent_sync(self, key: str, values: list[str], keep: int, ttl_seconds: float) -> None: ...
  @abstractmethod
  async def recent(self, key: str) -> list[str]: ...
  @abstractmethod
  def write_sync(self, writes: Writes, ttl_seconds: float) -> None: ...
  @abstractmethod
  async def read_samples(self, key: str, last: int | None = None) -> list[dict[str, Any]]: ...
  @abstractmethod
  async def read_counts(self, key: str) -> dict[str, int]: ...


class InMemoryStateStore(StateStore):
  def __init__(self):
    self.kv_store, self.expiries = {}, {}
    self.sets: dict[str, set[str]] = {}
    self.lists, self.hashes = {}, {}

  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    self.kv_store[key] = value
    self.expiries.pop(key, None)
    if ttl_seconds is not None:
      self.expiries[key] = time.monotonic() + ttl_seconds

  async def get_value(self, key: str) -> str | None:
    return self.get_value_sync(key)

  def get_value_sync(self, key: str) -> str | None:
    if key in self.expiries and time.monotonic() >= self.expiries[key]:
      self.kv_store.pop(key, None)
      self.expiries.pop(key, None)
    return self.kv_store.get(key)

  async def delete_values(self, *keys: str) -> None:
    for k in keys:
      for d in (self.kv_store, self.expiries, self.sets):
        d.pop(k, None)

  async def add_to_set(self, key: str, member: str) -> None:
    self.sets.setdefault(key, set()).add(member)

  async def remove_from_set(self, key: str, member: str) -> None:
    self.sets.get(key, set()).discard(member)

  async def set_members(self, key: str) -> set[str]:
    return set(self.sets.get(key, set()))

  async def scan_values(self, prefix: str) -> dict[str, str]:
    return {key: value for key in list(self.kv_store) if key.startswith(prefix) and (value := self.get_value_sync(key)) is not None}

  def push_recent_sync(self, key: str, values: list[str], keep: int, ttl_seconds: float) -> None:
    self.lists[key] = (values[::-1] + self.lists.get(key, []))[:keep]

  async def recent(self, key: str) -> list[str]:
    return list(self.lists.get(key, []))

  def write_sync(self, writes: Writes, ttl_seconds: float) -> None:
    for key, (items, limit) in writes.appends.items():
      self.lists[key] = [*self.lists.get(key, []), *items][-limit:]
    for key, amount in writes.counts.items():
      self.kv_store[key] = str(int(self.get_value_sync(key) or 0) + amount)
      self.expiries[key] = time.monotonic() + ttl_seconds
    for key, counts in writes.fields.items():
      self.hashes.setdefault(key, Counter()).update(counts)
    for key, value in writes.values.items():
      self.kv_store[key] = value
      self.expiries[key] = time.monotonic() + ttl_seconds

  async def read_samples(self, key: str, last: int | None = None) -> list[dict[str, Any]]:
    return [json.loads(item) for item in self.lists.get(key, [])[-(last or 0) :]]

  async def read_counts(self, key: str) -> dict[str, int]:
    return dict(self.hashes.get(key, {}))


class RedisStateStore(StateStore):
  def __init__(self, client: redis.Redis, sync_client: sync_redis.Redis):
    self.redis, self.sync_redis = client, sync_client

  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    await self.redis.set(key, value, px=None if ttl_seconds is None else int(ttl_seconds * 1000))

  async def get_value(self, key: str) -> str | None:
    return await self.redis.get(key)

  def get_value_sync(self, key: str) -> str | None:
    return self.sync_redis.get(key)

  async def delete_values(self, *keys: str) -> None:
    if keys:
      await self.redis.delete(*keys)

  async def add_to_set(self, key: str, member: str) -> None:
    await self.redis.sadd(key, member)

  async def remove_from_set(self, key: str, member: str) -> None:
    await self.redis.srem(key, member)

  async def set_members(self, key: str) -> set[str]:
    return set(await self.redis.smembers(key))

  async def scan_values(self, prefix: str) -> dict[str, str]:
    keys = [key async for key in self.redis.scan_iter(match=prefix + "*", count=1000)]
    return {key: value for key, value in zip(keys, await self.redis.mget(keys)) if value is not None} if keys else {}

  def push_recent_sync(self, key: str, values: list[str], keep: int, ttl_seconds: float) -> None:
    self.sync_redis.pipeline().lpush(key, *values).ltrim(key, 0, keep - 1).expire(key, int(ttl_seconds)).execute()

  async def recent(self, key: str) -> list[str]:
    return await self.redis.lrange(key, 0, -1)

  def write_sync(self, writes: Writes, ttl_seconds: float) -> None:
    ttl = int(ttl_seconds)
    pipe = self.sync_redis.pipeline(transaction=False)
    for key, (items, limit) in writes.appends.items():
      pipe.rpush(key, *items).ltrim(key, -limit, -1).expire(key, ttl)
    for key, amount in writes.counts.items():
      pipe.incrby(key, amount).expire(key, ttl)
    for key, counts in writes.fields.items():
      for name, amount in counts.items():
        pipe.hincrby(key, name, amount)
      pipe.expire(key, ttl)
    for key, value in writes.values.items():
      pipe.set(key, value, ex=ttl)
    pipe.execute()

  async def read_samples(self, key: str, last: int | None = None) -> list[dict[str, Any]]:
    raw = await self.redis.lrange(key, -(last or 0), -1)
    return await asyncio.to_thread(lambda: [json.loads(item) for item in raw])

  async def read_counts(self, key: str) -> dict[str, int]:
    return {name: int(count) for name, count in (await self.redis.hgetall(key)).items()}


@cache
def redis_client(redis_url: str) -> redis.Redis:
  # Socket timeout outlasts 10s XREAD BLOCK chunks so live streams never drop on read timeout.
  return redis.from_url(redis_url, decode_responses=True, health_check_interval=2, max_connections=10000, socket_timeout=30)


@cache
def get_store() -> RequestStore:
  return RedisStore(redis_client(url)) if (url := os.environ.get("REDIS_URL")) else InMemoryStore()


@cache
def get_state_store() -> StateStore:
  if not (url := os.environ.get("REDIS_URL")):
    return InMemoryStateStore()
  sync_client = sync_redis.Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
  return RedisStateStore(redis_client(url), sync_client)
