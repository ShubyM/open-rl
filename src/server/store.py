"""Generic state store backends, sharing Redis connections."""

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


@dataclass
class Writes:
  """Writes applied together, each key expiring a TTL after: JSON items appended to lists that keep their newest
  `limit`, amounts added to counters and to counts in hashes, and values set."""

  appends: dict[str, tuple[list[str], int]] = field(default_factory=dict)  # key -> (items, limit)
  counts: Counter[str] = field(default_factory=Counter)
  fields: dict[str, Counter[str]] = field(default_factory=dict)
  values: dict[str, str] = field(default_factory=dict)


class StateStore(ABC):
  """Application state stored as values with expiry and sets of members."""

  @abstractmethod
  async def set_value(self, key: str, value: str, ttl_seconds: float | None = None) -> None:
    """Store a simple string value by key, gone after ttl_seconds if given."""
    pass

  @abstractmethod
  async def get_value(self, key: str) -> str | None:
    """Fetch a string value by key."""
    pass

  @abstractmethod
  def get_value_sync(self, key: str) -> str | None:
    """Synchronously fetch a string value by key."""
    pass

  @abstractmethod
  async def delete_values(self, *keys: str) -> None:
    """Delete one or more keys."""
    pass

  @abstractmethod
  async def add_to_set(self, key: str, member: str) -> None:
    pass

  @abstractmethod
  async def remove_from_set(self, key: str, member: str) -> None:
    pass

  @abstractmethod
  async def set_members(self, key: str) -> set[str]:
    pass

  @abstractmethod
  async def scan_values(self, prefix: str) -> dict[str, str]:
    """Every live value whose key starts with prefix."""
    pass

  @abstractmethod
  def push_recent_sync(self, key: str, values: list[str], keep: int, ttl_seconds: float) -> None:
    """Synchronously put values at the head of the list at key and keep its newest `keep`. Redis drops the list
    ttl_seconds after the last push."""
    pass

  @abstractmethod
  async def recent(self, key: str) -> list[str]:
    """The list push_recent_sync keeps at key, newest first."""
    pass

  @abstractmethod
  def write_sync(self, writes: Writes, ttl_seconds: float) -> None:
    """Apply writes in one round trip, without the event loop, for a thread."""
    pass

  @abstractmethod
  async def read_samples(self, key: str, last: int | None = None) -> list[dict[str, Any]]:
    """The JSON items of a list write_sync appends to, oldest first, or only its last ones."""
    pass

  @abstractmethod
  async def read_counts(self, key: str) -> dict[str, int]:
    """The counts in a hash write_sync adds to."""
    pass


class InMemoryStateStore(StateStore):
  def __init__(self):
    self.kv_store: dict[str, str] = {}
    self.expiries: dict[str, float] = {}
    self.sets: dict[str, set[str]] = {}
    self.lists: dict[str, list[str]] = {}
    self.hashes: dict[str, Counter[str]] = {}

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
      self.kv_store.pop(k, None)
      self.expiries.pop(k, None)
      self.sets.pop(k, None)

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
    # Lists and hashes here never expire.
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
    self.redis = client
    self.sync_redis = sync_client

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
    # Decoded off the event loop, since a list holds up to thousands of records.
    return await asyncio.to_thread(lambda: [json.loads(item) for item in raw])

  async def read_counts(self, key: str) -> dict[str, int]:
    return {name: int(count) for name, count in (await self.redis.hgetall(key)).items()}


@cache
def redis_client(redis_url: str) -> redis.Redis:
  # The socket timeout must outlast the 10s XREAD BLOCK chunks: redis-py 8 defaults
  # it to 5s, and a read timing out on a live blocking read drops the wait early.
  return redis.from_url(redis_url, decode_responses=True, health_check_interval=2, max_connections=10000, socket_timeout=30)


@cache
def get_state_store() -> StateStore:
  redis_url = os.environ.get("REDIS_URL")
  if redis_url:
    # Timeouts, so a Redis that stops answering costs a thread its call rather than the thread.
    sync_client = sync_redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=5, socket_connect_timeout=5)
    return RedisStateStore(redis_client(redis_url), sync_client)
  return InMemoryStateStore()
