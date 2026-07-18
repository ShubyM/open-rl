"""Process-local storage wiring for Open-RL.

The server uses storage for three distinct jobs:

* commands are queued work for trainer and sampler workers;
* futures are asynchronous request results;
* models are the canonical per-model metadata records.

Redis and in-memory implementations intentionally live together in this file so
their behavior can be compared without hiding the different storage semantics.
"""

import asyncio
import json
import os
import time
from typing import Any

import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError


class InMemoryCommands:
  def __init__(self) -> None:
    # tenant_id -> queue of requests
    self.queues: dict[str, asyncio.Queue] = {}
    self.active_tenants: list[str] = []
    self.active_tenants_cv = asyncio.Condition()

  async def enqueue_training(self, command: dict[str, Any]) -> None:
    model_id = command.get("model_id", "default")

    async with self.active_tenants_cv:
      if model_id not in self.queues:
        self.queues[model_id] = asyncio.Queue()

      await self.queues[model_id].put(command)

      if model_id not in self.active_tenants:
        self.active_tenants.append(model_id)
        self.active_tenants_cv.notify()

  async def dequeue_training(self) -> list[dict[str, Any]]:
    async with self.active_tenants_cv:
      while not self.active_tenants:
        await self.active_tenants_cv.wait()

      model_id = self.active_tenants.pop(0)
      self.active_tenants.append(model_id)

      queue = self.queues[model_id]
      batch = [queue.get_nowait()]
      while not queue.empty():
        batch.append(queue.get_nowait())

      if queue.empty():
        self.active_tenants.remove(model_id)

      return batch

  async def enqueue_worker_launch(self, command: dict[str, Any]) -> None:
    raise RuntimeError("Worker launch requests require REDIS_URL; in-memory queues cannot be shared across processes")

  async def dequeue_worker_launch(self) -> list[dict[str, Any]]:
    raise RuntimeError("Worker launch requests require REDIS_URL; in-memory queues cannot be shared across processes")

  async def dequeue_training_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Per-model full fine-tuning workers require REDIS_URL; in-memory queues cannot be shared across processes")

  async def enqueue_sampling(self, command: dict[str, Any]) -> None:
    raise RuntimeError("Sampling queues require REDIS_URL")

  async def dequeue_sampling_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Sampling queues require REDIS_URL")


class InMemoryFutures:
  def __init__(self) -> None:
    self.results: dict[str, dict[str, Any]] = {}
    self.events: dict[str, asyncio.Event] = {}

  async def mark_pending(self, request_id: str) -> None:
    self.results[request_id] = {"status": "pending"}

  async def resolve(self, request_id: str, result: dict[str, Any]) -> None:
    self.results[request_id] = result
    if request_id in self.events:
      self.events[request_id].set()

  async def wait(self, request_id: str, timeout: float) -> dict[str, Any] | None:
    self.results.setdefault(request_id, {"status": "pending"})

    if self.results[request_id].get("status") != "pending":
      return self.results[request_id]

    event = asyncio.Event()
    self.events[request_id] = event

    try:
      await asyncio.wait_for(event.wait(), timeout=timeout)
      return self.results.get(request_id)
    except TimeoutError:
      return {"type": "try_again", "request_id": request_id, "queue_state": "active"}
    finally:
      self.events.pop(request_id, None)


class InMemoryModelMetadata:
  def __init__(self) -> None:
    self.records: dict[str, dict[str, Any]] = {}

  async def put(self, model_id: str, metadata: dict[str, Any]) -> None:
    self.records[model_id] = metadata

  async def get(self, model_id: str) -> dict[str, Any] | None:
    return self.records.get(model_id)

  def get_sync(self, model_id: str) -> dict[str, Any] | None:
    return self.records.get(model_id)

  async def delete(self, model_id: str) -> None:
    self.records.pop(model_id, None)


class InMemoryStore:
  def __init__(self) -> None:
    self.commands = InMemoryCommands()
    self.futures = InMemoryFutures()
    self.models = InMemoryModelMetadata()


class RedisCommands:
  def __init__(self, client: Any) -> None:
    self.redis = client
    self.active_list = "open_rl:active_tenants"
    self.active_set = "open_rl:active_tenants_set"
    self.worker_launch_queue = "open_rl:worker_launch_queue"

  async def enqueue_training(self, command: dict[str, Any]) -> None:
    model_id = command.get("model_id", "default")
    queue_key = f"open_rl:queue:{model_id}"

    await self.redis.rpush(queue_key, json.dumps(command))

    is_new = await self.redis.sadd(self.active_set, model_id)
    if is_new == 1:
      await self.redis.rpush(self.active_list, model_id)

  async def dequeue_training(self) -> list[dict[str, Any]]:
    try:
      result = await self.redis.brpoplpush(self.active_list, self.active_list, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    model_id = result
    queue_key = f"open_rl:queue:{model_id}"
    batch = []

    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    q_len = await self.redis.llen(queue_key)
    if q_len == 0:
      await self.redis.lrem(self.active_list, 0, model_id)
      await self.redis.srem(self.active_set, model_id)

    return batch

  async def enqueue_worker_launch(self, command: dict[str, Any]) -> None:
    await self.redis.rpush(self.worker_launch_queue, json.dumps(command))

  async def dequeue_worker_launch(self) -> list[dict[str, Any]]:
    try:
      result = await self.redis.blpop(self.worker_launch_queue, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    batch = [json.loads(result[1])]
    while True:
      item = await self.redis.lpop(self.worker_launch_queue)
      if not item:
        break
      batch.append(json.loads(item))

    return batch

  async def dequeue_training_for_model(self, model_id: str) -> list[dict[str, Any]]:
    queue_key = f"open_rl:queue:{model_id}"
    try:
      result = await self.redis.blpop(queue_key, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    batch = [json.loads(result[1])]
    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    q_len = await self.redis.llen(queue_key)
    if q_len == 0:
      await self.redis.lrem(self.active_list, 0, model_id)
      await self.redis.srem(self.active_set, model_id)

    return batch

  async def enqueue_sampling(self, command: dict[str, Any]) -> None:
    model_id = command.get("model_id", "default")
    queue_key = f"open_rl:sampler_queue:{model_id}"
    await self.redis.rpush(queue_key, json.dumps(command))

  async def dequeue_sampling_for_model(self, model_id: str) -> list[dict[str, Any]]:
    queue_key = f"open_rl:sampler_queue:{model_id}"
    try:
      result = await self.redis.blpop(queue_key, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    batch = [json.loads(result[1])]
    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    return batch


class RedisFutures:
  def __init__(self, client: Any) -> None:
    self.redis = client

  async def mark_pending(self, request_id: str) -> None:
    # Redis only stores resolved values; absence represents a pending future.
    return None

  async def resolve(self, request_id: str, result: dict[str, Any]) -> None:
    key = f"open_rl:future:{request_id}"
    await self.redis.rpush(key, json.dumps(result))
    await self.redis.expire(key, 300)

  async def wait(self, request_id: str, timeout: float) -> dict[str, Any] | None:
    key = f"open_rl:future:{request_id}"
    deadline = time.monotonic() + timeout

    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return {"type": "try_again", "request_id": request_id, "queue_state": "active"}
      try:
        result = await self.redis.blpop(key, timeout=min(3, max(1, int(remaining))))
      except RedisTimeoutError:
        result = None
      if result:
        payload = json.loads(result[1])
        await self.redis.rpush(key, result[1])
        await self.redis.expire(key, 300)
        return payload


class RedisModelMetadata:
  def __init__(self, redis_url: str, client: Any) -> None:
    self.redis = client

    import redis as sync_redis_mod

    self.sync_redis = sync_redis_mod.Redis.from_url(redis_url, decode_responses=True)

  @staticmethod
  def key(model_id: str) -> str:
    return f"open_rl:model_meta:{model_id}"

  async def put(self, model_id: str, metadata: dict[str, Any]) -> None:
    await self.redis.set(self.key(model_id), json.dumps(metadata))

  async def get(self, model_id: str) -> dict[str, Any] | None:
    value = await self.redis.get(self.key(model_id))
    return json.loads(value) if value else None

  def get_sync(self, model_id: str) -> dict[str, Any] | None:
    try:
      value = self.sync_redis.get(self.key(model_id))
      return json.loads(value) if value else None
    except Exception:
      return None

  async def delete(self, model_id: str) -> None:
    await self.redis.delete(self.key(model_id))


class RedisStore:
  def __init__(self, redis_url: str) -> None:
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2)
    self.commands = RedisCommands(self.redis)
    self.futures = RedisFutures(self.redis)
    self.models = RedisModelMetadata(redis_url, self.redis)


RequestStore = InMemoryStore | RedisStore


_store_instance: RequestStore | None = None


def get_store() -> RequestStore:
  global _store_instance
  if _store_instance is None:
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
      print(f"[RequestStore] Initializing Redis backend at {redis_url} with RR Tenant Queues")
      _store_instance = RedisStore(redis_url)
    else:
      print("[RequestStore] Initializing In-Memory backend with RR Tenant Queues")
      _store_instance = InMemoryStore()
  return _store_instance
