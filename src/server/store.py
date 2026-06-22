# This file contains the state management and request queue implementation for the Open-RL server, supporting both in-memory and Redis backends.

import asyncio
import json
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any

import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError


class RequestStore(ABC):
  @abstractmethod
  async def put_request(self, req_data: dict[str, Any]) -> None:
    """Push a request into the global queue."""
    pass

  @abstractmethod
  async def put_worker_launch_request(self, req_data: dict[str, Any]) -> None:
    """Push a create-model request onto the queue that starts dedicated FFT workers."""
    pass

  @abstractmethod
  async def get_requests(self) -> list[dict[str, Any]]:
    """Block until at least 1 request is available, then return all currently queued requests."""
    pass

  @abstractmethod
  async def get_worker_launch_requests(self) -> list[dict[str, Any]]:
    """Block until at least 1 worker-launch request is available, then drain that queue."""
    pass

  @abstractmethod
  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    """Block until this model has at least 1 request, then return all queued requests for it."""
    pass

  @abstractmethod
  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    """Push a sampling request into the queue for its model."""
    pass

  @abstractmethod
  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    """Block until this model has at least 1 sampling request, then return all queued requests for it."""
    pass

  @abstractmethod
  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    """Resolve a future by its request ID."""
    pass

  @abstractmethod
  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    """Block until the future resolves or the timeout is reached."""
    pass

  async def start(self) -> None:
    """Initialize any background tasks/connections required by the store."""
    return None

  async def stop(self) -> None:
    """Clean up background tasks/connections."""
    return None


class InMemoryStore(RequestStore):
  def __init__(self):
    # tenant_id -> queue of requests
    self.queues: dict[str, asyncio.Queue] = {}
    # Simple list for round-robin
    self.active_tenants: list[str] = []
    self.active_tenants_cv = asyncio.Condition()
    self.futures_store: dict[str, dict[str, Any]] = {}
    self.futures_events: dict[str, asyncio.Event] = {}

  async def put_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")

    async with self.active_tenants_cv:
      if model_id not in self.queues:
        self.queues[model_id] = asyncio.Queue()

      await self.queues[model_id].put(req_data)

      if model_id not in self.active_tenants:
        self.active_tenants.append(model_id)
        self.active_tenants_cv.notify()

  async def put_worker_launch_request(self, req_data: dict[str, Any]) -> None:
    raise RuntimeError("Worker launch requests require REDIS_URL; in-memory queues cannot be shared across processes")

  async def get_requests(self) -> list[dict[str, Any]]:
    async with self.active_tenants_cv:
      # Block until at least one tenant is active
      while not self.active_tenants:
        await self.active_tenants_cv.wait()

      # Pop left, push right (Round Robin)
      model_id = self.active_tenants.pop(0)
      self.active_tenants.append(model_id)

      queue = self.queues[model_id]
      batch = [queue.get_nowait()]

      # Drain the rest of this tenant's queue
      while not queue.empty():
        batch.append(queue.get_nowait())

      # If completely empty, remove from rotation
      if queue.empty():
        self.active_tenants.remove(model_id)

      return batch

  async def get_worker_launch_requests(self) -> list[dict[str, Any]]:
    raise RuntimeError("Worker launch requests require REDIS_URL; in-memory queues cannot be shared across processes")

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Per-model full fine-tuning workers require REDIS_URL; in-memory queues cannot be shared across processes")

  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    raise RuntimeError("Sampling queues require REDIS_URL")

  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    raise RuntimeError("Sampling queues require REDIS_URL")

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    self.futures_store[req_id] = result
    if req_id in self.futures_events:
      self.futures_events[req_id].set()

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    self.futures_store.setdefault(req_id, {"status": "pending"})

    if self.futures_store[req_id].get("status") != "pending":
      return self.futures_store[req_id]

    event = asyncio.Event()
    self.futures_events[req_id] = event

    try:
      await asyncio.wait_for(event.wait(), timeout=timeout)
      return self.futures_store.get(req_id)
    except TimeoutError:
      return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    finally:
      self.futures_events.pop(req_id, None)


class RedisStore(RequestStore):
  FUTURE_CHANNEL = "open_rl:completed_futures"

  def __init__(self, redis_url: str, max_connections: int = 1000):
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2, max_connections=max_connections)
    self.active_list = "open_rl:active_tenants"
    # We also keep a set to guarantee O(1) deduplication before RPushing
    self.active_set = "open_rl:active_tenants_set"
    self.worker_launch_queue = "open_rl:worker_launch_queue"
    self.future_waiters: dict[str, set[asyncio.Future[None]]] = {}
    self._future_listener_task: asyncio.Task | None = None
    self._future_listener_ready: asyncio.Event | None = None
    self._future_listener_lock = asyncio.Lock()

  async def put_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")
    queue_key = f"open_rl:queue:{model_id}"

    # 1. Add request to tenant-specific list
    await self.redis.rpush(queue_key, json.dumps(req_data))

    # 2. Add tenant to active set and list if not already there
    # SADD returns 1 if it was newly added, 0 if it already existed
    is_new = await self.redis.sadd(self.active_set, model_id)
    if is_new == 1:
      await self.redis.rpush(self.active_list, model_id)

  async def put_worker_launch_request(self, req_data: dict[str, Any]) -> None:
    await self.redis.rpush(self.worker_launch_queue, json.dumps(req_data))

  async def get_requests(self) -> list[dict[str, Any]]:
    # BRPOPLPUSH blocks until an item is available.
    # It atomically pops the rightmost element of src, pushes it to the left of dst, and returns it.
    # Wait max 5 seconds so we can check for connection death.
    try:
      result = await self.redis.brpoplpush(self.active_list, self.active_list, timeout=5)
    except RedisTimeoutError:
      return []

    if not result:
      return []

    model_id = result
    queue_key = f"open_rl:queue:{model_id}"
    batch = []

    # Drain the entire queue for this tenant non-blockingly
    while True:
      item = await self.redis.lpop(queue_key)
      if not item:
        break
      batch.append(json.loads(item))

    # If the queue was empty (or we just drained it all but nothing new arrived),
    # we check the length. If it's truly empty, we scrub it from the rotation.
    # This requires a tiny Lua script or a quick transaction to ensure we don't
    # delete a tenant just as a new request is pushed.

    # Quick check:
    q_len = await self.redis.llen(queue_key)
    if q_len == 0:
      # We remove it from the list AND set
      await self.redis.lrem(self.active_list, 0, model_id)
      await self.redis.srem(self.active_set, model_id)

    return batch

  async def get_worker_launch_requests(self) -> list[dict[str, Any]]:
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

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
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

  async def put_sampling_request(self, req_data: dict[str, Any]) -> None:
    model_id = req_data.get("model_id", "default")
    queue_key = f"open_rl:sampler_queue:{model_id}"
    await self.redis.rpush(queue_key, json.dumps(req_data))

  async def get_sampling_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
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

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return

    key = f"open_rl:future:{req_id}"
    payload = json.dumps(result)
    await self.redis.set(key, payload, ex=300)
    await self.redis.publish(self.FUTURE_CHANNEL, req_id)

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    result = await self._read_future_result(req_id)
    if result is not None:
      return result

    await self._ensure_future_listener()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()
    self.future_waiters.setdefault(req_id, set()).add(future)

    try:
      # Close the race where the result is written between the first Redis read
      # and waiter registration.
      result = await self._read_future_result(req_id)
      if result is not None:
        return result
      await asyncio.wait_for(future, timeout=timeout)
      result = await self._read_future_result(req_id)
      if result is not None:
        return result
      return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    except TimeoutError:
      return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    finally:
      waiters = self.future_waiters.get(req_id)
      if waiters is not None:
        waiters.discard(future)
        if not waiters:
          self.future_waiters.pop(req_id, None)

  async def _read_future_result(self, req_id: str) -> dict[str, Any] | None:
    payload = await self.redis.get(f"open_rl:future:{req_id}")
    return json.loads(payload) if payload else None

  async def _ensure_future_listener(self) -> None:
    async with self._future_listener_lock:
      if self._future_listener_task is None or self._future_listener_task.done():
        self._future_listener_ready = asyncio.Event()
        self._future_listener_task = asyncio.create_task(self._future_listener(self._future_listener_ready))
      ready = self._future_listener_ready

    assert ready is not None
    await ready.wait()

  async def _future_listener(self, ready: asyncio.Event) -> None:
    pubsub = self.redis.pubsub()
    try:
      await pubsub.subscribe(self.FUTURE_CHANNEL)
      ready.set()
      async for message in pubsub.listen():
        if message.get("type") != "message":
          continue
        req_id = message["data"]
        waiters = self.future_waiters.pop(req_id, set())
        for future in waiters:
          if not future.done():
            future.set_result(None)
    except asyncio.CancelledError:
      raise
    except Exception:
      ready.set()
      raise
    finally:
      ready.set()
      with suppress(Exception):
        await pubsub.unsubscribe(self.FUTURE_CHANNEL)
      with suppress(Exception):
        await pubsub.close()

  async def aclose(self) -> None:
    if self._future_listener_task is not None:
      self._future_listener_task.cancel()
      with suppress(asyncio.CancelledError, Exception):
        await self._future_listener_task
    for waiters in self.future_waiters.values():
      for future in waiters:
        future.cancel()
    self.future_waiters.clear()
    await self.redis.aclose()

  async def stop(self) -> None:
    await self.aclose()


# Global singleton factory
_store_instance = None


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
