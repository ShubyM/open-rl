# This file contains the state management and request queue implementation for the Open-RL server, supporting both in-memory and Redis backends.

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import redis.asyncio as redis


class RequestStore(ABC):
  @abstractmethod
  async def put_request(self, req_data: dict[str, Any]) -> None:
    """Push a request into the global queue."""
    pass

  @abstractmethod
  async def get_requests(self) -> list[dict[str, Any]]:
    """Block until at least 1 request is available, then return all currently queued requests."""
    pass

  @abstractmethod
  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    """Block for one model queue, then drain only that model's currently queued requests."""
    pass

  @abstractmethod
  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    """Resolve a future by its request ID."""
    pass

  @abstractmethod
  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    """Block until the future resolves or the timeout is reached."""
    pass


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

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    async with self.active_tenants_cv:
      while model_id not in self.queues or self.queues[model_id].empty():
        await self.active_tenants_cv.wait()

      queue = self.queues[model_id]
      batch = [queue.get_nowait()]
      while not queue.empty():
        batch.append(queue.get_nowait())

      if queue.empty() and model_id in self.active_tenants:
        self.active_tenants.remove(model_id)

      return batch

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
    except asyncio.TimeoutError:
      return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
    finally:
      self.futures_events.pop(req_id, None)


class RedisStore(RequestStore):
  def __init__(self, redis_url: str):
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2)
    self.active_list = "open_rl:active_tenants"
    # We also keep a set to guarantee O(1) deduplication before RPushing
    self.active_set = "open_rl:active_tenants_set"

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

  async def get_requests(self) -> list[dict[str, Any]]:
    # BRPOPLPUSH blocks until an item is available.
    # It atomically pops the rightmost element of src, pushes it to the left of dst, and returns it.
    # Wait max 5 seconds so we can check for connection death.
    result = await self.redis.brpoplpush(self.active_list, self.active_list, timeout=5)

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

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    queue_key = f"open_rl:queue:{model_id}"
    result = await self.redis.blpop(queue_key, timeout=5)
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

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return

    key = f"open_rl:future:{req_id}"
    await self.redis.rpush(key, json.dumps(result))
    await self.redis.expire(key, 300)

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    key = f"open_rl:future:{req_id}"

    result = await self.redis.blpop(key, timeout=max(1, int(timeout)))

    if result:
      payload = json.loads(result[1])
      await self.redis.rpush(key, result[1])
      await self.redis.expire(key, 300)
      return payload

    return {"type": "try_again", "request_id": req_id, "queue_state": "active"}


class FileStore(RequestStore):
  """Cross-process store backed by a shared directory — no external server (Redis) needed.

  Works across separately-launched worker subprocesses on the same machine via the filesystem,
  so it's the local-dev/no-Redis option for the dynamic-worker path. Requests and futures are
  written as atomic JSON files (write-temp + rename); consumers poll and drain.
  """

  def __init__(self, root: str):
    self.root = root
    self.qroot = os.path.join(root, "queue")
    self.froot = os.path.join(root, "future")
    os.makedirs(self.qroot, exist_ok=True)
    os.makedirs(self.froot, exist_ok=True)

  def _qdir(self, model_id: str) -> str:
    return os.path.join(self.qroot, model_id.replace("/", "__"))

  def _write_atomic(self, path: str, data: dict[str, Any]) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
      json.dump(data, f)
    os.rename(tmp, path)

  def _drain_dir(self, d: str) -> list[dict[str, Any]]:
    if not os.path.isdir(d):
      return []
    batch = []
    for fn in sorted(fn for fn in os.listdir(d) if fn.endswith(".json")):
      p = os.path.join(d, fn)
      try:
        with open(p) as f:
          data = json.load(f)
        os.unlink(p)
        batch.append(data)
      except (FileNotFoundError, json.JSONDecodeError):
        continue
    return batch

  async def put_request(self, req_data: dict[str, Any]) -> None:
    d = self._qdir(req_data.get("model_id", "default"))
    os.makedirs(d, exist_ok=True)
    name = f"{time.time_ns()}_{req_data.get('req_id', '')}.json"
    await asyncio.to_thread(self._write_atomic, os.path.join(d, name), req_data)

  async def get_requests(self) -> list[dict[str, Any]]:
    deadline = time.time() + 2.0
    while time.time() < deadline:
      for model_dir in sorted(os.listdir(self.qroot)) if os.path.isdir(self.qroot) else []:
        batch = await asyncio.to_thread(self._drain_dir, os.path.join(self.qroot, model_dir))
        if batch:
          return batch
      await asyncio.sleep(0.1)
    return []

  async def get_requests_for_model(self, model_id: str) -> list[dict[str, Any]]:
    d = self._qdir(model_id)
    deadline = time.time() + 2.0
    while time.time() < deadline:
      batch = await asyncio.to_thread(self._drain_dir, d)
      if batch:
        return batch
      await asyncio.sleep(0.1)
    return []

  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return
    await asyncio.to_thread(self._write_atomic, os.path.join(self.froot, f"{req_id}.json"), result)

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    p = os.path.join(self.froot, f"{req_id}.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
      if os.path.exists(p):
        try:
          with open(p) as f:
            return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
          pass
      await asyncio.sleep(0.05)
    return {"type": "try_again", "request_id": req_id, "queue_state": "active"}


# Global singleton factory
_store_instance = None


def get_store() -> RequestStore:
  global _store_instance
  if _store_instance is None:
    redis_url = os.environ.get("REDIS_URL")
    store_dir = os.environ.get("OPEN_RL_STORE_DIR")
    if redis_url:
      print(f"[RequestStore] Initializing Redis backend at {redis_url} with RR Tenant Queues")
      _store_instance = RedisStore(redis_url)
    elif store_dir:
      print(f"[RequestStore] Initializing File backend at {store_dir} (cross-process, no Redis)")
      _store_instance = FileStore(store_dir)
    else:
      print("[RequestStore] Initializing In-Memory backend with RR Tenant Queues")
      _store_instance = InMemoryStore()
  return _store_instance
