# This file contains the state management and request queue implementation for the Open-RL server, supporting both in-memory and Redis backends.

import asyncio
import json
import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from server.protocol import ClientSession, SamplerSnapshot


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

  @abstractmethod
  async def set_value(self, key: str, value: str, expires_seconds: int | None = None) -> None:
    """Store a simple string value by key."""
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
  async def list_values(self, prefix: str) -> dict[str, str]:
    """Return simple string values whose keys begin with ``prefix``."""
    pass

  @abstractmethod
  async def increment_value(self, key: str, amount: int = 1) -> int:
    """Atomically increment an integer value and return the new value."""
    pass

  @abstractmethod
  async def set_if_absent(self, key: str, value: str, expires_seconds: int | None = None) -> bool:
    """Atomically set a value only when the key does not exist."""
    pass

  @abstractmethod
  def lock(self, name: str) -> AbstractAsyncContextManager[None]:
    """Return a short-lived cross-task/process lock."""
    pass

  @abstractmethod
  async def queue_depths(self, model_id: str) -> dict[str, int]:
    """Return the currently queued training and sampling request counts."""
    pass

  @abstractmethod
  async def append_control_event(self, run_id: str, event: dict[str, Any]) -> str:
    """Append a bounded control-plane event and return its monotonic cursor."""
    pass

  @abstractmethod
  async def get_control_events(self, run_id: str, after: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    """Return control-plane events after ``after``, oldest first."""
    pass

  @abstractmethod
  async def list_control_run_ids(self) -> list[str]:
    """Return run ids that have emitted control-plane events."""
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
    self.kv_store: dict[str, str] = {}
    self.kv_expirations: dict[str, float] = {}
    self.locks: dict[str, asyncio.Lock] = {}
    self.control_events: dict[str, list[dict[str, Any]]] = {}
    self.control_event_sequence: dict[str, int] = {}

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

  async def set_value(self, key: str, value: str, expires_seconds: int | None = None) -> None:
    self.kv_store[key] = value
    if expires_seconds is None:
      self.kv_expirations.pop(key, None)
    else:
      self.kv_expirations[key] = time.time() + expires_seconds

  async def get_value(self, key: str) -> str | None:
    self._expire_value(key)
    return self.kv_store.get(key)

  def get_value_sync(self, key: str) -> str | None:
    return self.kv_store.get(key)

  async def delete_values(self, *keys: str) -> None:
    for k in keys:
      self.kv_store.pop(k, None)
      self.kv_expirations.pop(k, None)

  async def list_values(self, prefix: str) -> dict[str, str]:
    for key in list(self.kv_store):
      self._expire_value(key)
    return {key: value for key, value in self.kv_store.items() if key.startswith(prefix)}

  async def increment_value(self, key: str, amount: int = 1) -> int:
    async with self.lock(f"value:{key}"):
      current = int(await self.get_value(key) or "0")
      current += amount
      await self.set_value(key, str(current))
      return current

  async def set_if_absent(self, key: str, value: str, expires_seconds: int | None = None) -> bool:
    async with self.lock(f"value:{key}"):
      if await self.get_value(key) is not None:
        return False
      await self.set_value(key, value, expires_seconds=expires_seconds)
      return True

  @asynccontextmanager
  async def lock(self, name: str) -> AsyncIterator[None]:
    async with self.locks.setdefault(name, asyncio.Lock()):
      yield

  def _expire_value(self, key: str) -> None:
    expires_at = self.kv_expirations.get(key)
    if expires_at is not None and expires_at <= time.time():
      self.kv_store.pop(key, None)
      self.kv_expirations.pop(key, None)

  async def queue_depths(self, model_id: str) -> dict[str, int]:
    queue = self.queues.get(model_id)
    return {"training": queue.qsize() if queue is not None else 0, "sampling": 0}

  async def append_control_event(self, run_id: str, event: dict[str, Any]) -> str:
    sequence = self.control_event_sequence.get(run_id, 0) + 1
    self.control_event_sequence[run_id] = sequence
    cursor = str(sequence)
    stored = {**event, "id": event.get("id") or f"{run_id}:{cursor}", "cursor": cursor, "run_id": run_id}
    events = self.control_events.setdefault(run_id, [])
    events.append(stored)
    del events[:-1000]
    return cursor

  async def get_control_events(self, run_id: str, after: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    after_sequence = cursor_sequence(after)
    events = [event.copy() for event in self.control_events.get(run_id, []) if cursor_sequence(event.get("cursor")) > after_sequence]
    return events[: max(1, min(limit, 1000))]

  async def list_control_run_ids(self) -> list[str]:
    return sorted(self.control_events)


class RedisStore(RequestStore):
  def __init__(self, redis_url: str):
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2)
    import redis as sync_redis_mod

    self.sync_redis = sync_redis_mod.Redis.from_url(redis_url, decode_responses=True)
    self.active_list = "open_rl:active_tenants"
    # We also keep a set to guarantee O(1) deduplication before RPushing
    self.active_set = "open_rl:active_tenants_set"
    self.worker_launch_queue = "open_rl:worker_launch_queue"

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
    await self.redis.rpush(key, json.dumps(result))
    await self.redis.expire(key, 300)

  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    key = f"open_rl:future:{req_id}"

    # redis-py 8 defaults the client socket timeout to 5s, so a single BLPOP can
    # never block for the full long-poll window. Poll in slices shorter than the
    # socket timeout until the deadline so clients only see try_again when the
    # request genuinely outlived the window.
    deadline = time.monotonic() + timeout
    while True:
      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return {"type": "try_again", "request_id": req_id, "queue_state": "active"}
      try:
        result = await self.redis.blpop(key, timeout=min(3, max(1, int(remaining))))
      except RedisTimeoutError:
        result = None
      if result:
        payload = json.loads(result[1])
        await self.redis.rpush(key, result[1])
        await self.redis.expire(key, 300)
        return payload

  async def set_value(self, key: str, value: str, expires_seconds: int | None = None) -> None:
    await self.redis.set(key, value, ex=expires_seconds)

  async def get_value(self, key: str) -> str | None:
    return await self.redis.get(key)

  def get_value_sync(self, key: str) -> str | None:
    try:
      return self.sync_redis.get(key)
    except Exception:
      return None

  async def delete_values(self, *keys: str) -> None:
    if keys:
      await self.redis.delete(*keys)

  async def list_values(self, prefix: str) -> dict[str, str]:
    values: dict[str, str] = {}
    async for key in self.redis.scan_iter(match=f"{prefix}*", count=100):
      value = await self.redis.get(key)
      if value is not None:
        values[str(key)] = str(value)
    return values

  async def increment_value(self, key: str, amount: int = 1) -> int:
    return int(await self.redis.incrby(key, amount))

  async def set_if_absent(self, key: str, value: str, expires_seconds: int | None = None) -> bool:
    return bool(await self.redis.set(key, value, ex=expires_seconds, nx=True))

  @asynccontextmanager
  async def lock(self, name: str) -> AsyncIterator[None]:
    async with self.redis.lock(f"open_rl:lock:{name}", timeout=30, blocking_timeout=30):
      yield

  async def queue_depths(self, model_id: str) -> dict[str, int]:
    training, sampling = await self.redis.llen(f"open_rl:queue:{model_id}"), await self.redis.llen(f"open_rl:sampler_queue:{model_id}")
    return {"training": int(training), "sampling": int(sampling)}

  async def append_control_event(self, run_id: str, event: dict[str, Any]) -> str:
    sequence_key = f"open_rl:control:event_sequence:{run_id}"
    events_key = f"open_rl:control:events:{run_id}"
    sequence = int(await self.redis.incr(sequence_key))
    cursor = str(sequence)
    stored = {**event, "id": event.get("id") or f"{run_id}:{cursor}", "cursor": cursor, "run_id": run_id}
    payload = json.dumps(stored, separators=(",", ":"), default=str)
    async with self.redis.pipeline(transaction=False) as pipe:
      pipe.rpush(events_key, payload)
      pipe.ltrim(events_key, -1000, -1)
      pipe.sadd("open_rl:control:runs", run_id)
      await pipe.execute()
    return cursor

  async def get_control_events(self, run_id: str, after: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    raw_events = await self.redis.lrange(f"open_rl:control:events:{run_id}", 0, -1)
    after_sequence = cursor_sequence(after)
    decoded_events: list[dict[str, Any]] = []
    for raw in raw_events:
      try:
        event = json.loads(raw)
      except (TypeError, json.JSONDecodeError):
        continue
      decoded_events.append(event)
    # INCR assigns monotonic cursors before the non-transactional append
    # pipeline. Concurrent reporters can therefore reach RPUSH out of order;
    # cursor order remains the authoritative event order for watchers.
    decoded_events.sort(key=lambda event: cursor_sequence(event.get("cursor")))
    events: list[dict[str, Any]] = []
    for event in decoded_events:
      if cursor_sequence(event.get("cursor")) <= after_sequence:
        continue
      events.append(event)
      if len(events) >= max(1, min(limit, 1000)):
        break
    return events

  async def list_control_run_ids(self) -> list[str]:
    return sorted(str(run_id) for run_id in await self.redis.smembers("open_rl:control:runs"))


def cursor_sequence(cursor: object) -> int:
  try:
    return int(str(cursor or "0").rsplit(":", 1)[-1])
  except ValueError:
    return 0


CLIENT_SESSION_PREFIX = "open_rl:session:"
MODEL_REVISION_PREFIX = "open_rl:model_revision:"
REQUEST_CLAIM_PREFIX = "open_rl:request_claim:"
SAMPLER_ARTIFACT_PREFIX = "open_rl:sampler_artifact:"
SAMPLER_SNAPSHOT_PREFIX = "open_rl:sampler_snapshot:"
SAMPLER_IN_FLIGHT_PREFIX = "open_rl:sampler_in_flight:"


def client_session_key(session_id: str) -> str:
  return f"{CLIENT_SESSION_PREFIX}{session_id}"


async def put_client_session(store: RequestStore, session: ClientSession, ttl_seconds: int) -> None:
  await store.set_value(client_session_key(session.session_id), session.model_dump_json(), expires_seconds=ttl_seconds)


async def get_client_session(store: RequestStore, session_id: str) -> ClientSession | None:
  raw = await store.get_value(client_session_key(session_id))
  return ClientSession.model_validate_json(raw) if raw is not None else None


def model_revision_key(model_id: str) -> str:
  return f"{MODEL_REVISION_PREFIX}{model_id}"


async def get_model_revision(store: RequestStore, model_id: str) -> int:
  return int(await store.get_value(model_revision_key(model_id)) or "0")


async def bump_model_revision(store: RequestStore, model_id: str) -> int:
  return await store.increment_value(model_revision_key(model_id))


async def claim_request(store: RequestStore, request_id: str) -> bool:
  ttl_seconds = int(os.getenv("OPEN_RL_REQUEST_DEDUPE_SECONDS", "86400"))
  if ttl_seconds < 1:
    raise ValueError("OPEN_RL_REQUEST_DEDUPE_SECONDS must be positive")
  return await store.set_if_absent(f"{REQUEST_CLAIM_PREFIX}{request_id}", "1", expires_seconds=ttl_seconds)


async def release_request_claim(store: RequestStore, request_id: str) -> None:
  await store.delete_values(f"{REQUEST_CLAIM_PREFIX}{request_id}")


def sampler_snapshot_key(sampling_session_id: str) -> str:
  return f"{SAMPLER_SNAPSHOT_PREFIX}{sampling_session_id}"


def sampler_in_flight_key(sampling_session_id: str) -> str:
  return f"{SAMPLER_IN_FLIGHT_PREFIX}{sampling_session_id}"


def sampler_artifact_key(model_id: str, revision: int) -> str:
  return f"{SAMPLER_ARTIFACT_PREFIX}{model_id}:{revision}"


def sampler_revision_path(model_id: str, revision: int) -> str:
  return f"tinker://{model_id}/sampler_weights/revisions/{revision}"


def sampler_lock_name(model_id: str) -> str:
  return f"sampler:{model_id}"


async def get_sampler_artifact(store: RequestStore, model_id: str, revision: int) -> str | None:
  return await store.get_value(sampler_artifact_key(model_id, revision))


async def put_sampler_artifact(store: RequestStore, model_id: str, revision: int, storage_path: str) -> None:
  await store.set_value(sampler_artifact_key(model_id, revision), storage_path)


async def put_sampler_snapshot(store: RequestStore, snapshot: SamplerSnapshot) -> None:
  await store.set_value(sampler_snapshot_key(snapshot.sampling_session_id), snapshot.model_dump_json(exclude={"in_flight"}))


async def get_sampler_snapshot(store: RequestStore, sampling_session_id: str) -> SamplerSnapshot | None:
  raw = await store.get_value(sampler_snapshot_key(sampling_session_id))
  if raw is None:
    return None
  snapshot = SamplerSnapshot.model_validate_json(raw)
  snapshot.in_flight = max(0, int(await store.get_value(sampler_in_flight_key(sampling_session_id)) or "0"))
  return snapshot


async def list_sampler_snapshots(store: RequestStore, model_id: str | None = None) -> list[SamplerSnapshot]:
  snapshots: list[SamplerSnapshot] = []
  for raw in (await store.list_values(SAMPLER_SNAPSHOT_PREFIX)).values():
    try:
      snapshot = SamplerSnapshot.model_validate_json(raw)
    except ValueError:
      continue
    if model_id is not None and snapshot.model_id != model_id:
      continue
    snapshot.in_flight = max(0, int(await store.get_value(sampler_in_flight_key(snapshot.sampling_session_id)) or "0"))
    snapshots.append(snapshot)
  return snapshots


async def acquire_sampler_snapshot(store: RequestStore, sampling_session_id: str, now: float | None = None) -> SamplerSnapshot | None:
  snapshot = await get_sampler_snapshot(store, sampling_session_id)
  if snapshot is None:
    return None
  async with store.lock(sampler_lock_name(snapshot.model_id)):
    snapshot = await get_sampler_snapshot(store, sampling_session_id)
    if snapshot is None:
      return None
    if snapshot.expires_at is not None and snapshot.expires_at <= (time.time() if now is None else now):
      if snapshot.in_flight == 0:
        await store.delete_values(sampler_snapshot_key(sampling_session_id), sampler_in_flight_key(sampling_session_id))
      return None
    snapshot.in_flight = await store.increment_value(sampler_in_flight_key(sampling_session_id))
    return snapshot


async def release_sampler_snapshot(store: RequestStore, sampling_session_id: str) -> None:
  snapshot = await get_sampler_snapshot(store, sampling_session_id)
  if snapshot is None:
    return
  async with store.lock(sampler_lock_name(snapshot.model_id)):
    current = int(await store.get_value(sampler_in_flight_key(sampling_session_id)) or "0")
    if current <= 1:
      await store.delete_values(sampler_in_flight_key(sampling_session_id))
    else:
      await store.increment_value(sampler_in_flight_key(sampling_session_id), -1)


async def prune_sampler_snapshots(
  store: RequestStore,
  model_id: str,
  *,
  keep_ephemeral: int | None = None,
  now: float | None = None,
) -> list[str]:
  """Remove expired/old snapshot records and return newly orphaned artifact paths."""
  keep = int(os.getenv("OPEN_RL_SAMPLER_SNAPSHOT_RETENTION", "2")) if keep_ephemeral is None else keep_ephemeral
  if keep < 0:
    raise ValueError("OPEN_RL_SAMPLER_SNAPSHOT_RETENTION cannot be negative")
  current_time = time.time() if now is None else now
  async with store.lock(sampler_lock_name(model_id)):
    snapshots = await list_sampler_snapshots(store, model_id)
    ephemeral = sorted((snapshot for snapshot in snapshots if not snapshot.named), key=lambda snapshot: snapshot.created_at, reverse=True)
    retained_ephemeral = {snapshot.sampling_session_id for snapshot in ephemeral[:keep]}
    removed: list[SamplerSnapshot] = []
    for snapshot in snapshots:
      expired = snapshot.expires_at is not None and snapshot.expires_at <= current_time
      over_retention = not snapshot.named and snapshot.sampling_session_id not in retained_ephemeral
      if snapshot.in_flight == 0 and (expired or over_retention):
        removed.append(snapshot)
        await store.delete_values(
          sampler_snapshot_key(snapshot.sampling_session_id),
          sampler_in_flight_key(snapshot.sampling_session_id),
        )

    remaining_paths = {snapshot.storage_path for snapshot in await list_sampler_snapshots(store, model_id)}
    orphaned_paths: list[str] = []
    artifacts = await store.list_values(f"{SAMPLER_ARTIFACT_PREFIX}{model_id}:")
    for key, storage_path in artifacts.items():
      if storage_path not in remaining_paths:
        orphaned_paths.append(storage_path)
        await store.delete_values(key)
    return sorted(set(orphaned_paths))


def json_safe_control_value(value: Any, depth: int = 0) -> Any:
  """Keep lifecycle event details finite and JSON serializable."""
  if depth >= 8:
    return str(value)[:2048]
  if value is None or isinstance(value, bool | int | str):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  if isinstance(value, dict):
    return {str(key)[:256]: json_safe_control_value(item, depth + 1) for key, item in list(value.items())[:256]}
  if isinstance(value, list | tuple):
    return [json_safe_control_value(item, depth + 1) for item in value[:256]]
  return str(value)[:2048]


async def report_control_event(
  store: RequestStore,
  run_id: str | None,
  *,
  component: str,
  phase: str,
  message: str,
  status: str = "running",
  level: str = "info",
  duration_seconds: float | None = None,
  details: dict[str, Any] | None = None,
) -> str | None:
  """Publish a small, stable lifecycle event for the UI and agent CLI.

  Reporting must never become a second failure mode for a training run. Callers
  therefore get ``None`` if the observability backend is temporarily unavailable.
  """
  if not run_id:
    return None
  append_event = getattr(store, "append_control_event", None)
  if append_event is None:
    return None
  event: dict[str, Any] = {
    "timestamp": time.time(),
    "component": component,
    "phase": phase,
    "status": status,
    "level": level,
    "message": message,
  }
  if duration_seconds is not None:
    try:
      duration = float(duration_seconds)
    except (TypeError, ValueError):
      duration = None
    if duration is not None and math.isfinite(duration):
      event["duration_seconds"] = round(duration, 6)
  if details:
    event["details"] = json_safe_control_value(details)
  try:
    return await append_event(run_id, event)
  except Exception as exc:
    print(f"[CONTROL] Failed to report {component}/{phase} for {run_id}: {exc}")
    return None


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
