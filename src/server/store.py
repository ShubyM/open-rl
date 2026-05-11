# This file contains the state management and request queue implementation for the Open-RL server, supporting both in-memory and Redis backends.

import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import redis.asyncio as redis
from model_state import latest_training_state_alias, state_ref


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
  async def set_future(self, req_id: str, result: dict[str, Any]) -> None:
    """Resolve a future by its request ID."""
    pass

  @abstractmethod
  async def get_future(self, req_id: str, timeout: float) -> dict[str, Any] | None:
    """Block until the future resolves or the timeout is reached."""
    pass

  @abstractmethod
  async def publish_checkpoint(self, model_id: str, checkpoint_ref: str, metadata: dict[str, Any] | None = None) -> None:
    """Publish the latest durable checkpoint for a model."""
    pass

  @abstractmethod
  async def latest_checkpoint(self, model_id: str) -> dict[str, Any] | None:
    """Return the latest durable checkpoint for a model, if any."""
    pass

  @abstractmethod
  async def publish_model_state_alias(self, alias: str, state_id: str) -> None:
    """Point an alias at a durable model state."""
    pass

  @abstractmethod
  async def resolve_model_state_alias(self, alias: str) -> str | None:
    """Resolve a model state alias to its concrete state ID."""
    pass

  @abstractmethod
  async def publish_model_state(self, state_id: str, state: dict[str, Any]) -> None:
    """Publish a durable model state addressable by samplers and trainers."""
    pass

  @abstractmethod
  async def get_model_state(self, state_id: str) -> dict[str, Any] | None:
    """Return a durable model state by ID, if one has been published."""
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
    self.model_states: dict[str, dict[str, Any]] = {}
    self.model_state_aliases: dict[str, str] = {}

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

  async def publish_checkpoint(self, model_id: str, checkpoint_ref: str, metadata: dict[str, Any] | None = None) -> None:
    state = checkpoint_model_state(model_id, checkpoint_ref, metadata)
    await self.publish_model_state(state["state_id"], state)
    await self.publish_model_state_alias(latest_training_state_alias(model_id), state["state_id"])

  async def latest_checkpoint(self, model_id: str) -> dict[str, Any] | None:
    state = await self.get_model_state(latest_training_state_alias(model_id))
    return checkpoint_response(state) if state else None

  async def publish_model_state_alias(self, alias: str, state_id: str) -> None:
    self.model_state_aliases[alias] = state_id

  async def resolve_model_state_alias(self, alias: str) -> str | None:
    return self.model_state_aliases.get(alias)

  async def publish_model_state(self, state_id: str, state: dict[str, Any]) -> None:
    self.model_states[state_id] = state

  async def get_model_state(self, state_id: str) -> dict[str, Any] | None:
    state = self.model_states.get(state_id)
    if state:
      return state
    resolved = await self.resolve_model_state_alias(state_id)
    return self.model_states.get(resolved) if resolved else None


class RedisStore(RequestStore):
  def __init__(self, redis_url: str):
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2)
    self.active_list = "open_rl:active_tenants"
    # We also keep a set to guarantee O(1) deduplication before RPushing
    self.active_set = "open_rl:active_tenants_set"
    self.model_state_hash = "open_rl:model_states"
    self.model_state_alias_hash = "open_rl:model_state_aliases"

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

  async def publish_checkpoint(self, model_id: str, checkpoint_ref: str, metadata: dict[str, Any] | None = None) -> None:
    state = checkpoint_model_state(model_id, checkpoint_ref, metadata)
    await self.publish_model_state(state["state_id"], state)
    await self.publish_model_state_alias(latest_training_state_alias(model_id), state["state_id"])

  async def latest_checkpoint(self, model_id: str) -> dict[str, Any] | None:
    state = await self.get_model_state(latest_training_state_alias(model_id))
    return checkpoint_response(state) if state else None

  async def publish_model_state_alias(self, alias: str, state_id: str) -> None:
    await self.redis.hset(self.model_state_alias_hash, alias, state_id)

  async def resolve_model_state_alias(self, alias: str) -> str | None:
    return await self.redis.hget(self.model_state_alias_hash, alias)

  async def publish_model_state(self, state_id: str, state: dict[str, Any]) -> None:
    await self.redis.hset(self.model_state_hash, state_id, json.dumps(state))

  async def get_model_state(self, state_id: str) -> dict[str, Any] | None:
    payload = await self.redis.hget(self.model_state_hash, state_id)
    if payload:
      return json.loads(payload)
    resolved = await self.resolve_model_state_alias(state_id)
    if not resolved:
      return None
    payload = await self.redis.hget(self.model_state_hash, resolved)
    return json.loads(payload) if payload else None


def checkpoint_model_state(model_id: str, checkpoint_ref: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
  metadata = metadata or {}
  training_mode = metadata.get("training_mode", "lora")
  state = {
    "state_id": metadata.get("state_id") or checkpoint_ref,
    "model_id": model_id,
    "base_model": metadata.get("base_model", ""),
    "training_mode": training_mode,
    "version": int(metadata.get("version", 0) or 0),
    "optimizer_ref": metadata.get("optimizer_ref"),
    "delta_ref": metadata.get("delta_ref") or metadata.get("state_delta_ref"),
    "adapter_name": metadata.get("adapter_name"),
    "runtime_backend": metadata.get("runtime_backend") or metadata.get("inference_backend"),
    "checkpoint_metadata": metadata,
  }
  if training_mode == "lora":
    state["adapter_ref"] = metadata.get("adapter_ref") or checkpoint_ref
    state["full_weights_ref"] = metadata.get("full_weights_ref")
  else:
    state["adapter_ref"] = metadata.get("adapter_ref")
    state["full_weights_ref"] = metadata.get("full_weights_ref") or checkpoint_ref
  return {key: value for key, value in state.items() if value is not None}


def checkpoint_response(state: dict[str, Any]) -> dict[str, Any] | None:
  ref = state_ref(state)
  if not ref:
    return None
  return {"checkpoint_ref": ref, "metadata": state.get("checkpoint_metadata", {})}


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
