# Liveness and lifecycle state for client sessions and dedicated FFT workers.
#
# Kept apart from the request store on purpose: RequestStore moves training
# requests and futures (the data plane), while this store answers "which client
# sessions are alive and which models do they use" (the control plane the
# gateway's worker supervisor acts on). Worker health is deliberately not
# tracked here: a crashed container restarts in place and reports its own
# request failures, and eval traffic relaunches missing workers.

import os
import time
from abc import ABC, abstractmethod

import redis.asyncio as redis


class LifecycleStore(ABC):
  # *** Client sessions ***

  @abstractmethod
  async def touch_session(self, session_id: str, ttl: float) -> None:
    """Create or refresh a client session; it expires ttl seconds after the last touch."""
    pass

  @abstractmethod
  async def session_alive(self, session_id: str) -> bool:
    pass

  @abstractmethod
  async def bind_session_model(self, session_id: str, model_id: str) -> None:
    """Record that session_id uses model_id, so the model's worker stays alive with the session."""
    pass

  @abstractmethod
  async def model_session_state(self, model_id: str) -> tuple[bool, bool]:
    """Return (has_bound_sessions, has_live_session) for model_id."""
    pass

  # *** Supervised worker registry ***

  @abstractmethod
  async def register_fft_model(self, model_id: str) -> None:
    """Add the model to the supervisor's watch list. Called on every (idempotent) launch."""
    pass

  @abstractmethod
  async def unregister_fft_model(self, model_id: str) -> None:
    pass

  @abstractmethod
  async def list_fft_models(self) -> list[str]:
    pass


class InMemoryLifecycleStore(LifecycleStore):
  def __init__(self):
    self.session_deadlines: dict[str, float] = {}
    self.model_sessions: dict[str, set[str]] = {}
    self.fft_models: set[str] = set()

  async def touch_session(self, session_id: str, ttl: float) -> None:
    self.session_deadlines[session_id] = time.monotonic() + ttl

  async def session_alive(self, session_id: str) -> bool:
    return time.monotonic() < self.session_deadlines.get(session_id, 0.0)

  async def bind_session_model(self, session_id: str, model_id: str) -> None:
    self.model_sessions.setdefault(model_id, set()).add(session_id)

  async def model_session_state(self, model_id: str) -> tuple[bool, bool]:
    sessions = self.model_sessions.get(model_id, set())
    live = any([await self.session_alive(session_id) for session_id in sessions])
    return bool(sessions), live

  async def register_fft_model(self, model_id: str) -> None:
    self.fft_models.add(model_id)

  async def unregister_fft_model(self, model_id: str) -> None:
    self.fft_models.discard(model_id)

  async def list_fft_models(self) -> list[str]:
    return sorted(self.fft_models)


class RedisLifecycleStore(LifecycleStore):
  def __init__(self, redis_url: str):
    self.redis = redis.from_url(redis_url, decode_responses=True, health_check_interval=2)
    self.fft_models_set = "open_rl:fft_models"

  async def touch_session(self, session_id: str, ttl: float) -> None:
    await self.redis.set(f"open_rl:session:{session_id}", "1", px=max(1, int(ttl * 1000)))

  async def session_alive(self, session_id: str) -> bool:
    return bool(await self.redis.exists(f"open_rl:session:{session_id}"))

  async def bind_session_model(self, session_id: str, model_id: str) -> None:
    await self.redis.sadd(f"open_rl:model_sessions:{model_id}", session_id)

  async def model_session_state(self, model_id: str) -> tuple[bool, bool]:
    # Dead sessions stay in the set on purpose: a session key can lapse during a
    # transient client stall and come back on the next heartbeat, so membership
    # must outlive the liveness key.
    sessions = await self.redis.smembers(f"open_rl:model_sessions:{model_id}")
    for session_id in sessions:
      if await self.session_alive(session_id):
        return True, True
    return bool(sessions), False

  async def register_fft_model(self, model_id: str) -> None:
    await self.redis.sadd(self.fft_models_set, model_id)

  async def unregister_fft_model(self, model_id: str) -> None:
    await self.redis.srem(self.fft_models_set, model_id)

  async def list_fft_models(self) -> list[str]:
    return sorted(await self.redis.smembers(self.fft_models_set))


# Global singleton factory, mirroring server.store.get_store()
_lifecycle_instance = None


def get_lifecycle_store() -> LifecycleStore:
  global _lifecycle_instance
  if _lifecycle_instance is None:
    redis_url = os.environ.get("REDIS_URL")
    _lifecycle_instance = RedisLifecycleStore(redis_url) if redis_url else InMemoryLifecycleStore()
  return _lifecycle_instance
