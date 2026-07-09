"""Idle-session reaper: tears down trainer workers when client heartbeats stop.

The tinker SDK heartbeats /api/v1/session_heartbeat every ~10s for the lifetime
of the client process, so a session whose heartbeats stop is a client that
exited or lost connectivity for good. Once a session is idle past
OPEN_RL_SESSION_IDLE_TIMEOUT_SEC, each of its models gets the same graceful
SHUTDOWN_SENTINEL that delete_model sends to the trainer queue; after
OPEN_RL_SESSION_TEARDOWN_GRACE_SEC the trainer worker is also force-shut-down
through the worker manager, so a wedged or never-started worker cannot outlive
its client.

Sampler workers are deliberately left running: their cold start is expensive
and their lifecycle will be managed separately (e.g. GPU memory snapshots).
Model metadata and checkpoints are also kept, so reaped models remain loadable
and sampleable.
"""

import asyncio
import os
import time
import traceback

from server.store import RequestStore
from server.worker_manager import WorkerManager

DEFAULT_IDLE_TIMEOUT_SEC = 180.0
DEFAULT_POLL_INTERVAL_SEC = 15.0
DEFAULT_TEARDOWN_GRACE_SEC = 30.0


def _env_float(name: str, default: float) -> float:
  value = os.getenv(name)
  return float(value) if value else default


class SessionReaper:
  def __init__(
    self,
    store: RequestStore,
    worker_manager: WorkerManager,
    idle_timeout_sec: float | None = None,
    poll_interval_sec: float | None = None,
    teardown_grace_sec: float | None = None,
  ):
    if idle_timeout_sec is None:
      idle_timeout_sec = _env_float("OPEN_RL_SESSION_IDLE_TIMEOUT_SEC", DEFAULT_IDLE_TIMEOUT_SEC)
    if poll_interval_sec is None:
      poll_interval_sec = _env_float("OPEN_RL_SESSION_REAPER_INTERVAL_SEC", DEFAULT_POLL_INTERVAL_SEC)
    if teardown_grace_sec is None:
      teardown_grace_sec = _env_float("OPEN_RL_SESSION_TEARDOWN_GRACE_SEC", DEFAULT_TEARDOWN_GRACE_SEC)
    self.store = store
    self.worker_manager = worker_manager
    self.idle_timeout_sec = idle_timeout_sec
    self.poll_interval_sec = poll_interval_sec
    self.teardown_grace_sec = teardown_grace_sec
    # session_id -> when the graceful sentinel was enqueued. In-process only:
    # after a gateway restart an idle session is simply detected again and gets
    # a second (idempotent) sentinel before the forced teardown.
    self._sentinel_sent_at: dict[str, float] = {}

  async def run(self) -> None:
    print(
      f"[REAPER] Session reaper started (idle_timeout={self.idle_timeout_sec:.0f}s, "
      f"poll={self.poll_interval_sec:.0f}s, grace={self.teardown_grace_sec:.0f}s)"
    )
    while True:
      try:
        await self.run_once()
      except asyncio.CancelledError:
        raise
      except Exception:
        traceback.print_exc()
      await asyncio.sleep(self.poll_interval_sec)

  async def run_once(self, now: float | None = None) -> None:
    now = time.time() if now is None else now
    for session_id, last_seen in (await self.store.list_sessions()).items():
      if now - last_seen <= self.idle_timeout_sec:
        # A heartbeat arrived after we flagged the session; the sentinel may
        # already have stopped its trainers, but never follow up with force.
        self._sentinel_sent_at.pop(session_id, None)
        continue
      await self._reap(session_id, now)

  async def _reap(self, session_id: str, now: float) -> None:
    model_ids = await self.store.get_session_models(session_id)
    if not model_ids:
      await self.store.delete_session(session_id)
      self._sentinel_sent_at.pop(session_id, None)
      return

    sentinel_sent_at = self._sentinel_sent_at.get(session_id)
    if sentinel_sent_at is None:
      print(f"[REAPER] Session {session_id} idle past {self.idle_timeout_sec:.0f}s; requesting trainer shutdown for models {model_ids}")
      for model_id in model_ids:
        await self.store.put_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id, "op": "shutdown_workers"})
      self._sentinel_sent_at[session_id] = now
      return

    if now - sentinel_sent_at < self.teardown_grace_sec:
      return

    print(f"[REAPER] Session {session_id} grace period elapsed; deleting trainer workers for models {model_ids}")
    for model_id in model_ids:
      await asyncio.to_thread(self.worker_manager.shutdown_trainer, model_id)
    await self.store.delete_session(session_id)
    self._sentinel_sent_at.pop(session_id, None)


def start_session_reaper(store: RequestStore, worker_manager: WorkerManager) -> asyncio.Task | None:
  """Start the reaper loop; returns None when disabled via OPEN_RL_SESSION_IDLE_TIMEOUT_SEC<=0."""
  reaper = SessionReaper(store, worker_manager)
  if reaper.idle_timeout_sec <= 0:
    print("[REAPER] Session reaper disabled (OPEN_RL_SESSION_IDLE_TIMEOUT_SEC<=0)")
    return None
  return asyncio.create_task(reaper.run())
