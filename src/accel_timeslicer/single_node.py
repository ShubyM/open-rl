import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointRestorer
from .time_slicer import TimeSlicer
from .workload import WorkloadRef

logger = logging.getLogger(__name__)


@dataclass
class WorkloadState:
  connection_id: int | None
  workload: WorkloadRef
  checkpointed: bool = False
  failed: bool = False
  # The connection whose acquire holds the grant, while one does.
  holder: int | None = None


class SingleNodeTimeSlicer(TimeSlicer):
  """Grants accelerator turns. Claims are independent; within a claim one
  workload holds the grant at a time. A release checkpoints the workload off
  the devices and its next acquire restores it."""

  def __init__(self, restorer: CheckpointRestorer, scheduling_policy: str = "lrs"):
    self.restorer = restorer
    self.scheduling_policy = scheduling_policy.lower()
    self.workloads: dict[str, WorkloadState] = {}
    self.waiting_workloads: deque[str] = deque()
    # Per claim, the workload holding the grant (acquire..release).
    self.running: dict[str, str] = {}
    self.condition = asyncio.Condition()
    self.last_release_time: dict[str, float] = {}
    # How the server reaches each open connection with a line it did not ask for.
    self.connections: dict[int, Callable[[dict[str, Any]], None]] = {}

  def next_waiter(self, claim: str) -> str | None:
    """The earliest waiter under fifo, the least recently served otherwise."""
    waiting = [self.workloads[name].workload for name in self.waiting_workloads if self.workloads[name].workload.claim == claim]
    if not waiting:
      return None
    if self.scheduling_policy == "fifo" or len(waiting) == 1:
      return waiting[0].name
    return min(waiting, key=lambda w: self.last_release_time.get(w.name, 0.0)).name

  async def register(self, workload: WorkloadRef, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      state = self.workloads.get(workload.name)
      if state is None:
        self.workloads[workload.name] = WorkloadState(connection_id=connection_id, workload=workload)
      else:
        # A registration is a live process announcing itself, so a failure
        # recorded against an earlier process under this name is cleared.
        state.connection_id = connection_id
        state.workload = workload
        state.failed = False
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, workload: WorkloadRef, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      name, claim = workload.name, workload.claim
      state = self.workloads.get(name)
      if state is None:
        state = WorkloadState(connection_id=None, workload=workload)
        self.workloads[name] = state
      if state.failed:
        return {"ok": False, "error": f"workload {name} is failed"}
      if name in self.waiting_workloads or self.running.get(claim) == name:
        return {"ok": False, "error": f"workload {name} already has a pending or active acquire"}

      self.waiting_workloads.append(name)
      try:
        while name in self.waiting_workloads and (claim in self.running or self.next_waiter(claim) != name):
          await self.condition.wait()
      except BaseException:
        if name in self.waiting_workloads:
          self.waiting_workloads.remove(name)
        self.condition.notify_all()
        raise

      state = self.workloads.get(name)
      if state is None or state.failed or name not in self.waiting_workloads:
        self.clear_workload(name)
        self.condition.notify_all()
        return {"ok": False, "error": f"workload {name} is not available"}

      self.waiting_workloads.remove(name)
      self.running[claim] = name
      state.holder = connection_id
      self.condition.notify_all()

    if state.checkpointed:
      await self.run_restore(state)
      async with self.condition:
        state = self.workloads.get(name)
        if state is not None:
          state.checkpointed = False
        self.condition.notify_all()
    return {"ok": True}

  async def release(self, workload: WorkloadRef, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      name, claim = workload.name, workload.claim
      state = self.workloads.get(name)
      if state is None or self.running.get(claim) != name:
        return {"ok": False, "error": f"workload {name} does not hold an active acquire"}
      # A grant the workload took again over another connection, as a restarted launcher does for its boot turn, is
      # not this connection's to end. A caller in this process names no connection and is not checked.
      if connection_id is not None and state.holder != connection_id:
        return {"ok": False, "error": f"workload {name} is held by another connection; only the connection that acquired it can release it"}

    # The grant is held through the checkpoint, so the next waiter is not
    # granted until this workload is off the devices.
    try:
      checkpointed = await self.run_checkpoint(state)
    except Exception as exc:
      # A workload that could not be parked is still on the devices. It keeps
      # the grant so nobody is restored on top of its memory; the seat frees
      # when its process exits and the connection closes.
      logger.error("checkpoint failed for workload %s claim %s; it keeps the grant until its process exits: %s", name, claim, exc)
      error = f"checkpoint failed for workload {name}; it still holds the accelerator and must exit: {exc}"
      async with self.condition:
        state = self.workloads.get(name)
        if state is not None:
          state.failed = True
          state.checkpointed = False
          # Only the process that registered the workload can free the devices, by exiting, so it is told when another
          # connection's release failed, as a pool dispatcher's can for a sampler. One that released it itself has the reply.
          if state.connection_id != connection_id and (tell := self.connections.get(state.connection_id)):
            tell({"event": "FAULT", **workload.as_payload(), "error": error})
        self.condition.notify_all()
      return {"ok": False, "faulted": True, "error": error}

    async with self.condition:
      state = self.workloads.get(name)
      if state is not None:
        state.checkpointed = checkpointed is not False
      self.last_release_time[workload.name] = time.time()
      self.clear_workload(name)
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      name = workload.name
      if name not in self.workloads:
        return {"ok": False, "error": f"workload {name} is not registered"}

      self.clear_workload(name)
      del self.workloads[name]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    # A grant held over a connection other than the registering one is a lease another process took, the pool
    # dispatcher for a sampler. The workload is still on the devices, so it is released, which checkpoints it.
    async with self.condition:
      leased = [state.workload for state in self.workloads.values() if state.holder == connection_id and state.connection_id != connection_id]
    for workload in leased:
      logger.warning("connection holding workload %s claim %s closed; releasing it", workload.name, workload.claim)
      await self.release(workload, connection_id)
    async with self.condition:
      for name, state in self.workloads.items():
        if state.connection_id != connection_id:
          continue
        self.clear_workload(name)
        state.failed = True
        state.checkpointed = False
        state.connection_id = None
      self.condition.notify_all()

  def clear_workload(self, name: str) -> None:
    if name in self.waiting_workloads:
      self.waiting_workloads.remove(name)
    for claim, holder in list(self.running.items()):
      if holder == name:
        del self.running[claim]
    if (state := self.workloads.get(name)) is not None:
      state.holder = None

  async def run_checkpoint(self, state: WorkloadState) -> bool | None:
    """Parks the workload. Raises when the restorer could not, since the workload then still holds the devices."""
    workload = state.workload
    start = time.monotonic()
    checkpointed = await asyncio.to_thread(self.restorer.checkpoint, workload)
    if checkpointed is False:
      logger.info("released workload %s claim %s without checkpoint in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    else:
      logger.info("checkpointed workload %s claim %s in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    return checkpointed

  async def run_restore(self, state: WorkloadState) -> None:
    workload = state.workload
    start = time.monotonic()
    try:
      await asyncio.to_thread(self.restorer.restore, workload)
      logger.info("restored workload %s claim %s in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    except Exception as exc:
      logger.warning("restore failed for workload %s claim %s: %s", workload.name, workload.claim, exc)
