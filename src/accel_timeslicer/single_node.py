import asyncio
import logging
import time
from collections import deque
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

  async def acquire(self, workload: WorkloadRef) -> dict[str, Any]:
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
      self.condition.notify_all()

    if state.checkpointed:
      await self.run_restore(state)
      async with self.condition:
        state = self.workloads.get(name)
        if state is not None:
          state.checkpointed = False
        self.condition.notify_all()
    return {"ok": True}

  async def release(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      name, claim = workload.name, workload.claim
      state = self.workloads.get(name)
      if state is None or self.running.get(claim) != name:
        return {"ok": False, "error": f"workload {name} does not hold an active acquire"}

    # The grant is held through the checkpoint, so the next waiter is not
    # granted until this workload is off the devices.
    checkpointed = await self.run_checkpoint(state)

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

  async def run_checkpoint(self, state: WorkloadState) -> bool | None:
    workload = state.workload
    start = time.monotonic()
    try:
      checkpointed = await asyncio.to_thread(self.restorer.checkpoint, workload)
      if checkpointed is False:
        logger.info("released workload %s claim %s without checkpoint in %.2fs", workload.name, workload.claim, time.monotonic() - start)
      else:
        logger.info("checkpointed workload %s claim %s in %.2fs", workload.name, workload.claim, time.monotonic() - start)
      return checkpointed
    except Exception as exc:
      logger.warning("checkpoint failed for workload %s claim %s: %s", workload.name, workload.claim, exc)
      return False

  async def run_restore(self, state: WorkloadState) -> None:
    workload = state.workload
    start = time.monotonic()
    try:
      await asyncio.to_thread(self.restorer.restore, workload)
      logger.info("restored workload %s claim %s in %.2fs", workload.name, workload.claim, time.monotonic() - start)
    except Exception as exc:
      logger.warning("restore failed for workload %s claim %s: %s", workload.name, workload.claim, exc)
