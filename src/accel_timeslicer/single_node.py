import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .checkpoint import CheckpointRestorer
from .process_discovery import workload_process_exists
from .time_slicer import TimeSlicer
from .workload import WorkloadRef

logger = logging.getLogger(__name__)


class HandoffError(Exception):
  """A suspension or restore did not complete; the devices were not handed off."""


@dataclass
class WorkloadState:
  connection_id: int | None
  workload: WorkloadRef
  checkpointed: bool = False
  failed: bool = False


class SingleNodeTimeSlicer(TimeSlicer):
  """Grants accelerator turns. Groups are independent; within a group exactly
  one workload's state is on the devices: the resident worker.

  The resident is the whole model. Only a completed handoff changes who the
  resident is: suspend the current resident, then restore the next worker. A
  failed suspension leaves the resident in place and refuses the grant (fail
  closed); a failed restore marks the incoming worker failed and leaves the
  devices empty. Lifecycle events -- dropped connections, unregisters -- never
  guess at device state: the next handoff discovers it, because checkpointing
  a process that is gone finds nothing to move.

  Owners are the fairness unit: turns rotate to the least-recently-served
  owner, so an owner never gets extra turns for having more processes. A
  workload that names no owner is an owner of one.
  """

  def __init__(self, restorer: CheckpointRestorer, scheduling_policy: str = "lrs"):
    self.restorer = restorer
    self.scheduling_policy = scheduling_policy.lower()
    self.workloads: dict[str, WorkloadState] = {}
    self.waiting_workloads: deque[str] = deque()
    # Per group: who holds the grant (acquire..release), which groups have a
    # handoff in flight, and whose state is resident on the devices. Keying
    # by group makes "one per group" structural rather than a convention.
    self.running: dict[str, str] = {}
    self.handing_off: set[str] = set()
    self.resident: dict[str, str] = {}
    self.condition = asyncio.Condition()
    self.last_release_time: dict[str, float] = {}

  def next_waiter(self, group: str) -> str | None:
    """The workload whose turn is next: the earliest waiter under fifo, the
    one whose owner was served least recently otherwise."""
    waiting = [self.workloads[key].workload for key in self.waiting_workloads if self.workloads[key].workload.group == group]
    if not waiting:
      return None
    if self.scheduling_policy == "fifo" or len(waiting) == 1:
      return waiting[0].key
    return min(waiting, key=lambda w: self.last_release_time.get(w.owner_key, 0.0)).key

  def may_run(self, workload: WorkloadRef) -> bool:
    # A grant or an in-flight handoff closes the whole group; among the
    # waiters, the least-recently-served owner goes first, so a busy owner
    # cannot starve the others by re-acquiring.
    if workload.group in self.running or workload.group in self.handing_off:
      return False
    return self.next_waiter(workload.group) == workload.key

  async def register(self, workload: WorkloadRef, connection_id: int | None = None) -> dict[str, Any]:
    async with self.condition:
      state = self.workloads.get(workload.key)
      if state is None:
        self.workloads[workload.key] = WorkloadState(connection_id=connection_id, workload=workload)
      else:
        # A registration is a live process announcing itself: a failure
        # recorded against a previous incarnation does not outlive it.
        state.connection_id = connection_id
        state.workload = workload
        state.failed = False
      self.condition.notify_all()
      return {"ok": True}

  async def acquire(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      key, group = workload.key, workload.group
      state = self.workloads.get(key)
      if state is None:
        state = WorkloadState(connection_id=None, workload=workload)
        self.workloads[key] = state
      if state.failed:
        return {"ok": False, "error": f"workload {key} is failed"}
      if key in self.waiting_workloads or self.running.get(group) == key:
        return {"ok": False, "error": f"workload {key} already has a pending or active acquire"}

      self.waiting_workloads.append(key)
      try:
        while key in self.waiting_workloads and not self.may_run(workload):
          await self.condition.wait()
      except BaseException:
        if key in self.waiting_workloads:
          self.waiting_workloads.remove(key)
        self.condition.notify_all()
        raise

      state = self.workloads.get(key)
      if state is None or state.failed or key not in self.waiting_workloads:
        if key in self.waiting_workloads:
          self.waiting_workloads.remove(key)
        self.condition.notify_all()
        return {"ok": False, "error": f"workload {key} is not available"}

      self.waiting_workloads.remove(key)
      self.running[group] = key
      outgoing = self.resident.get(group)
      if outgoing == key:
        outgoing = None
      if outgoing is None and not state.checkpointed:
        # Already resident, or the devices are empty: nothing to move.
        self.resident[group] = key
        self.condition.notify_all()
        return {"ok": True}
      self.handing_off.add(group)

    try:
      await self.handoff(group, outgoing, state)
    except HandoffError as exc:
      async with self.condition:
        self.running.pop(group, None)
        return {"ok": False, "error": str(exc)}
    except BaseException:
      # Cancellation mid-handoff: give the grant back so the group is not
      # wedged. The resident map still tells the truth for the next handoff.
      async with self.condition:
        self.running.pop(group, None)
      raise
    finally:
      async with self.condition:
        self.handing_off.discard(group)
        self.condition.notify_all()

    async with self.condition:
      self.resident[group] = key
      if state.failed:
        # The process died while we were restoring it. Its state may be on
        # the devices, so the resident record stands; the grant does not.
        self.running.pop(group, None)
        return {"ok": False, "error": f"workload {key} is failed"}
      return {"ok": True}

  async def handoff(self, group: str, outgoing: str | None, incoming: WorkloadState) -> None:
    """Suspend the resident, then restore the incoming worker. Fail closed:
    an incomplete suspension leaves the resident in place, and a failed
    restore marks the incoming worker failed with the devices left empty."""
    if outgoing is not None:
      out_state = self.workloads.get(outgoing)
      if out_state is not None:
        start = time.monotonic()
        try:
          snapshot = await asyncio.to_thread(self.restorer.checkpoint, out_state.workload)
        except Exception as exc:
          # A resident whose process no longer exists -- finished and torn
          # down, or killed -- has nothing to suspend and holds no device
          # memory. Counting that as a failure wedges the group forever on a
          # ghost; a departed resident counts as suspended. When existence
          # cannot be established, assume alive and keep failing closed.
          try:
            vanished = not await asyncio.to_thread(workload_process_exists, out_state.workload)
          except Exception:
            vanished = False
          if vanished:
            logger.warning("resident %s vanished; treating its suspension as vacuous: %s", outgoing, exc)
            snapshot = False
          else:
            logger.warning("suspension of %s failed; it stays resident: %s", outgoing, exc)
            raise HandoffError(f"suspension of {outgoing} failed: {exc}") from exc
        logger.info("suspended %s in %.2fs%s", outgoing, time.monotonic() - start, "" if snapshot is not False else " (nothing to checkpoint)")
        async with self.condition:
          live = self.workloads.get(outgoing)
          if live is not None:
            live.checkpointed = snapshot is not False
      async with self.condition:
        self.evict_resident(group, outgoing)

    if incoming.checkpointed:
      start = time.monotonic()
      try:
        await asyncio.to_thread(self.restorer.restore, incoming.workload)
      except Exception as exc:
        logger.warning("restore of %s failed; marking it failed: %s", incoming.workload.key, exc)
        async with self.condition:
          incoming.failed = True
          incoming.checkpointed = False
        raise HandoffError(f"restore of {incoming.workload.key} failed: {exc}") from exc
      logger.info("restored %s in %.2fs", incoming.workload.key, time.monotonic() - start)
      incoming.checkpointed = False

  async def release(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      key, group = workload.key, workload.group
      if self.running.get(group) != key or group in self.handing_off:
        return {"ok": False, "error": f"workload {key} does not hold an active acquire"}

      # No checkpoint here: the workload stays resident until another one is
      # granted the group. If nobody ever is, it never pays the transfer.
      del self.running[group]
      self.last_release_time[workload.owner_key] = time.time()
      self.condition.notify_all()
      return {"ok": True}

  async def unregister(self, workload: WorkloadRef) -> dict[str, Any]:
    async with self.condition:
      key, group = workload.key, workload.group
      if key not in self.workloads:
        return {"ok": False, "error": f"workload {key} is not registered"}
      if self.running.get(group) == key and group in self.handing_off:
        return {"ok": False, "error": f"workload {key} has an acquire in flight"}

      if key in self.waiting_workloads:
        self.waiting_workloads.remove(key)
      if self.running.get(group) == key:
        del self.running[group]
      # An unregister is a deliberate teardown: the process is going away and
      # its device memory with it, so the residency record goes too.
      self.evict_resident(group, key)
      del self.workloads[key]
      self.condition.notify_all()
      return {"ok": True}

  async def connection_closed(self, connection_id: int) -> None:
    async with self.condition:
      for key, state in self.workloads.items():
        if state.connection_id != connection_id:
          continue
        group = state.workload.group
        if key in self.waiting_workloads:
          self.waiting_workloads.remove(key)
        if self.running.get(group) == key and group not in self.handing_off:
          del self.running[group]
        # The resident record is deliberately left alone: a dropped socket
        # does not mean the process or its device memory is gone. The next
        # handoff checkpoints it if it is still there, or finds nothing if
        # it is not.
        state.failed = True
        state.checkpointed = False
        state.connection_id = None
      self.condition.notify_all()

  def evict_resident(self, group: str, key: str) -> None:
    """Forget that key's state occupies group's devices, if it still does."""
    if self.resident.get(group) == key:
      del self.resident[group]
