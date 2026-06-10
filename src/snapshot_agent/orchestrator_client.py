"""Snapshot client backed by the llm-d accelerator orchestrator (cluster mode).

In a Kubernetes deployment the local snapshot agent is replaced by
llm-d-rl-time-slicing: a per-node privileged DaemonSet runs cuda-checkpoint and a
cluster-level orchestrator serializes GPU access per time-slice group. The
orchestrator discovers participants through pod labels (timeslice.io/group,
timeslice.io/job-id), so exclusivity is keyed by (job_id, group_id) rather than a
local pid, and there is no register/unregister handshake.

Upstream is building an official `timeslice` Python SDK (llm-d-rl-time-slicing
PR #25), but so far it only wraps the snapshot-agent API, which the orchestrator
drives — not workloads. Once that SDK grows an orchestrator client, this module
and the vendored stubs in snapshot_agent/orchestrator/ can be replaced by it.
"""

import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

# Acquire blocks while the orchestrator checkpoints the previous job and restores
# ours, which can take minutes for large contexts; keepalives stop idle proxies
# and load balancers from dropping the call mid-wait.
_CHANNEL_OPTIONS = [
  ("grpc.keepalive_time_ms", 30_000),
  ("grpc.keepalive_timeout_ms", 10_000),
  ("grpc.keepalive_permit_without_calls", 1),
]


class OrchestratorSnapshotClient:
  def __init__(self, address: str, job_id: str, group_id: str):
    self.address = address
    self.job_id = job_id
    self.group_id = group_id
    self._channel = None
    self._stub = None

  async def connect(self) -> None:
    if self._stub is not None:
      return
    # Imported lazily so the base install (no `cluster` extra) can still import
    # the snapshot_agent package.
    import grpc

    from snapshot_agent.orchestrator import accelerator_orchestrator_pb2_grpc as pb2_grpc

    self._channel = grpc.aio.insecure_channel(self.address, options=_CHANNEL_OPTIONS)
    self._stub = pb2_grpc.AcceleratorOrchestratorServiceStub(self._channel)

  async def close(self) -> None:
    if self._channel is not None:
      await self._channel.close()
    self._channel = None
    self._stub = None

  async def register(self, pid: int) -> dict[str, Any]:
    return {"ok": True}

  async def unregister(self, pid: int) -> dict[str, Any]:
    return {"ok": True}

  @asynccontextmanager
  async def acquire(self, pid: int) -> AsyncIterator[None]:
    del pid  # exclusivity is keyed by (job_id, group_id), not the local pid
    from snapshot_agent.orchestrator import accelerator_orchestrator_pb2 as pb2

    await self.connect()
    response = await self._stub.Acquire(pb2.AcquireRequest(job_id=self.job_id, group_id=self.group_id))
    if not response.success:
      raise RuntimeError(f"orchestrator denied Acquire for job {self.job_id} in group {self.group_id}")
    if response.context_restored or response.waited_ms:
      print(f"[SNAPSHOT] acquired group {self.group_id} after {response.waited_ms / 1000:.2f}s (context_restored={response.context_restored})")
    try:
      yield
    finally:
      try:
        await self._stub.Yield(pb2.YieldRequest(job_id=self.job_id, group_id=self.group_id))
      except Exception:
        # The orchestrator reconciles group state from pod/agent observation, so
        # a lost Yield delays the next waiter instead of corrupting anything.
        print(f"[SNAPSHOT] Yield failed for job {self.job_id} in group {self.group_id}")
        traceback.print_exc()
