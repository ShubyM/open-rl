import asyncio
import inspect
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from accel_timeslicer.checkpoint import CudaCheckpointRestorer, NoopCheckpointRestorer
from accel_timeslicer.llmd import LlmDCheckpointRestorer
from accel_timeslicer.serve import start_tcp_time_slicer, start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import WorkloadRef


class RecordingRestorer:
  def __init__(self):
    self.calls: list[tuple[str, WorkloadRef]] = []

  def checkpoint(self, target: WorkloadRef) -> None:
    self.calls.append(("checkpoint", target))

  def restore(self, target: WorkloadRef) -> None:
    self.calls.append(("restore", target))

  def labels(self) -> list[tuple[str, str, str]]:
    return [(op, target.job_id, target.group) for op, target in self.calls]

  def simple_labels(self) -> list[tuple[str, str]]:
    return [(op, target.job_id) for op, target in self.calls]


class BlockingRestorer(RecordingRestorer):
  def __init__(self):
    super().__init__()
    self.checkpoint_started = threading.Event()
    self.finish_checkpoint = threading.Event()
    self.restore_started = threading.Event()
    self.finish_restore = threading.Event()
    self.block_checkpoint = False
    self.block_restore = False

  def checkpoint(self, target: WorkloadRef) -> None:
    super().checkpoint(target)
    if self.block_checkpoint:
      self.checkpoint_started.set()
      self.finish_checkpoint.wait(timeout=5.0)

  def restore(self, target: WorkloadRef) -> None:
    super().restore(target)
    if self.block_restore:
      self.restore_started.set()
      self.finish_restore.wait(timeout=5.0)


class NoSnapshotRestorer(RecordingRestorer):
  def checkpoint(self, target: WorkloadRef) -> bool:
    super().checkpoint(target)
    return False


class FlakyRestorer(RecordingRestorer):
  """Raises from checkpoint or restore while the corresponding flag is set."""

  def __init__(self):
    super().__init__()
    self.fail_checkpoint = False
    self.fail_restore = False

  def checkpoint(self, target: WorkloadRef) -> None:
    super().checkpoint(target)
    if self.fail_checkpoint:
      raise RuntimeError("checkpoint exploded")

  def restore(self, target: WorkloadRef) -> None:
    super().restore(target)
    if self.fail_restore:
      raise RuntimeError("restore exploded")


class SingleNodeTimeSlicerTest(unittest.IsolatedAsyncioTestCase):
  async def test_vanished_resident_counts_as_suspended(self) -> None:
    # A resident that finished and was torn down leaves no process behind.
    # Its suspension must be vacuous, not a failure that wedges the group
    # forever on a ghost.
    class FailingRestorer(RecordingRestorer):
      def checkpoint(self, target: WorkloadRef) -> None:
        raise RuntimeError("gRPC: connection refused")

    agent = SingleNodeTimeSlicer(FailingRestorer())
    ghost = WorkloadRef(job_id="ghost")
    joiner = WorkloadRef(job_id="joiner")
    await agent.register(ghost)
    await agent.register(joiner)
    self.assertTrue((await agent.acquire(ghost))["ok"])

    with patch("accel_timeslicer.single_node.workload_process_exists", return_value=False):
      self.assertTrue((await agent.release(ghost))["ok"])
      granted = await agent.acquire(joiner)
    self.assertTrue(granted["ok"], granted)

  async def test_live_resident_with_failing_suspension_stays_resident(self) -> None:
    # The vacuous path is only for the dead: while the process exists, a
    # failed suspension still fails closed and the resident keeps the device.
    class FailingRestorer(RecordingRestorer):
      def checkpoint(self, target: WorkloadRef) -> None:
        raise RuntimeError("snapshot timed out")

    agent = SingleNodeTimeSlicer(FailingRestorer())
    resident = WorkloadRef(job_id="resident")
    joiner = WorkloadRef(job_id="joiner")
    await agent.register(resident)
    await agent.register(joiner)
    self.assertTrue((await agent.acquire(resident))["ok"])
    self.assertTrue((await agent.release(resident))["ok"])

    with patch("accel_timeslicer.single_node.workload_process_exists", return_value=True):
      denied = await agent.acquire(joiner)
    self.assertFalse(denied["ok"])
    self.assertIn("suspension", denied["error"])

  async def test_agent_grants_only_one_active_process_at_a_time(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    blocked = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    await asyncio.sleep(0.05)
    self.assertFalse(blocked.done())

    release = await agent.release(WorkloadRef(job_id="101"))
    self.assertTrue(release["ok"])
    granted_b = await asyncio.wait_for(blocked, timeout=1.0)
    self.assertTrue(granted_b["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])
    self.assertEqual(agent.running, {"shared-accelerator": "shared-accelerator:202"})

  async def test_owner_members_take_turns_one_at_a_time(self) -> None:
    # Same owner means one fairness slot, never co-residency: V1 keeps exactly
    # one workload loaded, so even siblings hand off through suspension.
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    lora_a = WorkloadRef(job_id="lora-a", owner="qwen")
    lora_b = WorkloadRef(job_id="lora-b", owner="qwen")
    await agent.register(lora_a)
    await agent.register(lora_b)

    self.assertTrue((await agent.acquire(lora_a))["ok"])
    blocked_b = asyncio.create_task(agent.acquire(lora_b))
    await asyncio.sleep(0.05)
    self.assertFalse(blocked_b.done())

    self.assertTrue((await agent.release(lora_a))["ok"])
    self.assertTrue((await asyncio.wait_for(blocked_b, timeout=1.0))["ok"])
    self.assertEqual(agent.running, {"shared-accelerator": "shared-accelerator:lora-b"})
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "lora-a")])

  async def test_owners_take_turns_and_a_waiting_owner_blocks_new_entrants(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    lora_a = WorkloadRef(job_id="lora-a", owner="qwen")
    lora_b = WorkloadRef(job_id="lora-b", owner="qwen")
    fft = WorkloadRef(job_id="fft")
    for w in (lora_a, lora_b, fft):
      await agent.register(w)

    self.assertTrue((await agent.acquire(lora_a))["ok"])
    self.assertTrue((await agent.release(lora_a))["ok"])
    # The fft has now waited longest, so the next lora acquire queues behind
    # it rather than extending the lora owner's turn.
    blocked_fft = asyncio.create_task(agent.acquire(fft))
    await asyncio.sleep(0.05)
    blocked_lora = asyncio.create_task(agent.acquire(lora_b))
    await asyncio.sleep(0.05)
    self.assertTrue(blocked_fft.done())
    self.assertTrue((await blocked_fft)["ok"])
    self.assertFalse(blocked_lora.done())

    self.assertTrue((await agent.release(fft))["ok"])
    self.assertTrue((await asyncio.wait_for(blocked_lora, timeout=1.0))["ok"])

  async def test_groups_do_not_wait_on_each_other(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    claim_a = WorkloadRef(job_id="trainer-a", group="claim-a")
    claim_b = WorkloadRef(job_id="trainer-b", group="claim-b")
    await agent.register(claim_a)
    await agent.register(claim_b)

    self.assertTrue((await agent.acquire(claim_a))["ok"])
    granted = await asyncio.wait_for(agent.acquire(claim_b), timeout=1.0)
    self.assertTrue(granted["ok"])
    self.assertEqual(agent.running, {"claim-a": "claim-a:trainer-a", "claim-b": "claim-b:trainer-b"})

  async def test_first_acquire_is_cold_and_later_acquire_restores_after_checkpoint(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    # Lazy suspension: nothing moves until a different workload takes over.
    self.assertEqual(restorer.simple_labels(), [])

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])
    self.assertTrue((await agent.release(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])

    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("checkpoint", "202"), ("restore", "101")])

  async def test_release_with_no_successor_leaves_the_process_resident(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    release = await agent.release(WorkloadRef(job_id="101"))

    self.assertTrue(release["ok"])
    self.assertFalse(agent.running)
    self.assertFalse(agent.workloads["shared-accelerator:101"].checkpointed)
    self.assertFalse(agent.workloads["shared-accelerator:101"].failed)
    self.assertEqual(restorer.simple_labels(), [])

    # Re-acquiring while still resident costs nothing: no restore.
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertEqual(restorer.simple_labels(), [])

  async def test_suspension_without_snapshot_does_not_restore_later(self) -> None:
    restorer = NoSnapshotRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    workload = WorkloadRef(job_id="101")
    other = WorkloadRef(job_id="202")

    await agent.register(workload)
    await agent.register(other)
    self.assertTrue((await agent.acquire(workload))["ok"])
    self.assertTrue((await agent.release(workload))["ok"])

    # The handoff finds nothing on the devices to snapshot, so 101 must not be
    # "restored" when it comes back.
    self.assertTrue((await agent.acquire(other))["ok"])
    self.assertFalse(agent.workloads[workload.key].checkpointed)
    self.assertTrue((await agent.release(other))["ok"])
    self.assertTrue((await agent.acquire(workload))["ok"])

    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("checkpoint", "202")])

  async def test_a_grant_waits_for_the_previous_residents_suspension(self) -> None:
    restorer = BlockingRestorer()
    restorer.block_checkpoint = True
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])

    # The successor's acquire performs the handoff, and does not return until
    # the outgoing checkpoint has finished: fail-closed, never two loaded.
    acquire_b = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    checkpoint_started = await asyncio.to_thread(restorer.checkpoint_started.wait, 1.0)
    self.assertTrue(checkpoint_started)
    await asyncio.sleep(0.05)
    self.assertFalse(acquire_b.done())

    restorer.finish_checkpoint.set()

    self.assertTrue((await asyncio.wait_for(acquire_b, timeout=1.0))["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])

  async def test_checkpointed_process_is_not_granted_until_restore_finishes(self) -> None:
    restorer = BlockingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="202")))["ok"])

    restorer.block_restore = True
    acquire_a = asyncio.create_task(agent.acquire(WorkloadRef(job_id="101")))

    restore_started = await asyncio.to_thread(restorer.restore_started.wait, 1.0)
    self.assertTrue(restore_started)
    self.assertFalse(acquire_a.done())

    restorer.finish_restore.set()

    self.assertTrue((await asyncio.wait_for(acquire_a, timeout=1.0))["ok"])
    self.assertFalse(agent.workloads["shared-accelerator:101"].checkpointed)
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101"), ("checkpoint", "202"), ("restore", "101")])

  async def test_unregister_waiting_process_prevents_later_grant(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    acquire_b = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    await asyncio.sleep(0.05)
    self.assertFalse(acquire_b.done())

    self.assertTrue((await agent.unregister(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])

    result = await asyncio.wait_for(acquire_b, timeout=1.0)
    self.assertFalse(result["ok"])
    self.assertFalse(agent.running)

  async def test_duplicate_commands_return_explicit_errors(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    await agent.register(WorkloadRef(job_id="101"))

    self.assertTrue((await agent.register(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertFalse((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertFalse((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.unregister(WorkloadRef(job_id="101")))["ok"])
    self.assertFalse((await agent.unregister(WorkloadRef(job_id="101")))["ok"])

  async def test_waiters_are_granted_in_fifo_order(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    for pid in [101, 202, 303, 404]:
      await agent.register(WorkloadRef(job_id=str(pid)))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])

    grant_order: list[int] = []

    async def acquire_then_release(pid: int) -> None:
      workload = WorkloadRef(job_id=str(pid))
      await agent.acquire(workload)
      grant_order.append(pid)
      await agent.release(workload)

    waiters = []
    for pid in [303, 202, 404]:
      waiters.append(asyncio.create_task(acquire_then_release(pid)))
      await asyncio.sleep(0.01)

    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)

    self.assertEqual(grant_order, [303, 202, 404])

  async def test_failed_suspension_refuses_the_grant_and_keeps_the_resident(self) -> None:
    # The spec's fail-closed clause: a handoff that cannot suspend the
    # resident must not grant the next worker.
    restorer = FlakyRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])

    restorer.fail_checkpoint = True
    refused = await agent.acquire(WorkloadRef(job_id="202"))
    self.assertFalse(refused["ok"])
    self.assertEqual(agent.resident, {"shared-accelerator": "shared-accelerator:101"})
    self.assertFalse(agent.running)

    # A transient failure heals: the next attempt performs the handoff.
    restorer.fail_checkpoint = False
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertEqual(agent.resident, {"shared-accelerator": "shared-accelerator:202"})

  async def test_failed_restore_marks_the_worker_failed_and_frees_the_group(self) -> None:
    restorer = FlakyRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    for job in ("101", "202", "303"):
      await agent.register(WorkloadRef(job_id=job))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="202")))["ok"])

    restorer.fail_restore = True
    refused = await agent.acquire(WorkloadRef(job_id="101"))
    self.assertFalse(refused["ok"])
    self.assertTrue(agent.workloads["shared-accelerator:101"].failed)
    # The devices were vacated by 202's suspension, so the group stays open.
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="303")))["ok"])

  async def test_lifecycle_ops_cannot_interrupt_a_handoff(self) -> None:
    restorer = BlockingRestorer()
    restorer.block_checkpoint = True
    agent = SingleNodeTimeSlicer(restorer)
    for job in ("101", "202", "303"):
      await agent.register(WorkloadRef(job_id=job))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    acquire_b = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    self.assertTrue(await asyncio.to_thread(restorer.checkpoint_started.wait, 1.0))

    # Neither a stray release nor an unregister may reopen the group while
    # the handoff is in flight, and no third acquire may be granted.
    self.assertFalse((await agent.release(WorkloadRef(job_id="202")))["ok"])
    self.assertFalse((await agent.unregister(WorkloadRef(job_id="202")))["ok"])
    acquire_c = asyncio.create_task(agent.acquire(WorkloadRef(job_id="303")))
    await asyncio.sleep(0.05)
    self.assertFalse(acquire_c.done())

    restorer.finish_checkpoint.set()
    self.assertTrue((await asyncio.wait_for(acquire_b, timeout=1.0))["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])
    self.assertTrue((await agent.release(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await asyncio.wait_for(acquire_c, timeout=1.0))["ok"])

  async def test_connection_blip_does_not_forget_the_resident(self) -> None:
    # A dropped socket says nothing about device memory: the released-but-
    # resident worker must still be suspended when the next worker arrives.
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"), connection_id=7)
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    await agent.connection_closed(7)

    self.assertEqual(agent.resident, {"shared-accelerator": "shared-accelerator:101"})
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])

  async def test_reregister_clears_a_previous_failure(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    await agent.register(WorkloadRef(job_id="101"), connection_id=7)
    await agent.connection_closed(7)
    self.assertFalse((await agent.acquire(WorkloadRef(job_id="101")))["ok"])

    # The restarted process announces itself again and is servable.
    await agent.register(WorkloadRef(job_id="101"), connection_id=8)
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])

  async def test_cancelled_acquire_does_not_wedge_the_group(self) -> None:
    restorer = BlockingRestorer()
    restorer.block_checkpoint = True
    agent = SingleNodeTimeSlicer(restorer)
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    acquire_b = asyncio.create_task(agent.acquire(WorkloadRef(job_id="202")))
    self.assertTrue(await asyncio.to_thread(restorer.checkpoint_started.wait, 1.0))

    acquire_b.cancel()
    restorer.finish_checkpoint.set()
    with self.assertRaises(asyncio.CancelledError):
      await acquire_b

    restorer.block_checkpoint = False
    self.assertTrue((await asyncio.wait_for(agent.acquire(WorkloadRef(job_id="202")), timeout=1.0))["ok"])

  async def test_noop_backend_keeps_the_handoff_protocol_without_moving_state(self) -> None:
    # For kind and CPU-only CI: turns still pass one at a time, and a
    # "checkpointed" workload is restored on its next turn -- the calls just
    # do nothing physical.
    agent = SingleNodeTimeSlicer(NoopCheckpointRestorer())
    await agent.register(WorkloadRef(job_id="101"))
    await agent.register(WorkloadRef(job_id="202"))

    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.release(WorkloadRef(job_id="101")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue(agent.workloads["shared-accelerator:101"].checkpointed)
    self.assertTrue((await agent.release(WorkloadRef(job_id="202")))["ok"])
    self.assertTrue((await agent.acquire(WorkloadRef(job_id="101")))["ok"])
    self.assertEqual(agent.resident, {"shared-accelerator": "shared-accelerator:101"})

  async def test_register_can_use_stable_snapshot_id_for_backend_calls(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    workload = WorkloadRef(job_id="job-a")
    successor = WorkloadRef(job_id="job-b")
    await agent.register(workload)
    await agent.register(successor)

    self.assertTrue((await agent.acquire(workload))["ok"])
    self.assertTrue((await agent.release(workload))["ok"])
    # The handoff names the outgoing workload by its stable job id.
    self.assertTrue((await agent.acquire(successor))["ok"])

    self.assertEqual(restorer.simple_labels(), [("checkpoint", "job-a")])


class SingleNodeTimeSlicerSocketTest(unittest.IsolatedAsyncioTestCase):
  async def test_persistent_socket_clients_alternate(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "accel-timeslicer.sock")
      server = await start_time_slicer(agent, socket_path)
      client_a = SocketTimeSlicerClient(socket_path)
      client_b = SocketTimeSlicerClient(socket_path)
      try:
        await client_a.register(WorkloadRef(job_id="101"))
        await client_b.register(WorkloadRef(job_id="202"))

        async with client_a.acquire(WorkloadRef(job_id="101")):
          blocked = asyncio.create_task(acquire_once(client_b, WorkloadRef(job_id="202")))
          await asyncio.sleep(0.05)
          self.assertFalse(blocked.done())

        self.assertEqual(await asyncio.wait_for(blocked, timeout=1.0), "202")
        # 101 was suspended on the handoff; 202 stays resident after its
        # release because nobody follows it.
        self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])
      finally:
        await client_a.close()
        await client_b.close()
        server.close()
        await server.wait_closed()

  async def test_closing_active_socket_marks_run_failed(self) -> None:
    agent = SingleNodeTimeSlicer(RecordingRestorer())
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "accel-timeslicer.sock")
      server = await start_time_slicer(agent, socket_path)
      client = SocketTimeSlicerClient(socket_path)
      try:
        await client.register(WorkloadRef(job_id="101"))
        await client.request({"command": "ACQUIRE", "job_id": "101"})
        await client.close()
        await asyncio.sleep(0.05)

        self.assertFalse(agent.running)
        self.assertTrue(agent.workloads["shared-accelerator:101"].failed)
      finally:
        server.close()
        await server.wait_closed()

  async def test_tcp_clients_share_single_node_agent(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    server = await start_tcp_time_slicer(agent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    client_a = SocketTimeSlicerClient(host="127.0.0.1", port=port)
    client_b = SocketTimeSlicerClient(host="127.0.0.1", port=port)
    try:
      await client_a.register(WorkloadRef(job_id="101"))
      await client_b.register(WorkloadRef(job_id="202"))

      async with client_a.acquire(WorkloadRef(job_id="101")):
        blocked = asyncio.create_task(acquire_once(client_b, WorkloadRef(job_id="202")))
        await asyncio.sleep(0.05)
        self.assertFalse(blocked.done())

      self.assertEqual(await asyncio.wait_for(blocked, timeout=1.0), "202")
      self.assertEqual(restorer.simple_labels(), [("checkpoint", "101")])
    finally:
      await client_a.close()
      await client_b.close()
      server.close()
      await server.wait_closed()

  async def test_env_client_registers_time_slice_job_id_and_group(self) -> None:
    restorer = RecordingRestorer()
    agent = SingleNodeTimeSlicer(restorer)
    server = await start_tcp_time_slicer(agent, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    env = {
      "OPEN_RL_ACCEL_TIMESLICER_HOST": "127.0.0.1",
      "OPEN_RL_ACCEL_TIMESLICER_PORT": str(port),
      "OPEN_RL_TIME_SLICE_JOB_ID": "job-a",
      # The launcher's stamp wins: under the scheduler this is the claim name.
      "OPEN_RL_TIME_SLICE_GROUP": "claim-a",
    }
    with patch.dict("os.environ", env, clear=True):
      client = time_slicer_client_from_env()
      workload = workload_from_env(101)
    try:
      await client.register(workload)
      async with client.acquire(workload):
        pass
      # A successor in the same group forces the handoff, which must name the
      # env-derived job id and group to the backend.
      successor = WorkloadRef(job_id="job-b", group="claim-a")
      await client.register(successor)
      async with client.acquire(successor):
        pass

      self.assertEqual(restorer.labels(), [("checkpoint", "job-a", "claim-a")])
    finally:
      await client.close()
      server.close()
      await server.wait_closed()

  async def test_workload_from_env_prefers_the_launchers_group_and_owner(self) -> None:
    env = {"OPEN_RL_TIME_SLICE_GROUP": "claim-a", "OPEN_RL_TIME_SLICE_OWNER": "qwen"}
    with patch.dict("os.environ", env, clear=True):
      workload = workload_from_env(101, job_id="model-a")

    self.assertEqual(workload.job_id, "model-a")
    self.assertEqual(workload.group, "claim-a")
    self.assertEqual(workload.owner, "qwen")

  async def test_workload_from_env_falls_back_to_the_callers_group(self) -> None:
    with patch.dict("os.environ", {}, clear=True):
      workload = workload_from_env(101, job_id="model-a", group="trainers")

    self.assertEqual(workload.job_id, "model-a")
    self.assertEqual(workload.group, "trainers")
    self.assertEqual(workload.owner, "")


class CudaCheckpointRestorerTest(unittest.TestCase):
  def test_checkpoint_discovers_pids_from_workload_identity(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with (
      patch.object(restorer, "discover_pids", return_value=[101, 202]),
      patch.object(restorer, "run_cuda_checkpoint") as run_cuda_checkpoint,
    ):
      restorer.checkpoint(workload)

    self.assertEqual(
      [call.args[0] for call in run_cuda_checkpoint.call_args_list],
      [
        ["--action", "lock", "--pid", "101"],
        ["--action", "lock", "--pid", "202"],
        ["--action", "checkpoint", "--pid", "101"],
        ["--action", "checkpoint", "--pid", "202"],
      ],
    )

  def test_restore_uses_checkpointed_pids_without_rediscovery(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with (
      patch.object(restorer, "discover_pids", return_value=[101, 202]) as discover_pids,
      patch.object(restorer, "run_cuda_checkpoint") as run_cuda_checkpoint,
    ):
      restorer.checkpoint(workload)
      discover_pids.side_effect = AssertionError("restore must not query nvidia-smi after checkpoint")
      restorer.restore(workload)

    self.assertEqual(discover_pids.call_count, 1)
    self.assertEqual(
      [call.args[0] for call in run_cuda_checkpoint.call_args_list],
      [
        ["--action", "lock", "--pid", "101"],
        ["--action", "lock", "--pid", "202"],
        ["--action", "checkpoint", "--pid", "101"],
        ["--action", "checkpoint", "--pid", "202"],
        ["--action", "restore", "--pid", "101"],
        ["--action", "restore", "--pid", "202"],
        ["--action", "unlock", "--pid", "101"],
        ["--action", "unlock", "--pid", "202"],
      ],
    )
    self.assertEqual(restorer.checkpointed_pids, {})

  def test_checkpoint_with_no_gpu_pids_skips_snapshot(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with (
      patch.object(restorer, "discover_pids", return_value=[]),
      patch.object(restorer, "run_cuda_checkpoint") as run_cuda_checkpoint,
    ):
      self.assertFalse(restorer.checkpoint(workload))

    run_cuda_checkpoint.assert_not_called()
    self.assertEqual(restorer.checkpointed_pids, {})

  def test_restore_without_prior_checkpoint_fails(self) -> None:
    restorer = CudaCheckpointRestorer("cuda-checkpoint")
    workload = WorkloadRef(job_id="trainer-model-a")

    with self.assertRaisesRegex(RuntimeError, "no checkpointed PIDs"):
      restorer.restore(workload)

  def test_process_discovery_checks_gpu_pids_and_process_group_leaders(self) -> None:
    from accel_timeslicer.process_discovery import discover_workload_gpu_pids, workload_root_pids

    workload = WorkloadRef(job_id="trainer-model-a")

    def environ(pid: int) -> dict[str, str]:
      if pid == 11:
        return {"OPEN_RL_TIME_SLICE_JOB_ID": "trainer-model-a", "OPEN_RL_TIME_SLICE_GROUP": "shared-accelerator"}
      if pid == 99:
        return {"OPEN_RL_TIME_SLICE_JOB_ID": "other"}
      return {}

    def pgid(pid: int) -> int | None:
      return {12: 11, 99: 98}.get(pid)

    with (
      patch("accel_timeslicer.process_discovery.process_environ", side_effect=environ),
      patch("accel_timeslicer.process_discovery.process_group_id", side_effect=pgid),
      patch("accel_timeslicer.process_discovery.nvidia_smi_compute_pids", return_value=[12, 99]),
    ):
      self.assertEqual(discover_workload_gpu_pids(workload), [12])
      self.assertEqual(workload_root_pids(workload), [11])


class LlmDCheckpointRestorerTest(unittest.TestCase):
  def test_installed_llmd_client_matches_checkpoint_restorer_contract(self) -> None:
    try:
      from timeslice.snapshot_agent import SnapshotAgentClient
      from timeslice.snapshot_agent.types import GetOperationResponse
    except ModuleNotFoundError as exc:
      if exc.name and exc.name.split(".")[0] == "timeslice":
        self.skipTest("timeslice cluster extra is not installed")
      raise

    for name in ["snapshot_and_wait", "restore_and_wait"]:
      parameters = inspect.signature(getattr(SnapshotAgentClient, name)).parameters
      self.assertEqual(
        list(parameters)[:5],
        ["self", "job_id", "group", "poll_interval_sec", "backend_config"],
      )
      self.assertEqual(parameters["poll_interval_sec"].default, 1.0)

    self.assertTrue(callable(SnapshotAgentClient.close))
    self.assertEqual(GetOperationResponse.__annotations__["status"], str)
    self.assertIn("error", GetOperationResponse.__annotations__)

  def test_checkpoint_and_restore_wait_for_llmd_operations_by_job_id(self) -> None:
    class Client:
      def __init__(self):
        self.calls = []

      def snapshot_and_wait(self, job_id, group="", poll_interval_sec=1.0, backend_config=None):
        self.calls.append(("snapshot", job_id, group, poll_interval_sec, backend_config))
        return SimpleNamespace(status="OPERATION_STATUS_COMPLETE")

      def restore_and_wait(self, job_id, group="", poll_interval_sec=1.0, backend_config=None):
        self.calls.append(("restore", job_id, group, poll_interval_sec, backend_config))
        return SimpleNamespace(status="OPERATION_STATUS_COMPLETE")

      def close(self):
        pass

    client = Client()
    cuda_config = object()
    restorer = LlmDCheckpointRestorer(client, cuda_config, 0.25)

    target = WorkloadRef(job_id="job-a")
    restorer.checkpoint(target)
    restorer.restore(target)

    self.assertEqual(
      client.calls,
      [
        ("snapshot", "job-a", "shared-accelerator", 0.25, cuda_config),
        ("restore", "job-a", "shared-accelerator", 0.25, cuda_config),
      ],
    )

  def test_workload_requires_job_id(self) -> None:
    class Client:
      def snapshot_and_wait(self, *_args, **_kwargs):
        raise AssertionError("must not call llm-d without job id")

      def restore_and_wait(self, *_args, **_kwargs):
        raise AssertionError("must not call llm-d without job id")

      def close(self):
        pass

    with self.assertRaisesRegex(ValueError, "job_id"):
      WorkloadRef(job_id="")


async def acquire_once(client: SocketTimeSlicerClient, workload: WorkloadRef) -> str:
  async with client.acquire(workload):
    return workload.job_id


if __name__ == "__main__":
  unittest.main()
