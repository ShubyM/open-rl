import asyncio
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._server_fixture import SERVER_DIR

sys.path.insert(0, str(SERVER_DIR))
redis_stub = types.ModuleType("redis")
redis_asyncio_stub = types.ModuleType("redis.asyncio")
redis_stub.asyncio = redis_asyncio_stub
sys.modules.setdefault("redis", redis_stub)
sys.modules.setdefault("redis.asyncio", redis_asyncio_stub)

from store import InMemoryStore  # noqa: E402
from trainer_scheduler import TrainerScheduler, TrainerSchedulerClient, serve_scheduler  # noqa: E402
from worker_launcher import launch_worker  # noqa: E402


class RecordingRestorer:
  def __init__(self):
    self.calls: list[tuple[str, int]] = []

  def checkpoint(self, pid: int) -> None:
    self.calls.append(("checkpoint", pid))

  def restore(self, pid: int) -> None:
    self.calls.append(("restore", pid))


class BlockingRestorer(RecordingRestorer):
  def __init__(self):
    super().__init__()
    self.checkpoint_started = threading.Event()
    self.finish_checkpoint = threading.Event()
    self.restore_started = threading.Event()
    self.finish_restore = threading.Event()
    self.block_checkpoint = False
    self.block_restore = False

  def checkpoint(self, pid: int) -> None:
    super().checkpoint(pid)
    if self.block_checkpoint:
      self.checkpoint_started.set()
      self.finish_checkpoint.wait(timeout=5.0)

  def restore(self, pid: int) -> None:
    super().restore(pid)
    if self.block_restore:
      self.restore_started.set()
      self.finish_restore.wait(timeout=5.0)


class TrainerSchedulerTest(unittest.IsolatedAsyncioTestCase):
  async def test_scheduler_grants_only_one_active_worker_at_a_time(self) -> None:
    restorer = RecordingRestorer()
    scheduler = TrainerScheduler(restorer)
    await scheduler.register("run-a", 101)
    await scheduler.register("run-b", 202)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    blocked = asyncio.create_task(scheduler.acquire("run-b"))
    await asyncio.sleep(0.05)
    self.assertFalse(blocked.done())

    release = await scheduler.release("run-a")
    self.assertTrue(release["ok"])
    granted_b = await asyncio.wait_for(blocked, timeout=1.0)
    self.assertTrue(granted_b["ok"])
    self.assertEqual(restorer.calls, [("checkpoint", 101)])
    self.assertEqual(scheduler.active_run_id, "run-b")

  async def test_first_acquire_is_cold_and_later_acquire_restores_after_checkpoint(self) -> None:
    restorer = RecordingRestorer()
    scheduler = TrainerScheduler(restorer)
    await scheduler.register("run-a", 101)
    await scheduler.register("run-b", 202)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    self.assertTrue((await scheduler.release("run-a"))["ok"])
    self.assertEqual(restorer.calls, [("checkpoint", 101)])

    self.assertTrue((await scheduler.acquire("run-b"))["ok"])
    self.assertTrue((await scheduler.release("run-b"))["ok"])
    self.assertTrue((await scheduler.acquire("run-a"))["ok"])

    self.assertEqual(restorer.calls, [("checkpoint", 101), ("checkpoint", 202), ("restore", 101)])

  async def test_release_with_no_waiters_checkpoints_worker(self) -> None:
    restorer = RecordingRestorer()
    scheduler = TrainerScheduler(restorer)
    await scheduler.register("run-a", 101)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    release = await scheduler.release("run-a")

    self.assertTrue(release["ok"])
    self.assertIsNone(scheduler.active_run_id)
    self.assertTrue(scheduler.workers["run-a"].checkpointed)
    self.assertFalse(scheduler.workers["run-a"].failed)
    self.assertEqual(restorer.calls, [("checkpoint", 101)])

  async def test_waiting_acquire_is_not_granted_until_release_checkpoint_finishes(self) -> None:
    restorer = BlockingRestorer()
    restorer.block_checkpoint = True
    scheduler = TrainerScheduler(restorer)
    await scheduler.register("run-a", 101)
    await scheduler.register("run-b", 202)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    release_a = asyncio.create_task(scheduler.release("run-a"))

    checkpoint_started = await asyncio.to_thread(restorer.checkpoint_started.wait, 1.0)
    self.assertTrue(checkpoint_started)

    acquire_b = asyncio.create_task(scheduler.acquire("run-b"))
    await asyncio.sleep(0.05)
    self.assertFalse(release_a.done())
    self.assertFalse(acquire_b.done())

    restorer.finish_checkpoint.set()

    self.assertTrue((await asyncio.wait_for(release_a, timeout=1.0))["ok"])
    self.assertTrue((await asyncio.wait_for(acquire_b, timeout=1.0))["ok"])
    self.assertEqual(restorer.calls, [("checkpoint", 101)])

  async def test_checkpointed_worker_is_not_granted_until_restore_finishes(self) -> None:
    restorer = BlockingRestorer()
    scheduler = TrainerScheduler(restorer)
    await scheduler.register("run-a", 101)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    self.assertTrue((await scheduler.release("run-a"))["ok"])

    restorer.block_restore = True
    acquire_a = asyncio.create_task(scheduler.acquire("run-a"))

    restore_started = await asyncio.to_thread(restorer.restore_started.wait, 1.0)
    self.assertTrue(restore_started)
    self.assertFalse(acquire_a.done())

    restorer.finish_restore.set()

    self.assertTrue((await asyncio.wait_for(acquire_a, timeout=1.0))["ok"])
    self.assertFalse(scheduler.workers["run-a"].checkpointed)
    self.assertEqual(restorer.calls, [("checkpoint", 101), ("restore", 101)])

  async def test_unregister_waiting_worker_prevents_later_grant(self) -> None:
    scheduler = TrainerScheduler(RecordingRestorer())
    await scheduler.register("run-a", 101)
    await scheduler.register("run-b", 202)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    acquire_b = asyncio.create_task(scheduler.acquire("run-b"))
    await asyncio.sleep(0.05)
    self.assertFalse(acquire_b.done())

    self.assertTrue((await scheduler.unregister("run-b"))["ok"])
    self.assertTrue((await scheduler.release("run-a"))["ok"])

    result = await asyncio.wait_for(acquire_b, timeout=1.0)
    self.assertFalse(result["ok"])
    self.assertIsNone(scheduler.active_run_id)

  async def test_duplicate_commands_return_explicit_errors(self) -> None:
    scheduler = TrainerScheduler(RecordingRestorer())
    await scheduler.register("run-a", 101)

    self.assertFalse((await scheduler.register("run-a", 999))["ok"])
    self.assertTrue((await scheduler.acquire("run-a"))["ok"])
    self.assertFalse((await scheduler.acquire("run-a"))["ok"])
    self.assertTrue((await scheduler.release("run-a"))["ok"])
    self.assertFalse((await scheduler.release("run-a"))["ok"])
    self.assertTrue((await scheduler.unregister("run-a"))["ok"])
    self.assertFalse((await scheduler.unregister("run-a"))["ok"])

  async def test_waiters_are_granted_in_fifo_order(self) -> None:
    scheduler = TrainerScheduler(RecordingRestorer())
    for run_id, pid in [("run-a", 101), ("run-b", 202), ("run-c", 303), ("run-d", 404)]:
      await scheduler.register(run_id, pid)

    self.assertTrue((await scheduler.acquire("run-a"))["ok"])

    grant_order: list[str] = []

    async def acquire_then_release(run_id: str) -> None:
      await scheduler.acquire(run_id)
      grant_order.append(run_id)
      await scheduler.release(run_id)

    waiters = []
    for run_id in ["run-c", "run-b", "run-d"]:
      waiters.append(asyncio.create_task(acquire_then_release(run_id)))
      await asyncio.sleep(0.01)

    self.assertTrue((await scheduler.release("run-a"))["ok"])
    await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)

    self.assertEqual(grant_order, ["run-c", "run-b", "run-d"])


class RunScopedQueueTest(unittest.IsolatedAsyncioTestCase):
  async def test_worker_drains_only_its_run_queue(self) -> None:
    store = InMemoryStore()
    await store.put_request({"req_id": "a1", "model_id": "run-a"})
    await store.put_request({"req_id": "b1", "model_id": "run-b"})
    await store.put_request({"req_id": "a2", "model_id": "run-a"})

    batch_a = await store.get_requests_for_model("run-a")
    batch_b = await store.get_requests_for_model("run-b")

    self.assertEqual([item["req_id"] for item in batch_a], ["a1", "a2"])
    self.assertEqual([item["req_id"] for item in batch_b], ["b1"])


class WorkerLauncherTest(unittest.TestCase):
  def test_launcher_starts_scheduled_worker_for_run(self) -> None:
    fake_process = types.SimpleNamespace(pid=1234)
    with (
      patch.dict(os.environ, {"REDIS_URL": "redis://redis:6379"}, clear=True),
      patch("worker_launcher.subprocess.Popen", return_value=fake_process) as popen,
    ):
      process = launch_worker("run-a", "/tmp/open-rl/scheduler.sock")

    self.assertIs(process, fake_process)
    command = popen.call_args.args[0]
    kwargs = popen.call_args.kwargs
    self.assertTrue(command[1].endswith("clock_cycle.py"))
    self.assertEqual(kwargs["env"]["OPEN_RL_WORKER_RUN_ID"], "run-a")
    self.assertEqual(kwargs["env"]["OPEN_RL_SCHEDULER_SOCKET"], "/tmp/open-rl/scheduler.sock")
    self.assertEqual(kwargs["env"]["OPEN_RL_DISABLE_WORKER_HEALTHZ"], "1")
    self.assertTrue(kwargs["start_new_session"])

  def test_launcher_requires_redis(self) -> None:
    with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      launch_worker("run-a", "/tmp/open-rl/scheduler.sock")


class TrainerSchedulerSocketTest(unittest.IsolatedAsyncioTestCase):
  async def test_persistent_socket_clients_alternate(self) -> None:
    restorer = RecordingRestorer()
    scheduler = TrainerScheduler(restorer)
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "scheduler.sock")
      server = await serve_scheduler(scheduler, socket_path)
      client_a = TrainerSchedulerClient(socket_path)
      client_b = TrainerSchedulerClient(socket_path)
      try:
        await client_a.register("run-a", 101)
        await client_b.register("run-b", 202)

        async with client_a.acquire("run-a"):
          blocked = asyncio.create_task(acquire_once(client_b, "run-b"))
          await asyncio.sleep(0.05)
          self.assertFalse(blocked.done())

        self.assertEqual(await asyncio.wait_for(blocked, timeout=1.0), "run-b")
        self.assertEqual(restorer.calls, [("checkpoint", 101), ("checkpoint", 202)])
      finally:
        await client_a.close()
        await client_b.close()
        server.close()
        await server.wait_closed()

  async def test_closing_active_worker_socket_marks_run_failed(self) -> None:
    scheduler = TrainerScheduler(RecordingRestorer())
    with tempfile.TemporaryDirectory() as tmp:
      socket_path = str(Path(tmp) / "scheduler.sock")
      server = await serve_scheduler(scheduler, socket_path)
      client = TrainerSchedulerClient(socket_path)
      try:
        await client.register("run-a", 101)
        await client.request({"command": "ACQUIRE", "run_id": "run-a"})
        await client.close()
        await asyncio.sleep(0.05)

        self.assertIsNone(scheduler.active_run_id)
        self.assertTrue(scheduler.workers["run-a"].failed)
      finally:
        server.close()
        await server.wait_closed()


async def acquire_once(client: TrainerSchedulerClient, run_id: str) -> str:
  async with client.acquire(run_id):
    return run_id


if __name__ == "__main__":
  unittest.main()
