"""Drives the sampler launcher, as its own process, for a replica that shares its GPU, against the real
time slicer over its socket and tests/vllm_stub.py on PATH as vllm."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from accel_timeslicer.serve import start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient
from accel_timeslicer.workload import WorkloadRef
from tests.test_accel_timeslicer import BlockingRestorer, FailingCheckpointRestorer

REPO = Path(__file__).resolve().parents[1]
CLAIM = "gpu-0"


def free_port() -> int:
  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


async def until(condition, timeout: float = 20.0) -> None:
  async with asyncio.timeout(timeout):
    while not condition():
      await asyncio.sleep(0.02)


class SamplerLauncherTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
    (self.dir / "vllm").symlink_to(REPO / "tests" / "vllm_stub.py")
    self.socket = str(self.dir / "slicer.sock")
    self.restorer = BlockingRestorer()
    self.agent = SingleNodeTimeSlicer(self.restorer)
    server = await start_time_slicer(self.agent, self.socket)
    self.addAsyncCleanup(server.wait_closed)
    self.addCleanup(server.close)

  def launch(self, name: str, *options: str, command: tuple[str, ...] = ()) -> tuple[subprocess.Popen, Path, Path]:
    """The launcher for replica `name` on CLAIM, running the stub with `options` unless given a command, with the stub's log and the marker."""
    log, awake, port = self.dir / f"{name}.log", self.dir / f"{name}.awake", free_port()
    env = {
      **os.environ,
      "PATH": f"{self.dir}:{Path(sys.executable).parent}:{os.environ['PATH']}",
      "PYTHONPATH": str(REPO / "src"),
      "OPEN_RL_SAMPLER_AWAKE_FILE": str(awake),
      "OPEN_RL_ACCEL_TIMESLICER_SOCKET": self.socket,
      "OPEN_RL_TIME_SLICE_JOB_ID": name,
      "OPEN_RL_TIME_SLICE_GROUP": CLAIM,
    }
    env.pop("OPEN_RL_ACCEL_TIMESLICER_HOST", None)
    command = command or ("vllm", "serve", "stub-model", "--port", str(port), "--log", str(log), *options)
    launcher = subprocess.Popen([sys.executable, "-m", "server.sampler_launcher", *command], env=env, start_new_session=True)
    # The stub exits on its own once the launcher is gone.
    self.addCleanup(subprocess.run, ["pkill", "-KILL", "-g", str(launcher.pid)])
    return launcher, log, awake

  def requests(self, log: Path) -> list[str]:
    return [line for line in log.read_text().splitlines() if not line.startswith("GET /health")]

  async def test_the_replica_boots_asleep_inside_its_own_turn_then_a_dispatcher_leases_it(self) -> None:
    launcher, log, awake = self.launch("sampler-x")
    await until(awake.exists)
    state = self.agent.workloads["sampler-x"]
    self.assertEqual((state.workload.claim, state.failed, state.checkpointed, self.agent.running), (CLAIM, False, True, {}))
    self.assertEqual((self.restorer.simple_labels(), self.requests(log)), ([("checkpoint", "sampler-x")], ["POST /sleep?level=1"]))
    self.assertIsNone(launcher.poll())
    # A dispatcher leases the replica over its own connection while the launcher keeps the registration.
    dispatcher = SocketTimeSlicerClient(socket_path=self.socket)
    self.addAsyncCleanup(dispatcher.close)
    async with dispatcher.acquire(WorkloadRef("sampler-x", CLAIM)):
      self.assertEqual(self.restorer.simple_labels()[-1], ("restore", "sampler-x"))
    self.assertEqual(self.restorer.simple_labels()[-1], ("checkpoint", "sampler-x"))
    # SIGTERM reaches the stub, whose status comes back, and the registration goes with the launcher.
    sent = time.monotonic()
    launcher.send_signal(signal.SIGTERM)
    self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 3)
    self.assertLess(time.monotonic() - sent, 3)
    self.assertEqual(self.requests(log)[-1], "SIGTERM")
    await until(lambda: self.agent.workloads["sampler-x"].failed)

  async def test_two_replicas_on_one_claim_boot_one_after_the_other(self) -> None:
    self.restorer.block_checkpoint = True
    replicas = {name: self.launch(name) for name in ("sampler-a", "sampler-b")}
    # The first to boot is checkpointed with the grant held, so the other's vLLM has not been started yet.
    await asyncio.to_thread(self.restorer.checkpoint_started.wait, 20)
    (holder,) = self.agent.running.values()
    (other,) = set(replicas) - {holder}
    self.assertFalse(replicas[other][1].exists())
    self.restorer.finish_checkpoint.set()
    await until(lambda: all(awake.exists() for _, _, awake in replicas.values()))
    self.assertEqual(self.restorer.simple_labels(), [("checkpoint", holder), ("checkpoint", other)])

  async def test_a_child_that_exits_before_it_is_healthy_gives_the_launcher_its_status_and_frees_the_claim(self) -> None:
    launcher, _, awake = self.launch("sampler-x", command=(sys.executable, "-c", "import sys; sys.exit(7)"))
    self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 7)
    # The launcher's exit closes the connection holding the turn, so the claim is free again.
    await until(lambda: self.agent.running == {} and self.agent.workloads["sampler-x"].failed)
    self.assertEqual((self.restorer.simple_labels(), awake.exists()), ([], False))

  async def test_a_vllm_that_does_not_sleep_is_stopped(self) -> None:
    launcher, log, awake = self.launch("sampler-x", "--fail-on", "/sleep?level=1")
    self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 1)
    self.assertEqual((self.requests(log), awake.exists()), (["POST /sleep?level=1", "SIGTERM"], False))
    await until(lambda: self.agent.running == {})

  async def test_a_replica_the_slicer_cannot_park_is_stopped(self) -> None:
    self.agent.restorer = FailingCheckpointRestorer()
    launcher, log, awake = self.launch("sampler-x")
    self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 1)
    self.assertEqual((self.requests(log), awake.exists()), (["POST /sleep?level=1", "SIGTERM"], False))
    await until(lambda: self.agent.running == {})

  async def test_a_turn_the_slicer_cannot_park_stops_vllm_and_the_launcher_so_the_trainer_goes_next(self) -> None:
    launcher, log, awake = self.launch("sampler-x")
    await until(awake.exists)
    trainer = SocketTimeSlicerClient(socket_path=self.socket)
    self.addAsyncCleanup(trainer.close)
    await trainer.register(WorkloadRef("trainer", CLAIM))
    # A dispatcher's turn ends in a park that fails, so vLLM still holds the GPU and the trainer waits.
    self.agent.restorer = FailingCheckpointRestorer()
    dispatcher = SocketTimeSlicerClient(socket_path=self.socket)
    self.addAsyncCleanup(dispatcher.close)
    async with dispatcher.acquire(WorkloadRef("sampler-x", CLAIM)):
      acquire = asyncio.create_task(trainer.request({"command": "ACQUIRE", "name": "trainer", "claim": CLAIM}))
      await until(lambda: "trainer" in self.agent.waiting_workloads)
    self.assertIn("must exit", dispatcher.faulted or "")
    # The launcher stops vLLM and exits 1, so the container restarts, and its exit hands the GPU to the trainer.
    self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 1)
    self.assertEqual(self.requests(log), ["POST /sleep?level=1", "SIGTERM"])
    self.assertTrue((await asyncio.wait_for(acquire, 5))["ok"])
    self.assertEqual((self.agent.running, self.agent.workloads["sampler-x"].failed), ({CLAIM: "trainer"}, True))
    # The restarted launcher registers again and boots vLLM in a turn of its own once the trainer releases.
    self.agent.restorer = self.restorer
    restarted, _, _ = self.launch("sampler-x")
    await until(lambda: "sampler-x" in self.agent.waiting_workloads)
    await trainer.request({"command": "RELEASE", "name": "trainer", "claim": CLAIM})
    await until(lambda: self.restorer.simple_labels()[-2:] == [("checkpoint", "trainer"), ("checkpoint", "sampler-x")])
    await until(awake.exists)
    self.assertEqual((restarted.poll(), self.agent.workloads["sampler-x"].failed, self.agent.running), (None, False, {}))

  async def test_sigterm_while_the_acquire_waits_behind_the_trainer_ends_the_launcher_at_once(self) -> None:
    trainer = SocketTimeSlicerClient(socket_path=self.socket)
    self.addAsyncCleanup(trainer.close)
    await trainer.register(WorkloadRef("trainer", CLAIM))
    async with trainer.acquire(WorkloadRef("trainer", CLAIM)):
      launcher, log, awake = self.launch("sampler-x")
      await until(lambda: "sampler-x" in self.agent.waiting_workloads)
      sent = time.monotonic()
      launcher.send_signal(signal.SIGTERM)
      self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 1)
      self.assertLess(time.monotonic() - sent, 3)
      self.assertEqual((log.exists(), awake.exists()), (False, False))  # vLLM was never started
    # The slicer resolves a queued ACQUIRE at grant time, so once the trainer releases, the grant
    # the dead launcher would have had is dropped with its connection and nothing holds the claim.
    await until(lambda: self.agent.running == {} and "sampler-x" not in self.agent.waiting_workloads)

  async def test_a_sigterm_vllm_ignores_while_booting_is_followed_by_sigkill_and_frees_the_turn(self) -> None:
    # vLLM ignores SIGTERM while its engine starts and would hold the GPU, and the turn, until kubelet's SIGKILL.
    launcher, log, awake = self.launch("sampler-x", "--boot-seconds", "60")
    await until(log.exists)  # the stub is up but not serving
    self.assertEqual(self.agent.running, {CLAIM: "sampler-x"})
    sent = time.monotonic()
    launcher.send_signal(signal.SIGTERM)
    self.assertEqual(await asyncio.to_thread(launcher.wait, 20), 137)
    self.assertLess(time.monotonic() - sent, 10)
    self.assertEqual((log.read_text().splitlines(), awake.exists()), (["SIGTERM ignored"], False))
    await until(lambda: self.agent.running == {} and self.agent.workloads["sampler-x"].failed)


if __name__ == "__main__":
  unittest.main()
