"""A pool whose replicas share their GPU with their trainer. The dispatcher holds each replica's lease from
the real time slicer, over its socket, for every call it makes to the replica."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from starlette.responses import Response
from starlette.routing import Route

from accel_timeslicer.serve import start_time_slicer
from accel_timeslicer.single_node import SingleNodeTimeSlicer
from accel_timeslicer.time_slicer import SocketTimeSlicerClient
from accel_timeslicer.workload import WorkloadRef
from server import sampling_dispatcher
from server.sampling_dispatcher import Dispatcher, revision_file
from tests.test_sampling_dispatcher import SAMPLE, DispatcherTestCase, FakeReplica, router, until

CLAIM = "gpu-0"
# What vLLM with sleep mode still answers while asleep. A checkpointed process answers nothing.
ASLEEP_OK = {"/health", "/weight_info", "/is_sleeping", "/wake_up", "/sleep"}


class LeasedReplica(FakeReplica):
  """A FakeReplica with vLLM's sleep mode. It starts asleep, as its launcher leaves it, is `parked` while the
  slicer holds it checkpointed, and logs any call that reaches it asleep or parked, which vLLM could not answer.
  `gpu` holds the names of everything awake on the device, so an overlap shows up in the log."""

  def __init__(self, name: str, log: list, version: str, expected, gpu: set):
    super().__init__(name, log, version, expected)
    self.awake, self.parked, self.gpu, self.wakes, self.delay = False, False, gpu, 0, 0.0
    self.app.router.routes += [Route("/sleep", self.sleep, methods=["POST"]), Route("/wake_up", self.wake_up, methods=["POST"])]
    inner = self.app

    async def guarded(scope, receive, send):
      path = scope.get("path")
      if scope["type"] == "http" and (self.parked or (not self.awake and path not in ASLEEP_OK)):
        self.log.append(("parked" if self.parked else "asleep", self.name, path))
        await Response(status_code=500)(scope, receive, send)
        return
      await inner(scope, receive, send)

    self.app = guarded

  async def sleep(self, request) -> Response:
    self.awake = False
    self.gpu.discard(self.name)
    self.log.append(("sleep", self.name))
    return Response()

  async def wake_up(self, request) -> Response:
    self.note_overlap()
    self.awake, self.wakes = True, self.wakes + 1
    self.gpu.add(self.name)
    self.log.append(("wake", self.name))
    return Response()

  async def generate(self, request) -> Response:
    self.note_overlap()
    await asyncio.sleep(self.delay)
    return await super().generate(request)

  def note_overlap(self) -> None:
    if others := self.gpu - {self.name}:
      self.log.append(("overlap", self.name, others))


class ParkingRestorer:
  """Records checkpoints and restores in the shared log and parks the replica, since a checkpointed process answers nothing.
  The checkpoint of a workload named in `fail` raises instead, as when the node's host memory is full."""

  def __init__(self, log: list, replicas: dict[str, LeasedReplica]):
    self.log, self.replicas, self.fail = log, replicas, set()

  def checkpoint(self, workload: WorkloadRef) -> None:
    if workload.name in self.fail:
      self.log.append(("checkpoint-failed", workload.name))
      raise RuntimeError("could not park the process")
    self.log.append(("checkpoint", workload.name))
    if replica := self.replicas.get(workload.name):
      replica.parked = True
      replica.gpu.discard(replica.name)

  def restore(self, workload: WorkloadRef) -> None:
    self.log.append(("restore", workload.name))
    if replica := self.replicas.get(workload.name):
      replica.parked = False


class Trainer:
  """A trainer on the same claim, holding the GPU for each step over its own slicer connection."""

  def __init__(self, socket_path: str, log: list, gpu: set):
    self.client, self.workload = SocketTimeSlicerClient(socket_path=socket_path), WorkloadRef("trainer", CLAIM)
    self.log, self.gpu, self.steps = log, gpu, 0

  async def run(self, steps: int, hold: float = 0.1) -> None:
    await self.client.register(self.workload)
    for _ in range(steps):
      async with self.client.acquire(self.workload):
        if self.gpu:
          self.log.append(("overlap", "trainer", set(self.gpu)))
        self.gpu.add("trainer")
        self.log.append(("step", "trainer"))
        await asyncio.sleep(hold)
        self.gpu.discard("trainer")
        self.steps += 1
      await asyncio.sleep(0.01)
    await self.client.close()


class SamplerGpuLeaseTest(DispatcherTestCase):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    self.enterContext(patch.object(sampling_dispatcher, "IDLE_SECONDS", 0.1))
    self.socket = str(self.tmp / "slicer.sock")
    self.gpu: set = set()
    self.leased: dict[str, LeasedReplica] = {}
    self.launchers: dict[str, SocketTimeSlicerClient] = {}
    self.agent = SingleNodeTimeSlicer(ParkingRestorer(self.log, self.leased))
    server = await start_time_slicer(self.agent, self.socket)
    self.addAsyncCleanup(server.wait_closed)
    self.addCleanup(server.close)

  def workload(self, replica: FakeReplica) -> WorkloadRef:
    return WorkloadRef(replica.name, CLAIM)

  async def replica(self, version: str = "default") -> LeasedReplica:
    """A time-sliced replica as its launcher leaves it: registered over the launcher's own connection,
    which stays open, asleep and checkpointed off the GPU."""
    replica = LeasedReplica(f"pod-{len(self.leased)}", self.log, version, revision_file(self.pool), self.gpu)
    replica.url = await self.serve(replica.app)
    self.addAsyncCleanup(replica.held.clear)
    await self.boot(replica)
    self.leased[replica.name] = replica
    return replica

  async def boot(self, replica: LeasedReplica) -> None:
    """What the replica's launcher does once vLLM is up. It registers over a connection of its own, kept in
    `launchers` and left open, and its boot turn ends in a release."""
    launcher = SocketTimeSlicerClient(socket_path=self.socket)
    self.addAsyncCleanup(launcher.close)
    await launcher.register(self.workload(replica))
    async with launcher.acquire(self.workload(replica)):
      pass
    self.launchers[replica.name] = launcher

  async def leased_dispatcher(self, replicas: list[LeasedReplica], concurrency: int = 4) -> tuple[Dispatcher, asyncio.Task]:
    router_url = await self.serve(router(self.log, replicas, self.http))
    leases = {}
    for replica in replicas:
      slicer = SocketTimeSlicerClient(socket_path=self.socket)
      self.addAsyncCleanup(slicer.close)
      leases[replica.url] = (self.workload(replica), slicer)
    dispatcher = Dispatcher(self.store, self.pool, self.http, router_url, [r.url for r in replicas], concurrency=concurrency)
    dispatcher.leases = leases
    run = asyncio.create_task(dispatcher.run())
    self.addAsyncCleanup(lambda: (run.cancel(), asyncio.gather(run, return_exceptions=True))[1])
    return dispatcher, run

  def events(self, *kinds: str) -> list[tuple]:
    return [entry for entry in self.log if entry[0] in kinds]

  def violations(self) -> list[tuple]:
    return self.events("asleep", "parked", "overlap", "unrouted")

  async def test_a_turn_restores_wakes_serves_and_sleeps_the_replica_before_releasing_it(self):
    replica = await self.replica()
    records = [await self.accepted() for _ in range(3)]
    await self.leased_dispatcher([replica])
    for r in records:
      self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    await until(lambda: len(self.events("checkpoint")) == 2)
    generates = [("generate", "pod-0", "default")] * 3
    expected = [("checkpoint", "pod-0"), ("restore", "pod-0"), ("wake", "pod-0"), *generates, ("sleep", "pod-0"), ("checkpoint", "pod-0")]
    self.assertEqual(self.events("checkpoint", "restore", "wake", "sleep", "generate"), expected)
    self.assertEqual((self.violations(), self.agent.running, replica.wakes), ([], {}, 1))

  async def test_a_busy_replica_gives_the_trainer_its_turn_and_comes_back(self):
    self.enterContext(patch.object(sampling_dispatcher, "MAX_TURN_SECONDS", 0.2))
    replica = await self.replica()
    replica.delay = 0.05  # a backlog of 20 outlasts the longest turn several times over
    trainer = Trainer(self.socket, self.log, self.gpu)
    records = [await self.accepted() for _ in range(20)]
    await self.leased_dispatcher([replica], concurrency=2)
    steps = asyncio.create_task(trainer.run(steps=3))
    for r in records:
      self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    await asyncio.wait_for(steps, timeout=20)
    self.assertEqual((self.violations(), trainer.steps), ([], 3))
    # The replica gave the GPU up with requests still pending and came back once the trainer had stepped.
    turns = [entry[0] for entry in self.events("step", "wake")]
    self.assertGreaterEqual(replica.wakes, 2)
    first, last = turns.index("wake"), len(turns) - 1 - turns[::-1].index("wake")
    self.assertIn("step", turns[first:last])

  async def test_two_replicas_on_one_gpu_take_turns(self):
    replicas = [await self.replica(), await self.replica()]
    records = [await self.accepted() for _ in range(8)]
    await self.leased_dispatcher(replicas, concurrency=2)
    for r in records:
      self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    self.assertEqual(self.violations(), [])
    self.assertEqual(sum(entry[0] == "generate" for entry in self.log), 8)

  async def test_a_weight_load_never_splits_across_turns(self):
    self.enterContext(patch.object(sampling_dispatcher, "MAX_TURN_SECONDS", 0.05))
    w1 = self.revision("w1", format="sparse_delta", parent="")
    w2 = self.revision("w2", format="sparse_delta", parent=w1)
    replica = await self.replica(w1)
    replica.held.set()  # the update waits to be let go, so the load outlasts the longest turn
    trainer = Trainer(self.socket, self.log, self.gpu)
    r = await self.accepted(w2)
    await self.leased_dispatcher([replica])
    await until(lambda: ("update", "pod-0", w2, w2) in self.log)
    step = asyncio.create_task(trainer.run(steps=1))
    await asyncio.sleep(0.3)
    self.assertFalse(step.done())
    replica.held.clear()
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    await asyncio.wait_for(step, timeout=10)
    sequence = [entry[:2] for entry in self.events("checkpoint", "restore", "update", "generate", "sleep", "step")]
    turn = [("restore", "pod-0"), ("update", "pod-0"), ("generate", "pod-0"), ("sleep", "pod-0"), ("checkpoint", "pod-0")]
    self.assertEqual(sequence, [("checkpoint", "pod-0"), *turn, ("step", "trainer"), ("checkpoint", "trainer")])
    self.assertEqual((self.violations(), replica.version), ([], w2))

  async def test_a_turn_after_a_clean_load_sends_no_finish_probe_and_a_turn_after_a_broken_load_does(self):
    # The probe is a bare finish_weight_update, and vLLM logs a 500 traceback for each one that finds no update open.
    w1, w2, w3 = self.revision("w1"), self.revision("w2"), self.revision("w3")
    replica = await self.replica()
    r1 = await self.accepted(w1)
    await self.leased_dispatcher([replica])
    self.assertEqual(await self.committed(r1["request_id"]), [SAMPLE])
    await until(lambda: len(self.events("checkpoint")) == 2)  # the first turn is over
    r2 = await self.accepted(w2)
    self.assertEqual(await self.committed(r2["request_id"]), [SAMPLE])
    self.assertEqual((replica.wakes, replica.probes), (2, 1))
    # The third turn's load breaks off and closes its own update with a bare finish, so the fourth turn probes.
    replica.fail_update = True
    r3 = await self.accepted(w3)
    self.assertEqual(await self.committed(r3["request_id"]), [SAMPLE])
    self.assertEqual((replica.wakes, replica.probes, self.violations()), (4, 3, []))

  async def test_a_dispatcher_that_dies_mid_turn_leaves_the_replica_checkpointed_before_the_next_grant(self):
    replica = await self.replica()
    replica.held.set()
    r = await self.accepted()
    dispatcher, run = await self.leased_dispatcher([replica])
    await until(lambda: ("generate", "pod-0", "default") in self.log)
    trainer = Trainer(self.socket, self.log, self.gpu)
    step = asyncio.create_task(trainer.run(steps=1))
    await asyncio.sleep(0.1)
    self.assertFalse(step.done())
    # The dispatcher dies: it makes no more calls and its slicer connection drops without a release.
    dead = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("the dispatcher is dead")))
    dispatcher.http = await self.enterAsyncContext(httpx.AsyncClient(transport=dead))
    dispatcher.leases[replica.url][1].writer.transport.abort()
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    await asyncio.wait_for(step, timeout=10)
    grants = [entry[:2] for entry in self.events("checkpoint", "restore", "wake", "step")]
    turn = [("restore", "pod-0"), ("wake", "pod-0"), ("checkpoint", "pod-0")]
    self.assertEqual(grants, [("checkpoint", "pod-0"), *turn, ("step", "trainer"), ("checkpoint", "trainer")])
    self.assertEqual(self.violations(), [])
    # The replica is registered and well behind its launcher's connection, so the next dispatcher leases it again.
    self.assertFalse(self.agent.workloads["pod-0"].failed)
    await self.leased_dispatcher([replica])
    replica.held.clear()
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    self.assertEqual(self.violations(), [])

  async def test_a_failed_park_at_the_end_of_a_turn_stops_the_replica_so_the_trainer_goes_next(self):
    replica = await self.replica()
    self.agent.restorer.fail.add(replica.name)
    first = await self.accepted()
    dispatcher, _ = await self.leased_dispatcher([replica])
    self.assertEqual(await self.committed(first["request_id"]), [SAMPLE])
    trainer = Trainer(self.socket, self.log, self.gpu)
    step = asyncio.create_task(trainer.run(steps=1))
    # The park at the end of the turn fails, so the replica keeps the GPU, and its launcher is told on its own connection.
    launcher = self.launchers[replica.name]
    self.assertIn("must exit", await asyncio.wait_for(launcher.fault(), 10) or "")
    await asyncio.sleep(0.2)
    self.assertFalse(step.done())
    # The launcher stops vLLM and exits, which closes its connection, and then the trainer is granted.
    await launcher.close()
    await asyncio.wait_for(step, timeout=5)
    # Restarted, the replica boots in a turn of its own again, and a new lease on it serves, as list_replicas makes one
    # once the restarted container has started.
    self.agent.restorer.fail.clear()
    await self.boot(replica)
    stale, fresh = dispatcher.leases[replica.url][1], SocketTimeSlicerClient(socket_path=self.socket)
    self.addAsyncCleanup(fresh.close)
    dispatcher.leases[replica.url] = (self.workload(replica), fresh)
    await stale.close()
    second = await self.accepted()
    self.assertEqual(await self.committed(second["request_id"]), [SAMPLE])
    self.assertEqual(self.violations(), [])

  async def test_a_turn_that_outlives_its_launcher_cannot_end_the_restarted_launchers_boot_turn(self):
    self.enterContext(patch.object(sampling_dispatcher, "MAX_TURN_SECONDS", 2.0))
    replica = await self.replica()
    await self.accepted()
    replica.held.set()  # a generate is in flight when vLLM dies
    dispatcher, _ = await self.leased_dispatcher([replica])
    await until(lambda: ("generate", "pod-0", "default") in self.log)
    with self.assertLogs("server.sampling_dispatcher", "WARNING") as logs:
      # vLLM dies mid-turn. The replica stops answering, and its launcher exits with it, which closes its connection.
      dead = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("vLLM is gone")))
      dispatcher.http = await self.enterAsyncContext(httpx.AsyncClient(transport=dead))
      replica.held.clear()
      await self.launchers[replica.name].close()
      await until(lambda: self.agent.workloads["pod-0"].failed)
      # kubelet restarts the container in the same pod, and the new launcher boots vLLM in a turn of its own.
      restarted = SocketTimeSlicerClient(socket_path=self.socket)
      self.addAsyncCleanup(restarted.close)
      await restarted.register(self.workload(replica))
      await restarted.request({"command": "ACQUIRE", **self.workload(replica).as_payload()})
      booting = len(self.log)
      # The old turn reaches MAX_TURN_SECONDS and releases the same workload, and the slicer refuses.
      await until(lambda: any("held by another connection" in line for line in logs.output))
    self.assertEqual(self.agent.running, {CLAIM: "pod-0"})
    self.assertNotIn(("checkpoint", "pod-0"), self.log[booting:])
    # The trainer waits for the boot turn, which the restarted launcher ends itself once vLLM sleeps.
    trainer = Trainer(self.socket, self.log, self.gpu)
    step = asyncio.create_task(trainer.run(steps=1))
    await asyncio.sleep(0.2)
    self.assertFalse(step.done())
    self.assertTrue((await restarted.request({"command": "RELEASE", **self.workload(replica).as_payload()}))["ok"])
    await asyncio.wait_for(step, timeout=5)

  async def test_a_generate_that_hangs_past_the_turn_is_cancelled_so_the_trainer_steps_and_it_is_sent_again(self):
    self.enterContext(patch.object(sampling_dispatcher, "MAX_TURN_SECONDS", 0.5))
    self.enterContext(patch.object(sampling_dispatcher, "TURN_GRACE_SECONDS", 0.5))
    replica = await self.replica()
    replica.held.set()  # a wedged replica, whose generate does not answer
    r = await self.accepted()
    await self.leased_dispatcher([replica])
    await until(lambda: ("generate", "pod-0", "default") in self.log)
    trainer = Trainer(self.socket, self.log, self.gpu)
    self.addAsyncCleanup(trainer.client.close)
    await asyncio.wait_for(trainer.run(steps=1), timeout=5)
    # Once the replica answers again, the record goes out in a later turn and is committed once.
    replica.held.clear()
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    generates = [i for i, entry in enumerate(self.log) if entry[0] == "generate"]
    self.assertLess(self.log.index(("step", "trainer")), generates[-1])
    self.assertEqual(self.violations(), [])

  async def test_an_exclusive_pool_makes_no_slicer_calls(self):
    replicas = await self.replicas("default")
    r = await self.accepted()
    await self.dispatcher(replicas)
    self.assertEqual(await self.committed(r["request_id"]), [SAMPLE])
    self.assertEqual((self.events("checkpoint", "restore"), self.agent.workloads), ([], {}))

  async def test_pods_marked_time_sliced_are_leased_through_the_slicer_on_their_node_once_started(self):
    def pod(name: str, ip: str | None, env: dict, started: bool = True) -> SimpleNamespace:
      container = SimpleNamespace(name="worker", env=[SimpleNamespace(name=k, value=v) for k, v in env.items()])
      status = SimpleNamespace(pod_ip=ip, host_ip="10.1.0.5", phase="Running", container_statuses=[SimpleNamespace(started=started)])
      return SimpleNamespace(metadata=SimpleNamespace(name=name), spec=SimpleNamespace(containers=[container]), status=status)

    identity = {"OPEN_RL_WORKLOAD_ID": "fft-job-sampler", "OPEN_RL_TIME_SLICE_GROUP": "claim-7", "OPEN_RL_ACCEL_TIMESLICER_PORT": "9753"}
    shared = pod("shared", "10.0.0.1", {**identity, "OPEN_RL_TIME_SLICED_SAMPLER": "true"})
    alone = pod("alone", "10.0.0.2", {"OPEN_RL_WORKLOAD_ID": "lora-x-0-sampler", "OPEN_RL_TIME_SLICING": "off"})
    # A time-sliced replica still booting has not passed its startup probe, so no turn is taken on it yet.
    booting = pod("booting", "10.0.0.3", {**identity, "OPEN_RL_TIME_SLICED_SAMPLER": "true"}, started=False)
    dispatcher = Dispatcher(self.store, self.pool, self.http, "http://router", [], concurrency=4)
    dispatcher.k8s = SimpleNamespace(
      list_namespaced_pod=lambda *args, **kwargs: SimpleNamespace(items=[shared, alone, booting, pod("no-ip", None, {})])
    )
    self.assertEqual(await dispatcher.list_replicas(), ["http://10.0.0.1:8000", "http://10.0.0.2:8000"])
    ((url, (workload, slicer)),) = dispatcher.leases.items()
    self.assertEqual((url, workload, slicer.host, slicer.port), ("http://10.0.0.1:8000", WorkloadRef("fft-job-sampler", "claim-7"), "10.1.0.5", 9753))
    # The lease goes when the pod does.
    dispatcher.k8s = SimpleNamespace(list_namespaced_pod=lambda *args, **kwargs: SimpleNamespace(items=[alone]))
    self.assertEqual(await dispatcher.list_replicas(), ["http://10.0.0.2:8000"])
    self.assertEqual(dispatcher.leases, {})
