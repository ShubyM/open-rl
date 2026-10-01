import asyncio
import contextlib
import types
import unittest

from server import observability
from server.store import InMemoryStore


class SlicerStub:
  def __init__(self):
    self.acquired = 0

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.acquired += 1
    yield


class GpuTurnTest(unittest.TestCase):
  def test_each_turn_is_recorded_under_the_workload(self) -> None:
    store = InMemoryStore()
    slicer = SlicerStub()
    workload = types.SimpleNamespace(name="fft-run-a-trainer")

    async def run():
      async with observability.gpu_turn(slicer, workload, store, "trainer", "run-a"):
        await asyncio.sleep(0.01)
      async with observability.gpu_turn(slicer, workload, store, "trainer", "run-a"):
        pass
      return await observability.read_turns(store, "fft-run-a-trainer")

    turns = asyncio.run(run())
    self.assertEqual(slicer.acquired, 2)
    self.assertEqual(len(turns), 2)
    self.assertEqual(turns[0]["operation"], "gpu_turn")
    self.assertEqual((turns[0]["role"], turns[0]["runtime_id"], turns[0]["workload"]), ("trainer", "run-a", "fft-run-a-trainer"))
    self.assertGreaterEqual(turns[0]["at"], turns[0]["started_at"] + 0.01)
    self.assertLessEqual(turns[0]["at"], turns[1]["started_at"])

  def test_a_turn_records_what_ran_inside_it(self) -> None:
    store = InMemoryStore()
    workload = types.SimpleNamespace(name="fft-run-a-trainer")

    async def op(name: str, run_id: str, request_id: str, delay: float = 0.0):
      async def call():
        await asyncio.sleep(delay)
        return {"type": "ok"}

      return await observability.observe_operation(store, {"op": name, "model_id": run_id, "request_id": request_id}, "trainer", run_id, call)

    async def run():
      await op("forward_backward", "run-a", "outside")  # no turn: must not leak into the next one
      async with observability.gpu_turn(SlicerStub(), workload, store, "trainer", "run-a"):
        with observability.turn_phase("wake_up"):
          await asyncio.sleep(0.005)
        await op("forward_backward", "run-a", "r1", 0.005)
        await op("optim_step", "run-a", "r2")
        await asyncio.gather(*(asyncio.create_task(op("sample", "run-a", f"s{i}", 0.01)) for i in range(5)))
        with observability.turn_phase("sleep"):
          pass
      return await observability.read_turns(store, "fft-run-a-trainer")

    (turn,) = asyncio.run(run())
    self.assertEqual([p["name"] for p in turn["phases"]], ["wake_up", "forward_backward", "optim_step", "sample", "sleep"])
    by_name = {p["name"]: p for p in turn["phases"]}
    self.assertEqual(by_name["forward_backward"]["request_id"], "r1")
    self.assertEqual(by_name["forward_backward"]["run_id"], "run-a")
    self.assertEqual(by_name["sample"]["count"], 5)
    self.assertLessEqual(turn["requested_at"], turn["started_at"])
    for phase in turn["phases"]:
      self.assertLessEqual(turn["started_at"], phase["start"])
      self.assertLessEqual(phase["start"], phase["end"])
      self.assertLessEqual(phase["end"], turn["at"])
