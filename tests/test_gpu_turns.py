import asyncio
import contextlib
import types
import unittest
from unittest import mock

from server.dashboard.snapshot import metadata_of
from server.store import InMemoryStateStore
from server.telemetry import ops


class SlicerStub:
  def __init__(self):
    self.acquired = 0

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    self.acquired += 1
    yield


class GpuTurnTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStateStore()
    self.enterContext(mock.patch.object(ops, "get_state_store", return_value=self.store))

  async def recorded(self, key: str) -> list[dict]:
    await asyncio.gather(*list(ops._writes))  # writes are fire-and-forget
    return await self.store.read_samples(key)

  async def test_each_turn_is_recorded_under_the_workload(self) -> None:
    slicer = SlicerStub()
    workload = types.SimpleNamespace(name="fft-run-a-trainer")
    async with ops.gpu_turn(slicer, workload, "trainer", "run-a"):
      await asyncio.sleep(0.01)
    async with ops.gpu_turn(slicer, workload, "trainer", "run-a"):
      pass
    turns = await self.recorded(ops.turn_key("fft-run-a-trainer"))
    self.assertEqual(slicer.acquired, 2)
    self.assertEqual(len(turns), 2)
    self.assertEqual(turns[0]["operation"], "gpu_turn")
    self.assertEqual((turns[0]["role"], turns[0]["runtime_id"], turns[0]["workload"]), ("trainer", "run-a", "fft-run-a-trainer"))
    self.assertGreaterEqual(turns[0]["at"], turns[0]["started_at"] + 0.01)
    self.assertLessEqual(turns[0]["at"], turns[1]["started_at"])

  async def test_a_turn_records_what_ran_inside_it(self) -> None:
    workload = types.SimpleNamespace(name="fft-run-a-trainer")

    async def op(name: str, run_id: str, request_id: str, delay: float = 0.0):
      async def call():
        await asyncio.sleep(delay)
        return {"type": "ok"}

      return await ops.observe_operation({"op": name, "model_id": run_id, "request_id": request_id}, "trainer", run_id, call)

    await op("forward_backward", "run-a", "outside")  # no turn: must not leak into the next one
    async with ops.gpu_turn(SlicerStub(), workload, "trainer", "run-a"):
      with ops.record_op("wake_up"):
        await asyncio.sleep(0.005)
      await op("forward_backward", "run-a", "r1", 0.005)
      await op("optim_step", "run-a", "r2")
      await asyncio.gather(*(asyncio.create_task(op("sample", "run-a", f"s{i}", 0.01)) for i in range(5)))
      with ops.record_op("sleep"):
        pass
    (turn,) = await self.recorded(ops.turn_key("fft-run-a-trainer"))
    self.assertEqual([o["name"] for o in turn["ops"]], ["wake_up", "forward_backward", "optim_step", "sample", "sleep"])
    by_name = {o["name"]: o for o in turn["ops"]}
    self.assertEqual(by_name["forward_backward"]["request_id"], "r1")
    self.assertEqual(by_name["forward_backward"]["run_id"], "run-a")
    self.assertEqual(by_name["sample"]["count"], 5)
    self.assertLessEqual(turn["requested_at"], turn["started_at"])
    for recorded in turn["ops"]:
      self.assertLessEqual(turn["started_at"], recorded["start"])
      self.assertLessEqual(recorded["start"], recorded["end"])
      self.assertLessEqual(recorded["end"], turn["at"])

  async def test_operations_and_turns_share_worker_identity(self) -> None:
    with mock.patch.dict("os.environ", {"POD_NAME": "pod", "POD_UID": "uid", "NODE_NAME": "node"}):
      async with ops.gpu_turn(SlicerStub(), types.SimpleNamespace(name="worker"), "trainer", "run-a"):
        await ops.observe_operation({"model_id": "run-a"}, "trainer", "run-a", mock.AsyncMock(return_value={}))
    (operation,), (turn,) = await self.recorded(ops.key("run-a")), await self.recorded(ops.turn_key("worker"))
    for field in ("pod", "pod_uid", "node"):
      self.assertEqual(operation[field], turn[field])

  async def test_cancellation_propagates_and_records_status_without_private_content(self) -> None:
    request = {"model_id": "run-a", "op": "train", "payload": "private input"}
    async with ops.gpu_turn(SlicerStub(), types.SimpleNamespace(name="worker"), "trainer", "run-a"):
      with self.assertRaises(asyncio.CancelledError), ops.record_op("block"):
        await ops.observe_operation(request, "trainer", "run-a", mock.AsyncMock(side_effect=asyncio.CancelledError("private error")))
    (sample,), (turn,) = await self.recorded(ops.key("run-a")), await self.recorded(ops.turn_key("worker"))
    self.assertEqual((sample["status"], sample["error_type"]), ("cancelled", "CancelledError"))
    self.assertEqual([(op["name"], op["status"]) for op in turn["ops"]], [("train", "cancelled"), ("block", "failed")])
    self.assertNotIn("private", repr((sample, turn)))

  async def test_store_failures_do_not_change_the_operation_result(self) -> None:
    result = {"type": "ok", "output": "private tokens"}
    with mock.patch.object(self.store, "append_sample", mock.AsyncMock(side_effect=ConnectionError("store offline"))):
      self.assertIs(await ops.observe_operation({"model_id": "r"}, "trainer", "r", mock.AsyncMock(return_value=result)), result)
      await asyncio.gather(*list(ops._writes), return_exceptions=True)

  async def test_successful_optimizer_steps_count_without_rewriting_model_metadata(self) -> None:
    metadata = '{"base_model": "Qwen/Qwen3-8B", "session_ids": ["live"]}'
    await self.store.set_value("open_rl:model_meta:r", metadata)
    for operation, result in (("optim_step", {"type": "ok"}), ("optim_step", {"type": "RequestFailedResponse"}), ("forward", {})):
      await ops.observe_operation({"model_id": "r", "op": operation}, "trainer", "r", mock.AsyncMock(return_value=result))
    with self.assertRaises(ValueError):
      await ops.observe_operation({"model_id": "r", "op": "optim_step"}, "trainer", "r", mock.AsyncMock(side_effect=ValueError()))
    await self.recorded(ops.key("r"))
    self.assertEqual(await self.store.get_value("open_rl:model_meta:r"), metadata)
    self.assertEqual(await self.store.get_value(ops.steps_key("r")), "1")
    self.assertEqual((await metadata_of(self.store))[0]["steps"], 1)

  async def test_shared_sampler_samples_are_keyed_by_logical_run(self) -> None:
    request = {"model_id": "base", "logical_run_id": "run-a", "lora_id": "tinker://run-a/sampler_weights/step-1"}
    await ops.observe_operation(request, "sampler", "base", mock.AsyncMock(return_value={}))
    self.assertEqual(await self.recorded(ops.key("base")), [])
    sample = (await self.recorded(ops.key("run-a")))[0]
    self.assertEqual((sample["run_id"], sample["runtime_id"], sample["operation"]), ("run-a", "base", "sample"))
