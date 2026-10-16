import asyncio
import contextlib
import json
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


class StoreStub:
  def __init__(self, fail=False):
    self.answers, self.fail = [], fail

  async def set_future(self, request_id, result):
    if self.fail:
      raise ConnectionError("store offline")
    self.answers.append(request_id)


class WorkerRecordTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStateStore()
    self.enterContext(mock.patch.object(ops, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.dict("os.environ", {"POD_NAME": "pod", "POD_UID": "uid", "NODE_NAME": "node"}))

  async def saved(self, worker_id: str) -> dict:
    await asyncio.sleep(0)  # let the heartbeat run
    return json.loads(await self.store.get_value(ops.worker_key(worker_id)))

  async def recorded(self, key: str) -> list[dict]:
    await asyncio.gather(*list(ops._writes))  # writes are fire-and-forget
    return await self.store.read_samples(key)

  async def test_the_record_shows_identity_phase_and_a_clean_exit(self) -> None:
    async with ops.serving("trainer", "run-a") as worker:
      record = await self.saved(worker.record["id"])
      self.assertEqual((record["pod"], record["node"], record["role"], record["runtime_id"]), ("pod", "node", "trainer", "run-a"))
      self.assertEqual(record["phase"], "starting")
      self.assertTrue(ops.alive(record, record["heartbeat_at"]))
      ops.set_phase("ready")
    record = await self.saved(worker.record["id"])
    self.assertEqual(record["phase"], "exited")
    self.assertFalse(ops.alive(record, record["heartbeat_at"]))
    self.assertIsNone(ops.current)

  async def test_popped_requests_stay_outstanding_until_their_answer_is_written(self) -> None:
    async with ops.serving("trainer", "base") as worker:
      ops.popped([{"request_id": f"f{i}", "op": "forward_backward", "model_id": "run-a"} for i in range(3)] + [{"op": "no id"}])
      ops.popped([{"request_id": "s1", "model_id": "base", "logical_run_id": "run-b"}])
      await ops.Answering(StoreStub()).set_future("f0", {})
      with self.assertRaises(ConnectionError):
        await ops.Answering(StoreStub(fail=True)).set_future("f1", {})
      groups = {(g["op"], g["run_id"]): g["count"] for g in worker.snapshot()["outstanding"]}
      self.assertEqual(groups, {("forward_backward", "run-a"): 2, ("unknown", "run-b"): 1})
    ops.popped([{"request_id": "after"}])  # outside a worker: nothing to note

  async def test_gpu_turns_and_the_wait_for_them_show_on_the_record(self) -> None:
    slicer, workload = SlicerStub(), types.SimpleNamespace(name="w")
    async with ops.serving("trainer", "run-a") as worker:
      async with ops.gpu_turn(slicer, workload):
        self.assertEqual(worker.record["phase"], "on_gpu")
        await asyncio.sleep(0.01)
      async with ops.gpu_turn(slicer, workload):
        pass
      self.assertEqual((slicer.acquired, worker.record["phase"]), (2, "ready"))
      (a, b), (c, d) = worker.record["turns"]
      self.assertGreaterEqual(b, a + 0.01)
      self.assertLessEqual(b, c)
      for _ in range(ops.MAX_TURNS):
        async with ops.gpu_turn(slicer, workload):
          pass
      self.assertEqual(len(worker.record["turns"]), ops.MAX_TURNS)

  async def test_worker_level_ops_are_recorded_under_the_runtime(self) -> None:
    with ops.record_op("wake_up"):
      pass  # not serving: nowhere to record it
    async with ops.serving("sampler", "base"):
      with self.assertRaises(RuntimeError), ops.record_op("weight_sync"):
        raise RuntimeError("bad weights")
    (sample,) = await self.recorded(ops.key("base"))
    self.assertEqual((sample["operation"], sample["status"], sample["role"], sample["pod_uid"]), ("weight_sync", "failed", "sampler", "uid"))

  async def test_cancellation_propagates_and_records_status_without_private_content(self) -> None:
    request = {"model_id": "run-a", "op": "train", "payload": "private input"}
    with self.assertRaises(asyncio.CancelledError):
      await ops.observe_operation(request, "trainer", "run-a", mock.AsyncMock(side_effect=asyncio.CancelledError("private error")))
    (sample,) = await self.recorded(ops.key("run-a"))
    self.assertEqual((sample["status"], sample["error_type"]), ("cancelled", "CancelledError"))
    self.assertNotIn("private", repr(sample))

  async def test_nonfinite_metrics_are_named_instead_of_dropped(self) -> None:
    result = {"metrics": {"loss": 1.5, "grad_norm": float("nan"), "kl": float("inf")}}
    await ops.observe_operation({"model_id": "r", "op": "optim_step"}, "trainer", "r", mock.AsyncMock(return_value=result))
    (sample,) = await self.recorded(ops.key("r"))
    self.assertEqual((sample["metrics"], sample["nonfinite"]), ({"loss": 1.5}, ["grad_norm", "kl"]))

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
    self.assertEqual((sample["run_id"], sample["runtime_id"], sample["operation"], sample["pod_uid"]), ("run-a", "base", "sample", "uid"))
