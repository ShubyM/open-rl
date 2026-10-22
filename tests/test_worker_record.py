import asyncio
import contextlib
import json
import types
import unittest
from unittest import mock

from server import telemetry
from server.dashboard.snapshot import metadata_of
from server.store import InMemoryStateStore


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
    self.enterContext(mock.patch.object(telemetry, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.dict("os.environ", {"POD_NAME": "pod", "POD_UID": "uid", "NODE_NAME": "node"}))
    self.enterContext(mock.patch.dict(telemetry.steps_done, clear=True))

  async def saved(self, worker_id: str) -> dict:
    await asyncio.sleep(0)  # let the heartbeat run
    return json.loads(await self.store.get_value(telemetry.worker_key(worker_id)))

  async def recorded(self, key: str) -> list[dict]:
    await asyncio.gather(*list(telemetry.writes))  # writes are fire-and-forget
    return await self.store.read_samples(key)

  async def test_the_record_shows_identity_phase_and_a_clean_exit(self) -> None:
    async with telemetry.serving("trainer", "run-a") as worker:
      record = await self.saved(worker.record["id"])
      self.assertEqual((record["pod"], record["node"], record["role"], record["runtime_id"]), ("pod", "node", "trainer", "run-a"))
      self.assertEqual(record["phase"], "starting")
      self.assertTrue(telemetry.alive(record, record["heartbeat_at"]))
      telemetry.set_phase("ready")
    record = await self.saved(worker.record["id"])
    self.assertEqual(record["phase"], "exited")
    self.assertFalse(telemetry.alive(record, record["heartbeat_at"]))
    self.assertIsNone(telemetry.current)

  async def test_each_process_echoes_its_settings_once_without_secrets(self) -> None:
    env = {"OPEN_RL_FINE_TUNING_TYPE": "lora", "VLLM_MAX_LORAS": "8", "OPEN_RL_API_KEY": "sk-private", "HOME": "/root"}
    with mock.patch.dict("os.environ", env), mock.patch("builtins.print") as printed:
      async with telemetry.serving("trainer", "run-a") as worker:
        record = await self.saved(worker.record["id"])
    self.assertEqual(record["config"]["env"], {"OPEN_RL_FINE_TUNING_TYPE": "lora", "VLLM_MAX_LORAS": "8"})
    (echo,) = [json.loads(c.args[0]) for c in printed.call_args_list if "openrl.config" in str(c.args[0])]
    self.assertEqual((echo["openrl"]["role"], echo["openrl"]["env"]), ("trainer", record["config"]["env"]))
    self.assertNotIn("sk-private", repr(printed.call_args_list))

  async def test_popped_requests_stay_outstanding_until_their_answer_is_written(self) -> None:
    async with telemetry.serving("trainer", "base") as worker:
      telemetry.popped([{"request_id": f"f{i}", "op": "forward_backward", "model_id": "run-a"} for i in range(3)] + [{"op": "no id"}])
      telemetry.popped([{"request_id": "s1", "model_id": "base", "logical_run_id": "run-b"}])
      await telemetry.Answering(StoreStub()).set_future("f0", {})
      with self.assertRaises(ConnectionError):
        await telemetry.Answering(StoreStub(fail=True)).set_future("f1", {})
      groups = {(g["op"], g["run_id"]): g["count"] for g in worker.snapshot()["outstanding"]}
      self.assertEqual(groups, {("forward_backward", "run-a"): 2, ("unknown", "run-b"): 1})
    telemetry.popped([{"request_id": "after"}])  # outside a worker: nothing to note

  async def test_gpu_turns_and_the_wait_for_them_show_on_the_record(self) -> None:
    slicer, workload = SlicerStub(), types.SimpleNamespace(name="w")
    async with telemetry.serving("trainer", "run-a") as worker:
      async with telemetry.gpu_turn(slicer, workload):
        self.assertEqual(worker.record["phase"], "on_gpu")
        await asyncio.sleep(0.01)
      async with telemetry.gpu_turn(slicer, workload):
        pass
      self.assertEqual((slicer.acquired, worker.record["phase"]), (2, "ready"))
      (a, b), (c, d) = worker.record["turns"]
      self.assertGreaterEqual(b, a + 0.01)
      self.assertLessEqual(b, c)
      for _ in range(telemetry.MAX_TURNS):
        async with telemetry.gpu_turn(slicer, workload):
          pass
      self.assertEqual(len(worker.record["turns"]), telemetry.MAX_TURNS)

  async def test_worker_level_ops_are_recorded_under_the_runtime(self) -> None:
    with telemetry.record_op("wake_up"):
      pass  # not serving: nowhere to record it
    async with telemetry.serving("sampler", "base"):
      with self.assertRaises(RuntimeError), telemetry.record_op("weight_sync"):
        raise RuntimeError("bad weights")
    (sample,) = await self.recorded(telemetry.ops_key("base"))
    self.assertEqual((sample["operation"], sample["status"], sample["role"], sample["pod_uid"]), ("weight_sync", "failed", "sampler", "uid"))

  async def test_cancellation_propagates_and_records_status_without_private_content(self) -> None:
    request = {"model_id": "run-a", "op": "train", "payload": "private input"}
    with self.assertRaises(asyncio.CancelledError):
      await telemetry.observe_operation(request, "trainer", "run-a", mock.AsyncMock(side_effect=asyncio.CancelledError("private error")))
    (sample,) = await self.recorded(telemetry.ops_key("run-a"))
    self.assertEqual((sample["status"], sample["error_type"]), ("cancelled", "CancelledError"))
    self.assertNotIn("private", repr(sample))

  async def test_nonfinite_metrics_are_named_instead_of_dropped(self) -> None:
    result = {"metrics": {"loss": 1.5, "grad_norm": float("nan"), "kl": float("inf")}}
    await telemetry.observe_operation({"model_id": "r", "op": "optim_step"}, "trainer", "r", mock.AsyncMock(return_value=result))
    (sample,) = await self.recorded(telemetry.ops_key("r"))
    self.assertEqual((sample["metrics"], sample["nonfinite"]), ({"loss": 1.5}, ["grad_norm", "kl"]))

  async def test_trainer_ops_carry_their_step_and_sampling_never_evicts_them(self) -> None:
    await self.store.set_value(telemetry.steps_key("r"), "5")  # a restarted trainer resumes the run's count
    for op in ("forward_backward", "optim_step", "forward_backward"):
      await telemetry.observe_operation({"model_id": "r", "op": op}, "trainer", "r", mock.AsyncMock(return_value={}))
    with mock.patch.object(telemetry, "SAMPLE_LIMIT", 3):
      for i in range(5):
        telemetry.record(
          {"op": "sample", "logical_run_id": "r", "request_id": f"s{i}"}, "sampler", None, 0.0, "succeeded", None, {}, where={"pod": "p"}
        )
    trainer = await self.recorded(telemetry.ops_key("r"))
    self.assertEqual([(s["operation"], s["step"]) for s in trainer], [("forward_backward", 5), ("optim_step", 5), ("forward_backward", 6)])
    samples = await self.recorded(telemetry.samples_key("r"))
    self.assertEqual([s["request_id"] for s in samples], ["s2", "s3", "s4"])
    self.assertNotIn("step", samples[0])

  async def test_forward_only_calls_are_recorded_apart_from_training(self) -> None:
    await telemetry.observe_operation(
      {"model_id": "r", "op": "forward_backward", "forward_only": True}, "trainer", "r", mock.AsyncMock(return_value={})
    )
    (sample,) = await self.recorded(telemetry.ops_key("r"))
    self.assertEqual(sample["operation"], "forward")

  async def test_the_ledger_counts_popped_and_answered_requests_per_run(self) -> None:
    async with telemetry.serving("trainer", "r"):
      telemetry.popped([{"request_id": "a", "model_id": "r", "op": "optim_step"}, {"request_id": "b", "model_id": "r", "op": "optim_step"}])
      await telemetry.Answering(StoreStub()).set_future("a", {})
      await asyncio.gather(*list(telemetry.writes))
    counts = [await self.store.get_value(telemetry.ledger_key("r", event)) for event in ("popped", "answered")]
    self.assertEqual(counts, ["2", "1"])

  async def test_store_failures_do_not_change_the_operation_result(self) -> None:
    result = {"type": "ok", "output": "private tokens"}
    with mock.patch.object(self.store, "append_sample", mock.AsyncMock(side_effect=ConnectionError("store offline"))):
      self.assertIs(await telemetry.observe_operation({"model_id": "r"}, "trainer", "r", mock.AsyncMock(return_value=result)), result)
      await asyncio.gather(*list(telemetry.writes), return_exceptions=True)

  async def test_successful_optimizer_steps_count_without_rewriting_model_metadata(self) -> None:
    metadata = '{"base_model": "Qwen/Qwen3-8B", "session_ids": ["live"]}'
    await self.store.set_value("open_rl:model_meta:r", metadata)
    for operation, result in (("optim_step", {"type": "ok"}), ("optim_step", {"type": "RequestFailedResponse"}), ("forward", {})):
      await telemetry.observe_operation({"model_id": "r", "op": operation}, "trainer", "r", mock.AsyncMock(return_value=result))
    with self.assertRaises(ValueError):
      await telemetry.observe_operation({"model_id": "r", "op": "optim_step"}, "trainer", "r", mock.AsyncMock(side_effect=ValueError()))
    await self.recorded(telemetry.ops_key("r"))
    self.assertEqual(await self.store.get_value("open_rl:model_meta:r"), metadata)
    self.assertEqual(await self.store.get_value(telemetry.steps_key("r")), "1")
    self.assertEqual((await metadata_of(self.store))[0]["steps"], 1)

  async def test_shared_sampler_samples_are_keyed_by_logical_run(self) -> None:
    request = {"model_id": "base", "logical_run_id": "run-a", "lora_id": "tinker://run-a/sampler_weights/step-1"}
    await telemetry.observe_operation(request, "sampler", "base", mock.AsyncMock(return_value={}))
    self.assertEqual(await self.recorded(telemetry.ops_key("base")), [])
    sample = (await self.recorded(telemetry.samples_key("run-a")))[0]
    self.assertEqual((sample["run_id"], sample["runtime_id"], sample["operation"], sample["pod_uid"]), ("run-a", "base", "sample", "uid"))
