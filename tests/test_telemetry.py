"""Run telemetry: what the recorder writes, and that recording never changes or slows what a worker answers."""

import asyncio
import json
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch

from server import api_server, telemetry
from server.store import InMemoryStateStore, InMemoryStore, StateStore
from server.training_requests_processor import TrainingRequestsProcessor
from tests.api_client import asgi_client
from tests.test_fft_batch_failure import BatchStore, SlicerStub

BOUND = 1.0  # an exiting process waits this long for its last flush; a serve must finish well inside it


class FaultyStore:
  """Every call raises, as with Redis down, or waits forever, as with a Redis that never answers."""

  def __init__(self, hangs: bool):
    self.hangs = hangs

  def __getattr__(self, name):
    if not self.hangs:
      raise ConnectionError(f"telemetry store {name} failed")
    return lambda *args, **kwargs: threading.Event().wait()


def no_store():
  raise RuntimeError("no telemetry store configured")


class RecorderOnlyStore(InMemoryStateStore):
  """Notes, and raises on, any use off the recorder's thread, since a Redis call on a request path is the bug."""

  def __init__(self):
    super().__init__()
    self.intrusions: list[str] = []

  def __getattribute__(self, name):
    if name in StateStore.__abstractmethods__ and threading.current_thread().name != "telemetry":
      super().__getattribute__("intrusions").append(name)
      raise AssertionError(f"{name} called on a request path")
    return super().__getattribute__(name)


def trainer() -> Mock:
  return Mock(cpu_offload=True, optim_step=lambda adam_params, model_id: {"metrics": {"grad_norm:mean": 0.5}})


async def train(worker: Mock, requests: list[dict]) -> dict:
  """One time-sliced trainer batch. Returns what each request was answered."""
  store = BatchStore(requests)
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    processor = TrainingRequestsProcessor(store, worker, "run-a", time_slicer=SlicerStub())
  await processor.run_once()
  return store.futures


def optim_steps(count: int) -> list[dict]:
  return [{"request_id": f"os-{i}", "op": "optim_step", "model_id": "run-a", "payload": {"adam_params": {}}} for i in range(count)]


class TelemetryTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store = InMemoryStateStore()
    self.enterContext(patch.object(telemetry, "get_state_store", lambda: self.store))
    self.enterContext(patch.object(telemetry, "FINISH_SECONDS", BOUND))

  def serve(self, work, role: str = "trainer") -> tuple:
    """Run work while recording. Returns its result, how long it took, and how long the exit after took."""

    async def run():
      async with telemetry.serving(role, "run-a"):
        started = time.monotonic()
        result = await asyncio.wait_for(work(), 2 * BOUND)
        exiting = time.monotonic()
      return result, exiting - started, time.monotonic() - exiting

    return asyncio.run(run())

  def worker_records(self) -> list[dict]:
    return [json.loads(v) for k, v in self.store.kv_store.items() if k.startswith(telemetry.key("worker", ""))]


class FaultTest(TelemetryTest):
  """The same work, with a healthy telemetry store and with one that fails, gets the same answers in about the same time."""

  def test_a_failing_hanging_or_missing_store_changes_no_answer(self) -> None:
    expected, _, _ = self.serve(lambda: train(trainer(), optim_steps(2)))
    self.assertEqual(self.store.hashes[telemetry.key("ledger", "run-a")], {"popped": 2, "answered": 2}, "the healthy serve recorded")
    for fault, store in {"raises": lambda: FaultyStore(False), "hangs": lambda: FaultyStore(True), "is missing": no_store}.items():
      with self.subTest(store=fault), patch.object(telemetry, "get_state_store", store):
        result, took, exiting = self.serve(lambda: train(trainer(), optim_steps(2)))
        self.assertEqual(result, expected)
        self.assertLess(took, BOUND)
        self.assertLess(exiting, 2 * BOUND)

  def test_no_request_path_touches_the_store_and_the_recorder_writes_what_it_saw(self) -> None:
    store = RecorderOnlyStore()
    with patch.object(telemetry, "get_state_store", lambda: store):
      result, _, _ = self.serve(lambda: train(trainer(), optim_steps(2)))
    self.assertEqual([r["type"] for r in result.values()], ["optim_step_completed"] * 2)
    self.assertEqual(store.intrusions, [])
    self.assertEqual(store.hashes[telemetry.key("ledger", "run-a")], {"popped": 2, "answered": 2})


class WorkerRecordTest(TelemetryTest):
  def test_the_record_shows_settings_turns_and_a_clean_exit(self) -> None:
    env = {"OPEN_RL_FINE_TUNING_TYPE": "full", "OPEN_RL_API_KEY": "sk-private", "HOME": "/root"}
    with patch.dict(os.environ, env, clear=True):
      self.serve(lambda: train(trainer(), optim_steps(1)))
    (record,) = self.worker_records()
    self.assertEqual((record["role"], record["runtime_id"], record["phase"]), ("trainer", "run-a", "exited"))
    self.assertEqual(record["config"]["env"], {"OPEN_RL_FINE_TUNING_TYPE": "full"})
    self.assertNotIn("sk-private", json.dumps(record))
    self.assertEqual((len(record["waits"]), len(record["turns"])), (1, 1))

  def test_the_record_stays_fresh_while_the_event_loop_is_blocked(self) -> None:
    async def block():
      blocked_at = time.time()
      time.sleep(0.3)  # loading weights holds the loop, and the recorder must not need it
      return blocked_at

    with patch.object(telemetry, "FLUSH_SECONDS", 0.02), patch.object(self.store, "write_sync", wraps=self.store.write_sync) as write:
      blocked_at, _, _ = self.serve(block)
    beats = [json.loads(v)["heartbeat_at"] for call in write.call_args_list for v in call.args[0].values.values()]
    self.assertTrue(any(beat - blocked_at > 0.2 for beat in beats[:-1]), "a beat landed while the loop was blocked, before the exit's")

  def test_popped_requests_stay_outstanding_until_answered(self) -> None:
    async def pop():
      telemetry.popped([{"request_id": f"f{i}", "op": "forward_backward", "model_id": "run-a"} for i in range(3)])
      # A shared trainer's delete_model was never counted as enqueued, so it is not outstanding either.
      telemetry.popped([{"request_id": "d1", "op": "delete_model", "model_id": "run-a"}])
      telemetry.answered("f0")
      return json.loads(telemetry.worker().snapshot())["outstanding"]

    outstanding, _, _ = self.serve(pop)
    self.assertEqual([(g["op"], g["run_id"], g["count"]) for g in outstanding], [("forward_backward", "run-a", 2)])
    self.assertEqual(self.store.hashes[telemetry.key("ledger", "run-a")], {"popped": 3, "answered": 1})


class LedgerTest(TelemetryTest):
  def setUp(self) -> None:
    super().setUp()
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch("server.worker_manager.get_state_store", return_value=api_server.state))
    self.enterContext(patch.dict(os.environ, {"SAMPLING_BACKEND": "torch"}))

  def test_a_shared_trainers_delete_is_counted_neither_enqueued_nor_popped(self) -> None:
    async def run():
      async with asgi_client() as client:
        created = (await client.post("/api/v1/create_model", json={"base_model": "m"})).json()
        (await client.post("/api/v1/delete_model", json={"model_id": created["request_id"]})).raise_for_status()
      requests = await api_server.store.get_requests(active_set_id="m-1")
      self.assertEqual([r["op"] for r in requests], ["create_model", "delete_model"])
      telemetry.popped(requests)
      return created["request_id"]

    run_id, _, _ = self.serve(run)
    self.assertEqual(self.store.hashes[telemetry.key("ledger", run_id)], {"enqueued": 1, "popped": 1})


if __name__ == "__main__":
  unittest.main()
