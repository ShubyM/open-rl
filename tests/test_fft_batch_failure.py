import asyncio
import contextlib
import os
import unittest
from unittest.mock import patch

from server import training_requests_processor as trp
from server.queue import queue_key, train_queue
from server.store import InMemoryStore, RedisStore
from server.training_requests_processor import TrainingRequestsProcessor
from tests.test_queue import FakeRedis
from training import commands


class SlicerStub:
  faulted = None

  @contextlib.asynccontextmanager
  async def acquire(self, workload):
    yield

  async def register(self, workload):
    return {"ok": True}

  async def unregister(self, workload):
    return {"ok": True}

  async def close(self):
    pass


class BrokenWorker:
  cpu_offload = True

  def wake_up(self):
    raise RuntimeError("CUDA out of memory")

  def sleep(self):
    pass


class BatchStore(InMemoryStore):
  def __init__(self, batch):
    super().__init__()
    self.batch = batch
    self.futures = {}

  async def get_requests_for_model(self, model_id):
    batch, self.batch = self.batch, []
    return batch

  async def set_result(self, request_id, result):
    self.futures[request_id] = result


def processor(store, slicer):
  with patch.dict(os.environ, {"REDIS_URL": "redis://test"}):
    return TrainingRequestsProcessor(store, BrokenWorker(), "run-a", time_slicer=slicer)


class FFTBatchFailureTest(unittest.TestCase):
  def test_a_batch_that_fails_before_answering_fails_every_request(self) -> None:
    store = BatchStore([{"request_id": "fb-1", "op": "forward_backward"}, {"request_id": "os-1", "op": "optim_step"}])
    proc = processor(store, SlicerStub())
    with self.assertRaises(RuntimeError):
      asyncio.run(proc.run_once())
    self.assertEqual(sorted(store.futures), ["fb-1", "os-1"])
    for result in store.futures.values():
      self.assertEqual(result["type"], "RequestFailedResponse")
      self.assertIn("CUDA out of memory", result["error_message"])

  def test_a_worker_the_slicer_could_not_park_exits_without_unregistering(self) -> None:
    store = BatchStore([{"request_id": "sv-1", "op": "save_state"}])
    slicer = SlicerStub()
    proc = processor(store, slicer)
    exits = []

    async def record_exit(unregister=True):
      exits.append(unregister)

    async def handled(request):
      slicer.faulted = "checkpoint failed for workload run-a; it still holds the accelerator and must exit"
      return request["request_id"], {"type": "SaveWeightsResponse"}

    proc.handle_request = handled
    proc.exit_gracefully = record_exit
    asyncio.run(proc.run_once())
    self.assertEqual(store.futures["sv-1"]["type"], "SaveWeightsResponse")
    self.assertEqual(exits, [False])

  def test_poison_pill_acked_without_double_answering_completed_entries(self) -> None:
    async def run() -> None:
      fake = FakeRedis()
      store = RedisStore(fake)
      await store.put_request(commands.wire(commands.OptimStep(request_id="ok-1", model_id="run-a", adam_params={"learning_rate": 1e-3})))
      await store.put_request({"op": "not_a_real_op", "model_id": "run-a"})  # malformed poison pill with request_id=None
      await store.put_request(commands.wire(commands.OptimStep(request_id="after-1", model_id="run-a", adam_params={"learning_rate": 1e-3})))

      class OkWorker:
        def optim_step(self, adam_params, model_id):
          return {"metrics": {"grad_norm:mean": 0.1}}

      answered_calls: list[tuple[str, str]] = []
      with (
        patch.dict(os.environ, {"REDIS_URL": "redis://test"}),
        patch.object(trp, "answered", side_effect=lambda rid, res: answered_calls.append((rid, res["type"]))),
      ):
        proc = TrainingRequestsProcessor(store, OkWorker(), "run-a")
        with self.assertRaises(ValueError):
          await proc.run_once()

      # ok-1 succeeded once (not re-failed or double-reported), poison pill was acked, after-1 failed cleanly.
      self.assertEqual(await store.queued("run-a"), 0)
      self.assertEqual((await store.get_result("ok-1", timeout=1))["type"], "optim_step_completed")
      self.assertEqual((await store.get_result("after-1", timeout=1))["type"], "RequestFailedResponse")
      self.assertEqual(answered_calls, [("ok-1", "optim_step_completed"), ("after-1", "RequestFailedResponse")])

    asyncio.run(run())

  def test_shutdown_survives_batch_failure_and_deletes_empty_train_stream(self) -> None:
    async def run() -> None:
      fake = FakeRedis()
      store = RedisStore(fake)
      await store.put_request(commands.wire(commands.OptimStep(request_id="opt-1", model_id="run-a", adam_params={"learning_rate": 1e-3})))
      await store.put_request({"op": "shutdown_workers", "model_id": "run-a"})
      exits: list[bool] = []

      async def record_exit(unregister=True):
        exits.append(unregister)

      proc = processor(store, SlicerStub())
      proc.exit_gracefully = record_exit
      with self.assertRaises(RuntimeError):
        await proc.run_once()
      self.assertEqual((await store.get_result("opt-1", timeout=1))["type"], "RequestFailedResponse")
      self.assertEqual(await store.queued("run-a"), 1, "shutdown_workers must remain queued after batch failure")
      self.assertEqual(exits, [])

      await proc.run_once()
      self.assertEqual(exits, [True])
      self.assertEqual(await fake.exists(queue_key(train_queue("run-a"))), 0)

    asyncio.run(run())


if __name__ == "__main__":
  unittest.main()
