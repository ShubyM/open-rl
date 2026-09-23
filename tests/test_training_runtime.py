"""Process ownership and command ordering without a training stack or Redis."""

import asyncio
import subprocess
import sys
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from server.store import InMemoryStore
from server.training_requests_processor import TrainingRequestsProcessor, run_training_requests_processor
from training import commands


class RecordingStore(InMemoryStore):
  def __init__(self):
    super().__init__()
    self.published = []

  async def set_future(self, request_id, result):
    self.published.append(request_id)
    await super().set_future(request_id, result)


class ResidentWorker:
  """An exclusive backend needs only command methods, with no sleep/wake API."""

  def __init__(self):
    self.value = 0
    self.saved_values = []

  def optim_step(self, adam_params, model_id):
    self.value += 1
    return {"metrics": {"step": self.value}}

  def save_state(self, model_id, path, include_optimizer, kind):
    self.saved_values.append(self.value)
    return {"path": path}


class OffloadedWorker(ResidentWorker):
  def __init__(self):
    super().__init__()
    self.cpu_offload = True
    self.asleep = True

  def wake_up(self):
    self.asleep = False

  def sleep(self):
    self.asleep = True

  def optim_step(self, adam_params, model_id):
    assert not self.asleep
    return super().optim_step(adam_params, model_id)

  def save_state(self, model_id, path, include_optimizer, kind):
    assert self.asleep
    return super().save_state(model_id, path, include_optimizer, kind)


class RecordingSlicer:
  def __init__(self, fault_on_release=False):
    self.faulted = None
    self.events = []
    self.fault_on_release = fault_on_release

  async def register(self, workload):
    self.events.append("register")

  async def unregister(self, workload):
    self.events.append("unregister")

  async def close(self):
    self.events.append("close")

  @asynccontextmanager
  async def acquire(self, workload):
    self.events.append("acquire")
    try:
      yield
    finally:
      self.events.append("release")
      if self.fault_on_release:
        self.faulted = "Could not park the trainer"


def step(request_id, model_id="model-a"):
  return commands.wire(commands.OptimStep(request_id=request_id, model_id=model_id))


class TrainingRuntimeTest(unittest.IsolatedAsyncioTestCase):
  async def test_dedicated_resident_runner_owns_only_its_model_and_stops(self):
    store = RecordingStore()
    worker = ResidentWorker()
    await store.put_request(step("other", "model-b"))
    await store.put_request(step("first"))
    await store.put_request(commands.wire(commands.SaveState(request_id="save", model_id="model-a", state_path="snapshot")))
    await store.put_request(step("second"))
    await store.put_request(commands.wire(commands.Shutdown(model_id="model-a")))
    await store.put_request(step("too-late"))

    with patch("server.training_requests_processor.time_slicer_client_from_env", side_effect=AssertionError("No lease on an exclusive GPU")):
      await asyncio.wait_for(run_training_requests_processor(worker, "model-a", store=store), timeout=2)

    self.assertEqual(worker.value, 2)
    self.assertEqual(worker.saved_values, [1])
    self.assertEqual(store.published, ["first", "save", "second", "too-late"])
    self.assertEqual(store.futures_store["too-late"]["type"], "RequestFailedResponse")
    self.assertNotIn("other", store.futures_store)
    self.assertEqual((await store.get_requests_for_model("model-b"))[0]["request_id"], "other")

  async def test_dedicated_reader_wakes_again_after_draining_a_queue(self):
    store = InMemoryStore()
    for request_id in ("first", "second"):
      reader = asyncio.create_task(store.get_requests_for_model("model-a"))
      await asyncio.sleep(0)
      self.assertFalse(reader.done())
      await store.put_request(step(request_id))
      batch = await asyncio.wait_for(reader, timeout=2)
      self.assertEqual([request["request_id"] for request in batch], [request_id])

  async def test_shared_queue_keeps_serving_both_adapters(self):
    store = RecordingStore()
    worker = ResidentWorker()
    processor = TrainingRequestsProcessor(store, worker, active_tenant_set_id="base-1")
    for model_id in ("adapter-a", "adapter-b"):
      await store.put_request(step(model_id, model_id), active_set_id="base-1")
    await processor.run_once()
    await processor.run_once()
    self.assertEqual(store.published, ["adapter-a", "adapter-b"])
    self.assertFalse(processor.stopping)

  async def test_rejects_ambiguous_queue_ownership(self):
    with self.assertRaisesRegex(ValueError, "not both"):
      TrainingRequestsProcessor(InMemoryStore(), ResidentWorker(), "model-a", active_tenant_set_id="base-1")

  async def test_rejects_commands_for_another_model(self):
    store = RecordingStore()
    worker = ResidentWorker()
    processor = TrainingRequestsProcessor(store, worker, "model-a")
    await processor.process_request(step("wrong", "model-b"))
    self.assertEqual(worker.value, 0)
    self.assertEqual(store.futures_store["wrong"]["type"], "RequestFailedResponse")

  async def test_leased_runner_preserves_save_position_and_cleans_up(self):
    store = RecordingStore()
    worker = OffloadedWorker()
    slicer = RecordingSlicer()
    await store.put_request(step("first"))
    await store.put_request(commands.wire(commands.SaveState(request_id="save", model_id="model-a", state_path="snapshot")))
    await store.put_request(step("second"))
    await store.put_request(commands.wire(commands.Shutdown(model_id="model-a")))
    await asyncio.wait_for(run_training_requests_processor(worker, "model-a", time_slicer=slicer, store=store), timeout=2)
    self.assertEqual(worker.saved_values, [1])
    self.assertEqual(worker.value, 2)
    self.assertTrue(worker.asleep)
    self.assertEqual(slicer.events, ["register", "acquire", "release", "acquire", "release", "unregister", "close"])
    self.assertEqual(store.published, ["first", "save", "second"])

  async def test_late_batch_failure_does_not_overwrite_published_success(self):
    store = RecordingStore()
    worker = ResidentWorker()
    processor = TrainingRequestsProcessor(store, worker, "model-a")
    await store.put_request(step("first"))
    await store.put_request({"model_id": "model-a", "op": "invalid"})
    await store.put_request(step("unanswered"))
    with self.assertRaises(ValueError):
      await processor.run_once()
    self.assertEqual(worker.value, 1)
    self.assertEqual(store.published, ["first", "unanswered"])
    self.assertEqual(store.futures_store["first"]["type"], "optim_step_completed")
    self.assertEqual(store.futures_store["unanswered"]["type"], "RequestFailedResponse")

  async def test_faulted_lease_stops_before_the_next_compute_group(self):
    store = RecordingStore()
    worker = OffloadedWorker()
    slicer = RecordingSlicer(fault_on_release=True)
    processor = TrainingRequestsProcessor(store, worker, "model-a", time_slicer=slicer)
    await store.put_request(step("first"))
    await store.put_request(commands.wire(commands.SaveState(request_id="save", model_id="model-a", state_path="snapshot")))
    await store.put_request(step("second"))
    with patch.object(processor, "exit_gracefully", new_callable=AsyncMock) as exit_worker:
      with self.assertRaisesRegex(RuntimeError, "Could not park"):
        await processor.run_once()
      exit_worker.assert_awaited_once_with(unregister=False)
    self.assertEqual(worker.value, 1)
    self.assertEqual(slicer.events, ["acquire", "release"])
    self.assertEqual(store.futures_store["first"]["type"], "optim_step_completed")
    self.assertEqual(store.futures_store["second"]["type"], "RequestFailedResponse")

  async def test_faulted_lease_answers_requests_after_shutdown_before_exiting(self):
    store = RecordingStore()
    worker = OffloadedWorker()
    slicer = RecordingSlicer(fault_on_release=True)
    processor = TrainingRequestsProcessor(store, worker, "model-a", time_slicer=slicer)
    await store.put_request(step("first"))
    await store.put_request(commands.wire(commands.Shutdown(model_id="model-a")))
    await store.put_request(step("too-late"))

    async def exit_worker(unregister=True):
      self.assertFalse(unregister)
      self.assertEqual(store.published, ["first", "too-late"])
      self.assertEqual(store.futures_store["too-late"]["type"], "RequestFailedResponse")
      raise SystemExit(0)

    with patch.object(processor, "exit_gracefully", side_effect=exit_worker), self.assertRaises(SystemExit):
      await processor.run_once()
    self.assertEqual(worker.value, 1)

  def test_importing_runtime_does_not_import_training_backends(self):
    subprocess.run(
      [
        sys.executable,
        "-c",
        "import sys; import server.training_requests_processor; "
        "assert all(name not in sys.modules for name in "
        "('torch', 'transformers', 'peft', 'training.lora_trainer_worker', 'training.fft_trainer_worker'))",
      ],
      check=True,
      timeout=30,
    )
