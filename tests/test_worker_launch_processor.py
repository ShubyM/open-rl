import sys
import unittest
from unittest.mock import patch

from tests._server_fixture import SERVER_DIR

sys.path.insert(0, str(SERVER_DIR))

import gateway  # noqa: E402
from store import InMemoryStore  # noqa: E402
from worker_launch_processor import CreateModelFromStateWorkerLaunchRequest, WorkerLaunchProcessor  # noqa: E402


class StoreStub:
  def __init__(self):
    self.forwarded_requests = []
    self.futures = {}

  async def put_request(self, req_data: dict) -> None:
    self.forwarded_requests.append(req_data)

  async def set_future(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result


class WorkerManagerStub:
  def __init__(self, error: Exception | None = None):
    self.error = error
    self.launched_model_ids = []

  def launch(self, model_id: str) -> None:
    self.launched_model_ids.append(model_id)
    if self.error is not None:
      raise self.error


class WorkerLaunchProcessorTest(unittest.IsolatedAsyncioTestCase):
  async def test_process_request_launches_worker_then_forwards_request(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub()
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {"req_id": "model-a", "model_id": "model-a", "type": "create_model", "base_model": "base-model"}

    await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, ["model-a"])
    self.assertEqual(len(store.forwarded_requests), 1)
    self.assertEqual(store.forwarded_requests[0]["type"], "create_model")
    self.assertEqual(store.forwarded_requests[0]["base_model"], "base-model")
    self.assertEqual(store.futures, {})

  async def test_process_request_sets_future_failure_when_launch_fails(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub(RuntimeError("boom"))
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {"req_id": "model-a", "model_id": "model-a", "type": "create_model", "base_model": "base-model"}

    with patch("worker_launch_processor.traceback.print_exc"):
      await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, ["model-a"])
    self.assertEqual(store.forwarded_requests, [])
    self.assertEqual(
      store.futures["model-a"],
      {"type": "RequestFailedResponse", "error_message": "boom"},
    )

  async def test_process_request_sets_future_failure_without_model_id(self) -> None:
    store = StoreStub()
    worker_manager = WorkerManagerStub()
    processor = WorkerLaunchProcessor(store, worker_manager)
    request = {"req_id": "model-a", "type": "create_model", "base_model": "base-model"}

    with patch("worker_launch_processor.traceback.print_exc"):
      await processor.process_request(request)

    self.assertEqual(worker_manager.launched_model_ids, [])
    self.assertEqual(store.forwarded_requests, [])
    self.assertEqual(store.futures["model-a"]["type"], "RequestFailedResponse")

  def test_create_model_from_state_launch_payload_defaults_restore_optimizer(self) -> None:
    request = CreateModelFromStateWorkerLaunchRequest(
      req_id="model-a",
      model_id="model-a",
      type="create_model_from_state",
      state_path="/tmp/checkpoint",
    )

    self.assertEqual(request.model_dump()["restore_optimizer"], False)


class WorkerLaunchQueueTest(unittest.IsolatedAsyncioTestCase):
  async def test_in_memory_worker_launch_queue_is_unsupported(self) -> None:
    store = InMemoryStore()

    with self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      await store.put_worker_launch_request({"model_id": "model-a"})


class GatewayWorkerLaunchQueueTest(unittest.IsolatedAsyncioTestCase):
  async def test_lifespan_full_mode_requires_redis(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      async with gateway.lifespan(gateway.app):
        pass

  async def test_create_model_full_mode_uses_worker_launch_queue(self) -> None:
    store = StoreStub()
    store.worker_launch_requests = []

    async def put_worker_launch_request(req_data: dict) -> None:
      store.worker_launch_requests.append(req_data)

    store.put_worker_launch_request = put_worker_launch_request
    old_store = gateway.store
    gateway.store = store

    try:
      with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
        result = await gateway.create_model({"base_model": "base-model"})
    finally:
      gateway.store = old_store

    self.assertEqual(result["request_id"], store.worker_launch_requests[0]["model_id"])
    self.assertEqual(store.worker_launch_requests[0]["type"], "create_model")
    self.assertEqual(store.worker_launch_requests[0]["base_model"], "base-model")
    self.assertEqual(store.forwarded_requests, [])


class GatewayFutureTranslationTest(unittest.TestCase):
  def test_create_model_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "create_model_result",
          "model_id": "model-a",
          "base_model": "base-model",
          "lora_rank": 16,
        }
      ),
      {
        "type": "create_model",
        "model_id": "model-a",
        "base_model": "base-model",
        "is_lora": True,
        "lora_rank": 16,
      },
    )

  def test_create_model_from_state_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "create_model_from_state_result",
          "model_id": "model-a",
          "base_model": "base-model",
          "lora_rank": 16,
        }
      ),
      {
        "type": "create_model_from_state",
        "model_id": "model-a",
        "base_model": "base-model",
        "is_lora": True,
        "lora_rank": 16,
      },
    )


if __name__ == "__main__":
  unittest.main()
