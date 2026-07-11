import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server import gateway
from server.worker_manager import FFTWorkerManager


class StoreStub:
  def __init__(self):
    self.forwarded_requests = []
    self.sampling_requests = []
    self.futures = {}

  async def put_request(self, req_data: dict) -> None:
    self.forwarded_requests.append(req_data)

  async def set_future(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result

  async def put_sampling_request(self, req_data: dict) -> None:
    self.sampling_requests.append(req_data)


class WorkerManagerStub:
  def __init__(self, error: Exception | None = None):
    self.error = error
    self.instance_id = None
    self.launched_model_ids = []
    self.shutdown_model_ids = []

  def launch(self, model_id: str, base_model: str | None = None) -> str | None:
    self.launched_model_ids.append(model_id)
    if self.error is not None:
      raise self.error
    return self.instance_id

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> str | None:
    return self.launch(model_id, base_model)

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> str | None:
    return self.launch(model_id, base_model)

  def shutdown(self, model_id: str) -> None:
    self.shutdown_model_ids.append(model_id)

  def shutdown_all(self) -> None:
    pass


class GatewayInlineWorkerLaunchTest(unittest.IsolatedAsyncioTestCase):
  """create_model in FFT mode launches the model's worker directly, then
  enqueues onto its per-model queue — there is no separate launch queue."""

  def setUp(self) -> None:
    self.store = StoreStub()
    self.worker_manager = WorkerManagerStub()
    self.old_store = gateway.store
    self.old_manager = gateway.fft_worker_manager
    gateway.store = self.store
    gateway.fft_worker_manager = self.worker_manager
    self.addCleanup(self.restore_gateway)

  def restore_gateway(self) -> None:
    gateway.store = self.old_store
    gateway.fft_worker_manager = self.old_manager

  async def test_create_model_launches_worker_then_enqueues(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    request = self.store.forwarded_requests[0]
    self.assertEqual(request["op"], "create_model")
    self.assertEqual(request["model_id"], model_id)
    self.assertEqual(request["payload"]["base_model"], "base-model")

  async def test_create_model_failed_launch_fails_future_and_enqueues_nothing(self) -> None:
    self.worker_manager.error = RuntimeError("boom")

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}), patch("server.gateway.traceback.print_exc"):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(self.store.forwarded_requests, [])
    self.assertEqual(self.store.futures[model_id], {"type": "RequestFailedResponse", "error_message": "boom"})

  async def test_create_model_from_state_launches_worker_then_enqueues(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model_from_state({"state_path": "/tmp/checkpoint"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    self.assertEqual(self.store.forwarded_requests[0]["op"], "create_model_from_state")

  async def test_create_model_without_fft_skips_launcher(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "false"}):
      await gateway.create_model({"base_model": "base-model"})

    self.assertEqual(self.worker_manager.launched_model_ids, [])
    self.assertEqual(len(self.store.forwarded_requests), 1)

  async def test_sampler_launch_errors_are_not_swallowed(self) -> None:
    self.worker_manager.error = RuntimeError("image pull failed")

    with (
      patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm"}),
      self.assertRaisesRegex(RuntimeError, "image pull failed"),
    ):
      await gateway.ensure_sampler_launched("model-a", "base-model")

  async def test_sampler_readiness_must_match_launched_instance(self) -> None:
    self.worker_manager.instance_id = "new-instance"

    class RedisStub:
      def __init__(self):
        self.values = [b"stale-instance", b"new-instance"]

      async def get(self, _key):
        return self.values.pop(0)

    with (
      patch.dict(
        "os.environ",
        {
          "OPEN_RL_ENABLE_FFT": "true",
          "OPEN_RL_SAMPLER_READY_TIMEOUT_SECONDS": "1",
          "SAMPLING_BACKEND": "vllm",
        },
      ),
      patch("server.gateway.get_store", return_value=SimpleNamespace(redis=RedisStub())),
      patch("server.gateway.asyncio.sleep", return_value=None),
    ):
      await gateway.wait_for_sampler_ready("model-a", "base-model")

    self.assertEqual(self.worker_manager.launched_model_ids, ["model-a", "model-a"])

  async def test_ephemeral_sampler_checkpoints_rotate_without_reusing_session_ids(self) -> None:
    with patch.dict(
      "os.environ",
      {
        "OPEN_RL_ENABLE_FFT": "true",
        "OPEN_RL_SAMPLER_SNAPSHOT_SLOTS": "2",
        "SAMPLING_BACKEND": "vllm",
      },
    ):
      await gateway.save_weights_for_sampler({"model_id": "model-a", "sampling_session_seq_id": 3})
      await gateway.save_weights_for_sampler({"model_id": "model-a", "sampling_session_seq_id": 5})

    first = self.store.forwarded_requests[0]["payload"]
    second = self.store.forwarded_requests[1]["payload"]
    self.assertEqual(first["path"], "tinker://model-a/sampler_weights/live-slot-1")
    self.assertEqual(second["path"], first["path"])
    self.assertEqual(first["sampling_session_id"], "tinker://model-a/sampler_weights/sampler-3")
    self.assertEqual(second["sampling_session_id"], "tinker://model-a/sampler_weights/sampler-5")
    self.assertEqual(gateway.sampler_storage_ref(second["sampling_session_id"]), second["path"])

  async def test_sampling_uses_rotating_path_and_unique_weight_revision(self) -> None:
    session_id = "tinker://model-a/sampler_weights/sampler-5"
    with patch.dict(
      "os.environ",
      {"OPEN_RL_ENABLE_FFT": "true", "OPEN_RL_SAMPLER_SNAPSHOT_SLOTS": "2"},
    ):
      await gateway.asample(
        {
          "model_id": session_id,
          "prompt": {"chunks": [{"tokens": [1, 2]}]},
          "sampling_params": {"max_tokens": 3},
        }
      )

    request = self.store.sampling_requests[0]
    self.assertTrue(request["weights_path"].endswith("/sampler_full/model-a/sampler_weights/live-slot-1"))
    self.assertEqual(request["weights_revision"], session_id)


class GatewayLifespanTest(unittest.IsolatedAsyncioTestCase):
  async def test_lifespan_full_mode_requires_redis(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      async with gateway.lifespan(gateway.app):
        pass


class FFTWorkerManagerTest(unittest.IsolatedAsyncioTestCase):
  async def test_requires_redis(self) -> None:
    with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
      FFTWorkerManager()

  async def test_local_launch_stamps_workload_tags_and_process_group(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = FFTWorkerManager()
      manager.launch("Model_A.1")

    _, kwargs = popen.call_args
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

  async def test_local_launch_uses_torchrun_for_fsdp(self) -> None:
    with (
      patch.dict(
        "os.environ",
        {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_FSDP_WORLD_SIZE": "2"},
        clear=True,
      ),
      patch("server.worker_manager.subprocess.Popen") as popen,
      patch("server.worker_manager.shutil.which", return_value="/usr/bin/uv"),
    ):
      FFTWorkerManager().launch("model-a")

    command = popen.call_args.args[0]
    self.assertIn("torchrun", command)
    self.assertIn("--nproc-per-node=2", command)
    self.assertEqual(command[-2:], ["--model-id", "model-a"])

  async def test_local_sampler_launch_stamps_workload_tags_and_process_group(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = FFTWorkerManager()
      instance_id = manager.launch_sampler("Model_A.1")

    _, kwargs = popen.call_args
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_MODEL_ID"], "Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_WORKER_INSTANCE_ID"], instance_id)
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "sampler-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "samplers")

  async def test_local_sampler_reuses_instance_id_while_process_is_running(self) -> None:
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm"}, clear=True),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      popen.return_value.poll.return_value = None
      manager = FFTWorkerManager()
      first_instance = manager.launch_sampler("model-a")
      second_instance = manager.launch_sampler("model-a")

    self.assertEqual(first_instance, second_instance)
    popen.assert_called_once()


class GatewayFutureTranslationTest(unittest.TestCase):
  def test_create_model_result_translates_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "model_created",
          "model_id": "model-a",
          "base_model": "base-model",
          "training_kind": "full",
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
          "type": "model_loaded_from_state",
          "model_id": "model-a",
          "base_model": "base-model",
          "training_kind": "full",
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

  def test_lora_create_model_result_translates_rank_to_tinker_shape(self) -> None:
    self.assertEqual(
      gateway.translate_future_result(
        {
          "type": "model_created",
          "model_id": "model-a",
          "base_model": "base-model",
          "rank": 4,
          "training_kind": "lora",
        }
      ),
      {
        "type": "create_model",
        "model_id": "model-a",
        "base_model": "base-model",
        "is_lora": True,
        "lora_rank": 4,
      },
    )

  def test_internal_future_result_types_translate_to_tinker_types(self) -> None:
    cases = [
      ("forward_backward_completed", "forward_backward"),
      ("optim_step_completed", "optim_step"),
      ("sample_completed", "sample"),
      ("state_saved", "save_weights"),
      ("weights_loaded", "load_weights"),
      ("sampler_weights_saved", "save_weights_for_sampler"),
      ("weights_saved", "save_weights"),
    ]

    for internal_type, public_type in cases:
      with self.subTest(internal_type=internal_type):
        self.assertEqual(
          gateway.translate_future_result({"type": internal_type, "path": "/tmp/x"}),
          {"type": public_type, "path": "/tmp/x"},
        )


if __name__ == "__main__":
  unittest.main()
