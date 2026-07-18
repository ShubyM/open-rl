import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server import gateway
from server.protocol import SamplerSnapshot
from server.run_metadata import MODEL_META_PREFIX, RUN_META_PREFIX, durable_metadata, update_run_metadata
from server.store import InMemoryStore, put_sampler_snapshot
from server.worker_manager import FFTWorkerManager


class StoreStub(InMemoryStore):
  def __init__(self):
    super().__init__()
    self.forwarded_requests = []
    self.sampling_requests = []
    self.futures = {}
    self.kv_store = {}

  async def put_request(self, req_data: dict) -> None:
    self.forwarded_requests.append(req_data)

  async def set_future(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result

  async def set_value(self, key: str, value: str, expires_seconds: int | None = None) -> None:
    self.kv_store[key] = value

  async def get_value(self, key: str) -> str | None:
    return self.kv_store.get(key)

  def get_value_sync(self, key: str) -> str | None:
    return self.kv_store.get(key)

  async def put_sampling_request(self, req_data: dict) -> None:
    self.sampling_requests.append(req_data)


class WorkerManagerStub:
  def __init__(self, error: Exception | None = None):
    self.error = error
    self.instance_id = None
    self.launched_model_ids = []
    self.launched_trainer_model_ids = []
    self.launched_sampler_model_ids = []
    self.shutdown_model_ids = []

  def launch(self, model_id: str, base_model: str | None = None) -> str | None:
    del base_model
    self.launched_model_ids.append(model_id)
    if self.error is not None:
      raise self.error
    return self.instance_id

  def launch_trainer(self, model_id: str, base_model: str | None = None) -> str | None:
    self.launched_trainer_model_ids.append(model_id)
    return self.launch(model_id, base_model)

  def launch_sampler(self, model_id: str, base_model: str | None = None) -> str | None:
    self.launched_sampler_model_ids.append(model_id)
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
    self.get_store_patch = patch.object(gateway, "get_store", return_value=self.store)
    self.get_store_patch.start()
    self.addCleanup(self.get_store_patch.stop)
    self.addCleanup(self._restore)

  def _restore(self) -> None:
    gateway.store = self.old_store
    gateway.fft_worker_manager = self.old_manager

  async def test_create_model_launches_worker_then_enqueues(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    request = self.store.forwarded_requests[0]
    self.assertEqual(request["op"], "create_model")
    self.assertEqual(request["model_id"], model_id)
    self.assertEqual(request["payload"], {})
    meta = json.loads(self.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "base-model")

  async def test_create_model_retry_reuses_identity_and_does_not_enqueue_twice(self) -> None:
    request = {"base_model": "base-model", "session_id": "session-a", "model_seq_id": 7}
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      first = await gateway.create_model(request)
      second = await gateway.create_model(request)

    self.assertEqual(first, second)
    self.assertEqual(self.worker_manager.launched_model_ids, [first["request_id"]])
    self.assertEqual(len(self.store.forwarded_requests), 1)

  async def test_optimizer_retry_with_the_same_sequence_is_enqueued_once(self) -> None:
    request = {"model_id": "model-a", "seq_id": 9, "adam_params": {"learning_rate": 1e-4}}
    first = await gateway.optim_step(request)
    second = await gateway.optim_step(request)

    self.assertEqual(first, {"request_id": "model-a:optim_step:9"})
    self.assertEqual(second, first)
    self.assertEqual(len(self.store.forwarded_requests), 1)

  async def test_create_model_keeps_only_valid_tracker_url_from_user_metadata(self) -> None:
    metadata_store = InMemoryStore()
    with (
      patch.object(gateway, "get_store", return_value=metadata_store),
      patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}),
    ):
      result = await gateway.create_model(
        {
          "base_model": "base-model",
          "user_metadata": {
            "run_name": "qwen smoke",
            "wandb_url": "https://wandb.ai/acme/project/runs/123",
            "secret": "must-not-be-stored",
          },
        }
      )

    for prefix in (MODEL_META_PREFIX, RUN_META_PREFIX):
      with self.subTest(prefix=prefix):
        raw = await metadata_store.get_value(f"{prefix}{result['request_id']}")
        metadata = json.loads(raw)
        self.assertEqual(metadata["name"], "qwen smoke")
        self.assertEqual(metadata["tracker_url"], "https://wandb.ai/acme/project/runs/123")
        self.assertNotIn("secret", metadata)

  async def test_delete_model_preserves_durable_identity_but_removes_active_model(self) -> None:
    metadata_store = InMemoryStore()
    with (
      patch.object(gateway, "get_store", return_value=metadata_store),
      patch.object(gateway, "store", metadata_store),
      patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "false"}),
    ):
      created = await gateway.create_model(
        {
          "base_model": "Qwen/Qwen2.5-0.5B",
          "user_metadata": {
            "name": "durable qwen run",
            "wandb_url": "https://wandb.ai/acme/project/runs/durable",
          },
        }
      )
      model_id = created["request_id"]
      result = await gateway.delete_model({"model_id": model_id})

    self.assertEqual(result, {"status": "ok"})
    self.assertIsNone(await metadata_store.get_value(f"{MODEL_META_PREFIX}{model_id}"))
    self.assertIsNone(await metadata_store.get_value(f"open_rl:model_base:{model_id}"))
    durable = json.loads(await metadata_store.get_value(f"{RUN_META_PREFIX}{model_id}"))
    self.assertEqual(durable["name"], "durable qwen run")
    self.assertEqual(durable["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(durable["training_kind"], "full")
    self.assertEqual(durable["tracker_url"], "https://wandb.ai/acme/project/runs/durable")
    self.assertIn("stopped_at", durable)

    # A late lifecycle update must not turn the stopped run into an active model.
    await update_run_metadata(
      metadata_store,
      model_id,
      {"training_kind": "full"},
      update_active=True,
    )
    self.assertIsNone(await metadata_store.get_value(f"{MODEL_META_PREFIX}{model_id}"))

  async def test_create_model_from_state_records_identity_without_known_base_model(self) -> None:
    metadata_store = InMemoryStore()
    with (
      patch.object(gateway, "get_store", return_value=metadata_store),
      patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}),
    ):
      result = await gateway.create_model_from_state(
        {
          "state_path": "/tmp/checkpoint",
          "user_metadata": {"display_name": "restored run"},
        }
      )

    model_id = result["request_id"]
    active = json.loads(await metadata_store.get_value(f"{MODEL_META_PREFIX}{model_id}"))
    durable = json.loads(await metadata_store.get_value(f"{RUN_META_PREFIX}{model_id}"))
    self.assertEqual(active["name"], "restored run")
    self.assertIsNone(active["base_model"])
    self.assertEqual(durable, durable_metadata(active))

  async def test_create_result_fills_active_and_durable_restored_identity(self) -> None:
    metadata_store = InMemoryStore()
    with (
      patch.object(gateway, "get_store", return_value=metadata_store),
      patch.object(gateway, "store", metadata_store),
      patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}),
    ):
      created = await gateway.create_model_from_state({"state_path": "/tmp/checkpoint"})
      model_id = created["request_id"]
      await metadata_store.set_future(
        model_id,
        {
          "type": "model_loaded_from_state",
          "model_id": model_id,
          "base_model": "Qwen/Qwen2.5-0.5B",
          "training_kind": "full",
        },
      )
      response = await gateway.retrieve_future({"request_id": model_id})

    self.assertEqual(response["base_model"], "Qwen/Qwen2.5-0.5B")
    for prefix in (MODEL_META_PREFIX, RUN_META_PREFIX):
      with self.subTest(prefix=prefix):
        metadata = json.loads(await metadata_store.get_value(f"{prefix}{model_id}"))
        self.assertEqual(metadata["base_model"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(metadata["training_kind"], "full")

  async def test_create_model_failed_launch_fails_future_and_enqueues_nothing(self) -> None:
    self.worker_manager.error = RuntimeError("boom")

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}), patch("server.gateway.traceback.print_exc"):
      result = await gateway.create_model({"base_model": "base-model"})

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(self.store.forwarded_requests, [])
    self.assertEqual(self.store.futures[model_id], {"type": "RequestFailedResponse", "error_message": "boom"})

  async def test_create_model_from_state_launches_worker_then_enqueues(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      result = await gateway.create_model_from_state(
        {
          "state_path": "/tmp/checkpoint",
          "base_model": "restored-base",
          "full_config": {"weight_sync_strategy": "delta"},
          "restore_optimizer": True,
        }
      )

    model_id = result["request_id"]
    self.assertEqual(self.worker_manager.launched_model_ids, [model_id])
    self.assertEqual(len(self.store.forwarded_requests), 1)
    req_forwarded = self.store.forwarded_requests[0]
    self.assertEqual(req_forwarded["op"], "create_model_from_state")
    self.assertEqual(req_forwarded["payload"]["state_path"], "/tmp/checkpoint")
    self.assertTrue(req_forwarded["payload"]["restore_optimizer"])

    # Assert canonical metadata persistence:
    meta = json.loads(self.store.get_value_sync(f"open_rl:model_meta:{model_id}"))
    self.assertEqual(meta["base_model"], "restored-base")
    self.assertEqual(meta["training_kind"], "restored")
    self.assertEqual(meta["full_config"]["weight_sync_strategy"], "delta")

    # Assert no dual-key writing:
    self.assertIsNone(self.store.get_value_sync(f"open_rl:model_base:{model_id}"))

  async def test_ensure_sampler_launched_delegates_to_worker_manager_with_model_id(self) -> None:
    import json

    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm"}):
      self.store.kv_store["open_rl:model_meta:model-x"] = json.dumps(
        {
          "base_model": "base-vllm",
          "weight_sync_strategy": "delta",
          "training_kind": "full",
        }
      )
      await gateway.ensure_sampler_launched("model-x")

    self.assertEqual(self.worker_manager.launched_sampler_model_ids, ["model-x"])

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

  async def test_ephemeral_sampler_saves_queue_unique_immutable_sessions(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "vllm"}):
      await gateway.save_weights_for_sampler({"model_id": "model-a", "sampling_session_seq_id": 3})
      await gateway.save_weights_for_sampler({"model_id": "model-a", "sampling_session_seq_id": 5})

    first = self.store.forwarded_requests[0]["payload"]
    second = self.store.forwarded_requests[1]["payload"]
    self.assertIsNone(first["path"])
    self.assertIsNone(second["path"])
    self.assertEqual(first["sampling_session_id"], "tinker://model-a/sampler_weights/sampler-3")
    self.assertEqual(second["sampling_session_id"], "tinker://model-a/sampler_weights/sampler-5")

  async def test_explicit_mock_sampler_preserves_order_without_checkpoint(self) -> None:
    with patch.dict(
      "os.environ",
      {
        "OPEN_RL_ENABLE_FFT": "true",
        "OPEN_RL_MOCK_SAMPLER": "1",
        "SAMPLING_BACKEND": "vllm",
      },
    ):
      await gateway.save_weights_for_sampler({"model_id": "model-a", "sampling_session_seq_id": 3})

    request = self.store.forwarded_requests[0]
    self.assertEqual(request["op"], "save_weights_for_sampler")
    self.assertTrue(request["payload"]["skip_checkpoint"])

  async def test_sampling_resolves_immutable_snapshot_record(self) -> None:
    session_id = "tinker://model-a/sampler_weights/sampler-5"
    await put_sampler_snapshot(
      self.store,
      SamplerSnapshot(
        sampling_session_id=session_id,
        model_id="model-a",
        revision=5,
        storage_path="tinker://model-a/sampler_weights/revisions/5",
        created_at=1.0,
      ),
    )
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}):
      await gateway.asample(
        {
          "model_id": session_id,
          "prompt": {"chunks": [{"tokens": [1, 2]}]},
          "sampling_params": {"max_tokens": 3},
        }
      )

    request = self.store.sampling_requests[0]
    self.assertTrue(request["weights_path"].endswith("/sampler_full/model-a/sampler_weights/revisions/5"))
    self.assertEqual(request["weights_revision"], "model-a:5")
    self.assertEqual(request["sampling_session_id"], session_id)


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
    self.assertIn("model_id=Model_A.1", popen.call_args.args[0])
    self.assertTrue(kwargs["start_new_session"])
    self.assertEqual(kwargs["env"]["OPEN_RL_ENABLE_FFT"], "true")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_JOB_ID"], "trainer-Model_A.1")
    self.assertEqual(kwargs["env"]["OPEN_RL_TIME_SLICE_GROUP"], "trainers")

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

  async def test_launch_fetches_metadata_from_store(self) -> None:
    import json

    from server.store import InMemoryStore

    s = InMemoryStore()
    s.kv_store["open_rl:model_meta:Model_A.1"] = json.dumps(
      {
        "base_model": "base-model-a",
        "weight_sync_strategy": "delta",
        "training_kind": "full",
      }
    )

    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "SAMPLING_BACKEND": "vllm"}, clear=True),
      patch("server.store.get_store", return_value=s),
      patch("server.worker_manager.subprocess.Popen") as popen,
    ):
      manager = FFTWorkerManager()
      manager.launch_trainer("Model_A.1")
      _, kwargs = popen.call_args
      self.assertEqual(kwargs["env"].get("BASE_MODEL"), "base-model-a")
      self.assertEqual(kwargs["env"].get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "delta")

      manager.launch_sampler("Model_A.1")
      _, kwargs_s = popen.call_args
      self.assertEqual(kwargs_s["env"].get("BASE_MODEL"), "base-model-a")
      self.assertEqual(kwargs_s["env"].get("OPEN_RL_WEIGHT_SYNC_STRATEGY"), "delta")

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

  async def test_cluster_snapshot_does_not_count_completed_process_as_failed(self) -> None:
    with patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379"}, clear=True):
      manager = FFTWorkerManager()
    manager.train_processes["completed"] = SimpleNamespace(poll=lambda: 0, pid=101)
    manager.train_processes["failed"] = SimpleNamespace(poll=lambda: 2, pid=102)

    snapshot = manager.cluster_snapshot()

    self.assertEqual(snapshot["summary"]["failed_pods"], 1)

  async def test_local_snapshot_reports_gpu_usage_and_configured_devices(self) -> None:
    gpu_output = SimpleNamespace(stdout="0, NVIDIA L4, 23034, 4096, 72\n")
    with (
      patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379", "CUDA_VISIBLE_DEVICES": "0"}, clear=True),
      patch("server.worker_manager.subprocess.run", return_value=gpu_output),
      patch("server.worker_manager.local_time_slicer_status", return_value={"ok": True, "workloads": []}),
    ):
      snapshot = FFTWorkerManager().cluster_snapshot()

    node = snapshot["nodes"][0]
    self.assertEqual(node["capacity"]["nvidia.com/gpu"], "1")
    self.assertEqual(node["configured_cuda_devices"], "0")
    self.assertEqual(node["gpus"][0]["memory_used_mib"], 4096)
    self.assertEqual(node["gpus"][0]["utilization_percent"], 72)

  async def test_vm_stack_exposes_bounded_log_tail_to_agents(self) -> None:
    with tempfile.TemporaryDirectory() as directory:
      log_path = Path(directory) / "gateway.log"
      log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
      with patch.dict(
        "os.environ",
        {"REDIS_URL": "redis://localhost:6379", "OPEN_RL_LOG_DIR": directory},
        clear=True,
      ):
        result = FFTWorkerManager().read_logs("model-a", "trainer", tail_lines=2)

    self.assertEqual(result["source"], "file")
    self.assertEqual(result["logs"], "two\nthree\n")
    self.assertIsNone(result["error"])


class GatewayMetadataExtractionTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = StoreStub()
    self.old_store = gateway.store
    gateway.store = self.store
    self.get_store_patch = patch.object(gateway, "get_store", return_value=self.store)
    self.get_store_patch.start()
    self.addCleanup(self.get_store_patch.stop)
    self.addCleanup(self._restore)

  def _restore(self) -> None:
    gateway.store = self.old_store

  async def test_extract_and_persist_metadata_from_headers(self) -> None:
    import json

    from fastapi import Request

    scope = {
      "type": "http",
      "headers": [
        (b"x-open-rl-weight-sync-strategy", b"delta"),
        (b"x-open-rl-training-kind", b"lora"),
      ],
    }
    request = Request(scope)
    model_id = await gateway._extract_and_persist_model_metadata(
      {"base_model": "Qwen/Qwen2.5-0.5B"},
      request,
      default_training_kind="full",
    )

    meta_val = self.store.kv_store.get(f"open_rl:model_meta:{model_id}")
    self.assertIsNotNone(meta_val)
    meta_dict = json.loads(meta_val)
    self.assertEqual(meta_dict["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(meta_dict["training_kind"], "lora")
    self.assertEqual(meta_dict["weight_sync_strategy"], "delta")


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
