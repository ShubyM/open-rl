import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

from server import gateway
from server.lifecycle_store import InMemoryLifecycleStore
from server.store import InMemoryStore
from server.training_requests_processor import FFTTrainingRequestsProcessor, sampler_full_local_path


class ManagerStub:
  def __init__(self):
    self.launched_model_ids = []
    self.shutdown_model_ids = []

  def launch(self, model_id: str) -> None:
    self.launched_model_ids.append(model_id)

  def shutdown(self, model_id: str) -> None:
    self.shutdown_model_ids.append(model_id)

  def shutdown_all(self) -> None:
    pass


class SessionEndpointTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.lifecycle = InMemoryLifecycleStore()
    patch.object(gateway, "store", InMemoryStore()).start()
    patch.object(gateway, "lifecycle", self.lifecycle).start()
    self.addCleanup(patch.stopall)

  async def test_create_session_mints_live_session(self) -> None:
    response = await gateway.create_session({})
    self.assertEqual(response["type"], "create_session")
    self.assertTrue(await self.lifecycle.session_alive(response["session_id"]))

  async def test_sessions_are_unique(self) -> None:
    first = await gateway.create_session({})
    second = await gateway.create_session({})
    self.assertNotEqual(first["session_id"], second["session_id"])

  async def test_heartbeat_keeps_session_alive(self) -> None:
    with patch.object(gateway, "SESSION_TTL_S", 0.2):
      session_id = (await gateway.create_session({}))["session_id"]
      await asyncio.sleep(0.15)
      await gateway.session_heartbeat({"session_id": session_id})
      await asyncio.sleep(0.15)
    self.assertTrue(await self.lifecycle.session_alive(session_id))

  async def test_session_expires_without_heartbeats(self) -> None:
    with patch.object(gateway, "SESSION_TTL_S", 0.05):
      session_id = (await gateway.create_session({}))["session_id"]
    await asyncio.sleep(0.1)
    self.assertFalse(await self.lifecycle.session_alive(session_id))

  async def test_create_model_binds_session(self) -> None:
    with patch.dict(os.environ, {}, clear=True):
      session_id = (await gateway.create_session({}))["session_id"]
      model_id = (await gateway.create_model({"base_model": "base", "session_id": session_id}))["request_id"]
    self.assertEqual(await self.lifecycle.model_session_state(model_id), (True, True))

  async def test_create_sampling_session_binds_session_to_sampled_model(self) -> None:
    session_id = (await gateway.create_session({}))["session_id"]
    await gateway.create_sampling_session({"session_id": session_id, "model_path": "tinker://model-a/sampler_weights/ckpt-1"})
    self.assertEqual(await self.lifecycle.model_session_state("model-a"), (True, True))


class EvalSamplePathTest(unittest.IsolatedAsyncioTestCase):
  """asample in FFT torch mode revives the model's worker and carries the weights ref."""

  def setUp(self) -> None:
    self.store = InMemoryStore()
    self.lifecycle = InMemoryLifecycleStore()
    self.manager = ManagerStub()
    patch.object(gateway, "store", self.store).start()
    patch.object(gateway, "lifecycle", self.lifecycle).start()
    self.addCleanup(patch.stopall)
    self.old_manager = gateway.fft_worker_manager
    gateway.fft_worker_manager = self.manager
    self.addCleanup(setattr, gateway, "fft_worker_manager", self.old_manager)

  async def test_sample_launches_worker_and_tags_weights_ref(self) -> None:
    env = {"OPEN_RL_ENABLE_FFT": "true", "SAMPLING_BACKEND": "torch", "REDIS_URL": "redis://stub"}
    with patch.dict(os.environ, env, clear=True):
      await gateway.asample(
        {
          "sampling_session_id": "tinker://model-a/sampler_weights/ckpt-1",
          "prompt": {"chunks": [{"tokens": [1, 2]}]},
          "sampling_params": {"max_tokens": 4},
        }
      )

    self.assertEqual(self.manager.launched_model_ids, ["model-a"])
    self.assertIn("model-a", await self.lifecycle.list_fft_models())
    [request] = await self.store.get_requests()
    self.assertEqual(request["op"], "sample")
    self.assertEqual(request["payload"]["sampler_weights_ref"], "tinker://model-a/sampler_weights/ckpt-1")

  async def test_sample_without_fft_does_not_launch(self) -> None:
    env = {"SAMPLING_BACKEND": "torch", "BASE_MODEL": "base"}
    with patch.dict(os.environ, env, clear=True):
      await gateway.asample({"sampling_session_id": "base", "prompt": {"chunks": []}})
    self.assertEqual(self.manager.launched_model_ids, [])


class ReaperTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    self.lifecycle = InMemoryLifecycleStore()
    self.manager = ManagerStub()
    patch.object(gateway, "store", self.store).start()
    patch.object(gateway, "lifecycle", self.lifecycle).start()
    self.addCleanup(patch.stopall)
    self.old_manager = gateway.fft_worker_manager
    gateway.fft_worker_manager = self.manager
    self.addCleanup(setattr, gateway, "fft_worker_manager", self.old_manager)

  async def supervise_once(self) -> None:
    await gateway.reap_idle_workers_once()

  async def test_worker_with_expired_sessions_is_shut_down(self) -> None:
    await self.lifecycle.register_fft_model("model-a")
    await self.lifecycle.bind_session_model("sess-1", "model-a")
    await self.lifecycle.touch_session("sess-1", 0.01)
    await asyncio.sleep(0.05)

    await self.supervise_once()

    self.assertEqual(self.manager.shutdown_model_ids, ["model-a"])
    self.assertEqual(await self.lifecycle.list_fft_models(), [])

  async def test_worker_with_live_session_is_kept(self) -> None:
    await self.lifecycle.register_fft_model("model-a")
    await self.lifecycle.bind_session_model("sess-1", "model-a")
    await self.lifecycle.touch_session("sess-1", 30.0)

    await self.supervise_once()

    self.assertEqual(self.manager.shutdown_model_ids, [])

  async def test_worker_never_bound_to_a_session_is_kept(self) -> None:
    await self.lifecycle.register_fft_model("model-a")

    await self.supervise_once()

    self.assertEqual(self.manager.shutdown_model_ids, [])
    self.assertEqual(await self.lifecycle.list_fft_models(), ["model-a"])

  async def test_queued_work_blocks_shutdown(self) -> None:
    await self.lifecycle.register_fft_model("model-a")
    await self.lifecycle.bind_session_model("sess-1", "model-a")
    await self.store.put_request({"request_id": "req-1", "op": "sample", "model_id": "model-a"})

    await self.supervise_once()

    self.assertEqual(self.manager.shutdown_model_ids, [])
    self.assertEqual(await self.lifecycle.list_fft_models(), ["model-a"])


class LazySamplerReloadTest(unittest.IsolatedAsyncioTestCase):
  def make_processor(self, worker) -> FFTTrainingRequestsProcessor:
    return FFTTrainingRequestsProcessor(InMemoryStore(), worker, "model-a", snapshot_client=MagicMock())

  def test_sampler_full_local_path_matches_save_side(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": "/tmp/x"}, clear=False):
      self.assertEqual(
        sampler_full_local_path("tinker://model-a/sampler_weights/ckpt-1"),
        "/tmp/x/sampler_full/model-a/sampler_weights/ckpt-1",
      )

  async def test_sample_reloads_checkpoint_when_model_is_gone(self) -> None:
    worker = MagicMock()
    worker.model = None
    worker.generate.return_value = {"sequences": []}

    with patch.dict(os.environ, {"REDIS_URL": "redis://stub", "OPEN_RL_TMP_DIR": "/tmp/x"}):
      processor = self.make_processor(worker)
      result = await processor.sample({"sampler_weights_ref": "tinker://model-a/sampler_weights/ckpt-1"}, "model-a")

    worker.load_from_state.assert_called_once_with("model-a", "/tmp/x/sampler_full/model-a/sampler_weights/ckpt-1")
    self.assertEqual(result["type"], "sample_completed")

  async def test_sample_without_ref_fails_with_actionable_error(self) -> None:
    worker = MagicMock()
    worker.model = None

    with patch.dict(os.environ, {"REDIS_URL": "redis://stub"}):
      processor = self.make_processor(worker)
      with self.assertRaisesRegex(RuntimeError, "no weights loaded"):
        await processor.sample({}, "model-a")

  async def test_sample_with_loaded_model_does_not_reload(self) -> None:
    worker = MagicMock()
    worker.generate.return_value = {"sequences": []}

    with patch.dict(os.environ, {"REDIS_URL": "redis://stub"}):
      processor = self.make_processor(worker)
      await processor.sample({"sampler_weights_ref": "tinker://model-a/sampler_weights/ckpt-1"}, "model-a")

    worker.load_from_state.assert_not_called()


if __name__ == "__main__":
  unittest.main()
