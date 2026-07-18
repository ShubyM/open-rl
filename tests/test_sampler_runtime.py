import os
import unittest
from unittest.mock import patch

from server import vllm_sampler
from server.protocol import SamplerSnapshot
from server.store import InMemoryStore, acquire_sampler_snapshot, get_sampler_snapshot, put_sampler_snapshot


class RedisStub:
  def __init__(self):
    self.values = {}

  async def set(self, key, value):
    self.values[key] = value

  async def get(self, key):
    return self.values.get(key)

  async def delete(self, key):
    self.values.pop(key, None)


class ReadyStore:
  def __init__(self):
    self.redis = RedisStub()


class SamplerRuntimeTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.old_engine = vllm_sampler.engine
    self.old_revision = vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS
    vllm_sampler.engine = None
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = None
    self.addCleanup(self.restore_globals)

  def restore_globals(self) -> None:
    vllm_sampler.engine = self.old_engine
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = self.old_revision

  def test_sleep_level_is_explicitly_configurable(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_VLLM_SLEEP_LEVEL": "2"}):
      self.assertEqual(vllm_sampler.vllm_sleep_level(), 2)
    with patch.dict(os.environ, {"OPEN_RL_VLLM_SLEEP_LEVEL": "3"}), self.assertRaisesRegex(ValueError, "must be 1 or 2"):
      vllm_sampler.vllm_sleep_level()

  async def test_mock_sampler_selects_revision_without_loading_checkpoint(self) -> None:
    with patch.dict(os.environ, {"MOCK_VLLM": "1"}):
      await vllm_sampler.prepare_engine("/missing/checkpoint", "model-a:7")

    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "model-a:7")
    self.assertIsNone(vllm_sampler.engine)

  async def test_ready_key_is_scoped_to_the_worker_instance(self) -> None:
    store = ReadyStore()
    with patch.dict(os.environ, {"MOCK_VLLM": "1"}):
      await vllm_sampler.publish_sampler_ready(store, "model-a", "new-instance")
      await vllm_sampler.clear_sampler_ready(store, "model-a", "stale-instance")
      self.assertEqual(store.redis.values["open_rl:sampler_ready:model-a"], "new-instance")
      await vllm_sampler.clear_sampler_ready(store, "model-a", "new-instance")

    self.assertNotIn("open_rl:sampler_ready:model-a", store.redis.values)

  async def test_sampling_releases_immutable_snapshot_after_completion(self) -> None:
    store = InMemoryStore()
    session_id = "tinker://model-a/sampler_weights/sampler-1"
    await put_sampler_snapshot(
      store,
      SamplerSnapshot(
        sampling_session_id=session_id,
        model_id="model-a",
        revision=1,
        storage_path="tinker://model-a/sampler_weights/revisions/1",
        created_at=1.0,
      ),
    )
    await acquire_sampler_snapshot(store, session_id)

    with patch.dict(os.environ, {"MOCK_VLLM": "1", "OPEN_RL_ENABLE_FFT": "true"}):
      await vllm_sampler.process_sampling_request(
        {
          "request_id": "request-a",
          "model_id": "model-a",
          "sampling_session_id": session_id,
          "weights_path": "/missing/checkpoint",
          "weights_revision": "model-a:1",
          "prompt_token_ids": [1, 2],
          "max_tokens": 2,
          "num_samples": 1,
        },
        store,
      )

    self.assertEqual((await get_sampler_snapshot(store, session_id)).in_flight, 0)
    self.assertEqual((await store.get_future("request-a", 0))["type"], "sample")
