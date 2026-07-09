import unittest
from unittest.mock import patch

from server import vllm_sampler


class FakeEngine:
  def __init__(self, sleeping: bool = True):
    self.sleeping = sleeping
    self.sleep_calls = []
    self.wake_calls = 0
    self.wake_tags = []
    self.reload_calls = []

  async def sleep(self, level: int) -> None:
    self.sleep_calls.append(level)
    self.sleeping = True

  async def is_sleeping(self) -> bool:
    return self.sleeping

  async def wake_up(self, tags=None) -> None:
    self.wake_calls += 1
    self.wake_tags.append(tags)
    if tags is None or tags == ["kv_cache"]:
      self.sleeping = False

  async def collective_rpc(self, method: str, kwargs: dict) -> list[None]:
    self.reload_calls.append((method, kwargs))
    return [None]


class PrepareEngineTest(unittest.IsolatedAsyncioTestCase):
  def tearDown(self) -> None:
    vllm_sampler.engine = None
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = None

  async def test_wakes_engine_when_checkpoint_is_unchanged(self) -> None:
    current = FakeEngine()
    vllm_sampler.engine = current
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "/checkpoints/step-1"

    with patch.object(vllm_sampler, "init_engine") as init_engine:
      await vllm_sampler.prepare_engine("/checkpoints/step-1")

    init_engine.assert_not_called()
    self.assertEqual(current.wake_calls, 1)
    self.assertEqual(current.reload_calls, [])

  async def test_reloads_engine_when_checkpoint_changes(self) -> None:
    current = FakeEngine(sleeping=False)
    vllm_sampler.engine = current
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "/checkpoints/step-1"

    with patch.object(vllm_sampler, "init_engine") as init_engine:
      await vllm_sampler.prepare_engine("/checkpoints/step-2")

    init_engine.assert_not_called()
    self.assertEqual(current.sleep_calls, [2])
    self.assertEqual(current.wake_tags, [["weights"], ["kv_cache"]])
    self.assertEqual(
      current.reload_calls,
      [("reload_weights", {"weights_path": "/checkpoints/step-2"})],
    )
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "/checkpoints/step-2")


if __name__ == "__main__":
  unittest.main()
