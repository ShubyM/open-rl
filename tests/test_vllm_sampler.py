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


class SleepLevelTest(unittest.TestCase):
  def test_defaults_to_level_one(self) -> None:
    with patch.dict("os.environ", {}, clear=True):
      self.assertEqual(vllm_sampler.vllm_sleep_level(), 1)

  def test_allows_level_two(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_VLLM_SLEEP_LEVEL": "2"}):
      self.assertEqual(vllm_sampler.vllm_sleep_level(), 2)

  def test_rejects_unsupported_level(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_VLLM_SLEEP_LEVEL": "3"}), self.assertRaisesRegex(ValueError, "must be 1 or 2"):
      vllm_sampler.vllm_sleep_level()


class LanguageModelOnlyTest(unittest.TestCase):
  def test_fft_defaults_to_text_only(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "true"}, clear=True):
      self.assertTrue(vllm_sampler.vllm_language_model_only())

  def test_lora_keeps_the_full_model_default(self) -> None:
    with patch.dict("os.environ", {"OPEN_RL_ENABLE_FFT": "false"}, clear=True):
      self.assertFalse(vllm_sampler.vllm_language_model_only())

  def test_explicit_override_wins(self) -> None:
    with patch.dict(
      "os.environ",
      {"OPEN_RL_ENABLE_FFT": "true", "OPEN_RL_VLLM_LANGUAGE_MODEL_ONLY": "0"},
      clear=True,
    ):
      self.assertFalse(vllm_sampler.vllm_language_model_only())


class PrepareEngineTest(unittest.IsolatedAsyncioTestCase):
  def tearDown(self) -> None:
    vllm_sampler.engine = None
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = None

  async def test_wakes_engine_when_checkpoint_is_unchanged(self) -> None:
    current = FakeEngine()
    vllm_sampler.engine = current
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "/checkpoints/step-1"

    with (
      patch.dict("os.environ", {"OPEN_RL_VLLM_SLEEP_LEVEL": "1"}),
      patch.object(vllm_sampler, "init_engine") as init_engine,
    ):
      await vllm_sampler.prepare_engine("/checkpoints/step-1")

    init_engine.assert_not_called()
    self.assertEqual(current.wake_calls, 1)
    self.assertEqual(current.reload_calls, [])

  async def test_reloads_unchanged_checkpoint_after_level_two_sleep(self) -> None:
    current = FakeEngine()
    vllm_sampler.engine = current
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "step-1"

    with patch.dict("os.environ", {"OPEN_RL_VLLM_SLEEP_LEVEL": "2"}):
      await vllm_sampler.prepare_engine("/checkpoints/slot-0", "step-1")

    self.assertEqual(current.wake_tags, [["weights"], ["kv_cache"]])
    self.assertEqual(
      current.reload_calls,
      [("reload_weights", {"weights_path": "/checkpoints/slot-0"})],
    )

  async def test_rejects_level_two_wake_without_checkpoint(self) -> None:
    current = FakeEngine()
    vllm_sampler.engine = current

    with (
      patch.dict("os.environ", {"OPEN_RL_VLLM_SLEEP_LEVEL": "2"}),
      self.assertRaisesRegex(RuntimeError, "checkpoint path is required"),
    ):
      await vllm_sampler.prepare_engine(None)

  async def test_reloads_engine_when_checkpoint_changes(self) -> None:
    current = FakeEngine(sleeping=False)
    vllm_sampler.engine = current
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "/checkpoints/step-1"

    with patch.object(vllm_sampler, "init_engine") as init_engine:
      await vllm_sampler.prepare_engine("/checkpoints/step-2")

    init_engine.assert_not_called()
    self.assertEqual(current.sleep_calls, [1])
    self.assertEqual(current.wake_tags, [["weights"], ["kv_cache"]])
    self.assertEqual(
      current.reload_calls,
      [("reload_weights", {"weights_path": "/checkpoints/step-2"})],
    )
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "/checkpoints/step-2")

  async def test_reloads_reused_checkpoint_path_when_revision_changes(self) -> None:
    current = FakeEngine()
    vllm_sampler.engine = current
    vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS = "step-1"

    await vllm_sampler.prepare_engine("/checkpoints/slot-0", "step-2")

    self.assertEqual(
      current.reload_calls,
      [("reload_weights", {"weights_path": "/checkpoints/slot-0"})],
    )
    self.assertEqual(vllm_sampler.CURRENT_LOADED_SAMPLER_WEIGHTS, "step-2")


if __name__ == "__main__":
  unittest.main()
