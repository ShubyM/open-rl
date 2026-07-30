import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROMPTS_PATH = Path(__file__).resolve().parents[1] / "examples" / "harvey_labs" / "prompts.py"


class _FakeQwenRenderer:
  strip_thinking_from_history = True

  @property
  def has_extension_property(self) -> bool:
    return not self.strip_thinking_from_history


def _load_prompts_with_stubbed_dependencies(renderer: _FakeQwenRenderer):
  gemma = types.ModuleType("gemma4_renderer")
  gemma.register_gemma4_tool_renderer = lambda: None
  reward = types.ModuleType("reward")
  reward.ARTIFACT_EXTENSIONS = ("txt",)
  tasks = types.ModuleType("tasks")
  tasks.LabTask = object

  model_info = types.ModuleType("tinker_cookbook.model_info")
  model_info.get_recommended_renderer_name = lambda _model_name: "qwen3_5"
  tokenizer_utils = types.ModuleType("tinker_cookbook.tokenizer_utils")
  tokenizer_utils.get_tokenizer = lambda _model_name: object()
  renderers = types.ModuleType("tinker_cookbook.renderers")
  renderers.get_renderer = lambda *_args, **_kwargs: renderer
  renderers_base = types.ModuleType("tinker_cookbook.renderers.base")
  renderers_base.Message = dict
  renderers_base.Renderer = object
  cookbook = types.ModuleType("tinker_cookbook")
  cookbook.model_info = model_info
  cookbook.tokenizer_utils = tokenizer_utils

  stubs = {
    "gemma4_renderer": gemma,
    "reward": reward,
    "tasks": tasks,
    "tinker_cookbook": cookbook,
    "tinker_cookbook.model_info": model_info,
    "tinker_cookbook.tokenizer_utils": tokenizer_utils,
    "tinker_cookbook.renderers": renderers,
    "tinker_cookbook.renderers.base": renderers_base,
  }
  spec = importlib.util.spec_from_file_location("harvey_prompts_under_test", PROMPTS_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(module)
  return module


class HarveyRendererTest(unittest.TestCase):
  def test_qwen_preserves_history_for_multiturn_rl(self) -> None:
    renderer = _FakeQwenRenderer()
    prompts = _load_prompts_with_stubbed_dependencies(renderer)

    result = prompts.lab_renderer("Qwen/Qwen3.5-9B", "qwen3_5")

    self.assertIs(result, renderer)
    self.assertFalse(renderer.strip_thinking_from_history)
    self.assertTrue(renderer.has_extension_property)

  def test_non_extending_renderer_is_rejected(self) -> None:
    renderer = _FakeQwenRenderer()
    prompts = _load_prompts_with_stubbed_dependencies(renderer)

    with self.assertRaisesRegex(ValueError, "prefix-extending renderer"):
      prompts.lab_renderer("example/model", "broken_renderer")


if __name__ == "__main__":
  unittest.main()
