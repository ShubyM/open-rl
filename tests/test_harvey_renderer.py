import importlib.util
import json
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


def _normalize():
  sys.path.insert(0, str(PROMPTS_PATH.parent))
  from gemma4_renderer import normalize_tool_call_args

  return normalize_tool_call_args


class NativeToolCallArgsTest(unittest.TestCase):
  """Gemma 4 emits tool arguments in an encoding its own parser rejects.

  Strings are delimited by the <|"|> special token and keys are left bare, so
  parse_response raises and the call is discarded as MALFORMED. run28 lost 83%
  of its parse errors this way, on calls that named a real tool.
  """

  Q = '<|"|>'

  def test_strict_json_is_left_alone(self) -> None:
    for body in ('{"command":"ls"}', '{"file_path":"a.docx","limit":5}', "{}"):
      self.assertEqual(_normalize()(body), body)

  def test_native_quotes_and_bare_keys_become_json(self) -> None:
    body = "{glob:%sb.docx%s,pattern:\\[.*\\]}" % (self.Q, self.Q)
    self.assertEqual(json.loads(_normalize()(body)), {"glob": "b.docx", "pattern": "\\[.*\\]"})

  def test_unclosed_string_keeps_the_arguments_own_quotes(self) -> None:
    # The trailing quote belongs to the shell command, not to the encoding.
    body = '{command:%sfind . -name "*.docx"}' % self.Q
    self.assertEqual(json.loads(_normalize()(body)), {"command": 'find . -name "*.docx"'})

  def test_mixed_closer_is_trusted_only_when_another_key_follows(self) -> None:
    body = '{file_path:%sa.docx",limit:500,offset:10}' % self.Q
    self.assertEqual(json.loads(_normalize()(body)), {"file_path": "a.docx", "limit": 500, "offset": 10})
    self.assertEqual(json.loads(_normalize()('{file_path:%sa.docx"}' % self.Q)), {"file_path": "a.docx"})

  def test_corrupt_bodies_are_returned_unchanged(self) -> None:
    # A space injected mid-key is a broken generation, not a format we decode.
    for body in ('{file_ apath:%sa.docx%s}' % (self.Q, self.Q), "not an object"):
      self.assertEqual(_normalize()(body), body)


class ToolCallBoundaryTest(unittest.TestCase):
  """A malformed call must not consume the calls next to it.

  With a plain `.*?` body an unterminated call kept scanning for a `}`, found
  the *next* call's, and swallowed it -- so a `write` that lost its brace to a
  truncated document also destroyed a valid `bash`. Replaying run29's 70
  captured parse errors, fixing this recovers 37 of them (52.9%).
  """

  GOOD = '<|tool_call>call:bash{"command":"ls"}<tool_call|>'
  GOOD2 = '<|tool_call>call:read{"file_path":"a.docx"}<tool_call|>'
  BAD = '<|tool_call>call:write{"file_path":"x.docx","content":"hello`)'

  def _names(self, text):
    sys.path.insert(0, str(PROMPTS_PATH.parent))
    from gemma4_renderer import _TOOL_CALL

    return [m.group("name") for m in _TOOL_CALL.finditer(text)]

  def test_unterminated_call_does_not_swallow_the_next_one(self) -> None:
    self.assertEqual(self._names(self.BAD + self.GOOD), ["bash"])
    self.assertEqual(self._names(self.GOOD + self.BAD + self.GOOD2), ["bash", "read"])

  def test_well_formed_calls_are_all_found(self) -> None:
    self.assertEqual(self._names(self.GOOD + self.GOOD2), ["bash", "read"])

  def test_unterminated_call_alone_matches_nothing(self) -> None:
    self.assertEqual(self._names(self.BAD), [])

  def test_long_unterminated_body_does_not_blow_up(self) -> None:
    big = '<|tool_call>call:write{"file_path":"big.docx","content":"' + "x" * 60000
    self.assertEqual(self._names(big + self.GOOD), ["bash"])


if __name__ == "__main__":
  unittest.main()
