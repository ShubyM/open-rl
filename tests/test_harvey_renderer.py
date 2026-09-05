import json
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("harvey_labs", reason="Harvey recipe tests require the examples environment")

from harvey_labs import prompts
from harvey_labs.gemma4_renderer import _TOOL_CALL, normalize_tool_call_args


class _FakeQwenRenderer:
  strip_thinking_from_history = True

  @property
  def has_extension_property(self) -> bool:
    return not self.strip_thinking_from_history


def _lab_renderer_with(renderer: _FakeQwenRenderer, model_name: str, renderer_name: str):
  # Swap the cookbook lookups for the fake so the test exercises only the
  # recipe's own renderer policy.
  with (
    patch.object(prompts.tokenizer_utils, "get_tokenizer", return_value=object()),
    patch.object(prompts, "get_renderer", return_value=renderer),
    patch.object(prompts, "verbatim_history_renderer", side_effect=lambda r: r),
  ):
    return prompts.lab_renderer(model_name, renderer_name)


class HarveyRendererTest(unittest.TestCase):
  def test_qwen_preserves_history_for_multiturn_rl(self) -> None:
    renderer = _FakeQwenRenderer()

    result = _lab_renderer_with(renderer, "Qwen/Qwen3.5-9B", "qwen3_5")

    self.assertIs(result, renderer)
    self.assertFalse(renderer.strip_thinking_from_history)
    self.assertTrue(renderer.has_extension_property)

  def test_non_extending_renderer_is_rejected(self) -> None:
    renderer = _FakeQwenRenderer()

    with self.assertRaisesRegex(ValueError, "prefix-extending renderer"):
      _lab_renderer_with(renderer, "example/model", "broken_renderer")


class NativeToolCallArgsTest(unittest.TestCase):
  """Gemma 4 emits tool arguments in an encoding its own parser rejects.

  Strings are delimited by the <|"|> special token and keys are left bare, so
  parse_response raises and the call is discarded as MALFORMED. run28 lost 83%
  of its parse errors this way, on calls that named a real tool.
  """

  Q = '<|"|>'

  def test_strict_json_is_left_alone(self) -> None:
    for body in ('{"command":"ls"}', '{"file_path":"a.docx","limit":5}', "{}"):
      self.assertEqual(normalize_tool_call_args(body), body)

  def test_native_quotes_and_bare_keys_become_json(self) -> None:
    body = f"{{glob:{self.Q}b.docx{self.Q},pattern:\\[.*\\]}}"
    self.assertEqual(json.loads(normalize_tool_call_args(body)), {"glob": "b.docx", "pattern": "\\[.*\\]"})

  def test_unclosed_string_keeps_the_arguments_own_quotes(self) -> None:
    # The trailing quote belongs to the shell command, not to the encoding.
    body = f'{{command:{self.Q}find . -name "*.docx"}}'
    self.assertEqual(json.loads(normalize_tool_call_args(body)), {"command": 'find . -name "*.docx"'})

  def test_mixed_closer_is_trusted_only_when_another_key_follows(self) -> None:
    body = f'{{file_path:{self.Q}a.docx",limit:500,offset:10}}'
    self.assertEqual(json.loads(normalize_tool_call_args(body)), {"file_path": "a.docx", "limit": 500, "offset": 10})
    self.assertEqual(json.loads(normalize_tool_call_args(f'{{file_path:{self.Q}a.docx"}}')), {"file_path": "a.docx"})

  def test_corrupt_bodies_are_returned_unchanged(self) -> None:
    # A space injected mid-key is a broken generation, not a format we decode.
    for body in (f"{{file_ apath:{self.Q}a.docx{self.Q}}}", "not an object"):
      self.assertEqual(normalize_tool_call_args(body), body)


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
