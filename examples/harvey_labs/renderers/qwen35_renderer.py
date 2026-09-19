"""Qwen3.5 renderer that keeps sampled assistant turns verbatim in history.

Multi-turn RL trains on one sequence per episode only if every observation
extends the previous observation plus the sampled action token for token.
trajectory_to_data checks exactly that and starts a new datum whenever it
fails, so a broken chain multiplies trainer work by the number of turns:
run43 pushed ~48M tokens per step through forward and backward to train on
1.3M action tokens, because 645 of its 645 turns were separate datums.

The stock Qwen3_5Renderer re-renders each assistant turn from the parsed
message. That canonical form does not reproduce what the model wrote. A
tool-calling turn comes back as ``</think>\\n\\n\\n\\n<tool_call>`` where the
model wrote ``</think>\\n\\n<tool_call>``, thinking loses its surrounding
whitespace, and parameters are re-padded, so every tool-calling turn breaks
the chain two tokens after ``</think>``. Only plain-text final answers extend.

This subclass records the sampled tokens on the parsed message and, when that
message is rendered back into history, emits the generation header plus those
tokens unchanged. The chain then holds by construction, for the sampler's next
prompt and for the trainer's merged datum alike. Messages without sampled
tokens, such as the seeded system and user turns, render exactly as before.
"""

from __future__ import annotations

import tinker
import tinker_cookbook.renderers as renderers
from tinker_cookbook.renderers.base import Message, ParseTermination, RenderContext, RenderedMessage
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
from tinker_cookbook.renderers.qwen3_8 import Qwen3_8Renderer

# Set on the parsed assistant Message. message_to_jsonable copies only the
# fields it knows, so transcripts never carry the token list.
SAMPLED_TOKENS_KEY = "sampled_tokens"


class VerbatimHistoryMixin:
  """Replay sampled assistant tokens verbatim when a turn is rendered back into history."""

  def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
    message, termination = super().parse_response(response)
    message[SAMPLED_TOKENS_KEY] = list(response)  # type: ignore[typeddict-unknown-key]
    return message, termination

  def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
    tokens = message.get(SAMPLED_TOKENS_KEY) if message.get("role") == "assistant" else None
    if not tokens:
      return super().render_message(message, ctx)
    # The same header the generation prompt used when these tokens were
    # sampled, so header + tokens is byte for byte the prefix the sampler saw.
    header = tinker.EncodedTextChunk(tokens=self._get_generation_suffix("assistant", ctx))
    stop = self.get_stop_sequences()[0]
    output = list(tokens)
    if output[-1] != stop:
      output.append(stop)
    return RenderedMessage(header=header, output=[tinker.EncodedTextChunk(tokens=output)])


class VerbatimHistoryQwen35Renderer(VerbatimHistoryMixin, Qwen3_5Renderer):
  """Qwen3_5Renderer whose history is what the model actually emitted."""


class VerbatimHistoryQwen38Renderer(VerbatimHistoryMixin, Qwen3_8Renderer):
  """Qwen3_8Renderer (reasoning-effort preamble, thinking kept) with verbatim history.

  Qwen3.8 shares Qwen3.5's tokens and tool format, so the re-render mismatch
  is the same; the cookbook renderer only adds the system-prompt preamble."""

  @property
  def has_extension_property(self) -> bool:
    # The cookbook class reports False because its canonical re-render of a
    # non-reasoning turn is not a token-level prefix of what was sampled.
    # Replaying the sampled tokens makes history that prefix by construction.
    return True


VERBATIM_RENDERER_NAME = "qwen3_5_verbatim"
QWEN38_REASONING_EFFORTS = ("xhigh", "medium", "low")
# Cookbook renderer names that the verbatim variants stand in for. Qwen3.8's
# reasoning effort is part of the renderer name upstream, so it is here too.
VERBATIM_FOR = {"qwen3_5": VERBATIM_RENDERER_NAME}
VERBATIM_FOR.update({f"qwen3_8_{effort}_reasoning": f"qwen3_8_{effort}_verbatim" for effort in QWEN38_REASONING_EFFORTS})


def register_verbatim_qwen35_renderer(name: str = VERBATIM_RENDERER_NAME) -> None:
  # Keeping thinking in history is necessary but not sufficient; see the module
  # docstring for why the sampled tokens are replayed as well.
  renderers.register_renderer(
    name,
    lambda tokenizer, image_processor=None: VerbatimHistoryQwen35Renderer(
      tokenizer, image_processor=image_processor, strip_thinking_from_history=False
    ),
  )
  for effort in QWEN38_REASONING_EFFORTS:
    renderers.register_renderer(
      f"qwen3_8_{effort}_verbatim",
      lambda tokenizer, image_processor=None, effort=effort: VerbatimHistoryQwen38Renderer(
        tokenizer, image_processor=image_processor, reasoning_effort=effort
      ),
    )


if __name__ == "__main__":
  import unittest

  from tinker_cookbook.tokenizer_utils import get_tokenizer

  class RendererTest(unittest.TestCase):
    def test_sampled_tokens_survive_history(self):
      for model, verbatim, stock_cls in (
        ("Qwen/Qwen3.5-9B", VerbatimHistoryQwen35Renderer, Qwen3_5Renderer),
        ("Qwen/Qwen3.8-27B", VerbatimHistoryQwen38Renderer, Qwen3_8Renderer),
      ):
        with self.subTest(model=model):
          self.check_family(get_tokenizer(model), verbatim, stock_cls)

    def check_family(self, tokenizer, verbatim, stock_cls):
      renderer = verbatim(tokenizer, strip_thinking_from_history=False)
      stock = stock_cls(tokenizer, strip_thinking_from_history=False)
      self.assertTrue(renderer.has_extension_property)
      messages = [{"role": "user", "content": "List the files."}]
      self.assertEqual(renderer.build_generation_prompt(messages).to_ints(), stock.build_generation_prompt(messages).to_ints())

      tool = "<tool_call>\n<function=bash>\n<parameter=command>ls -la</parameter>\n</function>\n</tool_call>"
      samples = {
        "inline_parameters": f"Check.\n</think>\n\n{tool}<|im_end|>",
        "multiline_parameters": "Check.\n</think>\n\n" + tool.replace("ls -la", "\nls -la\n") + "<|im_end|>",
        "two_tool_calls": f"Check both.\n</think>\n\n{tool}\n{tool}<|im_end|>",
        "thinking_whitespace": f"  Check...  \n\n</think>\n\n{tool}<|im_end|>",
        "text": "Done.\n</think>\n\nThere are three files.<|im_end|>",
        "trailing_newlines": "Done.\n</think>\n\nThere are three files.\n\n<|im_end|>",
      }
      for name, text in samples.items():
        with self.subTest(sample=name):
          prompt = renderer.build_generation_prompt(messages).to_ints()
          sampled = tokenizer.encode(text, add_special_tokens=False)
          message, termination = renderer.parse_response(sampled)
          self.assertTrue(termination.is_clean)
          messages.append(message)
          if message.get("tool_calls"):
            messages.extend({"role": "tool", "content": "a.txt", "tool_call_id": call.id} for call in message["tool_calls"])
          else:
            messages.append({"role": "user", "content": "Continue."})
          rendered = renderer.build_generation_prompt(messages).to_ints()
          expected = prompt + sampled
          self.assertEqual(rendered[: len(expected)], expected)
          self.assertGreater(len(rendered), len(expected))

  unittest.main()
