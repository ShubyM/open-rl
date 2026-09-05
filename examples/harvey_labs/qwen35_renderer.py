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
from tinker_cookbook.renderers.base import Message, ParseTermination, RenderContext, RenderedMessage
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer

# Set on the parsed assistant Message. message_to_jsonable copies only the
# fields it knows, so transcripts never carry the token list.
SAMPLED_TOKENS_KEY = "sampled_tokens"


class VerbatimHistoryQwen35Renderer(Qwen3_5Renderer):
  """Qwen3_5Renderer whose history is what the model actually emitted."""

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


def verbatim_history_renderer(renderer: Qwen3_5Renderer) -> VerbatimHistoryQwen35Renderer:
  """Rebuild a factory-made Qwen3_5Renderer as the verbatim-history variant."""
  return VerbatimHistoryQwen35Renderer(
    renderer.tokenizer,
    image_processor=getattr(renderer, "image_processor", None),
    strip_thinking_from_history=False,
    merge_text_chunks=getattr(renderer, "merge_text_chunks", True),
  )
