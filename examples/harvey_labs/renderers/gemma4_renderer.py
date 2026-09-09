"""Google's Gemma 4 template and HF parser, adapted to cookbook messages."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import tinker
import tinker_cookbook.renderers as renderers
from huggingface_hub import hf_hub_download
from tinker_cookbook.renderers.base import Message, ParseTermination, RenderContext, RenderedMessage, TextPart, ThinkingPart, ToolCall, ToolSpec
from transformers.utils.chat_parsing import parse_response

# Unmodified Google assets, including the tool-turn fixes from google/gemma-4-31B-it#118.
TEMPLATE_MODEL = "google/gemma-4-E2B-it"
TEMPLATE_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
SAMPLED_TOKENS_KEY = "sampled_tokens"


@cache
def google_assets() -> tuple[str, dict]:
  def read(filename: str) -> str:
    return Path(hf_hub_download(TEMPLATE_MODEL, filename, revision=TEMPLATE_REVISION)).read_text()

  return read("chat_template.jinja"), json.loads(read("tokenizer_config.json"))["response_template"]


class Gemma4ToolRenderer(renderers.Renderer):
  def __init__(self, tokenizer: Any, *, enable_thinking: bool = True):
    super().__init__(tokenizer)
    self.chat_template, self.response_template = google_assets()
    self.enable_thinking = enable_thinking

  @property
  def has_extension_property(self) -> bool:
    return True

  def get_stop_sequences(self) -> list[int]:
    return [self.tokenizer.convert_tokens_to_ids(token) for token in ("<turn|>", "<|tool_response>")]

  def template_tokens(self, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    return self.tokenizer.apply_chat_template(
      messages,
      chat_template=self.chat_template,
      tokenize=True,
      return_dict=False,
      add_generation_prompt=add_generation_prompt,
      enable_thinking=self.enable_thinking,
      preserve_thinking=True,
    )

  def build_generation_prompt(self, messages: list[Message], role: str = "assistant", prefill: str | None = None) -> tinker.ModelInput:
    if role != "assistant" or prefill is not None:
      raise ValueError("Gemma 4 supports assistant generation without a custom prefill")
    native = [self.to_openai_message(message) for message in messages]
    sampled_indices = [i for i, message in enumerate(messages) if SAMPLED_TOKENS_KEY in message]
    if not sampled_indices:
      return tinker.ModelInput.from_ints(self.template_tokens(native, add_generation_prompt=True))

    tokens = self.template_tokens(native[: sampled_indices[0]], add_generation_prompt=True)
    for i, end in zip(sampled_indices, [*sampled_indices[1:], len(messages)], strict=True):
      tokens.extend(messages[i][SAMPLED_TOKENS_KEY])
      # Let Google render the tool replies / next user turn. Only discard its
      # reconstructed assistant prefix: the actual sampled tokens are already above.
      assistant = native[i]
      canonical = self.template_tokens([assistant], add_generation_prompt=False)
      continuation = self.template_tokens(native[i:end], add_generation_prompt=True)
      newline = self.tokenizer.encode("\n", add_special_tokens=False)
      if canonical[-len(newline) :] == newline:
        canonical = canonical[: -len(newline)]
      if continuation[: len(canonical)] != canonical:
        raise ValueError("Google's Gemma template changed the assistant prefix before a continuation")
      if tokens[-1] != canonical[-1]:
        # EOS can finish sampling before the protocol's turn/tool delimiter.
        tokens.append(canonical[-1])
      tokens.extend(continuation[len(canonical) :])
    return tinker.ModelInput.from_ints(tokens)

  def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
    raise NotImplementedError("Use build_generation_prompt for Gemma RL; supervised masking is not supported")

  def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
    decoded = self.tokenizer.decode(response, skip_special_tokens=False)
    prefix = "<|turn>model\n"
    # Google pre-opens thinking after tool replies. Cookbook supplies only the
    # response to this method, so a closing channel without an opener identifies it.
    if self.enable_thinking and "<channel|>" in decoded and not decoded.startswith("<|channel>"):
      prefix += "<|channel>thought\n"
    message = Message(role="assistant", content="")
    message[SAMPLED_TOKENS_KEY] = list(response)  # type: ignore[typeddict-unknown-key]
    try:
      parsed = parse_response(decoded, self.response_template, prefix=prefix)
    except ValueError:
      message["content"] = decoded
      return message, ParseTermination.MALFORMED

    parts: list[TextPart | ThinkingPart] = []
    if thinking := parsed.get("thinking"):
      parts.append(ThinkingPart(type="thinking", thinking=thinking))
    if content := parsed.get("content"):
      parts.append(TextPart(type="text", text=content))
    message["content"] = parts
    if calls := parsed.get("tool_calls"):
      message["tool_calls"] = [
        ToolCall(
          id=f"call_{i}",
          function=ToolCall.FunctionBody(name=call["function"]["name"], arguments=json.dumps(call["function"]["arguments"])),
        )
        for i, call in enumerate(calls)
      ]
    if not response or ("<|tool_call>" in decoded and not calls):
      return message, ParseTermination.MALFORMED
    if response[-1] == self.tokenizer.eos_token_id:
      return message, ParseTermination.EOS
    termination = ParseTermination.STOP_SEQUENCE if response[-1] in self.get_stop_sequences() else ParseTermination.MALFORMED
    return message, termination

  def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
    rendered = self.tokenizer.apply_chat_template(
      [{"role": "system", "content": system_prompt}],
      tools=[{"type": "function", "function": tool} for tool in tools],
      chat_template=self.chat_template,
      tokenize=False,
      add_generation_prompt=False,
      enable_thinking=False,
    )
    prefix, suffix = "<bos><|turn>system\n", "<turn|>\n"
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
      raise ValueError("Unexpected Gemma 4 system/tool template")
    return [Message(role="system", content=rendered[len(prefix) : -len(suffix)])]

  def to_openai_message(self, message: Message) -> dict[str, Any]:
    result = super().to_openai_message(message)
    if SAMPLED_TOKENS_KEY in message:
      # Only tool metadata is needed to render the continuation. Assistant text
      # and thinking are replayed verbatim, including whitespace and empty channels.
      result["content"] = ""
    for call in result.get("tool_calls", []):
      call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return result


def register_gemma4_tool_renderer(name: str = "gemma4") -> None:
  renderers.register_renderer(name, lambda tokenizer, img_proc=None: Gemma4ToolRenderer(tokenizer))


if __name__ == "__main__":
  import unittest

  from tinker_cookbook.tokenizer_utils import get_tokenizer

  class RendererTest(unittest.TestCase):
    def test_sampled_tokens_survive_history(self):
      tokenizer = get_tokenizer(TEMPLATE_MODEL)
      tool = '<|tool_call>call:bash{command:<|"|>ls -la<|"|>}<tool_call|>'
      for thinking in (True, False):
        renderer = Gemma4ToolRenderer(tokenizer, enable_thinking=thinking)
        messages = renderer.create_conversation_prefix_with_tools(
          [{"name": "bash", "description": "Run a command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}]
        ) + [Message(role="user", content="List files.")]
        samples = [
          f"<|channel>thought\n Check. \n<channel|>{tool}<|tool_response>",
          f"<channel|>{tool}{tool}<|tool_response>" if thinking else f"{tool}<|tool_response>",
          "  Done. \n<channel|> Three files.  <turn|>" if thinking else " Three files.  <turn|>",
          "<|channel>thought\n\n<channel|>All done.\n\n<turn|>",
          f"{tool}<eos>",
          "<channel|>Done.<eos>" if thinking else "Done.<eos>",
        ]
        for text in samples:
          with self.subTest(thinking=thinking, response=text):
            prompt = renderer.build_generation_prompt(messages).to_ints()
            sampled = tokenizer.encode(text, add_special_tokens=False)
            message, termination = renderer.parse_response(sampled)
            self.assertTrue(termination.is_clean)
            messages.append(message)
            if calls := message.get("tool_calls"):
              for call in calls:
                self.assertEqual(json.loads(call.function.arguments), {"command": "ls -la"})
                messages.append(Message(role="tool", content="a.txt", tool_call_id=call.id))
            else:
              messages.append(Message(role="user", content="Continue."))
            expected = prompt + sampled
            actual = renderer.build_generation_prompt(messages).to_ints()
            self.assertEqual(actual[: len(expected)], expected)
            self.assertGreater(len(actual), len(expected))
            suffix = tokenizer.decode(actual[len(expected) :])
            if calls:
              self.assertEqual(suffix.count("response:bash"), len(calls))
              self.assertTrue(suffix.endswith("<|channel>thought\n" if thinking else "<tool_response|>"))
            else:
              self.assertIn("<|turn>user\nContinue.<turn|>\n<|turn>model\n", suffix)
        malformed = '<|tool_call>call:bash{command:"broken}<tool_call|><|tool_response>'
        _, termination = renderer.parse_response(tokenizer.encode(malformed, add_special_tokens=False))
        self.assertEqual(termination, ParseTermination.MALFORMED)

  unittest.main()
