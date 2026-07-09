"""Gemma 4 renderer backed by the checkpoint's native chat template."""

from __future__ import annotations

import json
from typing import Any

import tinker
import tinker_cookbook.renderers as renderers
from tinker_cookbook.renderers.base import (
  Message,
  RenderContext,
  RenderedMessage,
  TextPart,
  ThinkingPart,
  ToolCall,
  ToolSpec,
)


class Gemma4ToolRenderer(renderers.Renderer):
  """Render and parse Gemma 4's native function-calling protocol."""

  @property
  def has_extension_property(self) -> bool:
    return True

  @property
  def _bos_tokens(self) -> list[int]:
    return self.tokenizer.encode("<bos>", add_special_tokens=False)

  @property
  def _end_message_tokens(self) -> set[int]:
    return {
      self.tokenizer.encode("<turn|>", add_special_tokens=False)[0],
      self.tokenizer.encode("<|tool_response>", add_special_tokens=False)[0],
    }

  def get_stop_sequences(self) -> list[int]:
    return sorted(self._end_message_tokens)

  def template_tokens(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    rendered = self.tokenizer.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=add_generation_prompt,
      enable_thinking=False,
    )
    if hasattr(rendered, "keys"):
      rendered = rendered["input_ids"]
    return list(rendered)

  def build_generation_prompt(
    self,
    messages: list[Message],
    role: str = "assistant",
    prefill: str | None = None,
  ) -> tinker.ModelInput:
    if role != "assistant":
      raise ValueError(f"Gemma 4 only supports assistant generation, got {role!r}")
    prompt = self.template_tokens(
      [self.to_openai_message(message) for message in messages],
      add_generation_prompt=True,
    )
    if prefill:
      prompt.extend(self.tokenizer.encode(prefill, add_special_tokens=False))
    return tinker.ModelInput.from_ints(tokens=prompt)

  def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
    """Render one non-tool message for cookbook masking utilities.

    Live LAB rollouts use ``build_generation_prompt`` so the full conversation is
    always formatted by Hugging Face's checkpoint-provided template.
    """
    if message["role"] == "tool":
      raise NotImplementedError("Gemma 4 tool responses require their preceding tool call")

    rendered = self.template_tokens([self.to_openai_message(message)], add_generation_prompt=False)
    bos_len = len(self._bos_tokens)
    rendered = rendered[bos_len:]
    role = "model" if message["role"] in ("assistant", "model") else message["role"]
    header = self.tokenizer.encode(f"<|turn>{role}\n", add_special_tokens=False)
    if rendered[: len(header)] != header:
      raise ValueError(f"Unexpected Gemma 4 template output for role {message['role']!r}")
    return RenderedMessage(
      header=tinker.EncodedTextChunk(tokens=header),
      output=[tinker.EncodedTextChunk(tokens=rendered[len(header) :])],
    )

  def parse_response(self, response: list[int]) -> tuple[Message, bool]:
    eos_token_id = self.tokenizer.eos_token_id
    if eos_token_id is not None and response and response[-1] == eos_token_id:
      response = response[:-1]
    decoded = self.tokenizer.decode(response, skip_special_tokens=False)
    try:
      parsed = self.tokenizer.parse_response(decoded)
    except Exception:
      return Message(role="assistant", content=decoded), False

    tool_calls = []
    for tool_call in parsed.get("tool_calls", []):
      function = tool_call["function"]
      tool_calls.append(
        ToolCall(
          id=tool_call.get("id"),
          function=ToolCall.FunctionBody(
            name=function["name"],
            arguments=json.dumps(function.get("arguments", {}), separators=(",", ":")),
          ),
        )
      )

    parts: list[TextPart | ThinkingPart] = []
    if thinking := parsed.get("thinking"):
      parts.append(ThinkingPart(type="thinking", thinking=thinking))
    if content := parsed.get("content"):
      parts.append(TextPart(type="text", text=content))

    message = Message(role="assistant", content=parts if parts else "")
    if tool_calls:
      message["tool_calls"] = tool_calls

    malformed_tool_call = "<|tool_call>" in decoded and not tool_calls
    return message, not malformed_tool_call

  def create_conversation_prefix_with_tools(
    self,
    tools: list[ToolSpec],
    system_prompt: str = "",
  ) -> list[Message]:
    native_tools = [{"type": "function", "function": tool} for tool in tools]
    rendered = self.tokenizer.apply_chat_template(
      [{"role": "system", "content": system_prompt}],
      tools=native_tools,
      tokenize=False,
      add_generation_prompt=False,
      enable_thinking=False,
    )
    prefix = "<bos><|turn>system\n"
    suffix = "<turn|>\n"
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
      raise ValueError("Unexpected Gemma 4 system/tool template")
    return [Message(role="system", content=rendered[len(prefix) : -len(suffix)])]

  def to_openai_message(self, message: Message) -> dict[str, Any]:
    result = super().to_openai_message(message)
    if result["role"] == "model":
      result["role"] = "assistant"
    for tool_call in result.get("tool_calls", []):
      arguments = tool_call["function"].get("arguments")
      if isinstance(arguments, str):
        tool_call["function"]["arguments"] = json.loads(arguments)
    return result


def register_gemma4_tool_renderer(name: str = "gemma4") -> None:
  renderers.register_renderer(name, lambda tokenizer, img_proc=None: Gemma4ToolRenderer(tokenizer))
