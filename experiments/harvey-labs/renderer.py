"""Gemma 4 renderer with XML-style tool calls for cookbook tool envs."""

from __future__ import annotations

import json
from typing import Any

import tinker
import tinker_cookbook.renderers as renderers
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    _tool_call_payload,
    ensure_text,
    parse_content_blocks,
    parse_response_for_stop_token,
)


class Gemma4ToolRenderer(renderers.Renderer):
    """Gemma 4 chat renderer extended with cookbook tool-call messages."""

    @property
    def has_extension_property(self) -> bool:
        return True

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<bos>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<turn|>", add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Expected '<turn|>' to be one token, got {len(tokens)}")
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def role_for_message(self, message: Message) -> str:
        return "model" if message["role"] in ("assistant", "model") else "user"

    def format_content(self, message: Message) -> str:
        text = self.message_text(message)
        if message["role"] == "tool":
            name = message.get("name", "tool")
            call_id = message.get("tool_call_id", "")
            return f"<tool_response name={name} id={call_id}>\n{text}\n</tool_response>"

        if "tool_calls" in message and message["tool_calls"]:
            blocks = [
                f"<tool_call>{json.dumps(_tool_call_payload(tool_call))}</tool_call>"
                for tool_call in message["tool_calls"]
            ]
            text = text + ("\n" if text else "") + "\n".join(blocks)
        return text

    def message_text(self, message: Message) -> str:
        content = message["content"]
        if not isinstance(content, list):
            return content

        parts: list[str] = []
        for part in content:
            if part["type"] == "text":
                parts.append(part["text"])
            elif part["type"] == "thinking":
                parts.append(f"<think>{part['thinking']}</think>")
            else:
                raise NotImplementedError("Gemma4ToolRenderer does not support images")
        return "".join(parts)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        maybe_newline = "\n" if ctx.idx > 0 else ""
        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(f"{maybe_newline}<|turn>{self.role_for_message(message)}\n", add_special_tokens=False)
        )
        output = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(self.format_content(message) + "<turn|>", add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        content = ensure_text(assistant_message["content"])
        parsed = parse_content_blocks(content)
        if parsed is None:
            return assistant_message, True

        parts, tool_results = parsed
        assistant_message["content"] = parts
        tool_calls = [tool for tool in tool_results if isinstance(tool, ToolCall)]
        unparsed = [tool for tool in tool_results if isinstance(tool, UnparsedToolCall)]
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if unparsed:
            assistant_message["unparsed_tool_calls"] = unparsed
        return assistant_message, True

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        tools_text = ""
        if tools:
            tool_lines = "\n".join(
                json.dumps({"type": "function", "function": tool}, separators=(",", ":"))
                for tool in tools
            )
            tools_text = (
                "# Tools\n\n"
                "You can call tools by replying with exactly one or more tool_call blocks.\n"
                "Use this format and no trailing text after the final tool_call block:\n\n"
                "<tool_call>{\"name\":\"tool_name\",\"arguments\":{\"arg\":\"value\"}}</tool_call>\n\n"
                "Available tools:\n"
                "<tools>\n"
                f"{tool_lines}\n"
                "</tools>\n\n"
                "After tool results are returned, continue working. When the task is complete, "
                "call submit."
            )
        content = "\n\n".join(part for part in (tools_text, system_prompt) if part)
        return [Message(role="system", content=content)]

    def to_openai_message(self, message: Message) -> dict[str, Any]:
        result = super().to_openai_message(message)
        if message["role"] == "model":
            result["role"] = "assistant"
        return result


def register_gemma4_tool_renderer(name: str = "gemma4") -> None:
    renderers.register_renderer(name, lambda tokenizer, img_proc=None: Gemma4ToolRenderer(tokenizer))
