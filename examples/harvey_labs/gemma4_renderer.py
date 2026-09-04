"""Gemma 4 renderer backed by the checkpoint's native chat template."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import cache
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import tinker
import tinker_cookbook.renderers as renderers
from tinker_cookbook.renderers.base import (
  Message,
  ParseTermination,
  RenderContext,
  RenderedMessage,
  TextPart,
  ThinkingPart,
  ToolCall,
  ToolSpec,
)

# chat_template.jinja is pinned from Gemma 4 discussion #36 at HF revision
# 4e34fcbc4c9a95b92d6a8a97c2faed16dd783f91, with one local edit: the thinking
# block is gated on `is not none` rather than truthiness, so an empty thought
# channel round-trips as '<|channel>thought\n<channel|>'.
#
# Upstream needs this because after a tool response the template pre-opens
# <|channel>thought in the generation prompt; a model that closes it immediately
# produced a blank thought the re-render then dropped, breaking the prefix chain
# that trajectory merging depends on. The encoding matches upstream's own -- HF
# revision 707f0a3b emits exactly '<|channel>thought\n<channel|>' for the empty
# case on its `not enable_thinking` branch. That revision is otherwise identical
# to this one for tool-calling paths, so the pin stays on #36.
GEMMA4_CHAT_TEMPLATE_SHA256 = "ca51a48d0fe20cfe36f480d6cb0a691a60cf1bb4a34475183868e7d24eb8cd38"

# The body must not run past the next call's opener. With a plain `.*?` an
# unterminated call swallows the following one whole -- it keeps scanning for a
# `}` and finds the *next* call's, then finds that call's `<tool_call|>` right
# where it wants a closer, so both are consumed as one malformed call. A `write`
# that lost its brace to a truncated document therefore took a valid `bash` with
# it. No real body can contain the opener, so refusing to cross it is free.
_TOOL_CALL = re.compile(
  r"<\|tool_call>(?P<body>call:(?P<name>\w+)\{(?:(?!<\|tool_call>).)*?\})"
  r"(?:<tool_call\|>|<eos>|(?=<\|tool_response>)|$)",
  re.DOTALL,
)


def _valid_tool_calls(parsed: Any) -> list[dict[str, Any]]:
  calls = parsed.get("tool_calls") if isinstance(parsed, Mapping) else None
  if not isinstance(calls, list):
    return []
  return [
    call
    for call in calls
    if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping) and isinstance(call["function"].get("name"), str)
  ]


_NATIVE_QUOTE = '<|"|>'
_IDENT = re.compile(r"[A-Za-z_]\w*")
_JSON_SCALAR = re.compile(r"(?:-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|true|false|null)$")
_MIXED_CLOSER = re.compile(r'"\s*,\s*[A-Za-z_]\w*\s*:')


def _read_native_string(text: str, i: int) -> tuple[str, int]:
  """Read a <|"|>-delimited string at i, tolerating a missing or mixed closer."""
  i += len(_NATIVE_QUOTE)
  end = text.find(_NATIVE_QUOTE, i)
  if end != -1:
    return text[i:end], end + len(_NATIVE_QUOTE)
  # Opened with the native token and closed with a plain quote, or not closed at
  # all. Only trust a plain quote that a further key follows -- a quote before
  # the final brace is usually the argument's own, as in
  # {command:<|"|>find . -name "*.docx"}.
  mixed = _MIXED_CLOSER.search(text, i)
  if mixed:
    return text[i : mixed.start()], mixed.start() + 1
  end = len(text.rstrip())
  if end > i and text[end - 1] == "}":
    end -= 1
  value = text[i:end].rstrip()
  if value.endswith('"') and value.count('"') % 2:
    # An unpaired trailing quote is the closer; a balanced pair is the value's.
    value = value[:-1]
  return value, end


def _read_bare_value(text: str, i: int) -> tuple[Any, int]:
  """Read an unquoted value up to the top-level ',' or the closing '}'."""
  depth, start = 0, i
  while i < len(text):
    ch = text[i]
    if ch in "{[":
      depth += 1
    elif ch in "]}":
      if depth == 0:
        break
      depth -= 1
    elif ch == "," and depth == 0:
      break
    i += 1
  raw = text[start:i].strip()
  if _JSON_SCALAR.fullmatch(raw) or (raw[:1] in '"[{' and raw[-1:] in '"]}'):
    try:
      return json.loads(raw), i
    except ValueError:
      pass
  return raw, i


def normalize_tool_call_args(body: str) -> str:
  """Rebuild one `{...}` argument object leniently as strict JSON.

  Gemma 4 emits string arguments delimited by the <|"|> special token and leaves
  object keys bare -- {command:<|"|>find . -name "*.docx"}. That is the encoding
  the tokenizer produces, but its own parse_response rejects it, so run28 threw
  away 83% of its parse errors on calls that named a real tool and had a proper
  closing delimiter. Returns the input unchanged when it does not look like this
  encoding; callers must try the raw body first regardless.
  """
  text = body.strip()
  if not text.startswith("{"):
    return body
  i, out = 1, {}
  while i < len(text):
    while i < len(text) and text[i] in " \n\t,":
      i += 1
    if i >= len(text) or text[i] == "}":
      break
    if text.startswith(_NATIVE_QUOTE, i):
      key, i = _read_native_string(text, i)
    else:
      match = _IDENT.match(text, i)
      if not match:
        return body
      key, i = match.group(0), match.end()
    while i < len(text) and text[i] in " \n\t":
      i += 1
    if i >= len(text) or text[i] != ":":
      return body
    i += 1
    while i < len(text) and text[i] in " \n\t":
      i += 1
    if text.startswith(_NATIVE_QUOTE, i):
      value, i = _read_native_string(text, i)
    else:
      value, i = _read_bare_value(text, i)
    out[str(key)] = value
  return json.dumps(out, separators=(",", ":")) if out else body


@cache
def gemma4_chat_template() -> str:
  """Return the pinned canonical Gemma 4 tool-calling template."""
  template = Path(__file__).with_name("chat_template.jinja").read_text(encoding="utf-8")
  digest = sha256(template.encode()).hexdigest()
  if digest != GEMMA4_CHAT_TEMPLATE_SHA256:
    raise ValueError(f"Gemma 4 chat template hash mismatch: {digest}")
  return template


class Gemma4ToolRenderer(renderers.Renderer):
  """Render and parse Gemma 4's native function-calling protocol."""

  def __init__(self, tokenizer: Any, *, enable_thinking: bool = True):
    super().__init__(tokenizer)
    self.chat_template = gemma4_chat_template()
    self.enable_thinking = enable_thinking
    self.tool_names: set[str] = set()

  @property
  def has_extension_property(self) -> bool:
    return True

  @property
  def bos_tokens(self) -> list[int]:
    return self.tokenizer.encode("<bos>", add_special_tokens=False)

  @property
  def end_message_tokens(self) -> set[int]:
    return {
      self.tokenizer.encode("<turn|>", add_special_tokens=False)[0],
      self.tokenizer.encode("<|tool_response>", add_special_tokens=False)[0],
    }

  def get_stop_sequences(self) -> list[int]:
    return sorted(self.end_message_tokens)

  def template_tokens(
    self,
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
  ) -> list[int]:
    rendered = self.tokenizer.apply_chat_template(
      messages,
      chat_template=self.chat_template,
      tokenize=True,
      add_generation_prompt=add_generation_prompt,
      enable_thinking=self.enable_thinking if enable_thinking is None else enable_thinking,
    )
    if isinstance(rendered, Mapping):
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
    always formatted by the pinned canonical Hugging Face template.
    """
    if message["role"] == "tool":
      raise NotImplementedError("Gemma 4 tool responses require their preceding tool call")

    rendered = self.template_tokens(
      [self.to_openai_message(message)],
      add_generation_prompt=False,
      enable_thinking=False,
    )
    bos_len = len(self.bos_tokens)
    rendered = rendered[bos_len:]
    role = "model" if message["role"] in ("assistant", "model") else message["role"]
    header = self.tokenizer.encode(f"<|turn>{role}\n", add_special_tokens=False)
    if rendered[: len(header)] != header:
      raise ValueError(f"Unexpected Gemma 4 template output for role {message['role']!r}")
    return RenderedMessage(
      header=tinker.EncodedTextChunk(tokens=header),
      output=[tinker.EncodedTextChunk(tokens=rendered[len(header) :])],
    )

  def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
    eos_token_id = self.tokenizer.eos_token_id
    ended_with_eos = eos_token_id is not None and bool(response) and response[-1] == eos_token_id
    if ended_with_eos:
      response = response[:-1]
    decoded = self.tokenizer.decode(response, skip_special_tokens=False)
    parse_input = decoded
    if self.enable_thinking and "<channel|>" in decoded and not decoded.startswith("<|channel>"):
      # After a tool response the canonical template opens the thought channel
      # in the prompt. Sampling returns only its continuation, so reconstruct
      # that opener before handing the response to Gemma's native parser.
      parse_input = "<|channel>thought\n" + decoded
    try:
      parsed = self.tokenizer.parse_response(parse_input)
    except Exception:
      parsed = {"role": "assistant"}

    tool_calls = _valid_tool_calls(parsed)
    if "<|tool_call>" in parse_input and not tool_calls:
      # Gemma sometimes closes thought, emits final text, then opens a tool
      # call. That channel order is outside the response schema even when the
      # call block itself is complete and valid. Parse complete call blocks in
      # isolation so useful actions are not discarded with their prose.
      #
      # One call at a time, and keep whichever ones survive. Handing the whole
      # turn to the parser as a single concatenated block let one unrecoverable
      # call discard every valid call beside it: measured on run29 eval-0,
      # `good` alone parsed to 1 call and `bad` alone to 0, but `good`+`bad`
      # parsed to 0 in either order. Two of that eval's seven parse errors were
      # turns whose calls were individually well-formed.
      recovered: list[dict[str, Any]] = []
      for match in _TOOL_CALL.finditer(parse_input):
        name = match.group("name")
        if name not in self.tool_names:
          continue
        raw = match.group("body")[len("call:") + len(name) :]
        # Raw first, so a call that already parses is never touched; the lenient
        # rewrite only ever runs on text the tokenizer has already rejected.
        candidates = [raw]
        if (normalized := normalize_tool_call_args(raw)) != raw:
          candidates.append(normalized)
        for args in candidates:
          try:
            one = _valid_tool_calls(
              self.tokenizer.parse_response(f"<|tool_call>call:{name}{args}<tool_call|>")
            )
          except Exception:
            one = []
          if one:
            recovered.extend(one)
            break
      tool_calls = recovered

    parsed_tool_calls = []
    for tool_call in tool_calls:
      function = tool_call["function"]
      parsed_tool_calls.append(
        ToolCall(
          id=tool_call.get("id"),
          function=ToolCall.FunctionBody(
            name=function["name"],
            arguments=json.dumps(function.get("arguments", {}), separators=(",", ":")),
          ),
        )
      )

    parts: list[TextPart | ThinkingPart] = []
    # An empty thought is not the same as no thought. After a tool response the
    # template pre-opens <|channel>thought, so a model that closes it right away
    # still emitted a (blank) thought channel, and the re-render has to reproduce
    # it or the observation stops being a prefix of the next one. Only None --
    # the channel never opened -- means there is nothing to carry.
    thinking = parsed.get("thinking")
    if thinking is not None:
      parts.append(ThinkingPart(type="thinking", thinking=thinking))
    if content := parsed.get("content"):
      parts.append(TextPart(type="text", text=content))

    message = Message(role="assistant", content=parts if parts else "")
    if parsed_tool_calls:
      message["tool_calls"] = parsed_tool_calls

    malformed_tool_call = "<|tool_call>" in decoded and not parsed_tool_calls
    if malformed_tool_call:
      return message, ParseTermination.MALFORMED
    termination = ParseTermination.EOS if ended_with_eos else ParseTermination.STOP_SEQUENCE
    return message, termination

  def create_conversation_prefix_with_tools(
    self,
    tools: list[ToolSpec],
    system_prompt: str = "",
  ) -> list[Message]:
    self.tool_names = {str(tool["name"]) for tool in tools}
    native_tools = [{"type": "function", "function": tool} for tool in tools]
    rendered = self.tokenizer.apply_chat_template(
      [{"role": "system", "content": system_prompt}],
      tools=native_tools,
      chat_template=self.chat_template,
      tokenize=False,
      add_generation_prompt=False,
      # Tools are embedded into the returned system message. The final full
      # conversation render injects the single thinking marker.
      enable_thinking=False,
    )
    prefix = "<bos><|turn>system\n"
    suffix = "<turn|>\n"
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
      raise ValueError("Unexpected Gemma 4 system/tool template")
    return [Message(role="system", content=rendered[len(prefix) : -len(suffix)])]

  def to_openai_message(self, message: Message) -> dict[str, Any]:
    # Thinking must travel as reasoning_content, not as inline <think> tags.
    # The base implementation flattens content parts and wraps ThinkingPart in
    # <think>...</think>; Gemma 4 has no such tag. Its template reads thinking
    # from reasoning/reasoning_content and emits <|channel>thought ... <channel|>.
    # Left inline the tags land verbatim in the model channel, and because the
    # template renders tool_calls before content they land *after* the tool call
    # the thought was meant to precede.
    #
    # This also broke trajectory prefix merging. build_generation_prompt
    # re-renders the whole history every turn, so a thought block that migrates
    # across <|tool_response>| between turns stops observation k+1 from being a
    # token prefix of observation k, and data_processing flushes the accumulator
    # and starts a new training sequence -- run24 emitted 429 sequences for 48
    # trajectories over 469 env steps, ~8.9x the tokens it needed. Emitting
    # reasoning_content restores the canonical block, which also matches the
    # <|channel>thought opener the template appends to the generation prompt
    # after a tool response, so the chain holds across turns.
    content = message["content"]
    thinking: str | None = None
    if not isinstance(content, str) and any(p["type"] == "thinking" for p in content):
      # Pass the parser's text through verbatim, including the newline it keeps
      # from the closing delimiter. The template appends '\n' only when the text
      # does not already end in one, so raw round-trips exactly and an external
      # caller writing plain prose still gets the canonical form.
      thinking = "".join(p["thinking"] for p in content if p["type"] == "thinking")
      # Drop the parts unconditionally, even when what is left is empty or pure
      # whitespace. Leaving them for the base implementation to flatten puts a
      # literal <think> tag back into the model channel, which is the failure
      # this override exists to prevent.
      message = cast(Message, {**message, "content": [p for p in content if p["type"] != "thinking"]})

    result = super().to_openai_message(message)
    if thinking is not None:
      # Pass "" through as well: the template distinguishes none (no channel)
      # from empty (channel opened and closed immediately).
      result["reasoning_content"] = thinking
    if result["role"] == "model":
      result["role"] = "assistant"
    for tool_call in result.get("tool_calls", []):
      arguments = tool_call["function"].get("arguments")
      if isinstance(arguments, str):
        tool_call["function"]["arguments"] = json.loads(arguments)
    return result


def register_gemma4_tool_renderer(name: str = "gemma4") -> None:
  renderers.register_renderer(name, lambda tokenizer, img_proc=None: Gemma4ToolRenderer(tokenizer))
