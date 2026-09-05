"""The Qwen3.5 history chain, checked with the real tokenizer.

Invariant: for every turn, the next observation the env renders is exactly
prompt + sampled tokens + new tokens. trajectory_to_data merges a whole
episode into one datum if and only if that holds at every turn.

Run where the Qwen/Qwen3.5-9B tokenizer is cached and tinker_cookbook is
installed, e.g. on the box:

  cd ~/open-rl && examples/.venv/bin/python -m unittest tests.test_qwen35_verbatim_renderer -v
"""

from __future__ import annotations

import unittest

try:
  import tinker
  from harvey_labs.qwen35_renderer import SAMPLED_TOKENS_KEY, VerbatimHistoryQwen35Renderer, verbatim_history_renderer
  from tinker_cookbook import tokenizer_utils
  from tinker_cookbook.completers import TokensWithLogprobs
  from tinker_cookbook.renderers.base import message_to_jsonable
  from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
  from tinker_cookbook.rl.data_processing import trajectory_to_data
  from tinker_cookbook.rl.types import Trajectory, Transition

  HAVE_COOKBOOK = True
except ImportError:  # pragma: no cover - environment without the cookbook
  HAVE_COOKBOOK = False

MODEL = "Qwen/Qwen3.5-9B"

SEED_MESSAGES = [
  {"role": "system", "content": "You are a lab assistant."},
  {"role": "user", "content": "List the files in the working directory, then summarize them."},
]

# What a Qwen3.5 policy emits after the prefilled "<think>\n". Each one is a
# formatting the canonical re-render does not reproduce, except plain text.
SAMPLED = {
  "tool_call": (
    "Let me look around.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call><|im_end|>"
  ),
  "tool_call_one_line_params": (
    "Check.\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>ls -la</parameter>\n</function>\n</tool_call><|im_end|>"
  ),
  "two_tool_calls": (
    "Two things.\n"
    "</think>\n"
    "\n"
    "<tool_call>\n"
    "<function=bash>\n"
    "<parameter=command>\n"
    "ls\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n"
    "<tool_call>\n"
    "<function=bash>\n"
    "<parameter=command>\n"
    "pwd\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call><|im_end|>"
  ),
  "whitespace_in_thinking": (
    "  Let me think...  \n\n</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call><|im_end|>"
  ),
  "text_answer": "Done.\n</think>\n\nThere are three files: a.txt, b.txt and c.txt.<|im_end|>",
  "text_with_trailing_newlines": "Done.\n</think>\n\nThe summary follows.\n\n<|im_end|>",
}

TOOL_RESULT = {"role": "tool", "content": "total 0\n-rw-r--r-- 1 lab lab 0 a.txt\n", "tool_call_id": "call_0"}


def first_mismatch(expected: list[int], actual: list[int]) -> int | None:
  n = min(len(expected), len(actual))
  for i in range(n):
    if expected[i] != actual[i]:
      return i
  return None if len(actual) >= len(expected) else n


@unittest.skipUnless(HAVE_COOKBOOK, "tinker_cookbook is not installed")
class VerbatimHistoryTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    cls.tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    cls.stock = Qwen3_5Renderer(cls.tokenizer)
    cls.stock.strip_thinking_from_history = False
    cls.verbatim = verbatim_history_renderer(cls.stock)

  def sampled_tokens(self, text: str) -> list[int]:
    return self.tokenizer.encode(text, add_special_tokens=False)

  def next_observation(self, renderer, messages, sampled):
    message, termination = renderer.parse_response(sampled)
    self.assertTrue(termination.is_clean, termination)
    return renderer.build_generation_prompt(messages + [message, TOOL_RESULT]).to_ints(), message

  def test_verbatim_renderer_is_the_stock_class_plus_history(self) -> None:
    self.assertIsInstance(self.verbatim, Qwen3_5Renderer)
    self.assertIsInstance(self.verbatim, VerbatimHistoryQwen35Renderer)
    self.assertTrue(self.verbatim.has_extension_property)

  def test_every_sampled_form_extends_the_previous_observation(self) -> None:
    prompt = self.verbatim.build_generation_prompt(SEED_MESSAGES).to_ints()
    for name, text in SAMPLED.items():
      with self.subTest(sampled=name):
        sampled = self.sampled_tokens(text)
        nxt, _ = self.next_observation(self.verbatim, SEED_MESSAGES, sampled)
        expected = prompt + sampled
        i = first_mismatch(expected, nxt)
        self.assertIsNone(
          i,
          f"chain breaks at token {i}: expected {self.tokenizer.decode(expected[i : i + 8])!r}, got {self.tokenizer.decode(nxt[i : i + 8])!r}"
          if i is not None
          else None,
        )
        self.assertGreater(len(nxt), len(expected), "the tool result must follow")

  def test_stock_renderer_still_changes_inline_tool_parameters(self) -> None:
    """Pins the upstream behaviour this module exists for.

    If this starts failing, upstream fixed the re-render and the override can
    be retired.
    """
    prompt = self.stock.build_generation_prompt(SEED_MESSAGES).to_ints()
    sampled = self.sampled_tokens(SAMPLED["tool_call_one_line_params"])
    nxt, _ = self.next_observation(self.stock, SEED_MESSAGES, sampled)
    self.assertIsNotNone(first_mismatch(prompt + sampled, nxt))
    # And the plain-text form is the one case upstream gets right.
    sampled = self.sampled_tokens(SAMPLED["text_answer"])
    nxt, _ = self.next_observation(self.stock, SEED_MESSAGES, sampled)
    self.assertIsNone(first_mismatch(prompt + sampled, nxt))

  def test_parsed_message_keeps_its_semantics(self) -> None:
    sampled = self.sampled_tokens(SAMPLED["two_tool_calls"])
    ours, _ = self.verbatim.parse_response(sampled)
    theirs, _ = self.stock.parse_response(sampled)
    self.assertEqual(ours.get("tool_calls"), theirs.get("tool_calls"))
    self.assertEqual(ours.get("content"), theirs.get("content"))
    self.assertEqual(ours[SAMPLED_TOKENS_KEY], sampled)
    self.assertNotIn(SAMPLED_TOKENS_KEY, message_to_jsonable(ours))

  def test_messages_without_sampled_tokens_render_as_before(self) -> None:
    seeded_assistant = {"role": "assistant", "content": "Understood, starting now."}
    messages = SEED_MESSAGES + [seeded_assistant, {"role": "user", "content": "Go on."}]
    self.assertEqual(
      self.verbatim.build_generation_prompt(messages).to_ints(),
      self.stock.build_generation_prompt(messages).to_ints(),
    )

  def test_a_missing_stop_token_is_restored_in_history(self) -> None:
    sampled = self.sampled_tokens(SAMPLED["text_answer"])
    stop = self.verbatim.get_stop_sequences()[0]
    self.assertEqual(sampled[-1], stop)
    message, _ = self.verbatim.parse_response(sampled)
    message[SAMPLED_TOKENS_KEY] = sampled[:-1]
    ctx_messages = SEED_MESSAGES + [message, TOOL_RESULT]
    rendered = self.verbatim.build_generation_prompt(ctx_messages).to_ints()
    prompt = self.verbatim.build_generation_prompt(SEED_MESSAGES).to_ints()
    self.assertEqual(rendered[: len(prompt) + len(sampled)], prompt + sampled)

  def test_a_three_turn_episode_becomes_one_datum(self) -> None:
    renderer = self.verbatim
    messages = list(SEED_MESSAGES)
    transitions = []
    ob = renderer.build_generation_prompt(messages)
    for text in (SAMPLED["tool_call_one_line_params"], SAMPLED["whitespace_in_thinking"], SAMPLED["text_answer"]):
      sampled = self.sampled_tokens(text)
      message, termination = renderer.parse_response(sampled)
      self.assertTrue(termination.is_clean)
      messages.append(message)
      done = "tool_calls" not in message
      if not done:
        messages.append(TOOL_RESULT)
      ac = TokensWithLogprobs(tokens=sampled, maybe_logprobs=[-0.5] * len(sampled))
      transitions.append(Transition(ob=ob, ac=ac, reward=0.0, episode_done=done))
      ob = renderer.build_generation_prompt(messages) if not done else tinker.ModelInput.empty()
    traj = Trajectory(transitions=transitions, final_ob=ob)

    data = trajectory_to_data(traj, traj_advantage=1.0)

    self.assertEqual(len(data), 1, "every turn must extend the last, or the trainer pays per turn")
    datum = data[0]
    mask = datum.loss_fn_inputs["mask"].to_torch()
    targets = datum.loss_fn_inputs["target_tokens"].to_torch()
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
    action_tokens = [tok for t in transitions for tok in t.ac.tokens]
    self.assertEqual(int(mask.sum()), len(action_tokens))
    self.assertEqual(targets[mask.bool()].tolist(), action_tokens)
    self.assertTrue((advantages[mask.bool()] == 1.0).all())
    self.assertTrue((logprobs[mask.bool()] == -0.5).all())
    self.assertTrue((advantages[~mask.bool()] == 0.0).all())
    # The datum is the final context, not the sum of per-turn contexts.
    final_len = len(transitions[-1].ob.to_ints()) + len(transitions[-1].ac.tokens)
    self.assertEqual(datum.model_input.length + 1, final_len)


if __name__ == "__main__":
  unittest.main()
