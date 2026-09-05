"""Episode metrics and bounded diagnostics around cookbook's tool environment."""

import re
from dataclasses import replace

from tinker_cookbook.rl.types import Env, InitialObservationOverflow

# Missing keys are excluded from cookbook's metric averages. Fill zeroes on
# ordinary transitions so these flags measure rates rather than presence.
# The denominator is transitions, not episodes.
_TERMINAL_FLAG_METRICS = ("context_overflow", "parse_error", "max_tokens_reached")

_TOOL_CALL_NAME = re.compile(r"<\|tool_call>call:(\w+)")
SAMPLE_CHARS = 2000
REPETITION_WINDOW_TOKENS = 512


def failure_logs(tokenizer, action: list[int], kind: str) -> dict[str, str | int | float]:
  try:
    text = tokenizer.decode(action, skip_special_tokens=False)
  except Exception as exc:
    return {f"{kind}_decode_failed": repr(exc)}
  sample = text
  if len(text) > SAMPLE_CHARS:
    head = SAMPLE_CHARS // (4 if kind == "parse_error" else 2)
    sample = f"{text[:head]}\n...[{len(text) - SAMPLE_CHARS} chars elided]...\n{text[-(SAMPLE_CHARS - head) :]}"
  logs: dict[str, str | int | float] = {f"{kind}_text": sample, f"{kind}_chars": len(text)}
  if kind == "parse_error":
    logs.update(parse_error_call_tokens=text.count("<|tool_call>"), parse_error_call_names=",".join(sorted(set(_TOOL_CALL_NAME.findall(text)))))
  else:
    # Repetition loops have few distinct tokens; varied answers stay near 1.
    window = action[-REPETITION_WINDOW_TOKENS:]
    logs.update(max_tokens_tokens=len(action), max_tokens_distinct_frac=round(len(set(window)) / len(window), 4) if window else 0.0)
  return logs


class LabEpisodeEnv(Env):
  def __init__(self, env: Env, criteria_count: int, tokenizer):
    self.env = env
    self.criteria_count = criteria_count
    self.tokenizer = tokenizer

  def __getattr__(self, name):
    return getattr(self.env, name)

  def _metrics(self, metrics, episode_done):
    metrics = dict(metrics or {})
    for key in _TERMINAL_FLAG_METRICS:
      metrics.setdefault(key, 0.0)
    if episode_done and "lab/criteria_total" not in metrics:
      metrics.update(
        {
          "lab/criteria_passed": 0.0,
          "lab/criteria_total": float(self.criteria_count),
          "lab/criteria_pass_fraction": 0.0,
          "lab/graded": 0.0,
          "lab/reward_error": 0.0,
          "lab/failed_before_grading": 1.0,
        }
      )
    return metrics

  async def initial_observation(self):
    observation = await self.env.initial_observation()
    if isinstance(observation, InitialObservationOverflow):
      return replace(observation, metrics=self._metrics(observation.metrics, True))
    return observation

  async def step(self, action, *, extra=None):
    result = await self.env.step(action, extra=extra)
    metrics = self._metrics(result.metrics, result.episode_done)
    logs = dict(result.logs or {})
    if metrics["parse_error"]:
      logs.update(failure_logs(self.tokenizer, action, "parse_error"))
    if metrics["max_tokens_reached"]:
      logs.update(failure_logs(self.tokenizer, action, "max_tokens"))
    return replace(result, metrics=metrics, logs=logs)
