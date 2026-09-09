"""Count failed episodes in the rubric denominator and normalize terminal flags."""

from dataclasses import replace

from tinker_cookbook.rl.types import Env, InitialObservationOverflow

# Missing keys are excluded from cookbook's metric averages. Fill zeroes on
# ordinary transitions so these flags measure rates rather than presence.
# The denominator is transitions, not episodes.
_TERMINAL_FLAG_METRICS = ("context_overflow", "parse_error", "max_tokens_reached")


class LabEpisodeEnv(Env):
  def __init__(self, env: Env, criteria_count: int):
    self.env = env
    self.criteria_count = criteria_count

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
          "lab/all_pass": 0.0,
          "lab/graded": 0.0,
          "lab/no_output": 0.0,
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
    return replace(result, metrics=metrics)
