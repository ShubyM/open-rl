"""Budget-terminated LAB episodes are graded, not written off.

cd ~/open-rl && examples/.venv/bin/python -m unittest tests.test_harvey_episode_limits -v
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field

from harvey_labs.episode import LabEpisodeEnv
from harvey_labs.reward import reward_from_scores


@dataclass
class Result:
  reward: float
  episode_done: bool
  metrics: dict = field(default_factory=dict)


class MessageEnv:
  def __init__(self, reward, rubric):
    self.history = ["system", "user", "assistant"]
    self.graded_with = None

    async def reward_fn(history):
      self.graded_with = history
      return reward, rubric

    self.reward_fn = reward_fn


class Inner:
  def __init__(self, result, message_env):
    self.result = result
    self.message_env = message_env

  async def step(self, action, *, extra=None):
    return self.result


# The keys LabRubricReward.score returns.
RUBRIC = {
  "lab/criteria_passed": 30.0,
  "lab/criteria_total": 40.0,
  "lab/criteria_pass_fraction": 0.75,
  "lab/all_pass": 0.0,
  "lab/graded": 1.0,
  "lab/failed_before_grading": 0.0,
  "lab/no_output": 0.0,
  "lab/reward_error": 0.0,
}


class EpisodeLimitTest(unittest.TestCase):
  def run_step(self, result, reward=0.75, rubric=RUBRIC):
    message_env = MessageEnv(reward, rubric)
    env = LabEpisodeEnv(Inner(result, message_env), criteria_count=40)
    return asyncio.run(env.step([1, 2, 3])), message_env

  def test_max_tokens_terminal_is_graded(self):
    stepped, message_env = self.run_step(Result(-0.1, True, {"max_tokens_reached": 1.0}))
    self.assertEqual(stepped.reward, 0.75)
    self.assertEqual(stepped.metrics["lab/criteria_pass_fraction"], 0.75)
    self.assertEqual(stepped.metrics["lab/graded"], 1.0)
    self.assertEqual(stepped.metrics["lab/failed_before_grading"], 0.0)
    self.assertEqual(stepped.metrics["max_tokens_reached"], 1.0)
    self.assertIs(message_env.graded_with, message_env.history)

  def test_context_overflow_terminal_is_graded(self):
    stepped, _ = self.run_step(Result(-0.1, True, {"context_overflow": 1.0}))
    self.assertEqual(stepped.reward, 0.75)
    self.assertEqual(stepped.metrics["context_overflow"], 1.0)

  def test_already_graded_terminal_is_left_alone(self):
    graded = Result(0.4, True, {**RUBRIC, "lab/criteria_pass_fraction": 0.4})
    stepped, message_env = self.run_step(graded)
    self.assertEqual(stepped.reward, 0.4)
    self.assertIsNone(message_env.graded_with)

  def test_non_terminal_step_is_not_graded(self):
    stepped, message_env = self.run_step(Result(0.0, False, {}))
    self.assertEqual(stepped.reward, 0.0)
    self.assertIsNone(message_env.graded_with)
    self.assertEqual(stepped.metrics["max_tokens_reached"], 0.0)

  def test_parse_error_terminal_keeps_its_penalty(self):
    stepped, message_env = self.run_step(Result(-0.1, True, {"parse_error": 1.0}))
    self.assertEqual(stepped.reward, -0.1)
    self.assertIsNone(message_env.graded_with)
    self.assertEqual(stepped.metrics["lab/failed_before_grading"], 1.0)


class RewardFromScoresTest(unittest.TestCase):
  def test_reward_is_the_pass_fraction(self):
    reward, metrics = reward_from_scores({"n_criteria": 10, "n_passed": 9, "all_pass": False})
    self.assertAlmostEqual(reward, 0.9)
    self.assertEqual(metrics["lab/criteria_pass_fraction"], 0.9)
    self.assertEqual(metrics["lab/all_pass"], 0.0)

  def test_perfect_deliverable_scores_one(self):
    reward, metrics = reward_from_scores({"n_criteria": 10, "n_passed": 10, "all_pass": True})
    self.assertAlmostEqual(reward, 1.0)
    self.assertEqual(metrics["lab/all_pass"], 1.0)

  def test_no_criteria_scores_zero(self):
    reward, _ = reward_from_scores({})
    self.assertEqual(reward, 0.0)


if __name__ == "__main__":
  unittest.main()
