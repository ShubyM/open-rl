"""Dataset + rollout helpers for the text-to-SQL SFT + RL recipe.

Lives separate from `texttosql_sft_grpo.py` so the training loop there reads as
"one step = forward_backward + optim_step" without 70 lines of scoring
plumbing in between.
"""

from __future__ import annotations

import logging
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

SFT_TEXT_TO_SQL_DIR = Path(__file__).resolve().parents[2] / "sft" / "text-to-sql"
if str(SFT_TEXT_TO_SQL_DIR) not in sys.path:
  sys.path.insert(1, str(SFT_TEXT_TO_SQL_DIR))

from datasets import load_dataset
from texttosql_rewards import compute_sql_reward
from texttosql_sft import build_examples, clean_sql_for_execution
from transformers import PreTrainedTokenizerBase

Example = dict[str, Any]
Rollout = dict[str, Any]


# *** Dataset splits + batching ***
def shuffled_batches(examples: list[Example], batch_size: int, seed: int) -> Iterator[list[Example]]:
  """Yield shuffled mini-batches forever, reshuffling when the pool is exhausted."""
  if not examples:
    raise ValueError("Cannot batch an empty example list.")

  rng = random.Random(seed)
  batch_size = min(batch_size, len(examples))
  while True:
    shuffled = rng.sample(examples, k=len(examples))
    for i in range(0, len(shuffled) - batch_size + 1, batch_size):
      yield shuffled[i : i + batch_size]


def load_example_splits(config: Any, tokenizer: PreTrainedTokenizerBase) -> tuple[list[Example], list[Example], list[Example]]:
  """Return (sft_train, rl_train, eval), skipping phase-specific builds when disabled."""
  do_sft = config.phase in {"full", "sft_only"}
  do_rl = config.phase in {"full", "rl_only"}

  dataset = load_dataset(config.dataset.name, split="train").shuffle(seed=config.seed)
  dataset = dataset.select(range(min(config.dataset.limit, len(dataset))))
  if len(dataset) < 10:
    raise RuntimeError("dataset_limit is too small to create train/eval splits")

  split = dataset.train_test_split(test_size=min(2_500, max(1, len(dataset) // 5)), shuffle=False)

  sft_examples = build_examples(tokenizer, config.dataset.prompt_format, split["train"], config.dataset.train_limit) if do_sft else []
  rl_examples = (
    build_examples(
      tokenizer,
      config.dataset.prompt_format,
      split["train"],
      config.dataset.rl_train_limit,
      require_seed_data=True,
      require_target_rows=True,
    )
    if do_rl
    else []
  )
  eval_examples = build_examples(
    tokenizer,
    config.dataset.prompt_format,
    split["test"],
    config.dataset.eval_limit,
    require_seed_data=True,
    require_target_rows=True,
  )

  if do_sft and not sft_examples:
    raise RuntimeError("No SFT examples fit within the max sequence length.")
  if do_rl and not rl_examples:
    raise RuntimeError("No RL examples with executable target rows were found.")
  if not eval_examples:
    raise RuntimeError("No evaluation examples with executable seed data were found.")

  logging.info("Data: %s SFT train, %s RL train, %s eval", len(sft_examples), len(rl_examples), len(eval_examples))
  return sft_examples, rl_examples, eval_examples


# *** Rollout scoring + datum construction (PPO mechanics) ***
def score_rollout(example: Example, sequence: Any, tokenizer: PreTrainedTokenizerBase, reward_cfg: Any) -> Rollout:
  """Decode one sampled sequence, compute its SQL reward, return a flat rollout dict."""
  predicted_sql = clean_sql_for_execution(tokenizer.decode(sequence.tokens, skip_special_tokens=True))
  reward = compute_sql_reward(
    example,
    predicted_sql,
    compile_reward=reward_cfg.compile,
    match_reward=reward_cfg.match,
    compile_error_penalty=reward_cfg.error_penalty,
    similarity_reward=reward_cfg.similarity,
  )
  return {
    "question": example["question"],
    "target": example["target"],
    "predicted_sql": predicted_sql,
    "prompt_tokens": example["prompt_tokens"],
    "completion_tokens": list(sequence.tokens),
    "completion_logprobs": [float(v) for v in (sequence.logprobs or [])],
    "reward": reward["total"],
    "compile": reward["compile"],
    "execution_match": reward["execution_match"],
    "similarity": reward["similarity"],
    "sqlite_error": reward["sqlite_error"],
  }
