"""Dataset batching + splits for the text-to-SQL SFT + RL recipe."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import Any

from datasets import load_dataset
from texttosql_sft import build_examples
from transformers import PreTrainedTokenizerBase

Example = dict[str, Any]


def shuffled_batches(examples: list[Example], batch_size: int, seed: int) -> Iterator[list[Example]]:
  """Yield shuffled mini-batches forever, reshuffling when the pool is exhausted."""
  if not examples:
    raise ValueError("Cannot batch an empty example list.")

  rng = random.Random(seed)
  order = list(range(len(examples)))
  rng.shuffle(order)
  pos = 0
  batch_size = min(batch_size, len(examples))

  while True:
    if pos + batch_size > len(order):
      rng.shuffle(order)
      pos = 0
    yield [examples[order[i]] for i in range(pos, pos + batch_size)]
    pos += batch_size


def load_example_splits(config: Any, tokenizer: PreTrainedTokenizerBase) -> tuple[list[Example], list[Example], list[Example]]:
  """Return (sft_train, rl_train, eval), skipping phase-specific builds when disabled."""
  do_sft = config.phase in {"full", "sft_only"}
  do_rl = config.phase in {"full", "rl_only"}

  dataset = load_dataset(config.dataset_name, split="train").shuffle(seed=config.seed)
  dataset = dataset.select(range(min(config.dataset_limit, len(dataset))))
  if len(dataset) < 10:
    raise RuntimeError("dataset_limit is too small to create train/eval splits")

  split = dataset.train_test_split(test_size=min(2_500, max(1, len(dataset) // 5)), shuffle=False)

  sft_examples = build_examples(tokenizer, config.prompt_format, split["train"], config.train_limit) if do_sft else []
  rl_examples = (
    build_examples(
      tokenizer,
      config.prompt_format,
      split["train"],
      config.rl_train_limit,
      require_seed_data=True,
      require_target_rows=True,
    )
    if do_rl
    else []
  )
  eval_examples = build_examples(
    tokenizer,
    config.prompt_format,
    split["test"],
    config.eval_limit,
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
