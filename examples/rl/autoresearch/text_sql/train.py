"""Editable and runnable Text-SQL autoresearch attempt."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import chz
from tinker_cookbook.utils import ml_log

from rl.autoresearch.text_sql import prepare
from rl.autoresearch.tinker import force_rich_log_colors

TRAIN_STEPS = 40
LOGGER = logging.getLogger(__name__)


@chz.chz
class RunConfig:
  run_dir: Path = chz.field(doc="Attempt artifact directory written by run_attempt.")
  cache_root: Path = Path("artifacts/autoresearch/text_sql")
  attempt_timeout_minutes: float = float(os.getenv("ATTEMPT_TIMEOUT_MINUTES", "5"))


def generate_sql(example: dict[str, Any], state: dict[str, Any]) -> str:
  """Return one SQL query for the given question and schema."""

  _ = example, state
  return ""


def configure_logger() -> None:
  for handler in LOGGER.handlers:
    handler.close()
  LOGGER.handlers.clear()
  LOGGER.setLevel(logging.INFO)
  LOGGER.propagate = False
  handler = logging.StreamHandler(sys.stdout)
  handler.setFormatter(logging.Formatter("%(message)s"))
  LOGGER.addHandler(handler)


def close_logger() -> None:
  for handler in LOGGER.handlers:
    handler.flush()
    handler.close()
  LOGGER.handlers.clear()


def train(examples: list[dict[str, Any]], args: RunConfig) -> tuple[dict[str, Any], dict[str, float]]:
  state = {"steps": 0}
  deadline = time.monotonic() + args.attempt_timeout_minutes * 60
  start = time.monotonic()
  LOGGER.info("training loop: %s steps over %s examples", TRAIN_STEPS, len(examples))
  for step in range(1, TRAIN_STEPS + 1):
    if time.monotonic() >= deadline:
      LOGGER.info("attempt timeout reached at step %s", step - 1)
      break
    _ = generate_sql(examples[(step - 1) % len(examples)], state)
    state["steps"] = step
    if step == 1 or step == TRAIN_STEPS or step % 10 == 0:
      LOGGER.info("train step %s/%s", step, TRAIN_STEPS)
  seconds = time.monotonic() - start
  (args.run_dir / "training_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return state, {"train/steps": float(state["steps"]), "train/seconds": seconds}


def result_lines(row: dict[str, Any]) -> str:
  mark = "PASS" if row["correct"] else "FAIL"
  error = f"\n  error:     {row['execution_error']}" if row.get("execution_error") else ""
  return f"[dev] {mark} {row['question']}\n  predicted: {row['predicted']}\n  target:    {row['target']}{error}"


def write_train_examples(run_dir: Path, examples: list[dict[str, str]]) -> None:
  with (run_dir / "train_examples.jsonl").open("w", encoding="utf-8") as file:
    for example in examples:
      if example["split"] == "train":
        file.write(json.dumps(prepare.public_example(example), sort_keys=True) + "\n")


def run(args: RunConfig) -> Path:
  args.run_dir.mkdir(parents=True, exist_ok=True)
  configure_logger()
  ml_logger = ml_log.setup_logging(log_dir=str(args.run_dir), config=args, do_configure_logging_module=True)
  try:
    LOGGER.info("Open-RL text-to-SQL autoresearch run: %s", args.run_dir.name)
    LOGGER.info("loading dataset: %s", prepare.DEFAULT_DATASET)
    examples = prepare.load_examples(args.cache_root)
    write_train_examples(args.run_dir, examples)
    LOGGER.info("train/dev split: %s/%s rows from %s", prepare.TRAIN_LIMIT, prepare.DEV_LIMIT, prepare.DEFAULT_DATASET)
    state, train_metrics = train([prepare.public_example(example) for example in examples if example["split"] == "train"], args)
    results, metrics = prepare.score_dev(examples, lambda example: generate_sql(example, state))
    for row in results:
      LOGGER.info(result_lines(row))
    metrics.update(train_metrics)
    metrics["step"] = metrics["train/steps"]
    LOGGER.info("accuracy: %.3f", metrics["accuracy"])
    ml_logger.log_metrics({"phase": "eval", **metrics}, step=int(metrics["step"]))
    LOGGER.info("wrote %s", args.run_dir)
  finally:
    ml_logger.close()
    close_logger()
  return args.run_dir


def main() -> None:
  force_rich_log_colors()
  run(chz.entrypoint(RunConfig, allow_hyphens=True))


if __name__ == "__main__":
  main()
