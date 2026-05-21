"""Fixed Text-SQL data and scoring helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET = "philschmid/gretel-synthetic-text-to-sql"
DATASET_LIMIT = 12_500
TEST_SPLIT_SIZE = 2_500
TRAIN_LIMIT = 5_000
TEST_LIMIT = 50
SEED = 42
SQL_INTERRUPT_CHECK_STEPS = 100_000
SQL_INTERRUPT_AFTER_CHECKS = 50
ROW_RETURNING_KEYWORDS = ("select", "with")


def split_examples(examples: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
  train_examples = []
  test_examples = []
  for example in examples:
    if example["split"] == "train":
      train_examples.append(example)
    elif example["split"] == "test":
      test_examples.append(example)
  return train_examples, test_examples


def clean_sql(text: str) -> str:
  text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
  start = re.search(r"\b(select|with|update|insert|delete)\b", text, flags=re.IGNORECASE)
  if start:
    text = text[start.start() :]
  return re.sub(r";+\s*$", "", " ".join(text.split())).strip()


def is_row_returning_query(sql: str) -> bool:
  return clean_sql(sql).lower().startswith(ROW_RETURNING_KEYWORDS)


def run_sql(context: str, sql: str) -> tuple[list[tuple[Any, ...]] | None, str | None]:
  sql = clean_sql(sql)
  if not sql:
    return None, "empty query"
  if ";" in sql:
    return None, "multiple SQL statements"
  conn = sqlite3.connect(":memory:")
  steps = 0

  def progress() -> int:
    nonlocal steps
    steps += 1
    return int(steps > SQL_INTERRUPT_AFTER_CHECKS)

  try:
    conn.executescript(context)
    conn.set_progress_handler(progress, SQL_INTERRUPT_CHECK_STEPS)
    cursor = conn.execute(sql)
    rows = [tuple(row) for row in cursor.fetchall()] if cursor.description else []
    return rows, None
  except sqlite3.Error as exc:
    return None, str(exc)
  finally:
    conn.close()


def load_examples(data_dir: Path) -> list[dict[str, str]]:
  dataset_spec = {
    "dataset_name": DEFAULT_DATASET,
    "dataset_limit": DATASET_LIMIT,
    "test_split_size": TEST_SPLIT_SIZE,
    "train_limit": TRAIN_LIMIT,
    "test_limit": TEST_LIMIT,
    "seed": SEED,
  }
  dataset_sample_path = data_dir / "dataset_sample.json"
  if dataset_sample_path.exists():
    cached = json.loads(dataset_sample_path.read_text(encoding="utf-8"))
    if cached.get("spec") == dataset_spec:
      return cached["examples"]

  raw = load_dataset(DEFAULT_DATASET, split="train").shuffle(seed=SEED)
  raw = raw.select(range(min(DATASET_LIMIT, len(raw))))
  split_rows = raw.train_test_split(test_size=min(TEST_SPLIT_SIZE, max(1, len(raw) // 5)), shuffle=False)
  examples: list[dict[str, str]] = []
  examples.extend(build_dataset_examples(split_rows["train"], "train", TRAIN_LIMIT))
  examples.extend(build_dataset_examples(split_rows["test"], "test", TEST_LIMIT))

  train_examples, test_examples = split_examples(examples)
  if not train_examples:
    raise RuntimeError("No train examples with executable target rows were found.")
  if not test_examples:
    raise RuntimeError("No test examples with executable target rows were found.")

  dataset_sample_path.parent.mkdir(parents=True, exist_ok=True)
  dataset_sample_path.write_text(json.dumps({"spec": dataset_spec, "examples": examples}, indent=2) + "\n", encoding="utf-8")
  return examples


def build_dataset_examples(rows: Any, split: str, limit: int) -> list[dict[str, str]]:
  examples = []
  for row in rows:
    example = {
      "split": split,
      "question": row["sql_prompt"],
      "context": row["sql_context"],
      "target": clean_sql(row["sql"]),
    }

    if "insert into" not in example["context"].lower():
      continue
    if not is_row_returning_query(example["target"]):
      continue
    target_rows, error = run_sql(example["context"], example["target"])
    if error is not None or not target_rows:
      continue
    example["target_rows"] = target_rows

    examples.append(example)
    if len(examples) >= limit:
      break
  return examples


def score_test_predictions(examples: list[dict[str, str]], predictions: list[str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
  train_examples, test_examples = split_examples(examples)
  results = []
  for example, prediction in zip(test_examples, predictions, strict=True):
    predicted = clean_sql(prediction)
    target = clean_sql(example["target"])
    predicted_rows, execution_error = run_sql(example["context"], predicted)
    target_rows = [tuple(row) for row in example["target_rows"]]
    if not predicted_rows:
      correct = False
    elif re.search(r"\border\s+by\b", target, flags=re.IGNORECASE):
      correct = predicted_rows == target_rows
    else:
      correct = sorted(predicted_rows, key=repr) == sorted(target_rows, key=repr)
    results.append(
      {
        **example,
        "predicted": predicted,
        "correct": correct,
        "execution_error": execution_error or "",
      }
    )
  correct = sum(float(row["correct"]) for row in results)
  predicted_errors = sum(float(bool(row["execution_error"])) for row in results)
  metrics = {
    "accuracy": correct / len(results),
    "execution/predicted_errors": predicted_errors,
    "dataset/train_size": float(len(train_examples)),
    "dataset/test_size": float(len(results)),
  }
  return results, metrics
