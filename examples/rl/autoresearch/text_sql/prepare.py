"""Fixed Text-SQL data and scoring helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET = "philschmid/gretel-synthetic-text-to-sql"
DATASET_LIMIT = 5_050
DEV_LIMIT = 50
TRAIN_LIMIT = 5_000
SEED = 42


def clean_sql(text: str) -> str:
  text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
  start = re.search(r"\bselect\b", text, flags=re.IGNORECASE)
  if start:
    text = text[start.start() :]
  return re.sub(r";+\s*$", "", " ".join(text.split())).strip()


def public_example(example: dict[str, str]) -> dict[str, str]:
  return {"question": example["question"], "context": example["context"]}


def dataset_spec() -> dict[str, Any]:
  return {"dataset_name": DEFAULT_DATASET, "dataset_limit": DATASET_LIMIT, "train_limit": TRAIN_LIMIT, "dev_limit": DEV_LIMIT, "seed": SEED}


def dataset_cache_path(cache_root: Path) -> Path:
  return cache_root / "_dataset_sample.json"


def example_from_row(row: dict[str, Any], split: str) -> dict[str, str]:
  return {"split": split, "question": row["sql_prompt"], "context": row["sql_context"], "target": clean_sql(row["sql"])}


def load_examples(cache_root: Path) -> list[dict[str, str]]:
  spec = dataset_spec()
  cache_path = dataset_cache_path(cache_root)
  if cache_path.exists():
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    if cached.get("spec") == spec:
      return cached["examples"]

  raw = load_dataset(DEFAULT_DATASET, split="train").shuffle(seed=SEED)
  raw = raw.select(range(min(DATASET_LIMIT, len(raw))))
  examples: list[dict[str, str]] = []
  counts = {"train": 0, "dev": 0}
  for row in raw:
    split = "train" if counts["train"] < TRAIN_LIMIT else "dev"
    if counts[split] >= (TRAIN_LIMIT if split == "train" else DEV_LIMIT):
      break
    examples.append(example_from_row(row, split))
    counts[split] += 1

  if counts["train"] < TRAIN_LIMIT or counts["dev"] < DEV_LIMIT:
    raise RuntimeError(f"dataset sample only produced train={counts['train']} dev={counts['dev']}; increase DATASET_LIMIT above {DATASET_LIMIT}")

  cache_path.parent.mkdir(parents=True, exist_ok=True)
  cache_path.write_text(json.dumps({"spec": spec, "examples": examples}, indent=2) + "\n", encoding="utf-8")
  return examples


def score_dev(examples: list[dict[str, str]], predict) -> tuple[list[dict[str, Any]], dict[str, float]]:
  results = []
  for example in examples:
    if example["split"] != "dev":
      continue
    predicted = clean_sql(predict(public_example(example)))
    target = clean_sql(example["target"])
    results.append({**example, "predicted": predicted, "correct": predicted == target})
  metrics = {"accuracy": sum(float(row["correct"]) for row in results) / len(results)}
  metrics["dataset/train_size"] = float(sum(row["split"] == "train" for row in examples))
  metrics["dataset/dev_size"] = float(len(results))
  return results, metrics
