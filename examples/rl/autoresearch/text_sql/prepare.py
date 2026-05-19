"""Fixed Text-SQL data and scoring helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_DATASET = "philschmid/gretel-synthetic-text-to-sql"
DATASET_LIMIT = 6_000
DEV_LIMIT = 50
TRAIN_LIMIT = 5_000
SEED = 42
SQL_PROGRESS_STEPS = 100_000
SQL_PROGRESS_LIMIT = 50


def clean_sql(text: str) -> str:
  text = re.sub(r"^```(?:sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
  start = re.search(r"\b(select|with|update|insert|delete)\b", text, flags=re.IGNORECASE)
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


def ordered(sql: str) -> bool:
  return bool(re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE))


def query_kind(sql: str) -> str:
  match = re.match(r"\s*([a-z]+)", sql, flags=re.IGNORECASE)
  return match.group(1).lower() if match else ""


def quote_identifier(name: str) -> str:
  return '"' + name.replace('"', '""') + '"'


def database_state(conn: sqlite3.Connection) -> tuple:
  tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
  state = []
  for table in tables:
    quoted = quote_identifier(table)
    columns = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({quoted})"))
    rows = tuple(sorted(conn.execute(f"SELECT * FROM {quoted}").fetchall(), key=repr))
    state.append((table, columns, rows))
  return tuple(state)


def execute_sql(context: str, sql: str) -> dict[str, Any]:
  sql = clean_sql(sql)
  if not sql:
    return {"ok": False, "error": "empty query", "kind": "", "rows": [], "state": ()}
  conn = sqlite3.connect(":memory:")
  steps = 0

  def progress() -> int:
    nonlocal steps
    steps += 1
    return int(steps > SQL_PROGRESS_LIMIT)

  try:
    conn.executescript(context)
    conn.set_progress_handler(progress, SQL_PROGRESS_STEPS)
    cursor = conn.execute(sql)
    rows = [tuple(row) for row in cursor.fetchall()] if cursor.description else []
    conn.commit()
    return {"ok": True, "error": "", "kind": query_kind(sql), "rows": rows, "state": database_state(conn)}
  except sqlite3.Error as exc:
    return {"ok": False, "error": str(exc), "kind": query_kind(sql), "rows": [], "state": ()}
  finally:
    conn.close()


def same_result(target_sql: str, predicted: dict[str, Any], target: dict[str, Any]) -> bool:
  if not predicted["ok"] or not target["ok"]:
    return False
  if target["kind"] not in {"select", "with"}:
    return predicted["state"] == target["state"]
  if predicted["kind"] not in {"select", "with"}:
    return False
  predicted_rows, target_rows = predicted["rows"], target["rows"]
  if ordered(target_sql):
    return predicted_rows == target_rows
  return sorted(predicted_rows, key=repr) == sorted(target_rows, key=repr)


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
    example = example_from_row(row, split)
    if split == "dev" and not execute_sql(example["context"], example["target"])["ok"]:
      continue
    examples.append(example)
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
    predicted_result = execute_sql(example["context"], predicted)
    target_result = execute_sql(example["context"], target)
    results.append(
      {
        **example,
        "predicted": predicted,
        "correct": same_result(target, predicted_result, target_result),
        "execution_error": predicted_result["error"],
        "target_error": target_result["error"],
      }
    )
  metrics = {"accuracy": sum(float(row["correct"]) for row in results) / len(results)}
  metrics["execution/predicted_errors"] = sum(float(bool(row["execution_error"])) for row in results)
  metrics["execution/target_errors"] = sum(float(bool(row["target_error"])) for row in results)
  metrics["dataset/train_size"] = float(sum(row["split"] == "train" for row in examples))
  metrics["dataset/dev_size"] = float(len(results))
  return results, metrics
