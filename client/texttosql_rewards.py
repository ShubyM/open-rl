"""Composite reward for text-to-SQL GRPO.

Inspired by Reasoning-SQL (Pourreza et al. 2025). Combines a handful of
continuous partial signals so RL has useful gradient even before the
policy can produce exactly-correct SQL:

- Schema linking: Jaccard on tables/columns referenced
- N-gram similarity: Jaccard on bigrams of normalized SQL
- SequenceMatcher similarity: string-level
- Partial execution credit: column count, row count, value overlap

These sit on top of the binary compile + execution-match signals.
`compute_sql_reward` returns a dict with `total` plus each component so
the training loop can log / plot them individually.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from texttosql_sft import normalize_sql, run_sql, sql_results_match

_COLUMN_TYPE_PATTERN = re.compile(
  r"^\s+(\w+)\s+(?:TEXT|INTEGER|REAL|NUMERIC|BLOB|VARCHAR|CHAR|INT|FLOAT|DOUBLE|DECIMAL|BOOLEAN|DATE|TIMESTAMP)",
  re.IGNORECASE | re.MULTILINE,
)
_TABLE_PATTERN = re.compile(r"CREATE TABLE (\w+)", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"\b\w+\b")


def schema_items(context: str) -> set[str]:
  items = {m.group(1).lower() for m in _TABLE_PATTERN.finditer(context)}
  items |= {m.group(1).lower() for m in _COLUMN_TYPE_PATTERN.finditer(context)}
  return items


def schema_linking_reward(predicted_sql: str, target_sql: str, context: str) -> float:
  """Jaccard similarity of schema items (tables + columns) used in predicted vs gold SQL."""
  schema = schema_items(context)

  def used(sql: str) -> set[str]:
    return {w.lower() for w in _WORD_PATTERN.findall(sql)} & schema

  pred, gold = used(predicted_sql), used(target_sql)
  union = pred | gold
  return 1.0 if not union else len(pred & gold) / len(union)


def ngram_similarity(predicted_sql: str, target_sql: str, n: int = 2) -> float:
  """Jaccard similarity of n-grams between predicted and gold SQL (bigrams by default)."""

  def ngrams(text: str) -> set[tuple[str, ...]]:
    tokens = text.lower().split()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

  pred, gold = ngrams(predicted_sql), ngrams(target_sql)
  union = pred | gold
  return 1.0 if not union else len(pred & gold) / len(union)


def partial_execution_score(predicted_rows, target_rows) -> float:
  """Score 0-1 for partial execution match: column count, row count, value overlap."""
  if not predicted_rows or not target_rows:
    return 0.0

  score = 0.0
  # Column count match (25%)
  if len(predicted_rows[0]) == len(target_rows[0]):
    score += 0.25
  # Row count closeness (25%)
  score += 0.25 * (min(len(predicted_rows), len(target_rows)) / max(len(predicted_rows), len(target_rows)))
  # Value overlap across all cells (50%)
  pred_vals = {repr(v) for row in predicted_rows for v in row}
  target_vals = {repr(v) for row in target_rows for v in row}
  if target_vals:
    score += 0.5 * min(len(pred_vals & target_vals) / len(target_vals), 1.0)
  return score


def compute_sql_reward(
  example: dict[str, Any],
  predicted_sql: str,
  *,
  compile_reward: float,
  match_reward: float,
  compile_error_penalty: float,
  similarity_reward: float = 0.0,
) -> dict[str, Any]:
  """Composite reward: compile/match + weighted sum of continuous partial signals.

  `similarity_reward` is the overall weight applied to the continuous components
  (schema_linking 30%, ngram 20%, similarity 30%, partial_exec 20%). The weights
  were picked so a compile+match query always out-scores a non-matching one at
  any reasonable `similarity_reward <= match_reward`.
  """
  execution_match, execution_error = sql_results_match(
    example["context"],
    predicted_sql,
    example["target"],
    target_rows=example["target_rows"],
  )
  compiles = execution_error is None

  total = compile_error_penalty
  if compiles:
    total = compile_reward
  if execution_match:
    total += match_reward

  schema_score = schema_linking_reward(predicted_sql, example["target"], example["context"])
  ngram_score = ngram_similarity(predicted_sql, example["target"], n=2)
  similarity = SequenceMatcher(None, normalize_sql(predicted_sql), normalize_sql(example["target"])).ratio()

  partial_score = 0.0
  if compiles and not execution_match:
    predicted_rows, _ = run_sql(example["context"], predicted_sql)
    if predicted_rows is not None and example.get("target_rows") is not None:
      partial_score = partial_execution_score(predicted_rows, example["target_rows"])

  total += similarity_reward * (0.3 * schema_score + 0.2 * ngram_score + 0.3 * similarity + 0.2 * partial_score)

  return {
    "total": total,
    "compile": float(compiles),
    "execution_match": float(execution_match),
    "similarity": similarity,
    "schema_score": schema_score,
    "ngram_score": ngram_score,
    "partial_score": partial_score,
    "sqlite_error": execution_error or "",
  }
