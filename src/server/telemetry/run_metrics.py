"""Client step history and its latest values, shared by inspection and Prometheus."""

import asyncio
import json
import os
import time
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from server.telemetry import ops
from server.telemetry.metric_values import METRICS_PER_ROW, RUN_ID, finite_metrics  # noqa: F401

PREFIX = "open_rl:run_steps:"
# The capped list earlier gateways kept per run; read when a run has no step history yet.
LEGACY_PREFIX = "open_rl:run_metrics:"
# The runs that have reported, so a scrape reads them instead of scanning the keyspace.
INDEX = "open_rl:run_steps_index"
# A scrape exports runs that reported recently, and a bounded number of metrics each.
EXPORT_SECONDS = float(os.getenv("OPEN_RL_METRICS_EXPORT_HOURS", "6")) * 3600
EXPORT_METRICS_PER_RUN = int(os.getenv("OPEN_RL_METRICS_EXPORT_LIMIT", "64"))


class Row(BaseModel):
  step: int = Field(ge=0, le=2**53 - 1, strict=True)
  metrics: dict[str, Any]


class Request(BaseModel):
  rows: list[Row] = Field(max_length=1000)


async def ingest(store, run_id: str, body: Request) -> dict:
  if not RUN_ID.fullmatch(run_id):
    raise HTTPException(400, "run_id must be 1-128 letters, digits, '.', '_', ':' or '-'")
  at = time.time()
  rows = [{"step": row.step, "at": at, "metrics": values} for row in body.rows if (values := finite_metrics(row.metrics))]
  await store.put_steps(PREFIX + run_id, rows, ops.SAMPLE_LIMIT, ops.TTL_SECONDS)
  if rows:
    await store.add_to_set(INDEX, run_id)
  return {"accepted": len(rows)}


async def read(store, run_id: str, known: tuple[str, ...] = ()) -> tuple[str | None, list[dict] | None]:
  """The run's step history and its revision, or (revision, None) when the caller has it.
  A run sent before the step history existed is read from the earlier list, newest row per step."""
  revision, rows = await store.read_steps(PREFIX + run_id, known)
  if revision is not None or rows:
    return revision, rows
  latest = {}
  for record in await store.read_samples(LEGACY_PREFIX + run_id):
    if isinstance(record, dict) and isinstance(record.get("step"), int) and isinstance(record.get("metrics"), dict):
      latest[record["step"]] = {"step": record["step"], "at": record.get("at", 0.0), "metrics": finite_metrics(record["metrics"])}
  return None, [latest[step] for step in sorted(latest)]


def label(value: str) -> str:
  return '"' + value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '"'


async def exposition(store) -> str:
  """A scrape reads the latest step of each run that reported recently, never its history."""
  metrics, steps, received = [], [], []
  runs = sorted(await store.set_members(INDEX))
  latest = await asyncio.gather(*(store.get_value("latest:" + PREFIX + run_id) for run_id in runs))
  now = time.time()
  for run_id, raw in zip(runs, latest):
    if raw is None:
      # Its history expired; forget it.
      await store.remove_from_set(INDEX, run_id)
      continue
    row = json.loads(raw)
    if now - row["at"] > EXPORT_SECONDS:
      continue
    run = label(run_id)
    names = sorted(row["metrics"])[:EXPORT_METRICS_PER_RUN]
    metrics.extend(f"openrl_run_metric{{run_id={run},metric={label(name)}}} {row['metrics'][name]}" for name in names)
    steps.append(f"openrl_run_step{{run_id={run}}} {row['step']}")
    received.append(f"openrl_run_metrics_received_timestamp_seconds{{run_id={run}}} {row['at']}")
  return (
    "\n".join(
      [
        "# HELP openrl_run_metric Latest reported training metric for a run.",
        "# TYPE openrl_run_metric gauge",
        *metrics,
        "# HELP openrl_run_step Latest reported training step for a run.",
        "# TYPE openrl_run_step gauge",
        *steps,
        "# HELP openrl_run_metrics_received_timestamp_seconds Gateway receipt time of the latest step.",
        "# TYPE openrl_run_metrics_received_timestamp_seconds gauge",
        *received,
      ]
    )
    + "\n"
  )
