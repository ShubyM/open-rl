"""Client step history and its latest values, shared by inspection and Prometheus."""

import json
import math
import re
import time
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, Field

from server.telemetry import ops

PREFIX = "open_rl:run_steps:"
RUN_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
METRICS_PER_ROW = 500


class Row(BaseModel):
  step: int = Field(ge=0, le=2**53 - 1, strict=True)
  metrics: dict[str, Any]


class Request(BaseModel):
  rows: list[Row] = Field(max_length=1000)


def finite_metrics(values: dict) -> dict[str, float]:
  """Normalize once at each untrusted boundary, before JSON encoding or storage."""
  result = {}
  for name, value in values.items():
    if len(name) > 200 or not isinstance(value, int | float) or isinstance(value, bool):
      continue
    try:
      value = float(value)
    except OverflowError:
      continue
    if math.isfinite(value):
      result[name] = value
      if len(result) == METRICS_PER_ROW:
        break
  return result


async def ingest(store, run_id: str, body: Request) -> dict:
  if not RUN_ID.fullmatch(run_id):
    raise HTTPException(400, "run_id must be 1-128 letters, digits, '.', '_', ':' or '-'")
  at = time.time()
  rows = [{"step": row.step, "at": at, "metrics": values} for row in body.rows if (values := finite_metrics(row.metrics))]
  await store.put_steps(PREFIX + run_id, rows, ops.SAMPLE_LIMIT, ops.TTL_SECONDS)
  return {"accepted": len(rows)}


def label(value: str) -> str:
  return '"' + value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"') + '"'


async def exposition(store) -> str:
  """A scrape reads one bounded latest-step snapshot per retained run, never its history."""
  metrics, steps, received = [], [], []
  prefix = "latest:" + PREFIX
  for key, raw in sorted((await store.scan_values(prefix)).items()):
    run = label(key[len(prefix) :])
    row = json.loads(raw)
    metrics.extend(f"openrl_run_metric{{run_id={run},metric={label(name)}}} {value}" for name, value in sorted(row["metrics"].items()))
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
