"""A training client's own output, sent through the gateway for its run.

A run is its model: the run id is the model id the client trains, the host of its tinker://
paths. The gateway writes each line to its own output as structured JSON labelled with the
run, so on GKE Cloud Logging keeps it beside the run's worker logs and the dashboard shows it
as the run's client source. A short tail also stays in the run store, for a deployment
without Cloud Logging.
"""

import hashlib
import json
import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from server.telemetry import ops
from server.telemetry.gke import CLIENT, RUN_LABEL, SOURCE_LABEL
from server.telemetry.run_metrics import known_run

PREFIX = "open_rl:run_logs:"
TAIL = 5000
MAX_MESSAGE = 16384
SEVERITIES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
# A client clock is trusted within this much of the gateway's.
CLOCK_SKEW_SECONDS = 86400


class Line(BaseModel):
  message: str = Field(max_length=MAX_MESSAGE)
  severity: str = "INFO"
  at: float | None = None


class Request(BaseModel):
  lines: list[Line] = Field(max_length=1000)


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


async def ingest(store, run_id: str, body: Request) -> dict:
  await known_run(store, run_id)
  now = time.time()
  for line in body.lines:
    severity = line.severity.upper() if line.severity.upper() in SEVERITIES else "INFO"
    at = line.at if line.at is not None and abs(line.at - now) < CLOCK_SKEW_SECONDS else now
    labels = {RUN_LABEL: run_id, SOURCE_LABEL: CLIENT}
    print(json.dumps({"severity": severity, "message": line.message, "time": iso(at), "logging.googleapis.com/labels": labels}), flush=True)
    await store.append_sample(PREFIX + run_id, {"at": at, "severity": severity, "message": line.message}, TAIL, ops.TTL_SECONDS)
  return {"accepted": len(body.lines)}


async def tail(store, run_id: str, limit: int) -> list[dict]:
  """The newest client lines kept in the run store, newest first, as dashboard log records."""
  records = []
  for line in reversed(await store.read_samples(PREFIX + run_id, last=limit)):
    identity = hashlib.sha256(json.dumps([run_id, line.get("at"), line.get("message")]).encode()).hexdigest()
    records.append(
      {
        "id": identity,
        "run_id": run_id,
        "timestamp": iso(line.get("at", 0.0)),
        "pod": CLIENT,
        "role": CLIENT,
        "severity": line.get("severity", "INFO"),
        "message": line.get("message", ""),
      }
    )
  return records
