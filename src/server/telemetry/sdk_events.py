"""What the tinker SDK reports about itself through POST /api/v1/telemetry.

The gateway prints each event as one JSON line, without credentials. Errors also stay in a
short tail per session, so a run's page can show the ones from its own sessions as incidents.
"""

import json
import re
import time
from datetime import datetime

from server.telemetry import ops
from server.telemetry.gke import CLIENT

PREFIX = "open_rl:sdk_errors:"
TAIL = 200
ERRORS = {"ERROR", "CRITICAL"}
# The SDK attaches full request and response headers to fatal errors, and the request carries the API key.
DROPPED = {"request_headers", "response_headers"}
SECRET = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|secret|password", re.I)


def redact(value):
  if isinstance(value, dict):
    return {k: "[redacted]" if SECRET.search(str(k)) else redact(v) for k, v in value.items() if k not in DROPPED}
  if isinstance(value, list):
    return [redact(v) for v in value]
  return value


def describe(event: dict) -> tuple[str, str]:
  """A short reason and a message for an SDK error event."""
  if event.get("event") == "UNHANDLED_EXCEPTION":
    return f"SDK {event.get('error_type') or 'exception'}", str(event.get("error_message") or "")
  data = event.get("event_data") if isinstance(event.get("event_data"), dict) else {}
  name = str(event.get("event_name") or event.get("event") or "event").rsplit(".", 1)[-1]
  status = f"HTTP {data['status_code']}: " if data.get("status_code") else ""
  return f"SDK {name}", status + str(data.get("detail") or data.get("exception") or data.get("message") or "")


def at_of(event: dict) -> float:
  try:
    return datetime.fromisoformat(str(event.get("timestamp"))).timestamp()
  except ValueError:
    return time.time()


async def ingest(store, body: dict) -> None:
  context = {key: body.get(key) for key in ("session_id", "process_uuid", "sdk_version", "platform")}
  session_id = body.get("session_id")
  events = body.get("events")
  for event in events if isinstance(events, list) else []:
    if not isinstance(event, dict):
      continue
    event = redact(event)
    severity = str(event.get("severity") or "INFO").upper()
    # One JSON line per event: on GKE, Cloud Logging indexes it with the event's severity.
    message = f"tinker {event.get('event_name') or event.get('event')}"
    print(json.dumps({"severity": severity, "message": message, **context, "event": event}, default=str), flush=True)
    if severity in ERRORS and isinstance(session_id, str) and session_id:
      reason, text = describe(event)
      sample = {"at": at_of(event), "severity": severity, "reason": reason, "message": text[:2000]}
      await store.append_sample(PREFIX + session_id, sample, TAIL, ops.TTL_SECONDS)


async def incidents(store, session_ids: list[str], start: float, end: float) -> list[dict]:
  """The SDK errors of these sessions within [start, end], newest of each reason first."""
  found = []
  for session_id in session_ids:
    for sample in await store.read_samples(PREFIX + session_id):
      if start <= sample.get("at", 0) <= end:
        found.append({"at": sample["at"], "pod": CLIENT, "reason": sample["reason"], "tone": "error", "message": sample.get("message")})
  found.sort(key=lambda incident: incident["at"], reverse=True)
  newest = {}
  for incident in found:
    newest.setdefault(incident["reason"], {**incident, "count": 0})["count"] += 1
  return list(newest.values())
