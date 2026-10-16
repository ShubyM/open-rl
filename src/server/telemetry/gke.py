"""Cloud Logging as the run-log source on GKE.

The gateway learns its project, cluster and location from the metadata
server, authenticates with Application Default Credentials (Workload Identity
in the cluster), and queries entries for the pods a run has owned. Paging is
stateless: the cursor is Cloud Logging's own page token, valid for the same
filters and window.
"""

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from datetime import UTC, datetime, timedelta

import google.auth
import httpx
from google.auth.transport.requests import Request

from server.telemetry import kubernetes

MAX_MESSAGE = 16 * 1024
MAX_SOURCES = 64
REFUSED = "Cloud Logging refused this gateway's credentials (missing logging read scope or IAM role)"
UNAVAILABLE = "Cloud Logging unavailable; check credentials, IAM and telemetry configuration"

_METADATA = {"project": "project/project-id", "cluster": "instance/attributes/cluster-name", "location": "instance/attributes/cluster-location"}
_detected: dict | None = None
_discovery_at = 0.0
_discovery_lock = asyncio.Lock()
_credentials = None
_LOCK = threading.Lock()
_refused_until = 0.0
# GKE files every stderr line as ERROR, though most are libraries' progress
# bars, INFO logs and warnings. Such a line takes the level it prints instead.
_RELABELED = ("DEBUG", "INFO", "WARNING")
_PROCESS = re.compile(r"^\([\w ]+ pid=\d+\)\s*")  # vLLM's "(EngineCore pid=826) "
_PROGRESS = re.compile(r"\r|\d+%(\||\s+Completed)")
_LEVEL = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b")
_WARNING = re.compile(r"Warning\b|warnings\.warn\(")


# ---- identity -------------------------------------------------------------------


def configuration() -> dict:
  mode = os.getenv("OPEN_RL_GKE_TELEMETRY", "auto").strip().lower()
  fields = {
    "project": os.getenv("OPEN_RL_GKE_PROJECT") or (_detected or {}).get("project", ""),
    "cluster": os.getenv("OPEN_RL_GKE_CLUSTER") or (_detected or {}).get("cluster", ""),
    "location": os.getenv("OPEN_RL_GKE_LOCATION") or (_detected or {}).get("location", ""),
    "mode": mode,
  }
  complete = all(fields[key] for key in ("project", "cluster", "location"))
  fields["enabled"] = mode in {"1", "true"} or (mode == "auto" and complete)
  fields["configured"] = fields["enabled"] and complete
  return fields


async def discover() -> dict:
  """Resolve GKE identity once; retry unavailable metadata after a minute."""
  global _detected, _discovery_at
  config = configuration()
  if config["mode"] not in {"auto", "1", "true"} or config["configured"] or not os.getenv("KUBERNETES_SERVICE_HOST"):
    return config
  async with _discovery_lock:
    if time.monotonic() - _discovery_at < 60:
      return configuration()
    async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as client:

      async def read(key, path):
        response = await client.get(f"http://metadata.google.internal/computeMetadata/v1/{path}", headers={"Metadata-Flavor": "Google"})
        response.raise_for_status()
        value = response.text.strip()
        if response.headers.get("Metadata-Flavor") != "Google" or not value or len(value) > 256:
          raise ValueError("Invalid metadata response")
        return key, value

      values = await asyncio.gather(*(read(key, path) for key, path in _METADATA.items()), return_exceptions=True)
    # Never enable from partial metadata: generic GCE also exposes project ID.
    if all(isinstance(value, tuple) for value in values):
      _detected = dict(values)
    _discovery_at = time.monotonic()
    return configuration()


def access_token() -> str:
  global _credentials
  with _LOCK:
    if _credentials is None:
      _credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
    if not _credentials.valid:
      _credentials.refresh(Request())
    return _credentials.token


async def request(method: str, url: str, **kwargs) -> dict:
  token = await asyncio.wait_for(asyncio.to_thread(access_token), timeout=10)
  async with httpx.AsyncClient(timeout=15) as client:
    response = await client.request(method, url, headers={"Authorization": f"Bearer {token}"}, **kwargs)
    response.raise_for_status()
    return response.json()


# ---- time and sources -----------------------------------------------------------


def timestamp(value) -> str:
  """ISO-8601 in UTC with microseconds, the form Cloud Logging filters compare."""
  if isinstance(value, int | float):
    return datetime.fromtimestamp(value, UTC).isoformat(timespec="microseconds")
  parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=UTC)
  return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def time_range(since=None, until=None) -> tuple[str, str]:
  end = datetime.fromisoformat(timestamp(until)) if until else datetime.now(UTC)
  start = datetime.fromisoformat(timestamp(since)) if since else end - timedelta(minutes=30)
  if start >= end or end - start > timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")


def pod_sources(run: dict | None, lanes: list[dict], observed_at: str) -> list[dict]:
  """The pods a run's logs can come from: live ones until now, and each worker
  process of the run for its lifetime. Names bound to lifetimes, since Cloud
  Logging labels carry pod names and not UIDs."""
  sources: dict[tuple, dict] = {}
  for lane in lanes:
    if lane.get("pod"):
      sources[lane["pod"], lane.get("pod_uid") or lane["start"]] = {
        "pod": lane["pod"],
        "node": lane.get("node"),
        "role": lane.get("role") or "unknown",
        "created_at": timestamp(lane["start"]),
        "until": timestamp(lane["end"] + 60),
      }
  for pod in (run or {}).get("pods", []):
    if pod.get("created_at"):
      sources[pod["name"], pod.get("uid") or pod["created_at"]] = {
        "pod": pod["name"],
        "node": pod.get("node"),
        "role": pod.get("role", "unknown"),
        "created_at": timestamp(pod["created_at"]),
        "until": timestamp(observed_at),
        "shared_runtime": pod.get("shared_runtime", False),
      }
  return sorted(sources.values(), key=lambda source: source["created_at"], reverse=True)


# ---- query -------------------------------------------------------------------------


def scope_filter() -> list[str]:
  config = configuration()
  labels = (
    ("project_id", config["project"]),
    ("location", config["location"]),
    ("cluster_name", config["cluster"]),
    ("namespace_name", kubernetes.namespace()),
  )
  return ['resource.type="k8s_container"', *[f"resource.labels.{label}={json.dumps(value)}" for label, value in labels]]


def source_filter(sources: list[dict]) -> str:
  def clause(s: dict) -> str:
    return f"(resource.labels.pod_name={json.dumps(s['pod'])} AND timestamp>={json.dumps(s['created_at'])} AND timestamp<={json.dumps(s['until'])})"

  return "(" + " OR ".join(clause(s) for s in sources) + ")"


def logs_filter(sources: list[dict], start: str, end: str, severity: str | None, q: str) -> str:
  clauses = scope_filter() + [source_filter(sources), f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity in _RELABELED:
    stderr = json.dumps(f"projects/{configuration()['project']}/logs/stderr")
    clauses.append(f'(severity={json.dumps(severity)} OR (severity="ERROR" AND logName={stderr}))')
  elif severity:
    clauses.append(f"severity={json.dumps('DEFAULT' if severity == 'UNKNOWN' else severity)}")
  if q:
    clauses.append(f"(textPayload:{json.dumps(q)} OR jsonPayload.message:{json.dumps(q)})")
  return " AND ".join(clauses)


def stderr_severity(text: str) -> str:
  body = _PROCESS.sub("", text)
  if not body.strip() or _PROGRESS.search(body):
    return "INFO"
  if level := _LEVEL.search(body[:80]):
    return "WARNING" if level[1] == "WARN" else level[1]
  return "WARNING" if _WARNING.search(body) else "ERROR"


def entry_record(entry: dict, sources: list[dict], run_id: str) -> dict | None:
  """One log entry as a dashboard record, or None when it belongs to no source lifetime."""
  labels = entry.get("resource", {}).get("labels", {})
  at = timestamp(entry["timestamp"])
  match = next((s for s in sources if s["pod"] == labels.get("pod_name") and s["created_at"] <= at <= s["until"]), None)
  if match is None:
    return None
  fields = entry.get("jsonPayload") or {}
  text = entry.get("textPayload")
  if text is None:
    text = json.dumps(fields or entry.get("protoPayload", {}), ensure_ascii=False)
  identity = hashlib.sha256(json.dumps([entry.get("logName"), entry.get("insertId"), entry.get("timestamp"), text]).encode()).hexdigest()
  severity = entry.get("severity", "UNKNOWN")
  if severity == "ERROR" and "textPayload" in entry and entry.get("logName", "").endswith("/logs/stderr"):
    severity = stderr_severity(text)
  return {
    "id": identity,
    "run_id": run_id,
    "timestamp": entry.get("timestamp"),
    "pod": match["pod"],
    "container": labels.get("container_name"),
    "node": match.get("node"),
    "role": match.get("role", "unknown"),
    "severity": severity,
    "rank": fields.get("rank"),
    "request_id": fields.get("request_id"),
    "message": text[:MAX_MESSAGE],
    "message_truncated": len(text) > MAX_MESSAGE,
  }


async def run_logs(
  run_id: str, sources: list[dict], *, since=None, until=None, cursor=None, limit=200, q="", pod=None, node=None, severity=None
) -> dict:
  global _refused_until
  result = {"run_id": run_id, "source": "gke", "records": [], "sources": sources, "order": "newest_first", "next_cursor": None, "available": False}
  if not configuration()["configured"]:
    return {**result, "error": "GKE telemetry is not configured"}
  if time.monotonic() < _refused_until:
    return {**result, "error": REFUSED}
  selected = [s for s in sources if (pod is None or s["pod"] == pod) and (node is None or s.get("node") == node)]
  if not selected:
    return {**result, "available": True, "note": "No pod lifetimes match this run and filter"}
  if len(selected) > MAX_SOURCES:
    raise ValueError(f"Select a pod to narrow this query to at most {MAX_SOURCES} pod lifetimes")
  start, end = time_range(since, until)
  body = {
    "resourceNames": [f"projects/{configuration()['project']}"],
    "filter": logs_filter(selected, start, end, severity, q),
    "orderBy": "timestamp desc",
    "pageSize": limit,
  }
  if cursor:
    body["pageToken"] = cursor
  try:
    payload = await request("POST", "https://logging.googleapis.com/v2/entries:list", json=body)
    records = [record for entry in payload.get("entries", []) if (record := entry_record(entry, selected, run_id))]
    if severity in (*_RELABELED, "ERROR"):
      records = [record for record in records if record["severity"] == severity]
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code in (401, 403):
      # A scope or IAM refusal does not clear itself; stop asking for a while.
      _refused_until = time.monotonic() + 600
      return {**result, "error": REFUSED}
    if exc.response.status_code == 400 and cursor:
      raise ValueError("Invalid or expired log cursor; repeat the query without it") from None
    return {**result, "error": UNAVAILABLE}
  except Exception:
    return {**result, "error": UNAVAILABLE}
  return {**result, "records": records, "available": True, "next_cursor": payload.get("nextPageToken")}
