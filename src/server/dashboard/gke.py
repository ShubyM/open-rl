"""Optional, read-only GKE telemetry using ADC and automatically detected cluster scope."""

import asyncio
import hashlib
import json
import math
import os
import secrets
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime, timedelta

import google.auth
import httpx
from google.auth.transport.requests import Request

from server.dashboard import kubernetes, logs

_LOCK = threading.Lock()
_credentials = None
_pages = OrderedDict()


_detected = {}
_discovery_at = float("-inf")
_discovery_lock = asyncio.Lock()
_METADATA = {
  "project": "project/project-id",
  "cluster": "instance/attributes/cluster-name",
  "location": "instance/attributes/cluster-location",
}


def configuration() -> dict:
  mode = os.getenv("OPEN_RL_GKE_TELEMETRY", "auto").lower()
  fields = {key: os.getenv(f"OPEN_RL_GKE_{key.upper()}", "") or _detected.get(key, "") for key in _METADATA}
  complete = all(fields.values())
  fields["mode"] = mode
  fields["enabled"] = mode in {"1", "true"} or (mode == "auto" and complete)
  fields["configured"] = fields["enabled"] and complete
  fields["discovery"] = "metadata" if _detected else "explicit" if complete else "unavailable"
  return fields


async def discover() -> dict:
  """Resolve GKE identity once; retry unavailable metadata after a minute."""
  global _detected, _discovery_at
  config = configuration()
  if config["mode"] not in {"auto", "1", "true"} or config["configured"]:
    return config
  if not os.getenv("KUBERNETES_SERVICE_HOST"):
    return config
  async with _discovery_lock:
    if time.monotonic() - _discovery_at < 60:
      return configuration()
    async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as client:

      async def read(key, path):
        response = await client.get(
          f"http://metadata.google.internal/computeMetadata/v1/{path}",
          headers={"Metadata-Flavor": "Google"},
        )
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


def time_range(since=None, until=None) -> tuple[str, str]:
  end = datetime.fromisoformat(logs.timestamp(until)) if until else datetime.now(UTC)
  start = datetime.fromisoformat(logs.timestamp(since)) if since else end - timedelta(minutes=30)
  if start >= end or end - start > timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(), end.isoformat()


def scope_filter() -> list[str]:
  config = configuration()
  return [
    'resource.type="k8s_container"',
    *[
      f"resource.labels.{label}={json.dumps(value)}"
      for label, value in (
        ("project_id", config["project"]),
        ("location", config["location"]),
        ("cluster_name", config["cluster"]),
        ("namespace_name", kubernetes.k8s_namespace()),
      )
    ],
  ]


def matches_resource(labels: dict) -> bool:
  config = configuration()
  return all(
    labels.get(key) == value
    for key, value in (
      ("project_id", config["project"]),
      ("location", config["location"]),
      ("cluster_name", config["cluster"]),
      ("namespace_name", kubernetes.k8s_namespace()),
    )
  )


def pod_sources(run: dict | None, archived: list[dict], observed_at: str) -> list[dict]:
  sources = {}
  for source in archived:
    if source.get("created_at") and source.get("collected_at"):
      sources[(source["pod"], source.get("pod_uid"), source["container"])] = {**source, "until": source["collected_at"]}
  for pod in (run or {}).get("pods", []):
    for container in pod.get("containers", []):
      if pod.get("created_at"):
        sources[(pod["name"], pod.get("uid"), container["name"])] = {
          "pod": pod["name"],
          "pod_uid": pod.get("uid"),
          "container": container["name"],
          "node": pod.get("node"),
          "role": pod.get("role", "unknown"),
          "created_at": pod["created_at"],
          "until": observed_at,
          "shared_runtime": pod.get("shared_runtime", False),
        }
  return list(sources.values())


def source_filter(sources: list[dict]) -> str:
  # Standard GKE resource labels identify names, not pod UIDs. Bound names to observed lifetimes.
  return (
    "("
    + " OR ".join(
      "("
      + " AND ".join(
        [
          f"resource.labels.pod_name={json.dumps(s['pod'])}",
          f"resource.labels.container_name={json.dumps(s['container'])}",
          f"timestamp>={json.dumps(logs.timestamp(s['created_at']))}",
          f"timestamp<={json.dumps(logs.timestamp(s['until']))}",
        ]
      )
      + ")"
      for s in sources
    )
    + ")"
  )


async def run_logs(
  run_id: str,
  sources: list[dict],
  *,
  since=None,
  until=None,
  cursor=None,
  limit=200,
  q="",
  pod=None,
  container=None,
  node=None,
  severity=None,
  attempt=None,
) -> dict:
  result = {
    "schema_version": 1,
    "run_id": run_id,
    "source": "gke",
    "records": [],
    "sources": sources,
    "order": "newest_first",
    "next_cursor": None,
    "available": False,
    "coverage": {"history_complete": False, "correlation": "pod_name_container_observed_lifetime", "pod_uid_verified": False},
  }
  if not configuration()["configured"]:
    return {**result, "error": "GKE telemetry is not fully configured"}
  if attempt is not None:
    return {**result, "error": "GKE logs do not reliably identify restart attempts; use source=local"}
  selected = [
    s for s in sources if all(value is None or s.get(key) == value for key, value in (("pod", pod), ("container", container), ("node", node)))
  ]
  if not selected:
    return {**result, "error": "No observed pod lifetimes match this run and filter"}
  if len(selected) > 64:
    return {**result, "error": "Select a pod to narrow this query to at most 64 container lifetimes"}
  fingerprint = hashlib.sha256(
    json.dumps([configuration(), kubernetes.k8s_namespace(), run_id, since, until, q, pod, container, node, severity, limit]).encode()
  ).hexdigest()
  page_token = None
  if cursor:
    try:
      token = _pages[cursor]
      if token["fingerprint"] != fingerprint or time.monotonic() > token["expires"]:
        raise ValueError()
      start, end, page_token = token["start"], token["end"], token["page"]
      selected = token["sources"]
    except (ValueError, TypeError, KeyError):
      raise ValueError("Invalid or expired GKE cursor, or changed query scope") from None
  else:
    start, end = time_range(since, until)
  clauses = scope_filter() + [source_filter(selected), f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity:
    clauses.append(f"severity={json.dumps('DEFAULT' if severity == 'UNKNOWN' else severity)}")
  if q:
    clauses.append(f"(textPayload:{json.dumps(q)} OR jsonPayload.message:{json.dumps(q)})")
  body = {
    "resourceNames": [f"projects/{configuration()['project']}"],
    "filter": " AND ".join(clauses),
    "orderBy": "timestamp desc",
    "pageSize": limit,
  }
  if page_token:
    body["pageToken"] = page_token
  try:
    payload = await request("POST", "https://logging.googleapis.com/v2/entries:list", json=body)
    records = []
    for entry in payload.get("entries", []):
      resource = entry.get("resource", {}).get("labels", {})
      if not matches_resource(resource):
        continue
      fields = entry.get("jsonPayload") or {}
      text = entry.get("textPayload")
      if text is None:
        text = json.dumps(fields, ensure_ascii=False) if fields else json.dumps(entry.get("protoPayload", {}), ensure_ascii=False)
      match = next(
        (
          s
          for s in selected
          if s["pod"] == resource.get("pod_name")
          and s["container"] == resource.get("container_name")
          and logs.timestamp(s["created_at"]) <= logs.timestamp(entry["timestamp"]) <= logs.timestamp(s["until"])
        ),
        None,
      )
      if match is None:
        continue
      identity = hashlib.sha256(json.dumps([entry.get("logName"), entry.get("insertId"), entry.get("timestamp"), text]).encode()).hexdigest()
      records.append(
        {
          "id": identity,
          "run_id": run_id,
          "timestamp": entry.get("timestamp"),
          "pod": match["pod"],
          "pod_uid": None,
          "observed_pod_uid": match.get("pod_uid"),
          "container": match["container"],
          "node": match.get("node"),
          "role": match.get("role", "unknown"),
          "attempt": None,
          "severity": entry.get("severity", "UNKNOWN"),
          "rank": fields.get("rank"),
          "request_id": fields.get("request_id"),
          "message": text[: logs.MAX_MESSAGE],
          "message_truncated": len(text) > logs.MAX_MESSAGE,
        }
      )
    result.update(records=records, available=True)
    if payload.get("nextPageToken"):
      cursor_id = "gke." + secrets.token_urlsafe(24)
      _pages[cursor_id] = {
        "fingerprint": fingerprint,
        "start": start,
        "end": end,
        "page": payload["nextPageToken"],
        "sources": selected,
        "expires": time.monotonic() + 900,
      }
      while len(_pages) > 256:
        _pages.popitem(last=False)
      result["next_cursor"] = cursor_id

  except Exception:
    result["error"] = "Cloud Logging unavailable; check credentials, IAM and telemetry configuration"
  return result


METRICS = {
  "gpu_utilization": ("container/accelerator/duty_cycle", "%"),
  "gpu_memory": ("container/accelerator/memory_used", "bytes"),
  "cpu_usage": ("container/cpu/core_usage_time", "CPU seconds"),
  "memory_usage": ("container/memory/used_bytes", "bytes"),
}


async def resource_metrics(sources: list[dict], since=None, until=None) -> dict:
  start, end = time_range(since, until)
  result = {
    "source": "gke",
    "available": False,
    "series": [],
    "since": start,
    "until": end,
    "coverage": {"history_complete": False, "correlation": "pod_name_container_observed_lifetime", "pod_uid_verified": False},
  }
  if not configuration()["configured"] or not sources:
    return {**result, "error": "GKE configuration or observed pod lifetimes unavailable"}
  if len(sources) > 64:
    return {**result, "error": "Too many container lifetimes for one metrics query"}
  names = sorted({s["pod"] for s in sources})

  async def metric(name, definition):
    suffix, unit = definition
    clauses = scope_filter() + [
      f'metric.type="kubernetes.io/{suffix}"',
      "(" + " OR ".join(f"resource.labels.pod_name={json.dumps(pod)}" for pod in names) + ")",
    ]
    params = {"filter": " AND ".join(clauses), "interval.startTime": start, "interval.endTime": end, "view": "FULL", "pageSize": 1000}
    payload = await request("GET", f"https://monitoring.googleapis.com/v3/projects/{configuration()['project']}/timeSeries", params=params)
    series = []
    for item in payload.get("timeSeries", []):
      labels = item.get("resource", {}).get("labels", {})
      if not matches_resource(labels):
        continue
      matches = [s for s in sources if s["pod"] == labels.get("pod_name") and s["container"] == labels.get("container_name")]
      points = []
      for point in item.get("points", []):
        at = point["interval"]["endTime"]
        if not any(logs.timestamp(s["created_at"]) <= logs.timestamp(at) <= logs.timestamp(s["until"]) for s in matches):
          continue
        value = point.get("value", {})
        number = float(value.get("doubleValue", value.get("int64Value", "nan")))
        if math.isfinite(number):
          points.append([datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp(), number])
      if points:
        series.append(
          {
            "name": name,
            "unit": unit,
            "pod": labels.get("pod_name"),
            "container": labels.get("container_name"),
            "role": matches[0].get("role", "unknown"),
            "device": item.get("metric", {}).get("labels", {}).get("accelerator_id"),
            "points": sorted(points),
            "metric_kind": item.get("metricKind"),
          }
        )
    return series, bool(payload.get("nextPageToken"))

  values = await asyncio.gather(*(metric(name, definition) for name, definition in METRICS.items()), return_exceptions=True)
  errors = []
  for (name, _), value in zip(METRICS.items(), values, strict=True):
    if isinstance(value, Exception):
      errors.append(name)
    else:
      result["available"] = True
      result["series"].extend(value[0])
      if value[1]:
        errors.append(f"{name}: truncated")
  if errors:
    result["error"] = "Unavailable or incomplete metrics: " + ", ".join(errors)
  return result
