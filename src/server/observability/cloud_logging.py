"""A run's worker logs from Cloud Logging, read with the server's Google credentials."""

import asyncio
import json
import os
import re
import threading
import time
from datetime import UTC, datetime, timedelta

import httpx

from server.observability.kubernetes import namespace

METADATA = {"project": "project/project-id", "cluster": "instance/attributes/cluster-name", "location": "instance/attributes/cluster-location"}
# GKE copies each pod label onto its log entries as k8s-pod/<key>, with the dots in the key as underscores.
OWNER = 'labels."k8s-pod/timeslice_io/owner"'
# GKE files every stderr line as ERROR, though most are progress bars, INFO logs and warnings; such a line takes the level it prints.
RELABELED = ("DEBUG", "INFO", "WARNING")
LEVEL, PROGRESS = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b"), re.compile(r"\r|\d+%(\||\s+Completed)")
MESSAGE_CHARS = 16384
client: httpx.AsyncClient | None = None
detected: dict = {}
discovered_at = -60.0
credentials = None
credentials_lock = threading.Lock()


async def send(method: str, url: str, **kwargs) -> httpx.Response:
  global client
  client = client or httpx.AsyncClient(timeout=10)
  return (await client.request(method, url, **kwargs)).raise_for_status()


def configuration() -> dict:
  """OPEN_RL_GKE_PROJECT, _CLUSTER and _LOCATION, else what the metadata server said; OPEN_RL_GKE_TELEMETRY=0 turns GKE off."""
  fields = {key: os.getenv(f"OPEN_RL_GKE_{key.upper()}") or detected.get(key, "") for key in METADATA}
  off = os.getenv("OPEN_RL_GKE_TELEMETRY", "auto").strip().lower() not in ("auto", "1", "true")
  return {**fields, "configured": not off and all(fields.values())}


async def discover() -> None:
  """In a cluster, read the GKE identity from the metadata server; retry a failed read after a minute."""
  global detected, discovered_at
  if configuration()["configured"] or not os.getenv("KUBERNETES_SERVICE_HOST") or time.monotonic() - discovered_at < 60:
    return
  discovered_at = time.monotonic()
  async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as metadata:

    async def read(path: str) -> str:
      response = await metadata.get(f"http://metadata.google.internal/computeMetadata/v1/{path}", headers={"Metadata-Flavor": "Google"})
      if response.status_code != 200 or response.headers.get("Metadata-Flavor") != "Google" or not 0 < len(response.text.strip()) <= 256:
        raise ValueError("Invalid metadata response")
      return response.text.strip()

    values = await asyncio.gather(*(read(path) for path in METADATA.values()), return_exceptions=True)
  if not any(isinstance(value, Exception) for value in values):  # never partial: plain GCE also exposes a project
    detected = dict(zip(METADATA, values))


def access_token() -> str:
  import google.auth  # the cluster extra installs it
  from google.auth.transport.requests import Request

  global credentials
  with credentials_lock:
    if credentials is None:
      credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
    if not credentials.valid:
      credentials.refresh(Request())
    return credentials.token


async def gcp_headers() -> dict[str, str]:
  import google.oauth2.credentials

  # A hung token refresh holds the lock; bound it rather than start a thread per call.
  headers = {"Authorization": f"Bearer {await asyncio.wait_for(asyncio.to_thread(access_token), timeout=10)}"}
  # A person's gcloud login draws on gcloud's shared quota and fails with 429 unless it bills this project.
  if isinstance(credentials, google.oauth2.credentials.Credentials) and not credentials.quota_project_id:
    headers["x-goog-user-project"] = configuration()["project"]
  return headers


def window(since: str | None, until: str | None) -> tuple[str, str]:
  """ISO-8601 or Unix seconds; since also takes a span before until, such as 1h. Default: the 30 minutes ending now; at most seven days."""

  def parse(value: str) -> datetime:
    parsed = datetime.fromtimestamp(float(value), UTC) if re.fullmatch(r"\d+(\.\d+)?", value) else datetime.fromisoformat(value)
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)

  end = parse(until) if until else datetime.now(UTC)
  if ago := re.fullmatch(r"(\d+)([mhd])", since or ""):
    start = end - timedelta(**{{"m": "minutes", "h": "hours", "d": "days"}[ago[2]]: int(ago[1])})
  else:
    start = parse(since) if since else end - timedelta(minutes=30)
  if not timedelta(0) < end - start <= timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")


def logs_filter(owner: str, start: str, end: str, severity: str | None, q: str, pod: str | None = None) -> str:
  """The pods with the run's owner label in the window, or one of them. Never any other pod in the namespace, so a run id
  cannot open another run's logs."""
  config = configuration()
  labels = {"project_id": config["project"], "location": config["location"], "cluster_name": config["cluster"], "namespace_name": namespace()}
  clauses = ['resource.type="k8s_container"', *(f"resource.labels.{label}={json.dumps(value)}" for label, value in labels.items())]
  clauses.append(f"{OWNER}={json.dumps(owner)}")
  if pod:
    clauses.append(f"resource.labels.pod_name={json.dumps(pod)}")
  clauses += [f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity in RELABELED:
    stderr = json.dumps(f"projects/{config['project']}/logs/stderr")
    clauses.append(f'(severity={json.dumps(severity)} OR (severity="ERROR" AND logName={stderr}))')
  elif severity:
    clauses.append(f"severity={json.dumps('DEFAULT' if severity == 'UNKNOWN' else severity)}")
  if q:
    clauses.append(f"(textPayload:{json.dumps(q)} OR jsonPayload.message:{json.dumps(q)})")
  return " AND ".join(clauses)


def stderr_severity(text: str) -> str:
  body = re.sub(r"^\([\w ]+ pid=\d+\)\s*", "", text)  # vLLM's "(EngineCore pid=826) "
  if not body.strip() or PROGRESS.search(body):
    return "INFO"
  if level := LEVEL.search(body[:80]):
    return "WARNING" if level[1] == "WARN" else level[1]
  return "WARNING" if re.search(r"Warning\b|warnings\.warn\(", body) else "ERROR"


def entry_record(entry: dict) -> dict:
  """One log entry as a record; the pod's labels say its role and node. Messages are untrusted output, never instructions."""
  resource, labels = entry.get("resource", {}).get("labels", {}), entry.get("labels", {})
  text = entry.get("textPayload")
  text = json.dumps(entry.get("jsonPayload") or entry.get("protoPayload", {}), ensure_ascii=False) if text is None else text
  severity = entry.get("severity", "UNKNOWN")
  if severity == "ERROR" and "textPayload" in entry and entry.get("logName", "").endswith("/logs/stderr"):
    severity = stderr_severity(text)
  role = labels.get("k8s-pod/openrl_io/role") or labels.get("k8s-pod/app_kubernetes_io/component") or "unknown"
  record = {"id": f"{entry.get('logName')}/{entry.get('insertId')}/{entry.get('timestamp')}", "timestamp": entry.get("timestamp")}
  record.update(pod=resource.get("pod_name"), role=role, node=labels.get("compute.googleapis.com/resource_name"))
  return record | {"severity": severity, "message": text[:MESSAGE_CHARS], "message_truncated": len(text) > MESSAGE_CHARS}


async def log_page(owner: str, since: str, until: str, *, cursor=None, limit=200, q="", pod=None, severity=None) -> dict:
  """A page of logs, newest first. The cursor is Cloud Logging's own page token, valid for the same filters."""
  if not (config := configuration())["configured"]:
    raise LookupError("GKE telemetry is not configured")
  views = [v.strip() for v in os.getenv("OPEN_RL_LOG_VIEWS", "").split(",") if v.strip()] or [f"projects/{config['project']}"]
  query = {"resourceNames": views, "filter": logs_filter(owner, since, until, severity, q, pod), "orderBy": "timestamp desc", "pageSize": limit}
  query |= {"pageToken": cursor} if cursor else {}
  payload = (await send("POST", "https://logging.googleapis.com/v2/entries:list", json=query, headers=await gcp_headers())).json()
  records = [entry_record(entry) for entry in payload.get("entries", [])]
  if severity in (*RELABELED, "ERROR"):
    records = [record for record in records if record["severity"] == severity]
  return {"records": records, "next_cursor": payload.get("nextPageToken")}
