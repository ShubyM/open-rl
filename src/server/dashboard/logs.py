"""Run-scoped container logs, bounded collection, and a local retained archive.

The archive is a best-effort polling collector, not a lossless cluster logging pipeline.
The dashboard and agents share one trusted operator identity.
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from server.dashboard import data

MAX_SOURCES = 64
TAIL_LINES = 500
MAX_BYTES = 128 * 1024
MAX_MESSAGE = 4096
_LOCK = threading.Lock()


def timestamp(value: str) -> str:
  try:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
      raise ValueError()
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")
  except (ValueError, TypeError, AttributeError, OverflowError):
    raise ValueError("Times must be ISO 8601 timestamps with a timezone") from None


def archive_path() -> str:
  path = os.getenv("OPEN_RL_LOG_ARCHIVE", str(Path(data.tmp_dir()) / "logs.sqlite3"))
  return path + ".demo" if data.demo_mode_enabled() else path


def connect() -> sqlite3.Connection:
  path = Path(archive_path())
  path.parent.mkdir(parents=True, exist_ok=True)
  # Restrict the database before sqlite opens it, including on first creation.
  fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
  os.close(fd)
  os.chmod(path, 0o600)
  db = sqlite3.connect(path, timeout=10)
  db.row_factory = sqlite3.Row
  db.execute("PRAGMA journal_mode=DELETE")
  db.executescript("""
    CREATE TABLE IF NOT EXISTS records (
      id INTEGER PRIMARY KEY AUTOINCREMENT, identity TEXT UNIQUE, run_id TEXT NOT NULL,
      timestamp TEXT, pod TEXT, pod_uid TEXT, container TEXT, node TEXT,
      role TEXT, attempt INTEGER, severity TEXT, rank TEXT, request_id TEXT,
      message TEXT, message_truncated INTEGER, collected_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS log_scope ON records(run_id, id);
    CREATE INDEX IF NOT EXISTS log_time ON records(run_id, timestamp, id);
    CREATE TABLE IF NOT EXISTS sources (
      run_id TEXT, source TEXT, payload TEXT, PRIMARY KEY(run_id, source)
    );
    CREATE TABLE IF NOT EXISTS archive_meta (key TEXT PRIMARY KEY, value TEXT);
  """)
  return db


def parse_records(text: str, run_id: str, pod: dict, container: dict, previous: bool) -> list[dict]:
  records = []
  repeats = {}
  for line in text.splitlines():
    first, _, message = line.partition(" ")
    try:
      ts = timestamp(first)
    except ValueError:
      ts, message = None, line
    fields = {}
    try:
      candidate = json.loads(message)
      if isinstance(candidate, dict):
        fields = candidate
    except (ValueError, TypeError):
      pass
    severity = str(fields.get("level", fields.get("severity", ""))).upper()
    if not severity:
      match = re.search(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b", message)
      severity = match.group(1) if match else "UNKNOWN"
    severity = {"WARN": "WARNING", "FATAL": "CRITICAL"}.get(severity, severity)
    if severity not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
      severity = "UNKNOWN"
    attempt = max(0, int(container.get("restart_count") or 0) - int(previous))
    identity = json.dumps([run_id, pod.get("uid") or pod["name"], container["name"], attempt, first, message])
    occurrence = repeats.get(identity, 0)
    repeats[identity] = occurrence + 1
    records.append(
      {
        "identity": hashlib.sha256(f"{identity}:{occurrence}".encode()).hexdigest(),
        "run_id": run_id,
        "timestamp": ts,
        "pod": pod["name"],
        "pod_uid": pod.get("uid"),
        "container": container["name"],
        "node": pod.get("node"),
        "role": "sampler" if (pod.get("labels") or {}).get("timeslice.io/job-id", "").startswith("sampler-") else "trainer",
        "attempt": attempt,
        "severity": severity,
        "rank": str(fields["rank"])[:64] if "rank" in fields else None,
        "request_id": str(fields["request_id"])[:256] if "request_id" in fields else None,
        "message": message[:MAX_MESSAGE],
        "message_truncated": int(len(message) > MAX_MESSAGE),
        "collected_at": datetime.now(UTC).isoformat(timespec="microseconds"),
      }
    )
  return records


def retain(run_id: str, records: list[dict], sources: list[dict]) -> None:
  with _LOCK, closing(connect()) as db, db:
    for record in records:
      keys = list(record)
      db.execute(f"INSERT OR IGNORE INTO records ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", list(record.values()))
    for source in sources:
      key = json.dumps([source["pod"], source.get("pod_uid"), source["container"], source["attempt"]])
      db.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?)", (run_id, key, json.dumps(source)))
    cutoff = (datetime.now(UTC) - timedelta(days=7)).isoformat(timespec="microseconds")
    removed = db.execute("DELETE FROM records WHERE collected_at < ?", (cutoff,)).rowcount
    removed += db.execute("DELETE FROM records WHERE id IN (SELECT id FROM records ORDER BY id DESC LIMIT -1 OFFSET 20000)").rowcount
    db.execute("DELETE FROM sources WHERE json_extract(payload, '$.collected_at') < ?", (cutoff,))
    db.execute("DELETE FROM sources WHERE rowid IN (SELECT rowid FROM sources ORDER BY rowid DESC LIMIT -1 OFFSET 4096)")
    if removed:
      db.execute("INSERT OR REPLACE INTO archive_meta VALUES ('last_pruned_at', ?)", (datetime.now(UTC).isoformat(),))


def collect(run_id: str, pods: list[dict], demo_mode: bool = False, source_offset: int = 0) -> dict:
  work = [
    (pod, c, previous) for pod in pods for c in pod.get("containers", []) for previous in ([False, True] if c.get("restart_count", 0) else [False])
  ]
  if work:
    offset = source_offset % len(work)
    work = work[offset:] + work[:offset]
  sources, records = [], []
  api, error = (None, None) if demo_mode else data.k8s_core_v1()

  def read_source(item):
    pod, container, previous = item
    parsed = []
    source = {
      "pod": pod["name"],
      "pod_uid": pod.get("uid"),
      "container": container["name"],
      "node": pod.get("node"),
      "attempt": max(0, container.get("restart_count", 0) - int(previous)),
      "previous": previous,
      "status": "ok",
      "tail_limited": False,
      "collected_at": datetime.now(UTC).isoformat(timespec="microseconds"),
    }
    try:
      if demo_mode:
        now = datetime.now(UTC)
        text = "\n".join(
          f"{(now - timedelta(seconds=60 - i * 5)).isoformat()} "
          + json.dumps(
            {
              "level": "ERROR" if previous else "INFO",
              "rank": i % 2,
              "message": "CUDA out of memory" if previous else f"training step={i} loss={0.9 - i * 0.02:.3f}",
            }
          )
          for i in range(12)
        )
      else:
        if api is None:
          raise RuntimeError(error)
        text = api.read_namespaced_pod_log(
          pod["name"],
          data.k8s_namespace(),
          container=container["name"],
          previous=previous,
          timestamps=True,
          tail_lines=TAIL_LINES,
          limit_bytes=MAX_BYTES,
          _request_timeout=data.K8S_REQUEST_TIMEOUT + 4,
        )
      parsed = parse_records(text, run_id, pod, container, previous)
      source["tail_limited"] = len(parsed) >= TAIL_LINES or len(text.encode()) >= MAX_BYTES
      source["records"] = len(parsed)
    except Exception as exc:
      # Kubernetes exception bodies may contain response details; expose only the status.
      source.update(status="unavailable", error_code=str(getattr(exc, "status", None) or "collection_failed"))
    return source, parsed

  with ThreadPoolExecutor(max_workers=8) as executor:
    for source, parsed in executor.map(read_source, work[:MAX_SOURCES]):
      sources.append(source)
      records.extend(parsed)
  retain(run_id, records, sources)
  return {"sources_omitted": max(0, len(work) - MAX_SOURCES), "sources": sources}


def query(
  run_id: str,
  *,
  since: str | None = None,
  until: str | None = None,
  pod: str | None = None,
  container: str | None = None,
  node: str | None = None,
  severity: str | None = None,
  q: str = "",
  attempt: int | None = None,
  limit: int = 200,
  cursor: str | None = None,
) -> dict:
  since, until = timestamp(since) if since else None, timestamp(until) if until else None
  if since and until and since > until:
    raise ValueError("since must precede until")
  fingerprint = hashlib.sha256(json.dumps([run_id, since, until, pod, container, node, severity, q, attempt]).encode()).hexdigest()[:24]
  with _LOCK, closing(connect()) as db, db:
    high = db.execute("SELECT COALESCE(MAX(id),0) FROM records").fetchone()[0]
    anchor = None
    if cursor:
      try:
        token, high, anchor_time, anchor_id = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        if token != fingerprint or not isinstance(high, int) or not 0 <= high < 2**63 or not isinstance(anchor_id, int) or not 0 <= anchor_id < 2**63:
          raise ValueError()
        anchor = (timestamp(anchor_time), anchor_id)
      except (ValueError, TypeError, UnicodeError):
        raise ValueError("Invalid cursor or changed log filters") from None
    clauses, values = ["run_id = ?", "id <= ?"], [run_id, high]
    for key, value in [("pod", pod), ("container", container), ("node", node), ("severity", severity), ("attempt", attempt)]:
      if value is not None:
        clauses.append(f"{key} = ?")
        values.append(value)
    if since:
      clauses.append("timestamp >= ?")
      values.append(since)
    if until:
      clauses.append("timestamp <= ?")
      values.append(until)
    if q:
      clauses.append("instr(lower(message), lower(?)) > 0")
      values.append(q)
    if anchor:
      clauses.append("(COALESCE(timestamp, collected_at), id) < (?, ?)")
      values.extend(anchor)
    rows = db.execute(
      f"SELECT * FROM records WHERE {' AND '.join(clauses)} ORDER BY COALESCE(timestamp, collected_at) DESC, id DESC LIMIT ?", [*values, limit + 1]
    ).fetchall()
    entries = [{key: value for key, value in dict(row).items() if key != "identity"} for row in rows[:limit]]
    sources = [json.loads(row[0]) for row in db.execute("SELECT payload FROM sources WHERE run_id = ?", (run_id,))]
    prune = db.execute("SELECT value FROM archive_meta WHERE key='last_pruned_at'").fetchone()
  return {
    "schema_version": 1,
    "run_id": run_id,
    "records": entries,
    "order": "newest_first",
    "next_cursor": base64.urlsafe_b64encode(
      json.dumps([fingerprint, high, entries[-1]["timestamp"] or entries[-1]["collected_at"], entries[-1]["id"]]).encode()
    ).decode()
    if len(rows) > limit
    else None,
    "sources": sources,
    "coverage": {
      "mode": "best_effort_polling",
      "retention_days": 7,
      "max_archive_records": 20000,
      "last_pruned_at": prune[0] if prune else None,
      "tail_lines_per_source": TAIL_LINES,
      "bytes_per_source": MAX_BYTES,
      "history_complete": False,
      "note": (
        "Only collected output is retained. Rotation, deletion, restarts between polls, and bounded tails can leave gaps. "
        "Pagination excludes new collection; records pruned by retention are no longer available."
      ),
    },
  }


async def collector(app) -> None:
  """One collector per gateway process; use one replica with a persistent volume."""
  from server.store import get_store

  sweep = 0
  while True:
    try:
      k8s = await asyncio.to_thread(data.k8s_snapshot)
      if not k8s.get("available"):
        raise RuntimeError("Kubernetes discovery unavailable")
      runs = await data.runs_snapshot(get_store(), app.state.fft_worker_manager, k8s["pods"], k8s.get("scheduler"))
      for run in runs["runs"]:
        pods = data.model_pods(run["run_id"], k8s["pods"])
        if pods:
          await asyncio.to_thread(collect, run["run_id"], pods, False, sweep * MAX_SOURCES)
    except asyncio.CancelledError:
      raise
    except Exception:
      # Query responses also expose collection failures; don't emit credential-bearing errors.
      app.state.log_collector_error = "collection_failed"
    else:
      app.state.log_collector_error = None
    sweep += 1
    await asyncio.sleep(15)
