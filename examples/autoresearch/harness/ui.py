"""Tiny live UI server for autoresearch runs."""
# ruff: noqa: E501

from __future__ import annotations

import http.server
import json
import math
import os
import re
import socketserver
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import chz

UI_DIR = Path(__file__).parents[1] / "ui"
AGENT_TAIL_LIMIT = 12000
STALE_GRACE_SECONDS = 60
STATIC_FILES = {
  "/": "experiments.html",
  "/experiments.html": "experiments.html",
  "/app.js": "app.js",
  "/style.css": "style.css",
  "/assets/pierre-diffs.min.js": "assets/pierre-diffs.min.js",
}


@chz.chz
class UiConfig:
  log_root: Path = Path("artifacts/autoresearch/text_sql")
  interval_seconds: float = 15
  host: str = "0.0.0.0"
  port: int = 8080
  serve: bool = False


def read_text(path: Path | str, limit: int | None = None) -> str:
  if not path or not (path := Path(path)).exists():
    return ""
  if limit:
    with path.open("rb") as f:
      f.seek(0, os.SEEK_END)
      size = f.tell()
      f.seek(max(0, size - limit * 4))
      text = f.read().decode("utf-8", errors="replace")
    return text[-limit:]
  text = path.read_text(encoding="utf-8", errors="replace")
  return text


def read_json(path: Path) -> dict:
  return json.loads(path.read_text(encoding="utf-8"))


def size(path: Path | str) -> int:
  return path.stat().st_size if path and (path := Path(path)).exists() else 0


def clean(value):
  return value if not isinstance(value, float) or math.isfinite(value) else None


def number(value, default: float = 0.0) -> float:
  return float(value if value is not None else default)


def fmt(value) -> str:
  return "" if value is None else f"{value:.4g}" if isinstance(value, int | float) else str(value)


def title_label(value: str) -> str:
  return " ".join(part.capitalize() for part in str(value or "score").replace("/", " ").replace("_", " ").split())


def researcher_id(value: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:40] or "researcher"


def artifact_path(root: Path, boundary: Path, value: str | Path | None) -> Path | str:
  if not value:
    return ""
  path = Path(value)
  path = (path if path.is_absolute() else root / path).resolve()
  try:
    path.relative_to(boundary.resolve())
  except ValueError:
    return ""
  return path


def artifact_id(row: dict, name: str) -> str:
  return f"{row['researcher']}:{name}" if row["kind"] == "activity" else f"{row['researcher']}:{row['run']}:{name}"


def researcher_row(path: Path) -> dict:
  data = read_json(path)
  root = path.parent
  boundary = root.parents[1]
  artifacts = data.get("artifacts") or {}
  researcher = researcher_id(str(data["researcher"]))
  return {
    "researcher": researcher,
    "run": f"{researcher}-activity",
    "kind": "activity",
    "status": str(data["status"]),
    "git": {},
    "experiment": {},
    "description": "",
    "recipe": {},
    "attempt_timeout_minutes": data.get("agent_timeout_minutes"),
    "score": None,
    "score_metric": "missing_metric",
    "score_label": "score",
    "score_mode": "max",
    "history": [],
    "step": None,
    "metrics": 0,
    "updated_at": number(data.get("updated_at")),
    "order": data.get("started_at", 0),
    "paths": {name: artifact_path(root, boundary, artifacts.get(name)) for name in ("agent", "launcher", "notes")},
  }


def attempt_row(path: Path) -> dict:
  data = read_json(path)
  root = path.parent
  boundary = root.parents[3]
  recipe = data.get("recipe") or {}
  metric = data.get("metric") or {}
  git = data.get("git") or {}
  artifacts = data.get("artifacts") or {}
  metric_name = str(metric.get("name") or recipe.get("metric") or "missing_metric")
  metric_value = clean(number(metric["value"])) if "value" in metric and metric["value"] is not None else None
  return {
    "researcher": researcher_id(str(data["researcher"])),
    "run": str(data["id"]),
    "kind": "attempt",
    "status": str(data["status"]),
    "git": git,
    "experiment": {
      "name": "default-config" if data.get("baseline") else data.get("name"),
      "task": recipe.get("task"),
      "attempt": data.get("order"),
      "attempt_timeout_minutes": data.get("attempt_timeout_minutes"),
    },
    "description": str(git.get("description") or data.get("name") or ""),
    "recipe": {"name": recipe.get("task"), "editable": recipe.get("editable") or [], "metric": recipe.get("metric")},
    "attempt_timeout_minutes": data.get("attempt_timeout_minutes"),
    "score": metric_value,
    "score_metric": metric_name,
    "score_label": str(metric.get("label") or recipe.get("metric_label") or metric_name.rsplit("/", 1)[-1].replace("_", " ")),
    "score_mode": str(metric.get("mode") or recipe.get("metric_mode") or "max"),
    "history": [{"step": metric.get("step", 0), "score": metric_value}] if metric_value is not None else [],
    "step": metric.get("step"),
    "metrics": 1 if metric_value is not None else 0,
    "updated_at": number(data.get("updated_at")),
    "order": data.get("order", data.get("started_at", 0)),
    "paths": {name: artifact_path(root, boundary, artifacts.get(name)) for name in ("agent", "logs", "notes", "diff", "diff_full", "diff_files", "metrics")},
  }


def mark_stale(row: dict) -> None:
  if row["status"] != "running":
    return
  timeout_minutes = row.get("attempt_timeout_minutes")
  if not timeout_minutes:
    return
  latest = number(row.get("updated_at"))
  for key in ("logs", "agent", "notes"):
    value = row["paths"].get(key)
    if value and (path := Path(value)).exists():
      latest = max(latest, path.stat().st_mtime)
  if time.time() - latest > float(timeout_minutes) * 60 + STALE_GRACE_SECONDS:
    row["status"] = "stale"


def running(row: dict) -> bool:
  return str(row["status"]).lower() == "running"


def visible_score(row: dict):
  return clean(row.get("score")) if row.get("score") is not None and not running(row) else None


def score_rank(row: dict) -> float:
  score = visible_score(row)
  if score is None:
    return -math.inf
  return -float(score) if row.get("score_mode") == "min" else float(score)


def manifest_runs(log_root: Path) -> list[dict]:
  rows = []
  for path in sorted(log_root.glob("researchers/*/researcher.json")):
    try:
      rows.append(researcher_row(path))
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
      print(f"skipping invalid researcher manifest {path}: {exc}", file=sys.stderr, flush=True)
  for path in sorted(log_root.glob("researchers/*/attempts/*/attempt.json")):
    try:
      rows.append(attempt_row(path))
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
      print(f"skipping invalid attempt manifest {path}: {exc}", file=sys.stderr, flush=True)
  return rows


def tab(path: Path | str, label: str, fmt_name: str, live: bool, limit: int) -> dict:
  return {"label": label, "format": fmt_name, "tail": read_text(path, limit), "size": size(path), "live": live}


def artifact_tab(row: dict, name: str, label: str, fmt_name: str, live: bool, limit: int) -> dict:
  path = row["paths"].get(name)
  value = tab(path, label, fmt_name, live, limit)
  value["id"] = artifact_id(row, name) if path else ""
  return value


def diff_tab(row: dict) -> dict:
  compact = read_text(row["paths"].get("diff", ""))
  return {
    "label": "Diff",
    "id": artifact_id(row, "diff") if row["paths"].get("diff") else "",
    "files_id": artifact_id(row, "diff_files") if row["paths"].get("diff_files") else "",
    "compact": compact,
  }


def row_tabs(row: dict, latest_attempt: dict | None = None, notes_path: Path | str = "") -> dict:
  live = running(row)
  notes_path = row["paths"]["notes"] or notes_path
  logs_row = row if row["paths"].get("logs") or not latest_attempt else latest_attempt
  return {
    "agent": artifact_tab(row, "agent", "Agent", "agent", live, AGENT_TAIL_LIMIT),
    "logs": artifact_tab(logs_row, "logs", "Logs", "logs", live or bool(latest_attempt and running(latest_attempt)), 12000),
    "notes": artifact_tab(row, "notes", "Notes", "markdown", live or bool(latest_attempt and running(latest_attempt)), 20000),
    "diff": diff_tab(row),
  }


def ui_row(
  row: dict,
  researcher_label: str,
  number_value: int | None = None,
  latest_attempt: dict | None = None,
  notes_path: Path | str = "",
) -> dict:
  live_row = row["kind"] == "activity"
  kind = "live" if live_row else "attempt"
  label = "Live" if live_row else f"Attempt {number_value}"
  experiment_name = row.get("experiment", {}).get("name")
  tabs = row_tabs(row, latest_attempt, notes_path)
  tab_order = ["agent", "notes", "logs"] if live_row or experiment_name == "default-config" else ["agent", "notes", "logs", "diff"]
  score = visible_score(row)
  score_label = title_label(row.get("score_label") or row.get("score_metric") or "score")
  commit = f" · {row['git']['commit']}" if row.get("git", {}).get("commit") else ""
  score_text = f" · {score_label} {fmt(score)}" if not live_row and score is not None else ""
  meta_status = f"agent {row['status']}" if live_row else str(row["status"])
  description = f"agent {row['status']}" if live_row else row.get("description") or ""
  if experiment_name == "default-config":
    description = "Default config"
  return {
    "id": row["run"],
    "kind": kind,
    "label": label,
    "number": number_value,
    "status": row["status"],
    "live": running(row),
    "score": score,
    "score_label": score_label,
    "score_mode": row.get("score_mode") or "max",
    "description": description,
    "git": row.get("git") or {},
    "history": row.get("history") if score is not None else [],
    "tab_order": tab_order,
    "tabs": tabs,
    "meta": f"{label} · {researcher_label} · {meta_status}{score_text}{commit}",
    "order": row.get("order"),
    "updated_at": row.get("updated_at"),
  }


def improved(rows: list[dict]) -> list[dict]:
  best = -math.inf
  kept = []
  for row in rows:
    rank = score_rank(row)
    if rank == -math.inf:
      if running(row):
        kept.append(row)
      continue
    if rank > best:
      kept.append(row)
      best = rank
  return kept


def metric_label(rows: list[dict]) -> str:
  row = next((row for row in rows if row.get("score_label")), None)
  return title_label((row or {}).get("score_label") or "score")


def axis_label(rows: list[dict]) -> str:
  row = next((row for row in rows if row.get("score_metric") or row.get("score_label")), None) or {}
  return str(row.get("score_metric") or row.get("score_label") or "score").split("/")[-1].replace("_", " ").lower()


def view(name: str, attempts: list[dict], live_row: dict | None, researcher_label: str, notes_path: Path | str = "") -> dict:
  visible_attempts = attempts if name == "all" else improved(attempts)
  numbered = {row["run"]: idx for idx, row in enumerate(attempts, 1)}
  latest_attempt = next((row for row in reversed(attempts) if running(row)), None) or (attempts[-1] if attempts else None)
  table_rows = ([ui_row(live_row, researcher_label, latest_attempt=latest_attempt, notes_path=notes_path)] if live_row else []) + [
    ui_row(row, researcher_label, numbered[row["run"]], notes_path=notes_path) for row in visible_attempts
  ]
  chart_rows = [(row, visible_score(row)) for row in visible_attempts if visible_score(row) is not None]
  return {
    "name": name,
    "metric_label": metric_label(attempts),
    "axis_label": axis_label(attempts),
    "attempt_count": len(attempts),
    "table_rows": table_rows,
    "chart_points": [
      {
        "id": row["run"],
        "number": numbered[row["run"]],
        "label": f"A{numbered[row['run']]}",
        "title": f"Attempt {numbered[row['run']]} · {metric_label([row])} {fmt(score)}",
        "score": score,
      }
      for row, score in chart_rows
    ],
  }


def first_seen(rows: list[dict]) -> float:
  return min(number(row.get("experiment", {}).get("attempt") or row.get("order") or row.get("updated_at")) for row in rows)


def researcher_path(rows: list[dict], name: str) -> Path | str:
  return next((row["paths"][name] for row in rows if row["paths"].get(name)), "")


def researcher_payload(name: str, rows: list[dict], index: int) -> dict:
  ordered_rows = sorted(rows, key=lambda row: (number(row.get("order")), number(row.get("updated_at")), row["run"]))
  for row in ordered_rows:
    mark_stale(row)
  attempts = [row for row in ordered_rows if row["kind"] != "activity"]
  activity = next((row for row in ordered_rows if row["kind"] == "activity" and row["status"] != "completed"), None)
  notes_path = researcher_path(ordered_rows, "notes")
  label = f"Researcher {index}"
  live = bool(any(running(row) for row in ([activity] if activity else []) + attempts))
  attempt_label = "attempt" if len(attempts) == 1 else "attempts"
  meta = "running" if live else f"{len(attempts)} {attempt_label}"
  return {
    "id": name,
    "label": label,
    "status": activity["status"] if activity and not live else "running" if live else "complete",
    "live": live,
    "meta": meta,
    "views": {
      "all": view("all", attempts, activity, label, notes_path),
      "improvements": view("improvements", attempts, activity, label, notes_path),
    },
  }


def payload(log_root: Path) -> dict:
  log_root.mkdir(parents=True, exist_ok=True)
  runs = manifest_runs(log_root)
  groups = {}
  for run in runs:
    groups.setdefault(run["researcher"], []).append(run)
  names = sorted(groups, key=lambda name: (first_seen(groups[name]), name))
  return {"researchers": [researcher_payload(name, groups[name], idx) for idx, name in enumerate(names, 1)]}


def artifact_index(log_root: Path) -> dict[str, Path]:
  index = {}
  for row in manifest_runs(log_root):
    for name, path in row.get("paths", {}).items():
      if path:
        index[artifact_id(row, name)] = Path(path)
  return index


class Handler(http.server.SimpleHTTPRequestHandler):
  log_root: Path
  interval_seconds: float

  def end_headers(self) -> None:
    self.send_header("Cache-Control", "no-store, max-age=0")
    super().end_headers()

  def send_data(self, data: bytes, content_type: str = "application/json", status_code: int = 200) -> None:
    self.send_response(status_code)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)

  def stream_json(self, source) -> None:
    try:
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      for item in source:
        payload = b": keepalive\n\n" if item is None else f"data: {json.dumps(item, separators=(',', ':'))}\n\n".encode()
        self.wfile.write(payload)
        self.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
      return

  def do_GET(self) -> None:
    path = urlsplit(self.path).path
    if path == "/events":
      return self.events()
    if path == "/tail":
      return self.tail()
    if path == "/file":
      return self.send_file()
    return self.send_static(path)

  def send_static(self, path: str) -> None:
    filename = STATIC_FILES.get(unquote(path))
    if not filename:
      return self.send_error(404, "file not found")
    target = UI_DIR / filename
    if not target.is_file():
      return self.send_error(404, "file not found")
    return self.send_data(target.read_bytes(), self.guess_type(str(target)))

  def target(self) -> Path | None:
    value = parse_qs(urlsplit(self.path).query).get("id", [""])[0]
    if not value:
      self.send_error(400, "missing artifact id")
      return None
    target = artifact_index(self.log_root).get(value)
    if not target:
      self.send_error(404, "unknown artifact id")
      return None
    target = target.resolve()
    try:
      target.relative_to(self.log_root.resolve())
    except ValueError:
      self.send_error(403, "artifact outside log root")
      return None
    return target

  def send_file(self) -> None:
    target = self.target()
    if target is None:
      return
    if not target.is_file():
      return self.send_error(404, "file not found")
    self.send_data(target.read_bytes(), "text/plain; charset=utf-8")

  def tail(self) -> None:
    target = self.target()
    if not target:
      return
    query = parse_qs(urlsplit(self.path).query)
    offset = max(int(query.get("offset", ["0"])[0]), 0)
    replace_on_rewrite = query.get("replace", ["0"])[0] == "1"

    def source():
      nonlocal offset
      last_mtime = 0.0
      while True:
        if target.exists():
          stat = target.stat()
          if stat.st_size < offset:
            offset = 0
          if replace_on_rewrite and stat.st_mtime != last_mtime:
            text = target.read_text(encoding="utf-8", errors="replace")
            offset = stat.st_size
            last_mtime = stat.st_mtime
            yield {"text": text, "replace": True}
            continue
          with target.open("rb") as f:
            f.seek(offset)
            chunk, offset = f.read().decode("utf-8", errors="replace"), f.tell()
          if chunk:
            last_mtime = stat.st_mtime
            yield {"text": chunk}
        yield None
        time.sleep(1)

    self.stream_json(source())

  def events(self) -> None:
    def source():
      last = ""
      while True:
        text = json.dumps(payload(self.log_root), sort_keys=True, allow_nan=False)
        if text != last:
          last = text
          yield json.loads(text)
        else:
          yield None
        time.sleep(self.interval_seconds)

    self.stream_json(source())


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
  allow_reuse_address = True
  daemon_threads = True


def serve(log_root: Path, host: str, port: int, interval: float) -> None:
  Handler.log_root = log_root
  Handler.interval_seconds = interval
  print(f"serving UI at http://{host}:{port}/experiments.html", flush=True)
  ThreadingTCPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> None:
  args = chz.entrypoint(UiConfig, argv=argv, allow_hyphens=True)
  if args.serve:
    serve(args.log_root, args.host, args.port, args.interval_seconds)
  else:
    print(json.dumps(payload(args.log_root), indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
