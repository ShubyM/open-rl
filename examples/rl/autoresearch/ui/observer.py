"""Tiny live UI server for autoresearch runs."""
# ruff: noqa: E501

from __future__ import annotations

import http.server
import json
import math
import os
import socketserver
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import chz

UI_DIR = Path(__file__).parent
UI_EVENTS_FILE = "ui_events.jsonl"
AGENT_TAIL_LIMIT = 12000
RUN_PATHS = {"agent", "logs", "diff", "diff_full"}
STALE_GRACE_SECONDS = 60
DONE_STATUSES = {"completed", "complete", "keep", "kept", "discard", "discarded", "crash", "crashed", "failed", "stale", "stopped", "timed_out"}


@chz.chz
class UiConfig:
  log_root: Path = Path("artifacts/autoresearch/text_sql")
  out_dir: Path = Path("artifacts/autoresearch/ui")
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


def read_jsonl(path: Path) -> list[dict]:
  return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def size(path: Path | str) -> int:
  return path.stat().st_size if path and (path := Path(path)).exists() else 0


def rel(path: Path | str, root: Path) -> str:
  return os.path.relpath(path, root) if path else ""


def clean(value):
  return value if not isinstance(value, float) or math.isfinite(value) else None


def number(value, default: float = 0.0) -> float:
  return float(value if value is not None else default)


def fmt(value) -> str:
  return "" if value is None else f"{value:.4g}" if isinstance(value, int | float) else str(value)


def title_label(value: str) -> str:
  return " ".join(part.capitalize() for part in str(value or "score").replace("/", " ").replace("_", " ").split())


def event_paths(log_root: Path) -> list[Path]:
  roots = [*log_root.parent.glob("*/" + UI_EVENTS_FILE), *log_root.glob("*/" + UI_EVENTS_FILE)]
  return sorted({path for path in roots if path.is_file()})


def event_file_path(value: str | Path, event_file: Path) -> Path:
  path = Path(value)
  return path if path.is_absolute() else event_file.parent / path


def row_for(event: dict) -> dict:
  time_value = float(event["time"])
  work_id = str(event["work_id"])
  return {
    "researcher": str(event["researcher"]),
    "run": work_id,
    "kind": str(event["kind"]),
    "status": str(event["status"]),
    "git": event.get("git") or {},
    "experiment": event.get("experiment") or {},
    "description": str(event.get("description") or ""),
    "recipe": event.get("recipe") or {},
    "attempt_timeout_minutes": event.get("attempt_timeout_minutes"),
    "score": None,
    "score_metric": "missing_metric",
    "score_label": "score",
    "score_mode": "max",
    "history": [],
    "step": None,
    "metrics": 0,
    "updated_at": time_value,
    "order": event.get("order", time_value),
    "paths": {"agent": "", "logs": "", "diff": "", "diff_full": ""},
  }


def apply_event(row: dict, event: dict, event_file: Path) -> None:
  row["updated_at"] = max(number(row.get("updated_at")), float(event["time"]))
  for key in ("researcher", "kind", "status", "description", "attempt_timeout_minutes", "order"):
    if key in event and event[key] is not None:
      row[key] = event[key]
  for key in ("git", "experiment", "recipe"):
    row[key].update(event.get(key) or {})
  if "tab" in event:
    tab = str(event["tab"])
    if tab not in RUN_PATHS:
      raise ValueError(f"unknown UI tab: {tab}")
    row["paths"][tab] = event_file_path(event["path"], event_file)
  if metric := event.get("metric"):
    value = clean(number(metric["value"]))
    name = str(metric["name"])
    row["score"] = value
    row["score_metric"] = name
    row["score_label"] = str(metric.get("label", name.rsplit("/", 1)[-1].replace("_", " ")))
    row["score_mode"] = str(metric.get("mode") or row.get("score_mode") or "max")
    row["step"] = metric.get("step", row["step"])
    row["history"].append({"step": row["step"] if row["step"] is not None else len(row["history"]), "score": value})
    row["metrics"] = len(row["history"])


def mark_stale(row: dict) -> None:
  if row["kind"] != "attempt" or row["status"] != "running":
    return
  timeout_minutes = row.get("attempt_timeout_minutes")
  if not timeout_minutes:
    return
  latest = number(row.get("updated_at"))
  for key in ("logs", "agent"):
    value = row["paths"].get(key)
    if value and (path := Path(value)).exists():
      latest = max(latest, path.stat().st_mtime)
  if time.time() - latest > float(timeout_minutes) * 60 + STALE_GRACE_SECONDS:
    row["status"] = "stale"


def running(row: dict) -> bool:
  status = str(row.get("status") or "").lower()
  return status == "running" or (not row.get("metrics") and status not in DONE_STATUSES)


def visible_score(row: dict):
  return clean(row.get("score")) if row.get("score") is not None and not running(row) else None


def score_rank(row: dict) -> float:
  score = visible_score(row)
  if score is None:
    return -math.inf
  return -float(score) if row.get("score_mode") == "min" else float(score)


def event_runs(log_root: Path) -> list[dict]:
  rows = {}
  for path in event_paths(log_root):
    for event in read_jsonl(path):
      work_id = str(event["work_id"])
      if work_id not in rows:
        rows[work_id] = row_for(event)
      row = rows[work_id]
      apply_event(row, event, path)
  return drop_finished_activity(list(rows.values()))


def drop_finished_activity(rows: list[dict]) -> list[dict]:
  researchers_with_attempts = {row["researcher"] for row in rows if row["kind"] != "activity"}
  return [row for row in rows if row["kind"] != "activity" or row["status"] == "running" or row["researcher"] not in researchers_with_attempts]


def write_if_changed(path: Path, text: str) -> None:
  if path.exists() and path.read_text(encoding="utf-8", errors="replace") == text:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text, encoding="utf-8")


def tab(path: Path | str, out_dir: Path, label: str, fmt_name: str, live: bool, limit: int) -> dict:
  return {"label": label, "format": fmt_name, "path": rel(path, out_dir), "tail": read_text(path, limit), "size": size(path), "live": live}


def diff_tab(row: dict) -> dict:
  compact = read_text(row["paths"]["diff"])
  full = read_text(row["paths"]["diff_full"]) or compact
  return {"label": "Diff", "compact": compact, "full": full}


def row_tabs(row: dict, out_dir: Path, latest_attempt: dict | None = None) -> dict:
  live = running(row)
  logs_path = row["paths"]["logs"] or (latest_attempt or {}).get("paths", {}).get("logs", "")
  return {
    "agent": tab(row["paths"]["agent"], out_dir, "Agent", "agent", live, AGENT_TAIL_LIMIT),
    "logs": tab(logs_path, out_dir, "Logs", "logs", live or bool(latest_attempt and running(latest_attempt)), 12000),
    "diff": diff_tab(row),
  }


def ui_row(row: dict, out_dir: Path, researcher_label: str, number_value: int | None = None, latest_attempt: dict | None = None) -> dict:
  live_row = row["kind"] == "activity"
  kind = "live" if live_row else "attempt"
  label = "Live" if live_row else f"Attempt {number_value}"
  score = visible_score(row)
  score_label = title_label(row.get("score_label") or row.get("score_metric") or "score")
  commit = f" · {row['git']['commit']}" if row.get("git", {}).get("commit") else ""
  score_text = f" · {score_label} {fmt(score)}" if not live_row and score is not None else ""
  meta_status = f"agent {row['status']}" if live_row else str(row["status"])
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
    "description": f"agent {row['status']}" if live_row else row.get("description") or "",
    "git": row.get("git") or {},
    "history": row.get("history") if score is not None else [],
    "tab_order": ["agent", "logs"] if live_row else ["agent", "logs", "diff"],
    "tabs": row_tabs(row, out_dir, latest_attempt),
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


def view(name: str, attempts: list[dict], live_row: dict | None, out_dir: Path, researcher_label: str) -> dict:
  visible_attempts = attempts if name == "all" else improved(attempts)
  numbered = {row["run"]: idx for idx, row in enumerate(attempts, 1)}
  latest_attempt = next((row for row in reversed(attempts) if running(row)), None) or (attempts[-1] if attempts else None)
  table_rows = ([ui_row(live_row, out_dir, researcher_label, latest_attempt=latest_attempt)] if live_row else []) + [
    ui_row(row, out_dir, researcher_label, numbered[row["run"]]) for row in visible_attempts
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
        "label": f"E{numbered[row['run']]}",
        "title": f"Attempt {numbered[row['run']]} · {metric_label([row])} {fmt(score)}",
        "score": score,
      }
      for row, score in chart_rows
    ],
  }


def first_seen(rows: list[dict]) -> float:
  return min(number(row.get("experiment", {}).get("attempt") or row.get("order") or row.get("updated_at")) for row in rows)


def researcher_payload(name: str, rows: list[dict], index: int, out_dir: Path) -> dict:
  ordered_rows = sorted(rows, key=lambda row: (number(row.get("order")), number(row.get("updated_at")), row["run"]))
  for row in ordered_rows:
    mark_stale(row)
  attempts = [row for row in ordered_rows if row["kind"] != "activity"]
  activity = next((row for row in ordered_rows if row["kind"] == "activity" and running(row)), None)
  label = f"Researcher {index}"
  live = bool(activity or any(running(row) for row in attempts))
  best = max(attempts, key=score_rank) if attempts else None
  shown = len(improved(attempts))
  meta = "running" if live else f"{shown}/{len(attempts)} shown · best {metric_label(attempts)} {fmt(visible_score(best or {}))}"
  return {
    "id": name,
    "label": label,
    "status": "running" if live else "complete",
    "live": live,
    "meta": meta,
    "views": {
      "all": view("all", attempts, activity, out_dir, label),
      "improvements": view("improvements", attempts, activity, out_dir, label),
    },
  }


def payload(log_root: Path, out_dir: Path) -> dict:
  log_root.mkdir(parents=True, exist_ok=True)
  runs = event_runs(log_root)
  groups = {}
  for run in runs:
    groups.setdefault(run["researcher"], []).append(run)
  names = sorted(groups, key=lambda name: (first_seen(groups[name]), name))
  return {"researchers": [researcher_payload(name, groups[name], idx, out_dir) for idx, name in enumerate(names, 1)]}


def write_json(log_root: Path, out_dir: Path) -> list[dict]:
  out_dir.mkdir(parents=True, exist_ok=True)
  data = payload(log_root, out_dir)
  write_if_changed(out_dir / "ui.json", json.dumps(data, indent=2, allow_nan=False) + "\n")
  return data["researchers"]


def rebuild_loop(log_root: Path, out_dir: Path, interval: float) -> None:
  while True:
    try:
      researchers = write_json(log_root, out_dir)
      attempts = sum(len(researcher["views"]["all"]["chart_points"]) for researcher in researchers)
      print(f"ui updated: researchers={len(researchers)} scored_attempts={attempts}", flush=True)
    except Exception as exc:
      print(f"ui update failed: {exc}", flush=True)
    time.sleep(interval)


class Handler(http.server.SimpleHTTPRequestHandler):
  out_dir: Path

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
    if path == "/stream":
      return self.stream()
    if path == "/file":
      return self.send_file()
    return super().do_GET()

  def translate_path(self, path: str) -> str:
    path = unquote(path.split("?", 1)[0])
    if path in {"", "/", "/experiments.html", "/app.js", "/style.css"}:
      return str(UI_DIR / ("experiments.html" if path in {"", "/"} else path[1:]))
    return str(self.out_dir / path.lstrip("/"))

  def target(self) -> Path | None:
    value = parse_qs(urlsplit(self.path).query).get("path", [""])[0]
    if not value:
      self.send_error(400, "missing path")
      return None
    target = (self.out_dir / value).resolve()
    try:
      target.relative_to(self.out_dir.resolve().parent)
    except ValueError:
      self.send_error(403, "path outside ui root")
      return None
    return target

  def send_file(self) -> None:
    target = self.target()
    if not target or not target.is_file():
      return self.send_error(404, "file not found")
    self.send_data(target.read_bytes(), "text/plain; charset=utf-8")

  def stream(self) -> None:
    target = self.target()
    if not target:
      return
    offset = max(int(parse_qs(urlsplit(self.path).query).get("offset", ["0"])[0]), 0)

    def source():
      nonlocal offset
      while True:
        if target.exists():
          if target.stat().st_size < offset:
            offset = 0
          with target.open("rb") as f:
            f.seek(offset)
            chunk, offset = f.read().decode("utf-8", errors="replace"), f.tell()
          if chunk:
            yield {"text": chunk}
        yield None
        time.sleep(1)

    self.stream_json(source())

  def events(self) -> None:
    def source():
      last = ""
      while True:
        text = read_text(self.out_dir / "ui.json")
        if text and text != last:
          last = text
          yield json.loads(text)
        else:
          yield None
        time.sleep(1)

    self.stream_json(source())


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
  allow_reuse_address = True
  daemon_threads = True


def serve(log_root: Path, out_dir: Path, host: str, port: int, interval: float) -> None:
  Handler.out_dir = out_dir
  threading.Thread(target=rebuild_loop, args=(log_root, out_dir, interval), daemon=True).start()
  print(f"serving UI at http://{host}:{port}/experiments.html", flush=True)
  ThreadingTCPServer((host, port), Handler).serve_forever()


def main(argv: list[str] | None = None) -> None:
  args = chz.entrypoint(UiConfig, argv=argv, allow_hyphens=True)
  if args.serve:
    serve(args.log_root, args.out_dir, args.host, args.port, args.interval_seconds)
  else:
    researchers = write_json(args.log_root, args.out_dir)
    print(f"Found {len(researchers)} researchers under {args.log_root}")
    print(f"Wrote {args.out_dir / 'ui.json'}")


if __name__ == "__main__":
  main()
