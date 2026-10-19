"""Send a cookbook run's metrics and log to the OpenRL gateway, so the dashboard shows
them with the run: metrics as its training charts, log lines as its client logs.

  python -m server.scripts.forward_metrics LOG_PATH [--base-url URL] [--run-id ID] [--follow] [--no-logs]

Each line of LOG_PATH/metrics.jsonl is one step, and each new line of LOG_PATH/logs.log
one log line. The run is the model the cookbook trains, and its id is the model id;
without --run-id it is read from the first tinker:// path in LOG_PATH/checkpoints.jsonl,
which the cookbook writes at its first save, so everything waits until then. --follow
keeps sending until the process is stopped. The gateway keeps the newest row per step, so
a restart can resend metrics; log lines are sent once per run of this command.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

from server.telemetry.metric_values import finite_metrics

TINKER_PATH = re.compile(r"tinker://([^/\"]+)/")
BATCH = 200
LINE_BATCH = 500
MAX_LINE = 16384
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEVEL = re.compile(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]")


def run_id_of(log_path: Path) -> str | None:
  checkpoints = log_path / "checkpoints.jsonl"
  if checkpoints.exists() and (match := TINKER_PATH.search(checkpoints.read_text())):
    return match[1]
  return None


def read_rows(path: Path, offset: int) -> tuple[list[dict], int]:
  """The complete lines from offset on, and where the next read starts. A line the
  cookbook is still writing waits for the next read."""
  if not path.exists():
    return [], offset
  with path.open("rb") as f:
    f.seek(offset)
    data = f.read()
  end = data.rfind(b"\n") + 1
  rows = []
  for line in data[:end].splitlines():
    try:
      record = json.loads(line)
    except json.JSONDecodeError:
      continue
    if not isinstance(record, dict):
      continue
    step = record.get("step", record.get("progress/batch"))
    if isinstance(step, int) and not isinstance(step, bool) and 0 <= step <= 2**53 - 1:
      record.pop("step", None)
      if metrics := finite_metrics(record):
        rows.append({"step": step, "metrics": metrics})
  return rows, offset + end


def read_lines(path: Path, offset: int) -> tuple[list[dict], int]:
  """The complete log lines from offset on, with the severity the logger printed."""
  if not path.exists():
    return [], offset
  with path.open("rb") as f:
    f.seek(offset)
    data = f.read()
  end = data.rfind(b"\n") + 1
  lines = []
  for raw in data[:end].decode("utf-8", errors="replace").splitlines():
    text = ANSI.sub("", raw).rstrip()
    if text:
      level = LEVEL.search(text[:120])
      lines.append({"message": text[:MAX_LINE], "severity": level[1] if level else "INFO", "at": time.time()})
  return lines, offset + end


def send_lines(http: httpx.Client, base_url: str, run_id: str, lines: list[dict]) -> int:
  count = len(lines)
  while lines:
    http.post(f"{base_url}/api/v1/runs/{run_id}/logs", json={"lines": lines[:LINE_BATCH]}).raise_for_status()
    del lines[:LINE_BATCH]
  return count


def send(http: httpx.Client, base_url: str, run_id: str, rows: list[dict]) -> int:
  """Consume each acknowledged batch, leaving only unsent rows after a failure."""
  count = len(rows)
  while rows:
    http.post(f"{base_url}/api/v1/runs/{run_id}/metrics", json={"rows": rows[:BATCH]}).raise_for_status()
    del rows[:BATCH]
  return count


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("log_path", type=Path, help="The cookbook run's log_path")
  parser.add_argument(
    "--base-url", default=os.getenv("TINKER_BASE_URL") or os.getenv("BASE_URL"), help="The gateway; defaults to TINKER_BASE_URL or BASE_URL"
  )
  parser.add_argument("--run-id", default=None, help="The trained model's id; read from checkpoints.jsonl when omitted")
  parser.add_argument("--follow", action="store_true", help="Keep sending new steps")
  parser.add_argument("--interval", type=float, default=15.0, help="Seconds between reads with --follow")
  parser.add_argument("--no-logs", action="store_true", help="Send metrics only, not logs.log")
  args = parser.parse_args()
  if not args.base_url:
    parser.error("give --base-url or set TINKER_BASE_URL")
  base_url = args.base_url.rstrip("/")
  offset, pending, run_id = 0, [], args.run_id
  log_offset, pending_lines = 0, []
  with httpx.Client(timeout=30) as http:
    while True:
      rows, offset = read_rows(args.log_path / "metrics.jsonl", offset)
      pending += rows
      if not args.no_logs:
        lines, log_offset = read_lines(args.log_path / "logs.log", log_offset)
        pending_lines += lines
      run_id = run_id or run_id_of(args.log_path)
      if run_id and pending:
        try:
          count = send(http, base_url, run_id, pending)
          print(f"sent {count} steps of run {run_id}", flush=True)
        except httpx.HTTPError as exc:
          print(f"could not send {len(pending)} steps of run {run_id}: {exc}", file=sys.stderr, flush=True)
      if run_id and pending_lines:
        try:
          count = send_lines(http, base_url, run_id, pending_lines)
          print(f"sent {count} log lines of run {run_id}", flush=True)
        except httpx.HTTPError as exc:
          print(f"could not send {len(pending_lines)} log lines of run {run_id}: {exc}", file=sys.stderr, flush=True)
      if not args.follow:
        if pending or pending_lines:
          sys.exit("no run id yet: pass --run-id or wait for the first checkpoint" if not run_id else "not everything was sent")
        return
      time.sleep(args.interval)


if __name__ == "__main__":
  main()
