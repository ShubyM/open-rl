"""Send a cookbook run's metrics.jsonl to the OpenRL gateway, so the dashboard charts
them beside the run's GPUs and operations.

  python -m server.scripts.forward_metrics LOG_PATH [--base-url URL] [--run-id ID] [--follow]

Each line of LOG_PATH/metrics.jsonl is one step. The run is the model the cookbook
trains; without --run-id it is read from the first tinker:// path in
LOG_PATH/checkpoints.jsonl, which the cookbook writes at its first save, so steps wait
until then. --follow keeps sending new steps until the process is stopped. The gateway
keeps the newest row per step, so a restart can resend everything.
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

from server.telemetry.run_metrics import finite_metrics

TINKER_PATH = re.compile(r"tinker://([^/\"]+)/")
BATCH = 200


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
  args = parser.parse_args()
  if not args.base_url:
    parser.error("give --base-url or set TINKER_BASE_URL")
  base_url = args.base_url.rstrip("/")
  offset, pending, run_id = 0, [], args.run_id
  with httpx.Client(timeout=30) as http:
    while True:
      rows, offset = read_rows(args.log_path / "metrics.jsonl", offset)
      pending += rows
      run_id = run_id or run_id_of(args.log_path)
      if run_id and pending:
        try:
          count = send(http, base_url, run_id, pending)
          print(f"sent {count} steps of run {run_id}", flush=True)
        except httpx.HTTPError as exc:
          print(f"could not send {len(pending)} steps of run {run_id}: {exc}", file=sys.stderr, flush=True)
      if not args.follow:
        if pending:
          sys.exit("no run id yet: pass --run-id or wait for the first checkpoint" if not run_id else "steps were not sent")
        return
      time.sleep(args.interval)


if __name__ == "__main__":
  main()
