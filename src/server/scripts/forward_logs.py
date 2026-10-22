"""Send a cookbook run's log to the OpenRL gateway, so the dashboard shows it as the run's client logs.

  python -m server.scripts.forward_logs LOG_PATH [--base-url URL] [--run-id ID] [--follow]

Each new line of LOG_PATH/logs.log is one log line. The run is the model the cookbook trains, and its
id is the model id; without --run-id it is read from the first tinker:// path in LOG_PATH/checkpoints.jsonl,
which the cookbook writes at its first save, so sending waits until then. --follow keeps sending until the
process is stopped. Lines are sent once per run of this command. Metrics go through OpenTelemetry instead.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx

TINKER_PATH = re.compile(r"tinker://([^/\"]+)/")
LINE_BATCH = 500
MAX_LINE = 16384
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
LEVEL = re.compile(r"\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]")


def run_id_of(log_path: Path) -> str | None:
  checkpoints = log_path / "checkpoints.jsonl"
  if checkpoints.exists() and (match := TINKER_PATH.search(checkpoints.read_text())):
    return match[1]
  return None


def read_lines(path: Path, offset: int) -> tuple[list[dict], int]:
  """The complete log lines from offset on, with the severity the logger printed. A line the
  cookbook is still writing waits for the next read."""
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
  """Consume each acknowledged batch, leaving only unsent lines after a failure."""
  count = len(lines)
  while lines:
    http.post(f"{base_url}/api/v1/runs/{run_id}/logs", json={"lines": lines[:LINE_BATCH]}).raise_for_status()
    del lines[:LINE_BATCH]
  return count


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("log_path", type=Path, help="The cookbook run's log_path")
  parser.add_argument(
    "--base-url", default=os.getenv("TINKER_BASE_URL") or os.getenv("BASE_URL"), help="The gateway; defaults to TINKER_BASE_URL or BASE_URL"
  )
  parser.add_argument("--run-id", default=None, help="The trained model's id; read from checkpoints.jsonl when omitted")
  parser.add_argument("--follow", action="store_true", help="Keep sending new lines")
  parser.add_argument("--interval", type=float, default=15.0, help="Seconds between reads with --follow")
  args = parser.parse_args()
  if not args.base_url:
    parser.error("give --base-url or set TINKER_BASE_URL")
  base_url = args.base_url.rstrip("/")
  offset, pending, run_id = 0, [], args.run_id
  with httpx.Client(timeout=30) as http:
    while True:
      lines, offset = read_lines(args.log_path / "logs.log", offset)
      pending += lines
      run_id = run_id or run_id_of(args.log_path)
      if run_id and pending:
        try:
          count = send_lines(http, base_url, run_id, pending)
          print(f"sent {count} log lines of run {run_id}", flush=True)
        except httpx.HTTPError as exc:
          print(f"could not send {len(pending)} log lines of run {run_id}: {exc}", file=sys.stderr, flush=True)
      if not args.follow:
        if pending:
          sys.exit("no run id yet: pass --run-id or wait for the first checkpoint" if not run_id else "not everything was sent")
        return
      time.sleep(args.interval)


if __name__ == "__main__":
  main()
