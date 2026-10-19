#!/usr/bin/env python3
"""A stand-in for `vllm serve` with the endpoints the sampler launcher calls.

  vllm serve <model> --port <port> --log <file> [--boot-seconds S] [--fail-on PATH]

The launcher tests put this file on PATH as `vllm`. Every request lands in the log as one
line, `GET /health` or `POST /sleep?level=1`. Like vLLM, it ignores SIGTERM and SIGINT while
it boots, for `--boot-seconds`, logging the signal as ignored. Once it serves, a signal is
logged and exits 3, so a test can tell a forwarded signal from a crash. `--fail-on` answers
that path with a 500. It exits when its parent is gone, as vLLM's engine does.
"""

import argparse
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def watch_parent(parent: int) -> None:
  while os.getppid() == parent:
    time.sleep(0.2)
  os._exit(4)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("serve")
  parser.add_argument("model")
  parser.add_argument("--port", type=int, required=True)
  parser.add_argument("--log", required=True)
  parser.add_argument("--boot-seconds", type=float, default=0.3)
  parser.add_argument("--fail-on")
  options, _ = parser.parse_known_args()
  log = open(options.log, "a", buffering=1)
  serving = False

  class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
      log.write(f"GET {self.path}\n")
      self.reply()

    def do_POST(self) -> None:
      body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
      log.write(f"POST {self.path} {body}".rstrip() + "\n")
      self.reply(500 if self.path == options.fail_on else 200)

    def reply(self, status: int = 200) -> None:
      self.send_response(status)
      self.send_header("Content-Type", "application/json")
      self.end_headers()
      self.wfile.write(b"{}")

    def log_message(self, *args) -> None:
      pass

  def on_signal(signum, frame) -> None:
    name = signal.Signals(signum).name
    if not serving:
      log.write(f"{name} ignored\n")
      return
    log.write(f"{name}\n")
    sys.exit(3)

  signal.signal(signal.SIGTERM, on_signal)
  signal.signal(signal.SIGINT, on_signal)
  threading.Thread(target=watch_parent, args=(os.getppid(),), daemon=True).start()
  time.sleep(options.boot_seconds)
  server = HTTPServer(("127.0.0.1", options.port), Handler)
  serving = True
  server.serve_forever()


if __name__ == "__main__":
  main()
