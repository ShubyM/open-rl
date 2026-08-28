#!/usr/bin/env python3
# JSON CLI for operating the cluster. Exposes the same primitives as the /dashboard UI —
# health, problems, inspect, runs, logs, launch, stop — and always prints JSON so agents
# and scripts can consume the output directly. Stdlib only; no extra dependencies.

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def base_url() -> str:
  return os.environ.get("BASE_URL", "http://localhost:9003").rstrip("/")


def request(method: str, path: str, body: dict | None = None) -> dict:
  url = f"{base_url()}{path}"
  data = json.dumps(body).encode() if body is not None else None
  req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
  try:
    with urllib.request.urlopen(req, timeout=30) as resp:
      return json.load(resp)
  except urllib.error.HTTPError as exc:
    try:
      return {"error": True, "status": exc.code, **json.load(exc)}
    except Exception:
      return {"error": True, "status": exc.code, "message": exc.reason}
  except urllib.error.URLError as exc:
    return {"error": True, "message": f"gateway unreachable at {base_url()}: {exc.reason}"}


def emit(payload: dict) -> None:
  print(json.dumps(payload, indent=2))
  if payload.get("error"):
    sys.exit(1)


def main() -> None:
  parser = argparse.ArgumentParser(description="Open-RL cluster operations (JSON output). Set BASE_URL to target a gateway.")
  sub = parser.add_subparsers(dest="command", required=True)

  sub.add_parser("health", help="Gateway, storage, Kubernetes, and visibility checks")
  sub.add_parser("problems", help="Everything currently wrong, most severe first")
  sub.add_parser("diagnose", help="One coherent snapshot of cluster, runs, load, health, and problems")
  sub.add_parser("inspect", help="Cluster snapshot: pools, nodes, pods, gateway, services")
  sub.add_parser("runs", help="List runs with lifecycle state")

  run = sub.add_parser("run", help="Everything about one run: state, pods, queue depth, GPU claims, optional logs")
  run.add_argument("run_id")
  run.add_argument("log_lines", type=int, nargs="?", default=None, metavar="LOG_LINES", help="Include the last N log lines per pod")
  run.add_argument("--logs", dest="log_lines_flag", type=int, default=None, metavar="N", help=argparse.SUPPRESS)

  logs = sub.add_parser("logs", help="Fetch logs for a pod")
  logs.add_argument("pod")
  logs.add_argument("tail_lines", type=int, nargs="?", default=None, metavar="TAIL_LINES")
  logs.add_argument("--container")
  logs.add_argument("--tail", dest="tail_lines_flag", type=int, default=None)

  launch = sub.add_parser("launch", help="Launch a run (create_model)")
  launch.add_argument("base_model", nargs="?", metavar="BASE_MODEL")
  launch.add_argument("--base-model", dest="base_model_flag", help=argparse.SUPPRESS)

  stop = sub.add_parser("stop", help="Stop a run: its worker, queued work, and pods")
  stop.add_argument("run_id")

  args = parser.parse_args()

  if args.command == "health":
    emit(request("GET", "/api/v1/dashboard/health"))
  elif args.command == "problems":
    emit(request("GET", "/api/v1/dashboard/problems"))
  elif args.command == "diagnose":
    emit(request("GET", "/api/v1/dashboard/snapshot"))
  elif args.command == "inspect":
    emit(request("GET", "/api/v1/dashboard/cluster"))
  elif args.command == "runs":
    emit(request("GET", "/api/v1/dashboard/runs"))
  elif args.command == "run":
    path = f"/api/v1/dashboard/runs/{urllib.parse.quote(args.run_id)}"
    log_lines = args.log_lines_flag if args.log_lines_flag is not None else args.log_lines
    if log_lines:
      path += f"?logs={log_lines}"
    emit(request("GET", path))
  elif args.command == "logs":
    tail_lines = args.tail_lines_flag if args.tail_lines_flag is not None else args.tail_lines
    params = {"tail": str(tail_lines if tail_lines is not None else 500)}
    if args.container:
      params["container"] = args.container
    emit(request("GET", f"/api/v1/dashboard/pods/{urllib.parse.quote(args.pod)}/logs?{urllib.parse.urlencode(params)}"))
  elif args.command == "launch":
    base_model = args.base_model_flag or args.base_model
    if not base_model:
      parser.error("launch requires BASE_MODEL")
    emit(request("POST", "/api/v1/dashboard/runs", {"base_model": base_model}))
  elif args.command == "stop":
    emit(request("POST", f"/api/v1/dashboard/runs/{urllib.parse.quote(args.run_id)}/stop"))


if __name__ == "__main__":
  main()
