"""Agent-friendly CLI for the OpenRL control-plane API."""

from __future__ import annotations

import inspect
import json
import os
import shlex
import sys
import time
import webbrowser
from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal, NamedTuple, get_type_hints

import chz
import httpx
from chz.blueprint import ConstructionException, ExtraneousBlueprintArg, InvalidBlueprintArg, MissingBlueprintArg

from server.cluster_job import (
  DEFAULT_CLIENT_IMAGE,
  ClusterJobError,
  ClusterJobUsageError,
  JobConfig,
  SourceDeployConfig,
  deploy_source,
  launch_job,
)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_FOUND = 3
EXIT_TIMEOUT = 4
EXIT_UNHEALTHY = 5
EXIT_API = 6


def default_base_url() -> str:
  return os.getenv("OPENRL_BASE_URL") or os.getenv("TINKER_BASE_URL") or "http://127.0.0.1:9003"


def default_client_image() -> str:
  return os.getenv("OPENRL_CLIENT_IMAGE") or DEFAULT_CLIENT_IMAGE


def parse_duration(value: str) -> float:
  suffixes = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
  lowered = value.strip().lower()
  for suffix, multiplier in suffixes.items():
    if lowered.endswith(suffix):
      return float(lowered[: -len(suffix)]) * multiplier
  return float(lowered)


def json_dump(value: Any, *, stream: bool = False) -> None:
  if stream:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), flush=True)
  else:
    print(json.dumps(value, indent=2, sort_keys=True))


def format_age(seconds: Any) -> str:
  try:
    total = max(0, int(float(seconds)))
  except (TypeError, ValueError):
    return "-"
  if total < 60:
    return f"{total}s"
  if total < 3600:
    return f"{total // 60}m {total % 60:02d}s"
  return f"{total // 3600}h {(total % 3600) // 60:02d}m"


def table(headers: list[str], rows: list[list[Any]]) -> None:
  values = [[str(cell if cell is not None else "-") for cell in row] for row in rows]
  widths = [len(header) for header in headers]
  for row in values:
    for index, cell in enumerate(row):
      widths[index] = max(widths[index], len(cell))
  print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
  print("  ".join("-" * width for width in widths))
  for row in values:
    print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


class ControlClient:
  def __init__(self, base_url: str, timeout: float = 10.0):
    self.base_url = base_url.rstrip("/")
    self.client = httpx.Client(timeout=timeout)

  def close(self) -> None:
    self.client.close()

  def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
    response = self.client.get(f"{self.base_url}/api/v1/control{path}", params=params)
    if response.status_code == 404:
      raise LookupError(path)
    response.raise_for_status()
    return response.json()

  def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = f"{self.base_url}/api/v1/control{path}"
    response = self.client.post(url, json=payload) if payload is not None else self.client.post(url)
    if response.status_code == 404:
      raise LookupError(path)
    response.raise_for_status()
    return response.json()


def output_mode(args: Any, stream: bool = False) -> str:
  if getattr(args, "json", False):
    return "jsonl" if stream else "json"
  return getattr(args, "output", "table")


def run_list(client: ControlClient, args: Any) -> int:
  payload = client.get("/runs")
  runs = payload.get("runs", payload if isinstance(payload, list) else [])
  if output_mode(args) == "json":
    json_dump(payload)
    return EXIT_OK
  rows = []
  for run in runs:
    queues = run.get("queue") or {}
    rows.append(
      [
        run.get("id") or run.get("model_id"),
        run.get("name") or "-",
        run.get("status") or "unknown",
        run.get("phase") or "unknown",
        format_age(run.get("elapsed_seconds")),
        f"{queues.get('training', 0)}/{queues.get('sampling', 0)}",
        run.get("base_model") or "-",
      ]
    )
  table(["RUN", "NAME", "STATUS", "PHASE", "AGE", "QUEUES T/S", "MODEL"], rows)
  return EXIT_OK


def run_status(client: ControlClient, args: Any) -> int:
  run = client.get(f"/runs/{args.run_id}")
  if output_mode(args) == "json":
    json_dump(run)
  else:
    rows = [
      ["run", run.get("id") or args.run_id],
      ["name", run.get("name") or "-"],
      ["status", run.get("status") or "unknown"],
      ["phase", run.get("phase") or "unknown"],
      ["message", run.get("message") or "-"],
      ["elapsed", format_age(run.get("elapsed_seconds"))],
      ["model", run.get("base_model") or "-"],
    ]
    table(["FIELD", "VALUE"], rows)
    components = run.get("components") or []
    if components:
      print()
      table(
        ["COMPONENT", "STATUS", "PHASE", "POD", "NODE", "RESTARTS", "REASON"],
        [
          [
            item.get("name") or item.get("component") or item.get("role"),
            item.get("status") or "unknown",
            item.get("phase") or "unknown",
            item.get("pod") or item.get("pod_name") or "-",
            item.get("node") or item.get("node_name") or "-",
            item.get("restarts", 0),
            item.get("reason") or "-",
          ]
          for item in components
        ],
      )
  return EXIT_UNHEALTHY if str(run.get("status", "")).lower() in {"failed", "error"} else EXIT_OK


def terminal_status(run: dict[str, Any], until: set[str]) -> tuple[bool, int]:
  status = str(run.get("status") or "unknown").lower()
  phase = str(run.get("phase") or "unknown").lower()
  if status in {"failed", "error"}:
    return True, EXIT_UNHEALTHY
  if status in until or phase in until:
    return True, EXIT_OK
  return False, EXIT_OK


def run_watch(client: ControlClient, args: Any) -> int:
  deadline = time.monotonic() + parse_duration(args.timeout) if args.timeout else None
  until = {value.strip().lower() for value in args.until.split(",") if value.strip()}
  cursor = args.after
  seen: set[str] = set()
  stream_json = output_mode(args, stream=True) == "jsonl"
  last_phase: tuple[str, str] | None = None

  while True:
    if deadline is not None and time.monotonic() >= deadline:
      print(f"timed out waiting for run {args.run_id}", file=sys.stderr)
      return EXIT_TIMEOUT

    payload = client.get(f"/runs/{args.run_id}/events", {"after": cursor, "limit": args.limit})
    events = payload.get("events", payload if isinstance(payload, list) else [])
    for event in events:
      event_id = str(event.get("cursor") or event.get("id") or "")
      if event_id and event_id in seen:
        continue
      if event_id:
        cursor = event_id
        seen.add(event_id)
      if stream_json:
        json_dump(event, stream=True)
      else:
        stamp = event.get("timestamp") or datetime.now().astimezone().isoformat(timespec="seconds")
        component = event.get("component") or "run"
        phase = event.get("phase") or event.get("status") or "event"
        message = event.get("message") or ""
        print(f"{stamp}  {component:<12} {phase:<22} {message}", flush=True)

    run = client.get(f"/runs/{args.run_id}")
    state = (str(run.get("status") or "unknown"), str(run.get("phase") or "unknown"))
    if not events and not stream_json and state != last_phase:
      print(f"{datetime.now().astimezone().isoformat(timespec='seconds')}  run          {state[1]:<22} {run.get('message') or state[0]}", flush=True)
    last_phase = state
    done, code = terminal_status(run, until)
    if until and done:
      return code
    time.sleep(args.interval)


def run_cluster(client: ControlClient, args: Any) -> int:
  payload = client.get("/cluster")
  if output_mode(args) == "json":
    json_dump(payload)
    return EXIT_UNHEALTHY if payload.get("status") in {"degraded", "unhealthy", "unavailable"} else EXIT_OK
  nodes = payload.get("nodes", [])

  def resource(node: dict[str, Any], name: str) -> Any:
    return node.get(name) or (node.get("allocatable") or {}).get(name) or (node.get("capacity") or {}).get(name) or "-"

  def gpu_queue(node: dict[str, Any]) -> str:
    status = node.get("time_slicer") or {}
    if not status:
      return "-"
    active = 1 if status.get("active_workload") else 0
    return f"{active}/{len(status.get('waiting_workloads') or [])}"

  table(
    ["NODE", "STATUS", "GPU", "CPU", "MEMORY", "WORKLOADS", "GPU QUEUE", "MESSAGE"],
    [
      [
        node.get("name"),
        node.get("status") or ("ready" if node.get("ready") else "not-ready"),
        node.get("gpu") or node.get("gpu_type") or node.get("gpu_capacity") or resource(node, "nvidia.com/gpu"),
        resource(node, "cpu"),
        resource(node, "memory"),
        node.get("pod_count", node.get("workloads") if not isinstance(node.get("workloads"), list) else len(node["workloads"])),
        gpu_queue(node),
        node.get("message") or "-",
      ]
      for node in nodes
    ],
  )
  return EXIT_UNHEALTHY if payload.get("status") in {"degraded", "unhealthy", "unavailable"} else EXIT_OK


def run_problems(client: ControlClient, args: Any) -> int:
  payload = client.get("/problems")
  problems = payload.get("problems", [])
  if output_mode(args) == "json":
    json_dump(payload)
  else:
    rows = []
    for problem in problems:
      resources = problem.get("resources") or problem
      rows.append(
        [
          problem.get("severity") or "unknown",
          problem.get("code") or "unknown",
          resources.get("run_id") or "-",
          resources.get("component") or "-",
          resources.get("pod_name") or resources.get("node") or "-",
          problem.get("summary") or "-",
          problem.get("remediation") or "-",
        ]
      )
    table(
      ["SEVERITY", "CODE", "RUN", "COMPONENT", "POD/NODE", "SUMMARY", "REMEDIATION"],
      rows,
    )
  unhealthy = any(str(problem.get("severity") or "").lower() not in {"", "info"} for problem in problems)
  return EXIT_UNHEALTHY if unhealthy else EXIT_OK


def inspection_exit(kind: str, resource: dict[str, Any]) -> int:
  status = str(resource.get("status") or "").lower()
  if kind == "run" and status in {"failed", "error"}:
    return EXIT_UNHEALTHY
  if kind == "node" and (resource.get("ready") is False or status in {"unhealthy", "unavailable", "not-ready"}):
    return EXIT_UNHEALTHY
  if kind == "pod" and status in {"failed", "error"}:
    return EXIT_UNHEALTHY
  return EXIT_OK


def print_inspection(payload: dict[str, Any]) -> None:
  resource = payload["resource"]
  rows = [["kind", payload["kind"]], ["target", payload["target"]]]
  for key in ("name", "id", "status", "phase", "message", "reason", "node", "namespace", "model_id", "role"):
    if resource.get(key) is not None:
      rows.append([key, resource[key]])
  table(["FIELD", "VALUE"], rows)


def run_inspect(client: ControlClient, args: Any) -> int:
  try:
    run = client.get(f"/runs/{args.target}")
  except LookupError:
    run = None
  if run is not None:
    result = {"kind": "run", "target": args.target, "resource": run}
    if output_mode(args) == "json":
      json_dump(result)
    else:
      print_inspection(result)
    return inspection_exit("run", run)

  cluster = client.get("/cluster")
  nodes = cluster.get("nodes", [])
  pods = cluster.get("pods", [])
  cluster_events = cluster.get("events", [])
  node = next((item for item in nodes if str(item.get("name")) == args.target), None)
  if node is not None:
    result = {
      "kind": "node",
      "target": args.target,
      "resource": node,
      "related_pods": [pod for pod in pods if pod.get("node") == args.target],
      "related_events": [event for event in cluster_events if event.get("object_kind") == "Node" and event.get("object_name") == args.target],
    }
    if output_mode(args) == "json":
      json_dump(result)
    else:
      print_inspection(result)
    return inspection_exit("node", node)

  pod = next((item for item in pods if str(item.get("pod_name") or item.get("name")) == args.target), None)
  if pod is None:
    raise LookupError(args.target)
  result = {
    "kind": "pod",
    "target": args.target,
    "resource": pod,
    "related_events": [event for event in cluster_events if event.get("object_kind") == "Pod" and event.get("object_name") == args.target],
  }
  if output_mode(args) == "json":
    json_dump(result)
  else:
    print_inspection(result)
  return inspection_exit("pod", pod)


def run_events(client: ControlClient, args: Any) -> int:
  payload = client.get(f"/runs/{args.run_id}/events", {"after": args.after, "limit": args.limit})
  if output_mode(args) == "json":
    json_dump(payload)
  else:
    events = payload.get("events", payload if isinstance(payload, list) else [])
    table(
      ["CURSOR", "TIMESTAMP", "COMPONENT", "STATUS", "PHASE", "MESSAGE"],
      [
        [
          event.get("cursor") or event.get("id") or "-",
          event.get("timestamp") or "-",
          event.get("component") or "run",
          event.get("status") or "-",
          event.get("phase") or "-",
          event.get("message") or "-",
        ]
        for event in events
      ],
    )
  return EXIT_OK


def is_stopped(run: Any) -> bool:
  return isinstance(run, dict) and (str(run.get("status") or "").lower() == "stopped" or run.get("can_stop") is False)


def output_stop(payload: dict[str, Any], args: Any) -> None:
  if output_mode(args) == "json":
    json_dump(payload)
    return
  run = payload.get("run") or {}
  print(f"{payload.get('status', 'unknown')}: {run.get('id') or args.run_id} is {run.get('status') or 'unknown'}")


def run_stop(client: ControlClient, args: Any) -> int:
  payload = client.post(f"/runs/{args.run_id}/stop")
  operation = str(payload.get("status") or "").lower()
  if operation not in {"accepted", "noop"}:
    raise ValueError(f"unexpected stop response status {operation!r}")
  if not args.wait or is_stopped(payload.get("run")):
    output_stop(payload, args)
    return EXIT_OK

  timeout = parse_duration(str(args.timeout))
  interval = parse_duration(str(args.interval))
  deadline = time.monotonic() + timeout
  while True:
    if time.monotonic() >= deadline:
      print(f"timed out waiting for run {args.run_id} to stop", file=sys.stderr)
      return EXIT_TIMEOUT
    run = client.get(f"/runs/{args.run_id}")
    payload["run"] = run
    if is_stopped(run):
      output_stop(payload, args)
      return EXIT_OK
    time.sleep(interval)


def log_entries(payload: Any, run_id: str, component: str) -> list[dict[str, Any]]:
  raw_entries: list[Any] = []
  if isinstance(payload, str):
    raw_entries = payload.splitlines()
  elif isinstance(payload, list):
    raw_entries = payload
  elif isinstance(payload, dict):
    for key in ("lines", "logs", "entries"):
      value = payload.get(key)
      if isinstance(value, str):
        raw_entries = value.splitlines()
        break
      if isinstance(value, list):
        raw_entries = value
        break

  entries = []
  for entry in raw_entries:
    if isinstance(entry, dict):
      timestamp = entry.get("timestamp")
      stream = str(entry.get("stream") or "stdout")
      message = entry.get("message", entry.get("line", ""))
      message = message if isinstance(message, str) else json.dumps(message, sort_keys=True)
    else:
      timestamp = None
      stream = "stdout"
      message = str(entry)
    entries.append(
      {
        "run_id": run_id,
        "component": component,
        "timestamp": timestamp,
        "stream": stream,
        "message": message,
      }
    )
  return entries


def log_lines(payload: Any) -> list[str]:
  return [entry["message"] for entry in log_entries(payload, "", "")]


def run_logs(client: ControlClient, args: Any) -> int:
  last_entries: list[dict[str, Any]] = []
  while True:
    payload = client.get(
      f"/runs/{args.run_id}/logs",
      {"component": args.component, "tail": args.tail, "previous": args.previous},
    )
    entries = log_entries(payload, args.run_id, args.component)
    new = entries
    if args.follow and last_entries and entries[: len(last_entries)] == last_entries:
      new = entries[len(last_entries) :]
    if output_mode(args, stream=True) == "jsonl":
      for entry in new:
        json_dump(entry, stream=True)
    else:
      for entry in new:
        print(entry["message"], flush=True)
    error = payload.get("error") if isinstance(payload, dict) else None
    if error:
      print(f"OpenRL log error: {error}", file=sys.stderr)
      return EXIT_API
    last_entries = entries
    if not args.follow:
      return EXIT_OK
    time.sleep(args.interval)


def run_doctor(client: ControlClient, args: Any) -> int:
  payload = client.get("/doctor")
  if output_mode(args) == "json":
    json_dump(payload)
  else:
    checks = payload.get("checks", [])
    table(
      ["CHECK", "STATUS", "MESSAGE", "REMEDIATION"],
      [
        [
          check.get("name"),
          check.get("status"),
          check.get("message") or "-",
          check.get("remediation") or "-",
        ]
        for check in checks
      ],
    )
  healthy = payload.get("healthy")
  if healthy is None:
    healthy = payload.get("status") in {"ok", "healthy", "ready"}
  return EXIT_OK if healthy else EXIT_UNHEALTHY


def run_ui(client: ControlClient, args: Any) -> int:
  url = f"{client.base_url}/control/"
  print(url)
  if not args.no_open:
    webbrowser.open(url)
  return EXIT_OK


def run_launch(client: ControlClient, args: Any) -> int:
  del client
  config = JobConfig(
    source=args.source,
    entrypoint=args.entrypoint,
    args=args.args,
    image=args.image,
    image_pull_policy=args.image_pull_policy,
    context=args.context,
    namespace=args.namespace,
    gateway_namespace=args.gateway_namespace,
    gateway_url=args.gateway_url,
    name=args.name,
    detach=args.detach,
    timeout=args.timeout,
    max_source_bytes=args.max_source_bytes,
    request_cpu=args.request_cpu,
    request_ephemeral_storage=args.request_ephemeral_storage,
    request_memory=args.request_memory,
    limit_ephemeral_storage=args.limit_ephemeral_storage,
    limit_memory=args.limit_memory,
    workspace_size=args.workspace_size,
    env_secret=args.env_secret,
    active_deadline_seconds=args.active_deadline_seconds,
    ttl_seconds=args.ttl_seconds,
  )
  try:
    result = launch_job(config)
  except ClusterJobUsageError as exc:
    if args.json:
      json_dump({"status": "error", "error": str(exc)})
    else:
      print(f"OpenRL launch argument error: {exc}", file=sys.stderr)
    return EXIT_USAGE
  except ClusterJobError as exc:
    if args.json:
      json_dump({"status": "error", "error": str(exc)})
    else:
      print(f"OpenRL launch error: {exc}", file=sys.stderr)
    return EXIT_API

  if args.json:
    json_dump(result)
  else:
    print(f"job/{result['job']}  {result['status']}")
    print(f"namespace: {result['namespace']}")
    print(f"pod:       {result['pod']}")
    print(f"gateway:   {result['gateway_url']}")
    print(f"follow:    {shlex.join(result['follow_command'])}")
    print(f"stop:      {shlex.join(result['stop_command'])}")
  return EXIT_UNHEALTHY if result["status"] == "failed" else EXIT_OK


def run_deploy(client: ControlClient, args: Any) -> int:
  del client
  try:
    result = deploy_source(
      SourceDeployConfig(
        source=args.source,
        context=args.context,
        namespace=args.namespace,
        deployment=args.deployment,
        container=args.container,
        max_source_bytes=args.max_source_bytes,
        timeout=args.timeout,
        reset_workers=args.reset_workers,
      )
    )
  except ClusterJobUsageError as exc:
    print(f"OpenRL deploy argument error: {exc}", file=sys.stderr)
    return EXIT_USAGE
  except ClusterJobError as exc:
    print(f"OpenRL deploy error: {exc}", file=sys.stderr)
    return EXIT_API
  if args.json:
    json_dump(result)
  else:
    print(f"deployed source revision {result['revision']} ({result['source_bytes']} bytes)")
    print(f"gateway: {result['namespace']}/{result['deployment']}")
    workers = (
      "reset; relaunch active development runs" if result["workers_reset"] else "new workers use this revision; existing workers were left running"
    )
    print(f"workers: {workers}")
  return EXIT_OK


TableOutput = Literal["table", "json"]
StreamOutput = Literal["table", "jsonl"]
Component = Literal["all", "client", "gateway", "scheduler", "trainer", "sampler", "timeslicer"]


@chz.chz
class BaseArgs:
  base_url: str = chz.field(default_factory=default_base_url, doc="Gateway URL; defaults to OPENRL_BASE_URL.")
  request_timeout: float = chz.field(default=10.0, doc="HTTP request timeout in seconds.")


@chz.chz
class OutputArgs(BaseArgs):
  json: bool = chz.field(default=False, doc="Emit stable machine-readable JSON.")
  output: TableOutput = chz.field(default="table", doc="Output format.")


@chz.chz
class StreamArgs(BaseArgs):
  json: bool = chz.field(default=False, doc="Emit stable machine-readable JSON lines.")
  output: StreamOutput = chz.field(default="table", doc="Output format.")


@chz.chz
class StatusArgs(OutputArgs):
  run_id: str


@chz.chz
class WatchArgs(StreamArgs):
  run_id: str
  after: str = "0-0"
  until: str = ""
  timeout: str = ""
  interval: float = 1.0
  limit: int = 200


@chz.chz
class InspectArgs(OutputArgs):
  target: str


@chz.chz
class EventsArgs(OutputArgs):
  run_id: str
  after: str = "0-0"
  limit: int = 200


@chz.chz
class LogsArgs(StreamArgs):
  run_id: str
  component: Component = "trainer"
  tail: int = 200
  previous: bool = False
  follow: bool = False
  interval: float = 2.0


@chz.chz
class StopArgs(OutputArgs):
  run_id: str
  wait: bool = False
  timeout: float = chz.field(default=120.0, blueprint_cast=parse_duration)
  interval: float = chz.field(default=1.0, blueprint_cast=parse_duration)


@chz.chz
class UiArgs(BaseArgs):
  no_open: bool = False


PullPolicy = Literal["Always", "IfNotPresent", "Never"]


@chz.chz
class LaunchArgs:
  source: str
  entrypoint: str = ""
  args: str = ""
  image: str = chz.field(default_factory=default_client_image, doc="Client dependency image already available to the cluster.")
  image_pull_policy: PullPolicy = "IfNotPresent"
  context: str = ""
  namespace: str = ""
  gateway_namespace: str = ""
  gateway_url: str = ""
  name: str = ""
  detach: bool = False
  timeout: float = chz.field(default=600.0, blueprint_cast=parse_duration)
  max_source_bytes: int = 20 * 1024 * 1024
  request_cpu: str = "250m"
  request_ephemeral_storage: str = "256Mi"
  request_memory: str = "512Mi"
  limit_ephemeral_storage: str = "2Gi"
  limit_memory: str = "2Gi"
  workspace_size: str = "1Gi"
  env_secret: str = ""
  active_deadline_seconds: int = 86400
  ttl_seconds: int = 3600
  json: bool = False


@chz.chz
class DeployArgs:
  source: str = "src"
  context: str = ""
  namespace: str = ""
  deployment: str = "open-rl-gateway"
  container: str = "gateway"
  max_source_bytes: int = 50 * 1024 * 1024
  timeout: float = chz.field(default=300.0, blueprint_cast=parse_duration)
  reset_workers: bool = False
  json: bool = False


class CommandSpec(NamedTuple):
  options: type[Any]
  handler: Callable[[ControlClient, Any], int]
  positionals: tuple[str, ...]
  summary: str


COMMANDS: dict[str, CommandSpec] = {
  "runs": CommandSpec(OutputArgs, run_list, (), "List active and recent runs."),
  "status": CommandSpec(StatusArgs, run_status, ("run_id",), "Inspect one run and its components."),
  "watch": CommandSpec(WatchArgs, run_watch, ("run_id",), "Stream lifecycle events for a run."),
  "cluster": CommandSpec(OutputArgs, run_cluster, (), "List machines and scheduled workloads."),
  "problems": CommandSpec(OutputArgs, run_problems, (), "List actionable cluster and workload problems."),
  "inspect": CommandSpec(InspectArgs, run_inspect, ("target",), "Inspect an exact run ID, node name, or pod name."),
  "events": CommandSpec(EventsArgs, run_events, ("run_id",), "Read one page of lifecycle events for a run."),
  "logs": CommandSpec(LogsArgs, run_logs, ("run_id",), "Read or follow component logs."),
  "stop": CommandSpec(StopArgs, run_stop, ("run_id",), "Idempotently stop a live run and its cluster workers."),
  "doctor": CommandSpec(OutputArgs, run_doctor, (), "Check gateway, Redis, Kubernetes, GPU, and worker health."),
  "ui": CommandSpec(UiArgs, run_ui, (), "Print or open the browser control plane."),
  "launch": CommandSpec(LaunchArgs, run_launch, ("source",), "Run working-tree recipe code as a Kubernetes Job."),
  "deploy": CommandSpec(DeployArgs, run_deploy, (), "Deploy Python source without rebuilding GPU images."),
}

GLOBAL_FIELDS = {"base_url", "request_timeout"}
CHZ_PARSE_ERRORS = (ConstructionException, ExtraneousBlueprintArg, InvalidBlueprintArg, MissingBlueprintArg, TypeError, ValueError)


class UsageError(ValueError):
  pass


def public_name(name: str) -> str:
  return name.replace("_", "-")


def format_help(command: str | None = None) -> str:
  if command is None:
    width = max(len(name) for name in COMMANDS)
    commands = "\n".join(f"  {name.ljust(width)}  {spec.summary}" for name, spec in COMMANDS.items())
    return (
      "Usage: openrl [--base-url URL] [--request-timeout SECONDS] COMMAND [ARGS] [OPTIONS]\n\n"
      "Inspect cluster state, launch recipe jobs, and safely operate OpenRL runs.\n\n"
      f"Commands:\n{commands}\n\n"
      "Run 'openrl COMMAND --help' for command options.\n\n"
      "Exit codes: 0 success, 2 invalid arguments, 3 not found, 4 timeout, "
      "5 unhealthy state or problem at warning severity or higher, 6 API error, 130 interrupted.\n"
    )

  spec = COMMANDS[command]
  positional = " ".join(f"<{public_name(name)}>" for name in spec.positionals)
  usage = f"Usage: openrl {command}" + (f" {positional}" if positional else "") + " [OPTIONS]"
  hints = get_type_hints(spec.options)
  option_lines = []
  for name in inspect.signature(spec.options).parameters:
    if name == "__chz_args" or name in spec.positionals:
      continue
    flag = f"--{public_name(name)}"
    if name == "output":
      flag = f"-o, {flag}"
    value = "" if hints.get(name) is bool else " VALUE"
    option_lines.append(f"  {flag}{value}")
  option_lines.append("  -h, --help")
  return f"{usage}\n\n{spec.summary}\n\nOptions:\n" + "\n".join(option_lines) + "\n"


def split_command(argv: list[str]) -> tuple[str, list[str]]:
  prefix: list[str] = []
  index = 0
  while index < len(argv):
    token = argv[index]
    if token in COMMANDS:
      return token, prefix + argv[index + 1 :]
    option = token.split("=", 1)[0]
    if option not in {"--base-url", "--request-timeout"}:
      raise UsageError(f"unknown command or global option {token!r}")
    prefix.append(token)
    if "=" not in token:
      index += 1
      if index >= len(argv):
        raise UsageError(f"option {option} requires a value")
      prefix.append(argv[index])
    index += 1
  raise UsageError("missing command")


def normalize_argv(spec: CommandSpec, argv: list[str]) -> list[str]:
  fields = set(chz.chz_fields(spec.options))
  hints = get_type_hints(spec.options)
  bool_fields = {name for name, annotation in hints.items() if annotation is bool}
  positionals = iter(spec.positionals)
  normalized: list[str] = []
  index = 0

  while index < len(argv):
    token = argv[index]
    if token == "-o":
      token = "--output"
    elif token.startswith("-o="):
      token = f"--output={token[3:]}"

    if token.startswith("--"):
      option = token[2:]
      if "=" in option:
        raw_name, value = option.split("=", 1)
      else:
        raw_name, value = option, None
      name = raw_name.replace("-", "_")
      if name not in fields:
        raise UsageError(f"unknown option '--{raw_name}'")
      if value is None and name in bool_fields:
        value = "true"
      elif value is None:
        index += 1
        if index >= len(argv):
          raise UsageError(f"option '--{raw_name}' requires a value")
        value = argv[index]
      normalized.append(f"{name}={value}")
    elif "=" in token:
      raw_name, value = token.split("=", 1)
      name = raw_name.lstrip("-").replace("-", "_")
      if name not in fields:
        raise UsageError(f"unknown option '{raw_name}'")
      normalized.append(f"{name}={value}")
    else:
      try:
        name = next(positionals)
      except StopIteration:
        raise UsageError(f"unexpected positional argument {token!r}") from None
      normalized.append(f"{name}={token}")
    index += 1

  return normalized


def public_error(error: Exception, spec: CommandSpec | None) -> str:
  message = str(error)
  if spec is not None:
    for name in sorted(chz.chz_fields(spec.options), key=len, reverse=True):
      message = message.replace(name, public_name(name))
  return message


def main(argv: list[str] | None = None) -> int:
  raw_argv = list(sys.argv[1:] if argv is None else argv)
  if not raw_argv or raw_argv[0] in {"-h", "--help"}:
    print(format_help(), end="")
    return EXIT_OK

  command: str | None = None
  spec: CommandSpec | None = None
  try:
    command, command_argv = split_command(raw_argv)
    spec = COMMANDS[command]
    if any(token in {"-h", "--help"} for token in command_argv):
      print(format_help(command), end="")
      return EXIT_OK
    normalized = normalize_argv(spec, command_argv)
    args = chz.Blueprint(spec.options).make_from_argv(normalized)
  except CHZ_PARSE_ERRORS as exc:
    print(f"OpenRL argument error: {public_error(exc, spec)}", file=sys.stderr)
    print(format_help(command) if command in COMMANDS else format_help(), file=sys.stderr, end="")
    return EXIT_USAGE

  client = ControlClient(getattr(args, "base_url", default_base_url()), timeout=getattr(args, "request_timeout", 10.0))
  try:
    return int(spec.handler(client, args))
  except LookupError:
    target = getattr(args, "run_id", getattr(args, "target", command))
    print(f"not found: {target}", file=sys.stderr)
    return EXIT_NOT_FOUND
  except (httpx.HTTPError, ValueError) as exc:
    print(f"OpenRL API error: {exc}", file=sys.stderr)
    return EXIT_API
  except KeyboardInterrupt:
    return 130
  finally:
    client.close()


if __name__ == "__main__":
  raise SystemExit(main())
