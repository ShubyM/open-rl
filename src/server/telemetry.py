"""What OpenRL reports about itself: worker records and finished ops, client logs, SDK errors, and OpenTelemetry.

Each worker process keeps one record, open_rl:worker:<id>, rewritten every few seconds: where it runs, its GPUs,
its phase, the requests it popped and has not answered, and its recent GPU turns. A record that stops changing
belongs to a process that is gone. Each finished op is appended to a bounded list for its run, or for the runtime
when it served no single run. Recording is best effort: it never makes an op wait on the store, and never changes
what the op returns or raises. Nothing here reads the cluster; the dashboard does that.
"""

import asyncio
import importlib.metadata
import json
import math
import os
import re
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, contextmanager, suppress
from datetime import UTC, datetime
from numbers import Real

from fastapi import HTTPException
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field

from server.store import get_state_store

SAMPLE_LIMIT = int(os.getenv("OPEN_RL_OPS_SAMPLE_LIMIT", "2000"))
# How long a run's records outlive their last write; 0 keeps them.
TTL_SECONDS = float(os.getenv("OPEN_RL_OPS_TTL_DAYS", "7")) * 86400 or None
HEARTBEAT_SECONDS = 5
STALE_SECONDS = 6 * HEARTBEAT_SECONDS  # a record not rewritten for this long belongs to a process that is gone
MAX_TURNS = 200
# Cloud Logging labels on a client's lines, read back by the dashboard.
CLIENT, RUN_LABEL, SOURCE_LABEL = "client", "openrl_run_id", "openrl_source"
LOG_TAIL, MAX_LOG_LINE, SDK_ERROR_TAIL = 5000, 16384, 200
CLOCK_SKEW_SECONDS = 86400  # a client clock is trusted within this much of the gateway's
RUN_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
SEVERITIES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
# The SDK attaches full request and response headers to fatal errors, and the request carries the API key.
DROPPED = {"request_headers", "response_headers"}
SECRET = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|secret|password", re.I)
CONFIG_ENV = ("OPEN_RL_", "OPENRL_", "VLLM_", "NCCL_", "CUDA_", "PYTORCH_", "TORCH_")
PACKAGES = ("torch", "transformers", "peft", "vllm", "nemo_automodel", "megatron-core", "flash-attn", "open-rl")


def ops_key(run_id: str) -> str:
  return f"open_rl:operations:{run_id}"


def samples_key(run_id: str) -> str:
  """Sample commands keep their own list, so a run's sampling never evicts its training ops."""
  return f"open_rl:operations:{run_id}:samples"


def session_key(session_id: str) -> str:
  """A session's durable record; the live open_rl:session:<id> key expires two minutes after its last heartbeat."""
  return f"open_rl:session_record:{session_id}"


def ledger_key(run_id: str, event: str) -> str:
  """Counts of a run's trainer requests by lifecycle event: enqueued, popped, answered."""
  return f"open_rl:ledger:{run_id}:{event}"


def steps_key(run_id: str) -> str:
  return f"open_rl:steps:{run_id}"


def worker_key(worker_id: str) -> str:
  return f"open_rl:worker:{worker_id}"


def logs_key(run_id: str) -> str:
  return f"open_rl:run_logs:{run_id}"


def sdk_errors_key(session_id: str) -> str:
  return f"open_rl:sdk_errors:{session_id}"


def numeric_values(values: dict) -> dict[str, float]:
  """Native numbers by name, nonfinite included so an op can name its NaN."""
  result = {}
  for name, value in values.items():
    if isinstance(name, str) and 0 < len(name) <= 200 and isinstance(value, Real) and not isinstance(value, bool):
      with suppress(OverflowError):
        result[name] = float(value)
  return result


# *** worker records ***

current: "Worker | None" = None  # this process's record while it serves
writes: set[asyncio.Task] = set()  # in-flight writes, held so they are not collected mid-write
steps_done: dict[str, int] = {}  # run id -> optim_steps it had completed, stamped on each of its trainer ops


def worker_identity() -> dict:
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


def visible_gpus() -> list:
  """NVML handles of the GPUs this process can see, in CUDA order."""
  import pynvml

  pynvml.nvmlInit()
  handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
  visible = os.getenv("CUDA_VISIBLE_DEVICES")
  return [h for i, h in enumerate(handles) if visible is None or str(i) in visible.split(",") or pynvml.nvmlDeviceGetUUID(h) in visible]


def gpu_uuids() -> list[str]:
  """The GPUs this process can see, as the UUIDs DCGM labels its series with."""
  try:
    import pynvml

    return [pynvml.nvmlDeviceGetUUID(h) for h in visible_gpus()]
  except Exception:
    return []


def gpu_now() -> list[dict]:
  """Utilization and memory of each visible GPU now, so GPU checks work without a metrics backend."""
  try:
    import pynvml

    readings = []
    for h in visible_gpus():
      util, memory = pynvml.nvmlDeviceGetUtilizationRates(h).gpu, pynvml.nvmlDeviceGetMemoryInfo(h).used >> 20
      readings.append({"uuid": pynvml.nvmlDeviceGetUUID(h), "util": util, "mem_mib": memory})
    return readings
  except Exception:
    return []


def effective_config() -> dict:
  """What this process actually runs with: the settings the scheduler passed as env, and the package versions installed."""
  env = {k: v for k, v in os.environ.items() if k.startswith(CONFIG_ENV) and not SECRET.search(k)}
  versions = {}
  for package in PACKAGES:
    with suppress(Exception):
      versions[package] = importlib.metadata.version(package)
  return {"env": dict(sorted(env.items())), "versions": versions}


def alive(record: dict, now: float) -> bool:
  return record.get("phase") != "exited" and now - record.get("heartbeat_at", 0) < STALE_SECONDS


def operation_name(request: dict, role: str) -> str:
  # A forward-only call travels as forward_backward; its loss must not mix with training losses.
  if request.get("forward_only"):
    return "forward"
  return str(request.get("op") or ("sample" if role == "sampler" else "unknown"))


def run_of(request: dict) -> str | None:
  return request.get("logical_run_id") or request.get("model_id")


class Worker:
  def __init__(self, role: str, runtime_id: str | None) -> None:
    now = time.time()
    # Per process, not per pod: a container restarted in place must not overwrite what its predecessor held.
    self.record = {"id": uuid.uuid4().hex[:16], "node": socket.gethostname(), **worker_identity(), "role": role, "runtime_id": runtime_id}
    self.record.update(gpus=gpu_uuids(), started_at=now, phase="starting", phase_since=now, turns=[])
    self.outstanding: dict[str, dict] = {}  # request_id -> op, run_id and when it was popped

  def snapshot(self) -> dict:
    """The record as stored, outstanding requests grouped by op and run, oldest of each group first."""
    groups: dict[tuple, dict] = {}
    for entry in self.outstanding.values():
      groups.setdefault((entry["op"], entry["run_id"]), {**entry, "count": 0})["count"] += 1
    return {**self.record, "heartbeat_at": time.time(), "outstanding": list(groups.values())}


@asynccontextmanager
async def serving(role: str, runtime_id: str | None, **fields):
  """Keep this process's record fresh while the body runs; a clean exit says so."""
  global current
  current = worker = Worker(role, runtime_id)
  worker.record.update(fields, config=effective_config())
  # One structured line per process start: the effective-config check compares it with what the run asked for.
  identity = {k: worker.record.get(k) for k in ("id", "role", "runtime_id", "pod", "pod_uid", "node", "gpus")}
  echo = {"kind": "openrl.config", "v": 1, **identity, **worker.record["config"]}
  print(json.dumps({"severity": "INFO", "message": "openrl.config", "openrl": echo}), flush=True)

  async def save():
    with suppress(Exception):
      value = json.dumps(worker.snapshot())
      await asyncio.wait_for(get_state_store().set_value(worker_key(worker.record["id"]), value, ttl_seconds=TTL_SECONDS), timeout=5)

  async def heartbeat():
    while True:
      worker.record["gpu_now"] = gpu_now()
      await save()
      await asyncio.sleep(HEARTBEAT_SECONDS)

  task = asyncio.create_task(heartbeat())
  try:
    yield worker
  finally:
    task.cancel()
    set_phase("exited")
    await save()
    current = None


def note(**fields) -> None:
  if current is not None:
    current.record.update(fields)


def set_phase(phase: str) -> None:
  if current is not None and current.record["phase"] != phase:
    current.record.update(phase=phase, phase_since=time.time())


def popped(requests: list[dict]) -> None:
  """Requests this worker took from its queue; each stays outstanding until answered."""
  if current is not None:
    for r in requests:
      if r.get("request_id"):
        current.outstanding[r["request_id"]] = {"op": operation_name(r, current.record["role"]), "run_id": run_of(r), "since": time.time()}
        count("popped", run_of(r))


class Answering:
  """A request store that takes each request off the outstanding list once its answer is written."""

  def __init__(self, store) -> None:
    self.store = store

  def __getattr__(self, name: str):
    return getattr(self.store, name)

  async def set_future(self, request_id: str, result: dict) -> None:
    await self.store.set_future(request_id, result)
    if current is not None and (entry := current.outstanding.pop(request_id, None)):
      count("answered", entry["run_id"])


async def note_session(store, session_id: str, *, created: bool = False, model_id: str | None = None, metadata: dict | None = None) -> None:
  """When a session started and was last heard from, and every model it created or attached; one log line per change."""
  key, now = session_key(session_id), time.time()
  record = json.loads(await store.get_value(key) or "null") or {"session_id": session_id, "created_at": now, "model_ids": []}
  record["last_seen_at"] = now
  if metadata is not None:
    record["metadata"] = redact(metadata)
  event = "created" if created else None
  if model_id and model_id not in record["model_ids"]:
    record["model_ids"].append(model_id)
    event = "attached"
  await store.set_value(key, json.dumps(record), ttl_seconds=TTL_SECONDS)
  if event:
    labels = {SOURCE_LABEL: "openrl_event", **({RUN_LABEL: model_id} if model_id else {})}
    line = {"kind": "openrl.session", "v": 1, "event": event, "session_id": session_id, "model_id": model_id, "time": iso(now)}
    print(json.dumps({"severity": "INFO", "message": "openrl.session", "openrl": line, "logging.googleapis.com/labels": labels}), flush=True)


def count(event: str, run_id: str | None) -> None:
  """One more request of this run reached `event`; a request enqueued but never popped shows up as a gap."""
  if run_id:
    write(get_state_store().increment(ledger_key(run_id, event), TTL_SECONDS))


def write(coroutine: Coroutine) -> None:
  """Write in the background; a write that fails or times out is dropped."""
  task = asyncio.create_task(asyncio.wait_for(coroutine, timeout=5))
  writes.add(task)
  task.add_done_callback(lambda t: (writes.discard(t), t.cancelled() or t.exception()))


async def observe_operation(request: dict, role: str, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Run one API op and record how it went; returns or raises exactly as the op does."""
  if (run_id := run_of(request)) and run_id not in steps_done:
    with suppress(Exception):
      steps_done[run_id] = int(await get_state_store().get_value(steps_key(run_id)) or 0)
  started_at, status, error_type, result = time.time(), "succeeded", None, {}
  try:
    result = await call()
    if result.get("type") == "RequestFailedResponse":
      status, error_type = "failed", "RequestFailedResponse"
    return result
  except (Exception, asyncio.CancelledError) as error:
    status, error_type = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed", type(error).__name__
    raise
  finally:
    with suppress(Exception):
      record(request, role, runtime_id, started_at, status, error_type, result)


def record(
  request: dict,
  role: str,
  runtime_id: str | None,
  started_at: float,
  status: str,
  error_type: str | None,
  result: dict,
  where: dict | None = None,
  *,
  run_scoped: bool = True,
) -> None:
  """`where` names the pod the op ran on, when not this process's. No payloads, outputs or error messages are kept."""
  if not (run_id := run_of(request)):
    return
  operation, at = operation_name(request, role), time.time()
  values = numeric_values(result.get("metrics") or {})
  sample = {
    "at": at,
    "started_at": started_at,
    "operation": operation,
    "request_id": request.get("request_id"),
    "run_id": run_id,
    "runtime_id": runtime_id,
    "role": role,
    "status": status,
    "error_type": error_type,
    "elapsed_seconds": at - started_at,
    **(worker_identity() if where is None else where),
    "metrics": {k: v for k, v in values.items() if math.isfinite(v)},
  }
  if operation != "sample" and run_id in steps_done:
    sample["step"] = steps_done[run_id]
  # JSON has no NaN or infinity; name them so a NaN grad norm is not just missing.
  if nonfinite := sorted(k for k, v in values.items() if not math.isfinite(v)):
    sample["nonfinite"] = nonfinite
  if (span := trace.get_current_span().get_span_context()).is_valid:
    sample.update(trace_id=f"{span.trace_id:032x}", span_id=f"{span.span_id:016x}")
  write(get_state_store().append_sample(samples_key(run_id) if operation == "sample" else ops_key(run_id), sample, SAMPLE_LIMIT, TTL_SECONDS))
  if operation == "optim_step" and status == "succeeded":
    write(get_state_store().increment(steps_key(run_id), TTL_SECONDS))
    if run_id in steps_done:
      steps_done[run_id] += 1


@contextmanager
def record_op(name: str):
  """Record worker work that serves no one request (init, wake_up, sleep, weight_sync) under the worker's runtime."""
  started_at, status = time.time(), "failed"
  try:
    yield
    status = "succeeded"
  finally:
    if current is not None:
      runtime_id, role = current.record["runtime_id"], current.record["role"]
      with suppress(Exception):
        record({"op": name, "model_id": runtime_id}, role, runtime_id, started_at, status, None, {}, run_scoped=False)


@asynccontextmanager
async def gpu_turn(time_slicer, workload):
  """Hold the GPU through the time slicer, showing the wait and the turn on this worker's record."""
  set_phase("waiting_gpu")
  try:
    async with time_slicer.acquire(workload):
      started = time.time()
      set_phase("on_gpu")
      try:
        yield
      finally:
        if current is not None:
          current.record["turns"] = [*current.record["turns"][1 - MAX_TURNS :], [started, time.time()]]
  finally:
    set_phase("ready")


# *** client logs and SDK errors, through the gateway ***


class LogLine(BaseModel):
  message: str = Field(max_length=MAX_LOG_LINE)
  severity: str = "INFO"
  at: float | None = None


class LogRequest(BaseModel):
  lines: list[LogLine] = Field(max_length=1000)


async def known_run(store, run_id: str) -> None:
  """A run is its model: the run id is the model id the client trains."""
  if not RUN_ID.fullmatch(run_id):
    raise HTTPException(400, "run_id must be 1-128 letters, digits, '.', '_', ':' or '-'")
  if not await store.get_value(f"open_rl:model_meta:{run_id}"):
    raise HTTPException(404, f"No run {run_id}; a run is its model, so send the model id the client trains")


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


async def ingest_logs(store, run_id: str, body: LogRequest) -> dict:
  """Each line goes to stdout labelled with its run, so Cloud Logging keeps it beside the run's worker logs,
  and to a short tail in the run store for a deployment without Cloud Logging."""
  await known_run(store, run_id)
  now = time.time()
  for line in body.lines:
    severity = line.severity.upper() if line.severity.upper() in SEVERITIES else "INFO"
    at = line.at if line.at is not None and abs(line.at - now) < CLOCK_SKEW_SECONDS else now
    labels = {RUN_LABEL: run_id, SOURCE_LABEL: CLIENT}
    print(json.dumps({"severity": severity, "message": line.message, "time": iso(at), "logging.googleapis.com/labels": labels}), flush=True)
    await store.append_sample(logs_key(run_id), {"at": at, "severity": severity, "message": line.message}, LOG_TAIL, TTL_SECONDS)
  return {"accepted": len(body.lines)}


def redact(value):
  if isinstance(value, dict):
    return {k: "[redacted]" if SECRET.search(str(k)) else redact(v) for k, v in value.items() if k not in DROPPED}
  return [redact(v) for v in value] if isinstance(value, list) else value


def describe(event: dict) -> tuple[str, str]:
  """A short reason and a message for an SDK error event."""
  if event.get("event") == "UNHANDLED_EXCEPTION":
    return f"SDK {event.get('error_type') or 'exception'}", str(event.get("error_message") or "")
  data = event.get("event_data") if isinstance(event.get("event_data"), dict) else {}
  name = str(event.get("event_name") or event.get("event") or "event").rsplit(".", 1)[-1]
  status = f"HTTP {data['status_code']}: " if data.get("status_code") else ""
  return f"SDK {name}", status + str(data.get("detail") or data.get("exception") or data.get("message") or "")


async def ingest_sdk_events(store, body: dict) -> None:
  """The tinker SDK's own telemetry: one redacted JSON line per event, and error events kept per session,
  so a run's page can show the ones from its own sessions."""
  context = {key: body.get(key) for key in ("session_id", "process_uuid", "sdk_version", "platform")}
  session_id, events = body.get("session_id"), body.get("events")
  for event in [e for e in events if isinstance(e, dict)] if isinstance(events, list) else []:
    event = redact(event)
    severity = str(event.get("severity") or "INFO").upper()
    message = f"tinker {event.get('event_name') or event.get('event')}"
    print(json.dumps({"severity": severity, "message": message, **context, "event": event}, default=str), flush=True)
    if severity in ("ERROR", "CRITICAL") and isinstance(session_id, str) and session_id:
      reason, text = describe(event)
      try:
        at = datetime.fromisoformat(str(event.get("timestamp"))).timestamp()
      except ValueError:
        at = time.time()
      sample = {"at": at, "severity": severity, "reason": reason, "message": text[:2000]}
      await store.append_sample(sdk_errors_key(session_id), sample, SDK_ERROR_TAIL, TTL_SECONDS)


# *** OpenTelemetry ***


def configure_tracing(service_name: str) -> TracerProvider:
  """Install the global TracerProvider with each exporter named in OTEL_TRACES_EXPORTER."""
  tracer = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
  for name in filter(None, (n.strip() for n in os.environ.get("OTEL_TRACES_EXPORTER", "").split(","))):
    if name == "none":
      continue
    if not (eps := importlib.metadata.entry_points(group="opentelemetry_traces_exporter", name=name)):
      raise ValueError(f"Unknown OTEL_TRACES_EXPORTER: {name}")
    tracer.add_span_processor(BatchSpanProcessor(next(iter(eps)).load()()))
  trace.set_tracer_provider(tracer)
  return tracer
