"""What OpenRL records about itself, where it happens, for the dashboard to read. Recording is best effort: it never makes
an op wait on the store, and never changes what the op returns or raises."""

import asyncio
import json
import math
import os
import re
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from numbers import Real

from fastapi import HTTPException
from pydantic import BaseModel, Field

from server.store import get_state_store

SAMPLE_LIMIT = int(os.getenv("OPEN_RL_OPS_SAMPLE_LIMIT", "2000"))
TTL_SECONDS = float(os.getenv("OPEN_RL_OPS_TTL_DAYS", "7")) * 86400 or None  # 0 keeps records
HEARTBEAT_SECONDS, STALE_SECONDS, MAX_TURNS = 5, 30, 200  # a record not rewritten for STALE_SECONDS is a gone process
CLIENT, RUN_LABEL, SOURCE_LABEL = "client", "openrl_run_id", "openrl_source"  # Cloud Logging labels on client lines
SECRET = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|secret|password", re.I)
CONFIG_ENV = ("OPEN_RL_", "OPENRL_", "VLLM_", "NCCL_", "CUDA_", "PYTORCH_", "TORCH_")
OPEN_PLACEMENTS = "open_rl:placements:open"  # models holding a placement not yet released


def key(*parts: str) -> str:
  """operations:<run>, operations:<run>:samples, worker:<id>, steps:<run>, ledger:<run>:<event>, session_record:<id>, ..."""
  return ":".join(("open_rl", *parts))


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


current: "Worker | None" = None  # this process's record while it serves
writes: set[asyncio.Task] = set()  # in-flight writes, held so they are not collected mid-write
steps_done: dict[str, int] = {}  # run id -> optim_steps it had completed, stamped on each of its trainer ops


def identity() -> dict:
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


def gpu_uuids() -> list[str]:
  """The GPUs this process can see, in CUDA order, as the UUIDs DCGM labels its series with."""
  try:
    import pynvml

    pynvml.nvmlInit()
    uuids = [pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(pynvml.nvmlDeviceGetCount())]
  except Exception:
    return []
  visible = os.getenv("CUDA_VISIBLE_DEVICES")
  return [gpu for i, gpu in enumerate(uuids) if visible is None or str(i) in visible.split(",") or gpu in visible]


def alive(record: dict, now: float) -> bool:
  return record.get("phase") != "exited" and now - record.get("heartbeat_at", 0) < STALE_SECONDS


def operation_name(request: dict, role: str) -> str:
  # A forward-only call travels as forward_backward; its loss must not mix with training losses.
  return "forward" if request.get("forward_only") else str(request.get("op") or ("sample" if role == "sampler" else "unknown"))


def run_of(request: dict) -> str | None:
  return request.get("logical_run_id") or request.get("model_id")


class Worker:
  def __init__(self, role: str, runtime_id: str | None, fields: dict) -> None:
    now = time.time()
    # Per process, not per pod: a container restarted in place must not overwrite what its predecessor held.
    self.record = {"id": uuid.uuid4().hex[:16], "node": socket.gethostname(), **identity(), "role": role, "runtime_id": runtime_id}
    self.record.update(gpus=gpu_uuids(), started_at=now, phase="starting", phase_since=now, turns=[], **fields)
    # The settings the scheduler passed as env: what this process actually runs with.
    self.record["config"] = {"env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(CONFIG_ENV) and not SECRET.search(k)}}
    self.outstanding: dict[str, dict] = {}  # request_id -> op, run_id and when it was popped


@asynccontextmanager
async def serving(role: str, runtime_id: str | None, **fields):
  """Keep this process's record fresh while the body runs; a clean exit says so."""
  global current
  current = worker = Worker(role, runtime_id, fields)

  async def save():
    groups: dict[tuple, dict] = {}  # outstanding requests by op and run, the oldest of each first
    for entry in worker.outstanding.values():
      groups.setdefault((entry["op"], entry["run_id"]), {**entry, "count": 0})["count"] += 1
    with suppress(Exception):
      value = json.dumps({**worker.record, "heartbeat_at": time.time(), "outstanding": list(groups.values())})
      await asyncio.wait_for(get_state_store().set_value(key("worker", worker.record["id"]), value, ttl_seconds=TTL_SECONDS), timeout=5)

  async def heartbeat():
    while True:
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


def answered(request_id: str) -> None:
  """The request's answer is written, so it is no longer outstanding."""
  if current is not None and (entry := current.outstanding.pop(request_id, None)):
    count("answered", entry["run_id"])


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


def write(coroutine: Coroutine) -> None:
  """Write in the background; a write that fails or times out is dropped."""
  task = asyncio.create_task(asyncio.wait_for(coroutine, timeout=5))
  writes.add(task)
  task.add_done_callback(lambda t: (writes.discard(t), t.cancelled() or t.exception()))


def count(event: str, run_id: str | None) -> None:
  """One more request of this run reached `event`; a request enqueued but never popped shows up as a gap."""
  if run_id:
    write(get_state_store().increment(key("ledger", run_id, event), TTL_SECONDS))


async def observe_operation(request: dict, role: str, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Run one API op and record how it went; returns or raises exactly as the op does."""
  if (run_id := run_of(request)) and run_id not in steps_done:
    with suppress(Exception):
      steps_done[run_id] = int(await get_state_store().get_value(key("steps", run_id)) or 0)
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


def record(request: dict, role: str, runtime_id: str | None, started_at: float, status: str, error_type: str | None, result: dict, where=None):
  """`where` names the pod the op ran on, when not this process's. No payloads, outputs or error messages are kept."""
  if not (run_id := run_of(request)):
    return
  operation, at = operation_name(request, role), time.time()
  values = {k: float(v) for k, v in (result.get("metrics") or {}).items() if isinstance(v, Real) and not isinstance(v, bool)}
  sample = {"at": at, "started_at": started_at, "operation": operation, "request_id": request.get("request_id"), "run_id": run_id}
  sample.update(runtime_id=runtime_id, role=role, status=status, error_type=error_type, elapsed_seconds=at - started_at)
  sample.update(identity() if where is None else where, metrics={k: v for k, v in values.items() if math.isfinite(v)})
  if operation != "sample" and run_id in steps_done:
    sample["step"] = steps_done[run_id]
  # JSON has no NaN or infinity; name them so a NaN grad norm is not just missing.
  if nonfinite := sorted(k for k, v in values.items() if not math.isfinite(v)):
    sample["nonfinite"] = nonfinite
  # Samples keep their own list, so a run's sampling never evicts its training ops.
  list_key = key("operations", run_id, "samples") if operation == "sample" else key("operations", run_id)
  write(get_state_store().append_sample(list_key, sample, SAMPLE_LIMIT, TTL_SECONDS))
  if operation == "optim_step" and status == "succeeded":
    write(get_state_store().increment(key("steps", run_id), TTL_SECONDS))
    if run_id in steps_done:
      steps_done[run_id] += 1


async def note_session(store, session_id: str, *, created: bool = False, model_id: str | None = None, metadata: dict | None = None) -> None:
  """When a session started and was last heard from, and its models; it outlives the live key's two-minute expiry."""
  record_key, now = key("session_record", session_id), time.time()
  record = json.loads(await store.get_value(record_key) or "null") or {"session_id": session_id, "created_at": now, "model_ids": []}
  record["last_seen_at"] = now
  if metadata is not None:
    record["metadata"] = redact(metadata)
  event = "created" if created else None
  if model_id and model_id not in record["model_ids"]:
    record["model_ids"].append(model_id)
    event = "attached"
  await store.set_value(record_key, json.dumps(record), ttl_seconds=TTL_SECONDS)
  if event:
    labels = {SOURCE_LABEL: "openrl_event", **({RUN_LABEL: model_id} if model_id else {})}
    line = {"kind": "openrl.session", "v": 1, "event": event, "session_id": session_id, "model_id": model_id, "time": iso(now)}
    print(json.dumps({"severity": "INFO", "message": "openrl.session", "openrl": line, "logging.googleapis.com/labels": labels}), flush=True)


def epoch(stamp: str | None) -> float | None:
  return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp() if stamp else None


async def record_placements(store, workloads: list[dict]) -> None:
  """One record per (Workload, placedAt), kept after the Workload; a gone or re-placed Workload's open record is released now."""
  now, live = time.time(), {}
  for item in workloads:
    spec, status = item["spec"], item.get("status") or {}
    if status.get("placedAt"):  # the scheduler stamps it on bind
      name = item["metadata"]["name"]
      live[name] = {"workload": name, "role": spec["role"], "model_id": spec["modelID"], "owner": spec.get("ownerID"), "pod": status.get("podName")}
      live[name].update(node=status.get("nodeName"), gpu_uuids=status.get("gpuUUIDs") or [], placed_at=epoch(status["placedAt"]))
      live[name]["released_at"] = epoch(status.get("releasedAt"))
  opened = await store.set_members(OPEN_PLACEMENTS)
  for model_id in opened | {p["model_id"] for p in live.values()}:
    records = await store.read_samples(key("placements", model_id))
    fresh = {name: p for name, p in live.items() if p["model_id"] == model_id}
    updated = []
    for old in records:
      if (new := fresh.get(old["workload"])) and new["placed_at"] == old["placed_at"]:
        old = fresh.pop(old["workload"])
      elif old["released_at"] is None:
        old = {**old, "released_at": now}
      updated.append(old)
    updated += fresh.values()
    if updated != records:
      await store.replace_samples(key("placements", model_id), updated, TTL_SECONDS)
    held = any(r["released_at"] is None for r in updated)
    if held and model_id not in opened:
      await store.add_to_set(OPEN_PLACEMENTS, model_id)
    elif not held and model_id in opened:
      await store.remove_from_set(OPEN_PLACEMENTS, model_id)


class LogLine(BaseModel):
  message: str = Field(max_length=16384)
  severity: str = "INFO"
  at: float | None = None


class LogRequest(BaseModel):
  lines: list[LogLine] = Field(max_length=1000)


async def ingest_logs(store, run_id: str, body: LogRequest) -> dict:
  """Each line goes to stdout labelled with its run, so Cloud Logging keeps it beside the run's worker logs."""
  if not await store.get_value(f"open_rl:model_meta:{run_id}"):
    raise HTTPException(404, f"No run {run_id}; a run is its model, so send the model id the client trains")
  now = time.time()
  for line in body.lines:
    severity = line.severity.upper() if line.severity.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL") else "INFO"
    at = line.at if line.at is not None and abs(line.at - now) < 86400 else now  # a client clock is trusted within a day
    labels = {RUN_LABEL: run_id, SOURCE_LABEL: CLIENT}
    print(json.dumps({"severity": severity, "message": line.message, "time": iso(at), "logging.googleapis.com/labels": labels}), flush=True)
    await store.append_sample(f"open_rl:run_logs:{run_id}", {"at": at, "severity": severity, "message": line.message}, 5000, TTL_SECONDS)
  return {"accepted": len(body.lines)}


def redact(value):
  # The SDK attaches full request and response headers to fatal errors, and the request carries the API key.
  if isinstance(value, dict):
    return {k: "[redacted]" if SECRET.search(str(k)) else redact(v) for k, v in value.items() if k not in ("request_headers", "response_headers")}
  return [redact(v) for v in value] if isinstance(value, list) else value


async def ingest_sdk_events(store, body: dict) -> None:
  """The tinker SDK's own telemetry: one redacted JSON line per event, and error events kept per session."""
  context = {k: body.get(k) for k in ("session_id", "process_uuid", "sdk_version", "platform")}
  session_id, events = body.get("session_id"), body.get("events")
  for event in [redact(e) for e in events if isinstance(e, dict)] if isinstance(events, list) else []:
    severity = str(event.get("severity") or "INFO").upper()
    message = f"tinker {event.get('event_name') or event.get('event')}"
    print(json.dumps({"severity": severity, "message": message, **context, "event": event}, default=str), flush=True)
    if severity in ("ERROR", "CRITICAL") and isinstance(session_id, str) and session_id:
      data = event.get("event_data") if isinstance(event.get("event_data"), dict) else {}
      name = str(event.get("error_type") or event.get("event_name") or event.get("event") or "event").rsplit(".", 1)[-1]
      status = f"HTTP {data['status_code']}: " if data.get("status_code") else ""
      text = status + str(event.get("error_message") or data.get("detail") or data.get("exception") or data.get("message") or "")
      at = time.time()
      with suppress(ValueError):
        at = datetime.fromisoformat(str(event.get("timestamp"))).timestamp()
      sample = {"at": at, "severity": severity, "reason": f"SDK {name}", "message": text[:2000]}
      await store.append_sample(key("sdk_errors", session_id), sample, 200, TTL_SECONDS)
