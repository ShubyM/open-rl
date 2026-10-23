"""What OpenRL records about itself, where it happens, for the dashboard to read. Recording is best effort: no hook raises
into the code it observes or waits on the store, and none changes what an op returns or raises."""

import asyncio
import functools
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from numbers import Real

from server.store import get_state_store

SAMPLE_LIMIT, STEP_TIMES = int(os.getenv("OPEN_RL_OPS_SAMPLE_LIMIT", "2000")), 64
TTL_SECONDS = float(os.getenv("OPEN_RL_OPS_TTL_DAYS", "7")) * 86400 or None  # 0 keeps records
HEARTBEAT_SECONDS, STALE_SECONDS, MAX_TURNS = 5, 30, 200  # a record not rewritten for STALE_SECONDS is a gone process
WRITE_SECONDS, FINISH_SECONDS = 5, 5  # how long one write, and everything at exit, may take before it is dropped
CLIENT, RUN_LABEL, SOURCE_LABEL = "client", "openrl_run_id", "openrl_source"  # Cloud Logging labels on client lines
SECRET = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|secret|password", re.I)
CONFIG_ENV = ("OPEN_RL_", "OPENRL_", "VLLM_", "NCCL_", "CUDA_", "PYTORCH_", "TORCH_")
OPEN_PLACEMENTS = "open_rl:placements:open"  # models holding a placement not yet released


def key(*parts: str) -> str:
  """operations:<run>, operations:<run>:samples, worker:<id>, steps:<run>, ledger:<run> (a hash of counts), session_record:<id>, ..."""
  return ":".join(("open_rl", *parts))


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


current: "Worker | None" = None  # this process's record while it serves
writes: set[asyncio.Task] = set()  # in-flight writes, held so they are not collected mid-write
step_base: dict[str, asyncio.Future] = {}  # run id -> optim_steps the run had completed when this process first ran one of its ops
steps_here: defaultdict[str, int] = defaultdict(int)  # run id -> optim_steps this process has completed for it since
session_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)  # the API server is one replica
placements_lock = asyncio.Lock()


def best_effort(hook):
  """A hook that never raises into the code it observes."""

  @functools.wraps(hook)
  def quiet(*args, **kwargs):
    with suppress(Exception):
      hook(*args, **kwargs)

  return quiet


@best_effort
def write(coroutine: Coroutine, timeout: float | None = None) -> None:
  """Run a store write in the background; one that fails or takes longer than timeout (WRITE_SECONDS) is dropped."""
  try:
    task = asyncio.get_running_loop().create_task(asyncio.wait_for(coroutine, timeout or WRITE_SECONDS))
  except RuntimeError:
    coroutine.close()  # no loop to run it on
    raise
  writes.add(task)
  task.add_done_callback(lambda t: (writes.discard(t), t.cancelled() or t.exception()))


def identity() -> dict:
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


def gpu_uuids() -> list[str] | None:
  """The GPUs this process can see, in CUDA order, as the UUIDs DCGM labels its series with. None when NVML cannot be
  read. pynvml comes with vLLM, so a trainer may lack it, which says nothing about how many GPUs it has."""
  try:
    import pynvml

    pynvml.nvmlInit()
    uuids = [pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(pynvml.nvmlDeviceGetCount())]
  except Exception:
    return None
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
  def __init__(self, role: str, runtime_id: str | None) -> None:
    now = time.time()
    # Per process, not per pod: a container restarted in place must not overwrite what its predecessor held.
    self.record = {"id": uuid.uuid4().hex[:16], "node": socket.gethostname(), **identity(), "role": role, "runtime_id": runtime_id}
    self.record.update(gpus=gpu_uuids(), started_at=now, phase="starting", phase_since=now, turns=[])
    # The settings the scheduler passed as env: what this process actually runs with.
    self.record["config"] = {"env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(CONFIG_ENV) and not SECRET.search(k)}}
    self.outstanding: dict[str, dict] = {}  # request_id -> op, run_id and when it was popped
    self.observed: set[str] = set()  # outstanding requests whose op ran under observe_operation, which records it
    # A thread, so loading a model on the event loop does not make this worker look dead.
    self.stopped = threading.Event()
    self.heartbeat = threading.Thread(target=self.beat, name="worker-heartbeat", daemon=True)

  def snapshot(self) -> str:
    """The record as JSON. Copies first, because the heartbeat thread reads while the event loop writes."""
    groups: dict[tuple, dict] = {}  # outstanding requests by op and run, the oldest of each first
    for entry in list(self.outstanding.values()):
      groups.setdefault((entry["op"], entry["run_id"]), {**entry, "count": 0})["count"] += 1
    return json.dumps({**self.record, "heartbeat_at": time.time(), "outstanding": list(groups.values())})

  def save(self) -> None:
    write(get_state_store().set_value(key("worker", self.record["id"]), self.snapshot(), ttl_seconds=TTL_SECONDS))

  def beat(self) -> None:
    while not self.stopped.wait(HEARTBEAT_SECONDS):
      with suppress(Exception):
        get_state_store().set_value_sync(key("worker", self.record["id"]), self.snapshot(), ttl_seconds=TTL_SECONDS)


@best_effort
def start(role: str, runtime_id: str | None) -> None:
  global current
  worker = Worker(role, runtime_id)
  worker.save()
  worker.heartbeat.start()
  current = worker


@asynccontextmanager
async def serving(role: str, runtime_id: str | None):
  """Keep this process's record fresh while the body runs; a clean exit says so."""
  start(role, runtime_id)
  try:
    yield current
  finally:
    await finish()


async def finish() -> None:
  """Say this process exited and land the writes in flight, within FINISH_SECONDS. Also called before an os._exit."""
  global current
  worker, current = current, None

  async def close() -> None:
    if worker is not None:
      worker.stopped.set()  # stop the heartbeat first, so no late beat overwrites the exit
      await asyncio.to_thread(worker.heartbeat.join, WRITE_SECONDS)
      worker.record.update(phase="exited", phase_since=time.time())
      worker.save()
    await asyncio.gather(*list(writes), return_exceptions=True)

  with suppress(Exception):
    await asyncio.wait_for(close(), FINISH_SECONDS)


@best_effort
def set_phase(phase: str) -> None:
  if current is not None and current.record["phase"] != phase:
    current.record.update(phase=phase, phase_since=time.time())


@best_effort
def popped(requests: list[dict]) -> None:
  """Requests this worker took from its queue; each stays outstanding until answered. A shared trainer's delete_model
  is queued without being counted as enqueued, so it is not counted here either."""
  if current is not None:
    for r in requests:
      if r.get("request_id") and r.get("op") != "delete_model":
        current.outstanding[r["request_id"]] = {"op": operation_name(r, current.record["role"]), "run_id": run_of(r), "since": time.time()}
        count("popped", run_of(r))


@best_effort
def answered(request_id: str, result: dict | None = None) -> None:
  """The request's answer is written, so it is no longer outstanding. A failure answered without its op running, such as a
  wake or parse failure, gets its record here, so every failed answer has one."""
  if current is None or (entry := current.outstanding.pop(request_id, None)) is None:
    return
  count("answered", entry["run_id"])
  if request_id not in current.observed and (result or {}).get("type") == "RequestFailedResponse":
    request = {"op": entry["op"], "request_id": request_id, "logical_run_id": entry["run_id"]}
    record(request, current.record["role"], current.record["runtime_id"], entry["since"], "RequestFailedResponse")
  current.observed.discard(request_id)


@asynccontextmanager
async def gpu_turn():
  """Show a GPU turn on this worker's record. Enter it right after time_slicer.acquire, so the turn is the hold."""
  started = time.time()
  set_phase("on_gpu")
  try:
    yield
  finally:
    with suppress(Exception):  # no worker record outside a served process
      current.record["turns"] = [*current.record["turns"][1 - MAX_TURNS :], [started, time.time()]]
    set_phase("ready")


@best_effort
def count(event: str, run_id: str | None) -> None:
  """One more request of this run reached `event`; a request enqueued but never popped shows up as a gap."""
  if run_id:
    write(get_state_store().increment_field(key("ledger", run_id), event, TTL_SECONDS))


@best_effort
def observing(request: dict, role: str) -> None:
  """An op is about to run. Starts reading the run's step count, which its trainer records carry, without waiting on it."""
  if current is not None and request.get("request_id") in current.outstanding:
    current.observed.add(request["request_id"])
  if (run_id := run_of(request)) and operation_name(request, role) != "sample" and run_id not in step_base:
    step_base[run_id] = asyncio.ensure_future(asyncio.wait_for(get_state_store().get_value(key("steps", run_id)), WRITE_SECONDS))
    step_base[run_id].add_done_callback(lambda f: f.cancelled() or f.exception())


async def observe_operation(request: dict, role: str, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Run one API op and record how it went; returns or raises exactly as the op does."""
  if os.getenv("RANK", "0") != "0":  # torchrun ranks run the same op; rank 0 records it
    return await call()
  started_at = time.time()
  observing(request, role)
  try:
    result = await call()
  except Exception as error:
    record(request, role, runtime_id, started_at, type(error).__name__)
    raise
  record(request, role, runtime_id, started_at, None, result)
  return result


def numbers(metrics: dict | None) -> dict[str, float]:
  """The metric values that are numbers, as floats; a bool is not one."""
  return {k: float(v) for k, v in (metrics or {}).items() if isinstance(v, Real) and not isinstance(v, bool)}


def finite(metrics: dict | None) -> dict[str, float]:
  return {k: v for k, v in numbers(metrics).items() if math.isfinite(v)}


async def append_trainer_op(run_id: str, sample: dict, done_here: int) -> None:
  """Stamp the step the op belongs to, once the run's count is read, and keep the record. An unread count leaves it off."""
  with suppress(Exception):
    if run_id in step_base:
      sample["step"] = int(await step_base[run_id] or 0) + done_here
  await get_state_store().append_sample(key("operations", run_id), sample, SAMPLE_LIMIT, TTL_SECONDS)


@best_effort
def record(request: dict, role: str, runtime_id: str | None, started_at: float, error_type: str | None = None, result: dict | None = None):
  """One command record: failed with error_type, or as the result says. No payloads, outputs or error messages are kept."""
  if not (run_id := run_of(request)):
    return
  result = result if isinstance(result, dict) else {}
  if error_type is None and result.get("type") == "RequestFailedResponse":
    error_type = "RequestFailedResponse"
  operation, at, status = operation_name(request, role), time.time(), "failed" if error_type else "succeeded"
  values = numbers(result.get("metrics"))
  sample = {"at": at, "started_at": started_at, "operation": operation, "request_id": request.get("request_id"), "run_id": run_id}
  sample.update(runtime_id=runtime_id, role=role, status=status, error_type=error_type, elapsed_seconds=at - started_at)
  sample.update(identity(), metrics=finite(values))
  # JSON has no NaN or infinity; name them so a NaN metric is not just missing.
  if nonfinite := sorted(k for k, v in values.items() if not math.isfinite(v)):
    sample["nonfinite"] = nonfinite
  store = get_state_store()
  # Samples keep their own list, so a run's sampling never evicts its training ops.
  if operation == "sample":
    write(store.append_sample(key("operations", run_id, "samples"), sample, SAMPLE_LIMIT, TTL_SECONDS))
    return
  write(append_trainer_op(run_id, sample, steps_here[run_id]), 2 * WRITE_SECONDS)  # the step read may take one WRITE_SECONDS
  # A run's one-line form reads these few records instead of its command history.
  if status == "failed":
    count("failed", run_id)
  if operation == "optim_step" and status == "succeeded":
    steps_here[run_id] += 1
    write(store.increment(key("steps", run_id), TTL_SECONDS))
    write(store.append_sample(key("step_times", run_id), at, STEP_TIMES, TTL_SECONDS))


async def update_session(store, session_id: str, created: bool, model_id: str | None, metadata: dict | None) -> None:
  record_key, now = key("session_record", session_id), time.time()
  # Heartbeats and attaches rewrite the same record; one at a time, or a heartbeat drops a model attached meanwhile.
  async with session_locks[session_id]:
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


@best_effort
def note_session(store, session_id: str, *, created: bool = False, model_id: str | None = None, metadata: dict | None = None) -> None:
  """When a session started and was last heard from, and its models; it outlives the live key's two-minute expiry."""
  write(update_session(store, session_id, created, model_id, metadata))


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
  async with placements_lock:  # sweeps run in the background and may overlap
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
      held = any(r["released_at"] is None for r in updated)
      if updated != records:
        await store.replace_samples(key("placements", model_id), updated, TTL_SECONDS)
      elif held and TTL_SECONDS:  # a placement held longer than the TTL must outlive it
        await store.expire(key("placements", model_id), TTL_SECONDS)
      if held and model_id not in opened:
        await store.add_to_set(OPEN_PLACEMENTS, model_id)
      elif not held and model_id in opened:
        await store.remove_from_set(OPEN_PLACEMENTS, model_id)


def redact(value):
  # The SDK attaches full request and response headers to fatal errors, and the request carries the API key.
  if isinstance(value, dict):
    return {k: "[redacted]" if SECRET.search(str(k)) else redact(v) for k, v in value.items() if k not in ("request_headers", "response_headers")}
  return [redact(v) for v in value] if isinstance(value, list) else value


@best_effort
def ingest_sdk_events(store, body: dict) -> None:
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
      write(store.append_sample(key("sdk_errors", session_id), sample, 200, TTL_SECONDS))
