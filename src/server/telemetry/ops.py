"""What the workers were doing, recorded by the workers themselves.

Each worker process keeps one record about itself, open_rl:worker:<id>: where it
runs, which GPUs it has, whether it is waiting for or holding the GPU, the
requests it has popped and not yet answered, and its recent GPU turns. It
rewrites the record every few seconds, so a record that stops changing belongs
to a process that exited or died. Each finished op is also appended to a bounded
list for the run it served, or for the runtime when it served no single run.

Recording is best effort: it never makes an op wait on the store, and never
changes what the op returns or raises."""

import asyncio
import json
import math
import os
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, contextmanager, suppress

from opentelemetry import trace

from server.store import get_state_store

SAMPLE_LIMIT = 2000
TTL_SECONDS = 7 * 86400
HEARTBEAT_SECONDS = 5
# A record not rewritten for this long belongs to a process that is gone.
STALE_SECONDS = 6 * HEARTBEAT_SECONDS
MAX_TURNS = 200

# This process's record while it serves, or None outside a worker.
current: "Worker | None" = None
# In-flight writes, referenced so they are not garbage collected mid-write.
_writes: set[asyncio.Task] = set()


def key(run_id: str) -> str:
  return f"open_rl:operations:{run_id}"


def steps_key(run_id: str) -> str:
  return f"open_rl:steps:{run_id}"


def worker_key(worker_id: str) -> str:
  return f"open_rl:worker:{worker_id}"


def worker_identity() -> dict:
  """The pod identity ops are joined to their worker by."""
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


def gpu_uuids() -> list[str]:
  """The GPUs this process can see, as the UUIDs DCGM labels its series with."""
  try:
    import pynvml

    pynvml.nvmlInit()
    uuids = [pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(pynvml.nvmlDeviceGetCount())]
  except Exception:
    return []
  visible = os.getenv("CUDA_VISIBLE_DEVICES")
  return [uuid for i, uuid in enumerate(uuids) if visible is None or str(i) in visible.split(",") or uuid in visible]


def alive(record: dict, now: float) -> bool:
  """Whether a worker record still belongs to a running process."""
  return record.get("phase") != "exited" and now - record.get("heartbeat_at", 0) < STALE_SECONDS


def operation_name(request: dict, role: str) -> str:
  return str(request.get("op") or ("sample" if role == "sampler" else "unknown"))


def run_of(request: dict) -> str | None:
  return request.get("logical_run_id") or request.get("model_id")


class Worker:
  """The record this process keeps about itself, and the requests it has popped and not answered."""

  def __init__(self, role: str, runtime_id: str | None) -> None:
    identity, now = worker_identity(), time.time()
    self.record = {
      # Per process, not per pod: a container restarted in place must not overwrite the requests its predecessor held.
      "id": uuid.uuid4().hex[:16],
      "node": socket.gethostname(),
      **identity,
      "role": role,
      "runtime_id": runtime_id,
      "gpus": gpu_uuids(),
      "started_at": now,
      "phase": "starting",
      "phase_since": now,
      "turns": [],
    }
    self.outstanding: dict[str, dict] = {}  # request_id -> op, run_id and when it was popped

  def snapshot(self) -> dict:
    """The record as stored, outstanding requests grouped by op and run. Pop order puts each group's oldest first."""
    groups: dict[tuple, dict] = {}
    for entry in self.outstanding.values():
      groups.setdefault((entry["op"], entry["run_id"]), {**entry, "count": 0})["count"] += 1
    return {**self.record, "heartbeat_at": time.time(), "outstanding": list(groups.values())}


@asynccontextmanager
async def serving(role: str, runtime_id: str | None, **fields):
  """Keep this process's record fresh while the body runs; a clean exit says so."""
  global current
  current = worker = Worker(role, runtime_id)
  worker.record.update(fields)

  async def save():
    with suppress(Exception):
      value = json.dumps(worker.snapshot())
      await asyncio.wait_for(get_state_store().set_value(worker_key(worker.record["id"]), value, ttl_seconds=TTL_SECONDS), timeout=5)

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


def note(**fields) -> None:
  """Add what this process knows about its own state to its record."""
  if current is not None:
    current.record.update(fields)


def set_phase(phase: str) -> None:
  if current is not None and current.record["phase"] != phase:
    current.record.update(phase=phase, phase_since=time.time())


def popped(requests: list[dict]) -> None:
  """Note requests this worker took from its queue; each stays outstanding until answered."""
  if current is None:
    return
  for request in requests:
    if request.get("request_id"):
      current.outstanding[request["request_id"]] = {
        "op": operation_name(request, current.record["role"]),
        "run_id": run_of(request),
        "since": time.time(),
      }


class Answering:
  """A request store that takes each request off this worker's outstanding list once its answer is written."""

  def __init__(self, store) -> None:
    self.store = store

  def __getattr__(self, name: str):
    return getattr(self.store, name)

  async def set_future(self, request_id: str, result: dict) -> None:
    await self.store.set_future(request_id, result)
    if current is not None:
      current.outstanding.pop(request_id, None)


def write(coroutine: Coroutine) -> None:
  """Write in the background; a write that fails or times out is dropped."""
  task = asyncio.create_task(asyncio.wait_for(coroutine, timeout=5))
  _writes.add(task)
  task.add_done_callback(written)


def written(task: asyncio.Task) -> None:
  _writes.discard(task)
  if not task.cancelled():
    task.exception()  # retrieved, so a dropped write is not logged as an error


async def observe_operation(request: dict, role: str, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Run one API op and record how it went; returns or raises exactly as the op does."""
  started_at = time.time()
  status, error_type, result = "succeeded", None, {}
  try:
    result = await call()
    if result.get("type") == "RequestFailedResponse":
      status, error_type = "failed", "RequestFailedResponse"
    return result
  except (Exception, asyncio.CancelledError) as error:
    status = "cancelled" if isinstance(error, asyncio.CancelledError) else "failed"
    error_type = type(error).__name__
    raise
  finally:
    with suppress(Exception):
      record(request, role, runtime_id, started_at, status, error_type, result)


def record(
  request: dict, role: str, runtime_id: str | None, started_at: float, status: str, error_type: str | None, result: dict, where: dict | None = None
) -> None:
  """`where` names the pod the op ran on, when that is not this process's own pod.
  No request payloads, outputs or exception messages are recorded."""
  run_id = run_of(request)
  if not run_id:
    return
  operation, at = operation_name(request, role), time.time()
  metrics = {name: value for name, value in (result.get("metrics") or {}).items() if isinstance(value, int | float)}
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
    "metrics": {name: value for name, value in metrics.items() if math.isfinite(value)},
  }
  # JSON has no NaN or infinity; name them so a NaN grad norm is not just missing.
  if nonfinite := sorted(name for name, value in metrics.items() if not math.isfinite(value)):
    sample["nonfinite"] = nonfinite
  span = trace.get_current_span().get_span_context()
  if span.is_valid:
    sample.update(trace_id=f"{span.trace_id:032x}", span_id=f"{span.span_id:016x}")
  write(get_state_store().append_sample(key(run_id), sample, SAMPLE_LIMIT, TTL_SECONDS))
  if operation == "optim_step" and status == "succeeded":
    write(get_state_store().increment(steps_key(run_id), TTL_SECONDS))


@contextmanager
def record_op(name: str):
  """Record worker work that serves no one request (init, wake_up, sleep,
  weight_sync) under the worker's runtime."""
  started_at, status = time.time(), "failed"
  try:
    yield
    status = "succeeded"
  finally:
    if current is not None:
      runtime_id, role = current.record["runtime_id"], current.record["role"]
      with suppress(Exception):
        record({"op": name, "model_id": runtime_id}, role, runtime_id, started_at, status, None, {})


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
