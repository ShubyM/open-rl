"""What the workers were doing, recorded by the workers themselves: a bounded
list of finished ops per run, and one of exclusive GPU turns per workload with
the ops that ran inside each. Recording is best effort: it never makes an op
wait on the store, and never changes what the op returns or raises."""

import asyncio
import contextvars
import math
import os
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager, contextmanager, suppress

from opentelemetry import trace

from server.store import get_state_store

SAMPLE_LIMIT = 2000
TTL_SECONDS = 7 * 86400  # as long as placement history
MAX_TURN_OPS = 64

# The ops of the GPU turn the current task is inside, or None outside a turn.
# Tasks created inside a turn copy the context, so concurrent sample requests
# land in the turn that launched them.
_turn_ops: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar("openrl_turn_ops", default=None)
# In-flight writes, referenced so they are not garbage collected mid-write.
_writes: set[asyncio.Task] = set()


def key(run_id: str) -> str:
  return f"open_rl:operations:{run_id}"


def turn_key(workload: str) -> str:
  return f"open_rl:turns:{workload}"


def steps_key(run_id: str) -> str:
  return f"open_rl:steps:{run_id}"


def worker_identity() -> dict:
  """The pod identity the dashboard joins ops and GPU turns to placements by."""
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


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


def record(request: dict, role: str, runtime_id: str | None, started_at: float, status: str, error_type: str | None, result: dict) -> None:
  # No request payloads, outputs or exception messages are recorded.
  run_id = request.get("logical_run_id") or request.get("model_id")
  operation = str(request.get("op") or ("sample" if role == "sampler" else "unknown"))
  at = time.time()
  add_to_turn(operation, started_at, at, run_id=run_id, request_id=request.get("request_id"), status=status)
  if not run_id:
    return
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
    **worker_identity(),
    "metrics": {name: value for name, value in (result.get("metrics") or {}).items() if isinstance(value, int | float) and math.isfinite(value)},
  }
  span = trace.get_current_span().get_span_context()
  if span.is_valid:
    sample.update(trace_id=f"{span.trace_id:032x}", span_id=f"{span.span_id:016x}")
  write(get_state_store().append_sample(key(run_id), sample, SAMPLE_LIMIT, TTL_SECONDS))
  if operation == "optim_step" and status == "succeeded":
    write(get_state_store().increment(steps_key(run_id), TTL_SECONDS))


def add_to_turn(name: str, start: float, end: float, run_id: str | None = None, request_id: str | None = None, status: str = "succeeded") -> None:
  """Record an op inside the current GPU turn. Concurrent ops of the same name
  and run (a batch of sample requests) merge into one with a count."""
  ops = _turn_ops.get()
  if ops is None:
    return
  for op in reversed(ops):
    if op["name"] == name and op["run_id"] == run_id and start < op["end"]:
      op.update(start=min(start, op["start"]), end=max(end, op["end"]), count=op["count"] + 1)
      if status != "succeeded":
        op["status"] = status
      return
  if len(ops) < MAX_TURN_OPS:
    ops.append({"name": name, "start": start, "end": end, "count": 1, "status": status, "run_id": run_id, "request_id": request_id})


@contextmanager
def record_op(name: str):
  """Record worker work that is not an API request (init, wake_up, sleep,
  weight_sync) as an op of the current GPU turn, like the requests around it."""
  start, status = time.time(), "failed"
  try:
    yield
    status = "succeeded"
  finally:
    add_to_turn(name, start, time.time(), status=status)


@asynccontextmanager
async def gpu_turn(time_slicer, workload, role: str, runtime_id: str | None):
  """Hold the GPU through the time-slicer and record the turn: the interval the
  device was ours, which by construction never overlaps another workload's,
  and the ops that ran inside it."""
  requested = time.time()
  async with time_slicer.acquire(workload):
    started, ops = time.time(), []
    token = _turn_ops.set(ops)
    try:
      yield
    finally:
      _turn_ops.reset(token)
      turn = {
        "at": time.time(),
        "started_at": started,
        "requested_at": requested,
        "operation": "gpu_turn",
        "workload": workload.name,
        "role": role,
        "runtime_id": runtime_id,
        **worker_identity(),
        "ops": ops,
      }
      with suppress(Exception):
        write(get_state_store().append_sample(turn_key(workload.name), turn, SAMPLE_LIMIT, TTL_SECONDS))
