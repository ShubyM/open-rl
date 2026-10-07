"""What OpenRL was doing, recorded by the workers: one bounded list of ops per
run, and one of exclusive GPU turns per workload with the ops that ran inside
each. Recording is best effort and never changes what an op returns or raises."""

import asyncio
import contextlib
import contextvars
import math
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, contextmanager
from itertools import islice

from opentelemetry import context, propagate, trace
from opentelemetry.trace import StatusCode

SAMPLE_LIMIT = 2000


def key(run_id: str) -> str:
  return f"open_rl:operations:{run_id}"


@contextmanager
def operation_span(request: dict):
  span, token = trace.INVALID_SPAN, None
  try:
    parent = propagate.extract(request["trace_context"]) if request.get("trace_context") else None
    span = trace.get_tracer(__name__).start_span("openrl.operation", context=parent)
    token = context.attach(trace.set_span_in_context(span))
  except Exception:
    pass
  try:
    yield span
  finally:
    try:
      if token is not None:
        context.detach(token)
      span.end()
    except Exception:
      pass


async def observe_operation(store, request: dict, role: str, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Return/raise exactly as the operation does; recording is bounded and best effort."""
  started_at, started = time.time(), time.perf_counter()
  result, status, error_type = {}, "succeeded", None
  run_id = request.get("logical_run_id") or request.get("adapter_id") or request.get("lora_id") or request.get("model_id")
  # A sampler ref names a version of a run; its ops belong to the run.
  if isinstance(run_id, str) and run_id.startswith("tinker://"):
    run_id = run_id.removeprefix("tinker://").split("/")[0]
  with operation_span(request) as span:
    try:
      result = await call()
      if result.get("type") == "RequestFailedResponse":
        status, error_type = "failed", "RequestFailedResponse"
      return result
    except asyncio.CancelledError:
      status, error_type = "cancelled", "CancelledError"
      raise
    except Exception as error:
      status, error_type = "failed", type(error).__name__
      raise
    finally:
      elapsed = time.perf_counter() - started
      # No request payloads, output tokens, or exception messages enter samples/spans.
      try:
        operation = str(request.get("op") or ("sample" if role == "sampler" else "unknown"))[:128]
        runtime = request.get("runtime_id") or os.getenv("OPEN_RL_RUNTIME_ID") or runtime_id
        sample = {
          "at": time.time(),
          "started_at": started_at,
          "operation": operation,
          "request_id": request.get("request_id"),
          "run_id": run_id,
          "runtime_id": runtime,
          "role": os.getenv("OPEN_RL_PROCESS_ROLE") or role,
          "status": status,
          "error_type": error_type,
          "elapsed_seconds": elapsed,
          "metrics": {
            name[:128]: value
            for name, value in islice((result.get("metrics") or {}).items(), 32)
            if isinstance(name, str) and isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
          },
        }
        queued = request.get("enqueued_at")
        if isinstance(queued, int | float) and not isinstance(queued, bool) and math.isfinite(queued) and 0 < queued <= started_at:
          sample["enqueued_at"] = queued
          sample["queue_seconds"] = started_at - queued
        for field, env in (("pod_uid", "POD_UID"), ("node", "NODE_NAME")):
          if value := os.getenv(env):
            sample[field] = value
        span_context = span.get_span_context()
        if span_context.is_valid:
          sample.update(trace_id=f"{span_context.trace_id:032x}", span_id=f"{span_context.span_id:016x}")
        for field in ("operation", "run_id", "runtime_id", "request_id", "role", "status", "error_type"):
          if sample.get(field) is not None:
            span.set_attribute(f"openrl.{field}", str(sample[field]))
        if status != "succeeded":
          span.set_status(StatusCode.ERROR)
        add_to_turn(operation, started_at, sample["at"], run_id=run_id, request_id=sample["request_id"], status=status)
        if run_id:
          await asyncio.wait_for(store.append_sample(key(run_id), sample, SAMPLE_LIMIT), timeout=0.1)
      except Exception:
        pass


async def read(store, run_id: str) -> dict:
  try:
    samples = await asyncio.wait_for(store.read_samples(key(run_id)), timeout=1)
    return {
      "available": bool(samples),
      "samples": samples,
      "coverage": "Last 2000 finished operations; best effort, not exhaustive",
    }
  except Exception:
    return {"available": False, "samples": [], "error": "Operation telemetry unavailable"}


TURN_KEY_PREFIX = "open_rl:turns:"
MAX_TURN_OPS = 64

# The ops of the GPU turn the current task is inside, or None outside a turn.
# Tasks created inside a turn copy the context, so concurrent sample requests
# land in the turn that launched them.
_turn_ops: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar("openrl_turn_ops", default=None)


def turn_key(workload: str) -> str:
  return TURN_KEY_PREFIX + workload


def add_to_turn(name: str, start: float, end: float, run_id: str | None = None, request_id: str | None = None, status: str = "succeeded") -> None:
  """Record an op inside the current GPU turn. Concurrent ops of the same name
  and run (a batch of sample requests) merge into one with a count."""
  ops = _turn_ops.get()
  if ops is None:
    return
  for op in reversed(ops):
    if op["name"] == name and op.get("run_id") == run_id and start < op["end"]:
      op.update(start=min(start, op["start"]), end=max(end, op["end"]), count=op["count"] + 1)
      if status != "succeeded":
        op["status"] = status
      return
  if len(ops) >= MAX_TURN_OPS:
    return
  op = {"name": name[:64], "start": start, "end": end, "count": 1, "status": status}
  if run_id:
    op["run_id"] = run_id
  if request_id:
    op["request_id"] = request_id
  ops.append(op)


@contextmanager
def record_op(name: str):
  """Record a block of worker work that is not an API request (wake_up, sleep,
  weight_sync) as an op of the current GPU turn, like the requests around it."""
  start, status = time.time(), "succeeded"
  try:
    yield
  except BaseException:
    status = "failed"
    raise
  finally:
    add_to_turn(name, start, time.time(), status=status)


@asynccontextmanager
async def gpu_turn(time_slicer, workload, store, role: str, runtime_id: str | None):
  """Hold the GPU through the time-slicer and record the turn: the interval the
  device was ours, which by construction never overlaps another workload's,
  and the ops that ran inside it. Without a time-slicer the device is always
  ours, and the turn is just the work, recorded under the pod's Workload."""
  requested = time.time()
  name = workload.name if workload is not None else os.getenv("OPEN_RL_TIME_SLICE_JOB_ID")
  if not name:
    yield
    return
  async with time_slicer.acquire(workload) if time_slicer is not None else contextlib.nullcontext():
    started = time.time()
    ops: list[dict] = []
    token = _turn_ops.set(ops)
    try:
      yield
    finally:
      _turn_ops.reset(token)
      sample = {
        "at": time.time(),
        "started_at": started,
        "requested_at": requested,
        "operation": "gpu_turn",
        "workload": name,
        "role": os.getenv("OPEN_RL_PROCESS_ROLE") or role,
        "runtime_id": runtime_id,
        "node": os.getenv("NODE_NAME"),
        "pod_uid": os.getenv("POD_UID"),
        "ops": ops,
      }
      try:
        await asyncio.wait_for(store.append_sample(turn_key(name), sample, SAMPLE_LIMIT), timeout=0.1)
      except Exception:
        pass


async def read_turns(store, workload: str) -> list[dict]:
  try:
    return await asyncio.wait_for(store.read_samples(turn_key(workload)), timeout=1)
  except Exception:
    return []
