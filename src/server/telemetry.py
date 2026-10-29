"""The records OpenRL keeps about its runs, for the inspection API to read.

A hook on a request path only appends to this process's buffer in memory. It swallows its own errors and returns
nothing, so telemetry never blocks or changes what it observes. One thread writes the buffer to Redis in one pipeline
every FLUSH_SECONDS, with a worker's record of itself, so a failing or hanging Redis costs a flush and nothing more.

  open_rl:operations:<run>          trainer op records, each stamped with the run's step, the newest SAMPLE_LIMIT
  open_rl:operations:<run>:samples  the trainer's sample op records, kept apart so sampling never evicts training ops
  open_rl:steps:<run>               the run's optimizer steps, and open_rl:step_times:<run> when its last STEP_TIMES ended
  open_rl:ledger:<run>              a hash counting the run's trainer requests enqueued, popped, answered and failed
  open_rl:worker:<id>               a worker process's phase, GPU turns and the waits for them, outstanding requests and
                                    settings env
"""

import asyncio
import contextvars
import json
import math
import os
import re
import socket
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from functools import wraps
from numbers import Real

from server.store import Writes, get_state_store
from training.commands import Command

TTL_SECONDS = 7 * 86400
SAMPLE_LIMIT = 2000  # records kept per list
STEP_TIMES = 64
MAX_TURNS = 200
FLUSH_SECONDS = 5
FINISH_SECONDS = 5  # how long an exiting process waits for its last flush
FAILED = "RequestFailedResponse"
SECRET = re.compile(r"authorization|cookie|api[-_]?key|access[-_]?token|secret|password", re.I)
CONFIG_ENV = ("OPEN_RL_", "OPENRL_", "VLLM_", "NCCL_", "CUDA_", "PYTORCH_", "TORCH_")

# Entries waiting for the next flush, oldest first, each a tuple whose first item names its kind:
#   ("op", run, record)      a trainer op record, whose step counts this process's steps until a flush adds the run's
#                            count before them
#   ("sample", run, record)  a sample record
#   ("step", run, at)        an optimizer step that ended at `at`
#   ("count", run, event)    one more of the run's trainer requests reached event
# While flushes fail the oldest go.
buffer: deque[tuple] = deque(maxlen=100_000)


def key(*parts: str) -> str:
  return ":".join(("open_rl", *parts))


def best_effort(hook):
  @wraps(hook)
  def quiet(*args, **kwargs) -> None:
    with suppress(Exception):
      hook(*args, **kwargs)

  return quiet


def identity() -> dict[str, str]:
  """The pod this process runs in, as the scheduler passes it."""
  return {field: value for field, env in (("pod", "POD_NAME"), ("pod_uid", "POD_UID"), ("node", "NODE_NAME")) if (value := os.getenv(env))}


@dataclass
class Outstanding:
  """A request this worker popped and has not answered."""

  op: str
  run_id: str
  since: float
  observed: bool = False  # observe_operation records the op, so answered need not


class Worker:
  """A worker process's record of itself. The event loop changes it and the recorder's thread writes it."""

  def __init__(self, role: str, runtime_id: str | None) -> None:
    now = time.time()
    self.record = {
      # One record per process, so a container restarted in place does not overwrite its predecessor's.
      "id": uuid.uuid4().hex[:16],
      "node": socket.gethostname(),
      **identity(),
      "role": role,
      "runtime_id": runtime_id,
      "started_at": now,
      "phase": "starting",
      "phase_since": now,
      "turns": [],
      "waits": [],
      # The settings the scheduler passed as env, which is what this process runs with.
      "config": {"env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(CONFIG_ENV) and not SECRET.search(k)}},
    }
    self.outstanding: dict[str, Outstanding] = {}

  def snapshot(self) -> str:
    """The record as JSON, with outstanding requests counted by op and run, each group showing its oldest."""
    groups: dict[tuple, dict] = {}
    for entry in list(self.outstanding.values()):  # a copy, since the event loop changes it meanwhile
      group = groups.setdefault((entry.op, entry.run_id), {"op": entry.op, "run_id": entry.run_id, "since": entry.since, "count": 0})
      group["count"] += 1
    return json.dumps({**self.record, "heartbeat_at": time.time(), "outstanding": list(groups.values())})


def append(writes: Writes, list_key: str, item, limit: int) -> None:
  writes.appends.setdefault(list_key, ([], limit))[0].append(json.dumps(item))


def plan(entries: Iterable[tuple], step_starts: dict[str, int], pod: dict[str, str]) -> Writes:
  """The writes for buffered entries. step_starts holds each trainer op's run's step count before this process's
  first step, and pod is this process's pod, which every record names."""
  writes = Writes()
  for kind, run, *rest in entries:
    if kind == "op":
      record = rest[0] | pod
      record["step"] += step_starts[run]
      append(writes, key("operations", run), record, SAMPLE_LIMIT)
    elif kind == "sample":
      append(writes, key("operations", run, "samples"), rest[0] | pod, SAMPLE_LIMIT)
    elif kind == "step":
      writes.counts[key("steps", run)] += 1
      append(writes, key("step_times", run), rest[0], STEP_TIMES)
    elif kind == "count":
      writes.fields.setdefault(key("ledger", run), Counter())[rest[0]] += 1
  return writes


class Recorder:
  """This process's writer: a thread that drains the buffer into one pipeline every FLUSH_SECONDS, with the worker's
  record, and once more when stopped. It is the only thread that writes, so a flush never overlaps another."""

  def __init__(self, worker: Worker | None) -> None:
    self.worker, self.pod = worker, identity()
    # Each run's steps done in this process, counted on the event loop as its ops finish.
    self.steps_done: Counter[str] = Counter()
    # Each run's step count before this process's first step, read on this thread before its first op is written.
    self.step_starts: dict[str, int] = {}
    self.stopped = threading.Event()
    self.thread = threading.Thread(target=self.beat, name="telemetry", daemon=True)

  def beat(self) -> None:
    while not self.stopped.wait(FLUSH_SECONDS):
      self.flush()
    if self.worker is not None:
      self.worker.record.update(phase="exited", phase_since=time.time())
    self.flush()

  def flush(self) -> None:
    entries = []
    with suppress(IndexError):
      while True:
        entries.append(buffer.popleft())
    with suppress(Exception):
      store = get_state_store()
      # This process's steps for a run land with or after its first op, so the count read here precedes them.
      for run in {entry[1] for entry in entries if entry[0] == "op"} - self.step_starts.keys():
        self.step_starts[run] = int(store.get_value_sync(key("steps", run)) or 0)
      writes = plan(entries, self.step_starts, self.pod)
      if self.worker is not None:
        writes.values[key("worker", self.worker.record["id"])] = self.worker.snapshot()
      store.write_sync(writes, TTL_SECONDS)


recorder: Recorder | None = None


def add(*entry) -> None:
  # Nothing is buffered without a recorder to write it, so a process that does not record holds nothing.
  if recorder is not None:
    buffer.append(entry)


def worker() -> Worker | None:
  return recorder.worker if recorder is not None else None


def start(role: str, runtime_id: str | None = None, *, worker_record: bool = True) -> None:
  """Record from now on. Only rank 0 records, since every torchrun rank runs the same ops."""
  global recorder
  if recorder is not None or os.getenv("RANK", "0") != "0":
    return
  with suppress(Exception):
    started = Recorder(Worker(role, runtime_id) if worker_record else None)
    started.thread.start()
    recorder = started


async def finish() -> None:
  """Stop recording: a worker's record says it exited, and the buffer is flushed a last time within FINISH_SECONDS.
  Also called before an os._exit."""
  global recorder
  stopping, recorder = recorder, None
  if stopping is None:
    return
  stopping.stopped.set()
  deadline = time.monotonic() + FINISH_SECONDS
  while stopping.thread.is_alive() and time.monotonic() < deadline:
    await asyncio.sleep(0.02)


@asynccontextmanager
async def serving(role: str, runtime_id: str | None):
  """Keep this process's worker record while the body runs, and mark it exited after."""
  start(role, runtime_id)
  try:
    yield
  finally:
    await finish()


@best_effort
def set_phase(phase: str) -> None:
  if (w := worker()) is not None and w.record["phase"] != phase:
    w.record.update(phase=phase, phase_since=time.time())


@best_effort
def note(kind: str, started: float, ended: float) -> None:
  if (w := worker()) is not None:
    w.record[kind] = [*w.record[kind][1 - MAX_TURNS :], [started, ended]]


@asynccontextmanager
async def gpu_turn(slot):
  """Take the time slicer's slot for the body and show it on the worker record: the wait from asking to the grant, then
  the turn, which holds the wake and the sleep around the work."""
  asked = time.time()
  async with slot:
    granted = time.time()
    note("waits", asked, granted)
    set_phase("on_gpu")
    try:
      yield
    finally:
      note("turns", granted, time.time())
      set_phase("ready")


@best_effort
def enqueued(run_id: str) -> None:
  add("count", run_id, "enqueued")


@best_effort
def popped(requests: list[dict]) -> None:
  """Requests this worker took from its queue, each outstanding until answered. A shared trainer's delete_model is
  queued without being counted as enqueued, so it is not counted here either."""
  if (w := worker()) is None:
    return
  for request in requests:
    if request.get("request_id") and request.get("op") != "delete_model":
      w.outstanding[request["request_id"]] = Outstanding(request["op"], request["model_id"], time.time())
      add("count", request["model_id"], "popped")


@best_effort
def answered(request_id: str, result: dict) -> None:
  """The request's answer is written. A failure answered without its op running, such as a failed wake or parse, gets
  its op record here, so every failed answer has one."""
  if (w := worker()) is None or (entry := w.outstanding.pop(request_id, None)) is None:
    return
  add("count", entry.run_id, "answered")
  if not entry.observed and result.get("type") == FAILED:
    keep_op(op_record(entry.op, entry.run_id, request_id, w.record["runtime_id"], entry.since, FAILED))


# Trainer op records

# The names of the metrics an op's worker found NaN or infinite, before it replaced them for JSON.
nonfinite_names: contextvars.ContextVar[set[str] | None] = contextvars.ContextVar("nonfinite_names", default=None)


@best_effort
def nonfinite(names: Iterable[str]) -> None:
  if (found := nonfinite_names.get()) is not None:
    found.update(names)


def op_record(op: str, run_id: str, request_id: str | None, runtime_id: str | None, started_at: float, error_type: str | None) -> dict:
  now = time.time()
  return {
    "at": now,
    "started_at": started_at,
    "operation": op,
    "request_id": request_id,
    "run_id": run_id,
    "runtime_id": runtime_id,
    "role": "trainer",
    "status": "failed" if error_type else "succeeded",
    "error_type": error_type,
    "elapsed_seconds": now - started_at,
    "metrics": {},
  }


def keep_op(record: dict) -> None:
  if recorder is None:
    return
  run = record["run_id"]
  if record["operation"] == "sample":
    add("sample", run, record)
    return
  record["step"] = recorder.steps_done[run]  # before this op's own step
  add("op", run, record)
  if record["status"] == "failed":
    add("count", run, "failed")
  elif record["operation"] == "optim_step":
    recorder.steps_done[run] += 1
    add("step", run, record["at"])


@best_effort
def op_finished(command: Command, runtime_id: str | None, started_at: float, found: set[str], error_type: str | None = None, result=None) -> None:
  """Record how one op went: failed with error_type, or as its result says. No payload, output or error message is
  kept. A metric is either a finite number or named in nonfinite, from the raw value its worker saw."""
  result = result if isinstance(result, dict) else {}
  if error_type is None and result.get("type") == FAILED:
    error_type = FAILED
  op = "forward" if getattr(command, "forward_only", False) else command.op  # an eval loss must not mix with training losses
  record = op_record(op, command.model_id, command.request_id, runtime_id, started_at, error_type)
  numbers = {k: float(v) for k, v in (result.get("metrics") or {}).items() if isinstance(v, Real) and not isinstance(v, bool)}
  found = found | {k for k, v in numbers.items() if not math.isfinite(v)}
  record["metrics"] = {k: v for k, v in numbers.items() if k not in found}
  if found:
    record["nonfinite"] = sorted(found)
  if command.enqueued_at is not None:
    record["queue_seconds"] = started_at - command.enqueued_at
  keep_op(record)


async def observe_operation(command: Command, runtime_id: str | None, call: Callable[[], Awaitable[dict]]) -> dict:
  """Run one trainer op and record how it went. Returns or raises exactly as the op does."""
  started_at, found = time.time(), set()
  mark_observed(command.request_id)
  token = nonfinite_names.set(found)  # the op's worker threads copy this context, so they add to found
  try:
    result = await call()
  except Exception as error:
    op_finished(command, runtime_id, started_at, found, type(error).__name__)
    raise
  finally:
    nonfinite_names.reset(token)
  op_finished(command, runtime_id, started_at, found, result=result)
  return result


@best_effort
def mark_observed(request_id: str) -> None:
  if (w := worker()) is not None and (entry := w.outstanding.get(request_id)) is not None:
    entry.observed = True
