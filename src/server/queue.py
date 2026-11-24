"""Unified Redis Streams queue for training and sampling requests, with XREAD BLOCK futures.

Each pending queue is a Redis stream (`open_rl:q:<name>`), read without removal so an interrupted
worker leaves its entries behind for the next one. Results land in a single-entry reply stream
(`open_rl:f:<request_id>`) waited on via `XREAD BLOCK`, staying readable across retried reads."""

from __future__ import annotations

import json
import time
from array import array
from typing import Any

import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError

from server.telemetry import sample_answered

RESULT_TTL_SECONDS = 300
# An accepted request may never get a result, so its idempotency key expires instead of staying until Redis is full.
PENDING_TTL_SECONDS = 3 * 24 * 3600
# Sweep streams a page at a time so a backlog of long prompts never sits in memory at once.
SWEEP_PAGE = 100
# Keep each XREAD BLOCK chunk well below redis_client's 30s socket timeout.
BLOCK_CHUNK_MS = 10_000


def queue_key(name: str) -> str:
  return f"open_rl:q:{name}"


def train_queue(model_id: str) -> str:
  return f"train:{model_id}"


def sample_queue(pool: str) -> str:
  return f"sample:{pool}"


def stream_key(pool: str) -> str:
  return queue_key(sample_queue(pool))


def idempotency_key(queue_name: str, key: str) -> str:
  return f"open_rl:q:{queue_name}:idem:{key}"


def future_key(request_id: str) -> str:
  return f"open_rl:f:{request_id}"


def compact_prompt(record: dict[str, Any]) -> dict[str, Any]:
  """A json.loads object_hook that holds a prompt as array('I'), four bytes a token against about 36 as a list of ints,
  and drops each list as soon as its record is read. A prompt array('I') cannot hold stays the list, for vLLM to refuse."""
  if isinstance(tokens := record.get("prompt_token_ids"), list):
    try:
      record["prompt_token_ids"] = array("I", tokens)
    except (OverflowError, TypeError):
      pass
  return record


def job_of(record: dict[str, Any]) -> str:
  """The job that asked for the record. Records accepted before the API server named the job group under their model."""
  return record.get("job") or record["model"]


def accepted_at(record: dict[str, Any]) -> float:
  """When the record's request was accepted. A stream id starts with that time in milliseconds."""
  return int(record["entry_id"].split("-")[0]) / 1000


def _entry_record(entry_id: str, fields: dict[str, str], *, compact: bool = False) -> dict[str, Any]:
  hook = compact_prompt if compact else None
  return {**json.loads(fields["record"], object_hook=hook), "entry_id": entry_id}


def _job_and_revision(fields: dict[str, str]) -> tuple[str, str]:
  """An entry's job and revision without decoding its prompt. Older entries without both fields decode the record."""
  if "job" in fields:
    return fields["job"], fields["revision"]
  record = _entry_record("0-0", fields, compact=True)
  return job_of(record), record["revision"]


class StreamQueue:
  """Redis Streams queue supporting per-model reads, multi-tenant round-robin groups, and blocking futures."""

  def __init__(self, client: redis.Redis):
    self.redis = client
    self._rr_cursor: dict[str, str] = {}

  async def enqueue(
    self,
    queue_name: str,
    record: dict[str, Any],
    *,
    group: str | None = None,
    idempotency_key: str | None = None,
    fields: dict[str, str] | None = None,
  ) -> str:
    """Append `record` to `queue_name`, deduplicating by `idempotency_key` if given. Returns the request ID."""
    entry = {"record": json.dumps(record), **(fields or {})}
    stream = queue_key(queue_name)
    request_id = record.get("request_id", "")

    if not idempotency_key:
      await self.redis.xadd(stream, entry)
    else:
      idem_key = f"{stream}:idem:{idempotency_key}"

      async def add(pipe) -> str:
        if existing := await pipe.get(idem_key):
          return existing
        pipe.multi()
        pipe.set(idem_key, request_id, ex=PENDING_TTL_SECONDS)
        pipe.xadd(stream, entry)
        return request_id

      request_id = await self.redis.transaction(add, idem_key, value_from_callable=True)

    if group is not None:
      group_set = f"open_rl:set:{group}"
      wake_stream = f"open_rl:q:wake:{group}"
      async with self.redis.pipeline(transaction=False) as pipe:
        pipe.sadd(group_set, queue_name)
        pipe.xadd(wake_stream, {"q": queue_name}, maxlen=1)
        pipe.expire(wake_stream, PENDING_TTL_SECONDS)
        await pipe.execute()

    return request_id

  async def pending(
    self,
    queue_name: str,
    limit: int | None = 1000,
    after: str = "0-0",
    *,
    compact: bool = False,
  ) -> list[dict[str, Any]]:
    """Pending records in accept order without removal."""
    entries = await self.redis.xrange(queue_key(queue_name), min=f"({after}", count=limit)
    return [_entry_record(entry_id, fields, compact=compact) for entry_id, fields in entries]

  async def wait_for_new(self, queue_name: str, after: str = "0-0", timeout_ms: int = 5000) -> None:
    await self.redis.xread({queue_key(queue_name): after}, count=1, block=timeout_ms)

  async def read_batch(self, queue_name: str, timeout_ms: int = 5000) -> list[dict[str, Any]]:
    """Return pending entries for `queue_name`, waiting up to `timeout_ms` if the stream is currently empty."""
    if batch := await self.pending(queue_name):
      return batch
    try:
      await self.wait_for_new(queue_name, after="0-0", timeout_ms=timeout_ms)
      return await self.pending(queue_name)
    except RedisTimeoutError:
      return []

  async def next_batch(self, group: str | None = None, *, timeout_ms: int = 5000) -> list[dict[str, Any]]:
    """Round-robin across the active queues in `group`, waiting on its wake stream when all are empty."""
    group_name = group or "default"
    group_set = f"open_rl:set:{group_name}"
    wake_stream = f"open_rl:q:wake:{group_name}"
    deadline = time.monotonic() + timeout_ms / 1000.0

    while True:
      # Capture the wake stream's head before scanning membership so an enqueue landing mid-scan wakes XREAD.
      tail = await self.redis.xrange(wake_stream, min="-", max="+")
      wake_id = tail[-1][0] if tail else "0-0"

      members = sorted(await self.redis.smembers(group_set))
      cursor = self._rr_cursor.get(group_name)
      if cursor is not None and members:
        start = next((i for i, member in enumerate(members) if member > cursor), 0)
        members = members[start:] + members[:start]

      for queue_name in members:
        if batch := await self.pending(queue_name):
          self._rr_cursor[group_name] = queue_name
          return batch
        await self.redis.srem(group_set, queue_name)
        if await self.queued(queue_name) > 0:
          await self.redis.sadd(group_set, queue_name)
        else:
          await self.delete_if_empty(queue_name)

      remaining = deadline - time.monotonic()
      if remaining <= 0:
        return []
      block_ms = max(1, min(int(remaining * 1000), BLOCK_CHUNK_MS))
      try:
        await self.redis.xread({wake_stream: wake_id}, count=1, block=block_ms)
      except RedisTimeoutError:
        pass

  async def complete(
    self,
    queue_name: str,
    entry_id: str,
    request_id: str,
    result: dict[str, Any],
    *,
    idempotency_key: str | None = None,
  ) -> bool:
    """Save the first final result to `open_rl:f:<request_id>` and drop the pending entry. False if already answered."""
    if result.get("status") == "pending":
      return False
    stream = queue_key(queue_name)
    future = future_key(request_id)

    async def save(pipe) -> bool:
      if not await pipe.xrange(stream, entry_id, entry_id):
        return False
      pipe.multi()
      pipe.xadd(future, {"data": json.dumps(result)}, maxlen=1)
      pipe.expire(future, RESULT_TTL_SECONDS)
      pipe.xdel(stream, entry_id)
      if idempotency_key:
        pipe.expire(f"{stream}:idem:{idempotency_key}", RESULT_TTL_SECONDS)
      return True

    # Every completion writes the future key, so watching it lets only the first winner commit.
    return await self.redis.transaction(save, future, value_from_callable=True)

  async def ack(self, queue_name: str, entry_id: str) -> bool:
    """Remove an entry without writing a future (used for fire-and-forget commands like Shutdown)."""
    return bool(await self.redis.xdel(queue_key(queue_name), entry_id))

  async def complete_record(self, queue_name: str, record: dict[str, Any], result: dict[str, Any]) -> bool:
    """Complete a record read from `queue_name`, or fall back to set_result when no `entry_id` is present."""
    request_id = record.get("request_id")
    entry_id = record.get("entry_id")
    if entry_id is None:
      if request_id:
        await self.set_result(request_id, result)
      return True
    if not request_id:
      return await self.ack(queue_name, entry_id)
    return await self.complete(queue_name, entry_id, request_id, result)

  async def ack_record(self, queue_name: str, record: dict[str, Any]) -> bool:
    entry_id = record.get("entry_id")
    if entry_id is None:
      return True
    return await self.ack(queue_name, entry_id)

  async def set_result(self, request_id: str, result: dict[str, Any]) -> None:
    """Write a result directly to `open_rl:f:<request_id>` when no queue entry exists."""
    if result.get("status") == "pending":
      return
    future = future_key(request_id)
    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.xadd(future, {"data": json.dumps(result)}, maxlen=1)
      pipe.expire(future, RESULT_TTL_SECONDS)
      await pipe.execute()

  async def get_result(self, request_id: str, timeout: float) -> dict[str, Any]:
    """Wait for `request_id`'s result via `XREAD BLOCK`, returning `try_again` on timeout."""
    future = future_key(request_id)
    deadline = time.monotonic() + timeout
    while True:
      remaining = deadline - time.monotonic()
      block_ms = max(1, min(int(remaining * 1000), BLOCK_CHUNK_MS)) if remaining > 0 else None
      try:
        response = await self.redis.xread({future: "0-0"}, count=1, block=block_ms)
      except RedisTimeoutError:
        response = None
      if response:
        payload = json.loads(response[0][1][0][1]["data"])
        await self.redis.expire(future, RESULT_TTL_SECONDS)
        return payload
      if remaining <= 0 or time.monotonic() >= deadline:
        return {"type": "try_again", "request_id": request_id, "queue_state": "active"}

  async def queued(self, queue_name: str) -> int:
    return await self.redis.xlen(queue_key(queue_name))

  async def delete_if_empty(self, queue_name: str) -> bool:
    """Delete an empty stream atomically; watching it preserves any entry enqueued concurrently."""
    stream = queue_key(queue_name)

    async def delete(pipe) -> bool:
      if await pipe.xlen(stream):
        return False
      pipe.multi()
      pipe.delete(stream)
      return True

    return await self.redis.transaction(delete, stream, value_from_callable=True)

  async def entries(self, queue_name: str, until: str = "+"):
    """Yield `(entry_id, fields)` from `queue_name` up to `until`, one page at a time."""
    stream = queue_key(queue_name)
    after = "-"
    while page := await self.redis.xrange(stream, min=after, max=until, count=SWEEP_PAGE):
      for entry in page:
        yield entry
      after = f"({page[-1][0]}"

  async def answered(self, request_ids: list[str]) -> set[str]:
    """Return the subset of `request_ids` that already have a saved result."""
    if not request_ids:
      return set()
    async with self.redis.pipeline(transaction=False) as pipe:
      for request_id in request_ids:
        pipe.exists(future_key(request_id))
      found = await pipe.execute()
    return {request_id for request_id, exists in zip(request_ids, found) if exists}

  async def accept(self, record: dict[str, Any]) -> str:
    """Accept a sampling request into its pool stream, indexing job and revision for prompt-free sweeps."""
    fields = {"job": job_of(record), "revision": record["revision"]}
    return await self.enqueue(
      sample_queue(record["pool"]),
      record,
      idempotency_key=record.get("idempotency_key"),
      fields=fields,
    )

  async def complete_sample(self, record: dict[str, Any], result: dict[str, Any], started_at: float | None = None) -> bool:
    """Commit a sample result and record telemetry if this call was the first winner."""
    saved = await self.complete(
      sample_queue(record["pool"]),
      record["entry_id"],
      record["request_id"],
      result,
      idempotency_key=record.get("idempotency_key"),
    )
    if saved:
      sample_answered(job_of(record), record["pool"], record["request_id"], result, started_at or accepted_at(record))
    return saved

  async def active_pools(self) -> list[str]:
    """Every sampling pool with a stream, whether or not it currently has pending requests."""
    prefix = queue_key(sample_queue(""))
    return [stream.removeprefix(prefix) async for stream in self.redis.scan_iter(match=f"{prefix}*", _type="stream")]

  async def pending_jobs(self, pool: str) -> set[str]:
    """The jobs with a pending request in `pool`."""
    return {_job_and_revision(fields)[0] async for _, fields in self.entries(sample_queue(pool))}

  async def job_records(self, pool: str, job: str) -> list[dict[str, Any]]:
    records = []
    async for entry_id, fields in self.entries(sample_queue(pool)):
      if _job_and_revision(fields)[0] == job:
        records.append(_entry_record(entry_id, fields, compact=True))
    return records

  async def fail_job(self, pool: str, job: str, result: dict[str, Any]) -> int:
    """Save `result` for every pending request of `job` in `pool`. Returns how many were failed."""
    failed = 0
    for record in await self.job_records(pool, job):
      if await self.complete_sample(record, result):
        failed += 1
    return failed

  async def fail_older_than(self, seconds: float, result: dict[str, Any]) -> int:
    """Save `result` for every pending sampling request accepted more than `seconds` ago. Returns how many."""
    cutoff = str(int((time.time() - seconds) * 1000))
    failed = 0
    for pool in await self.active_pools():
      async for entry_id, fields in self.entries(sample_queue(pool), cutoff):
        if await self.complete_sample(_entry_record(entry_id, fields, compact=True), result):
          failed += 1
    return failed

  async def pinned_revisions(self, pool: str) -> set[str]:
    """Revisions still named by pending requests in `pool`, so their weights must remain on disk."""
    return {_job_and_revision(fields)[1] async for _, fields in self.entries(sample_queue(pool))} - {""}
