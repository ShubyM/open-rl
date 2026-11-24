"""Unified Redis Streams queue and XREAD BLOCK future-reply layer for training and sampling."""

from __future__ import annotations

import json
import time
from array import array
from typing import Any

import redis.asyncio as redis
from redis.exceptions import TimeoutError as RedisTimeoutError

RESULT_TTL_SECONDS = 300
PENDING_TTL_SECONDS = 3 * 24 * 3600
SWEEP_PAGE = 100
BLOCK_CHUNK_MS = 10_000


def queue_key(name: str) -> str:
  return f"open_rl:q:{name}"


def train_queue(model_id: str) -> str:
  return f"train:{model_id}"


def sample_queue(pool: str) -> str:
  return f"sample:{pool}"


def stream_key(pool: str) -> str:
  return queue_key(sample_queue(pool))


def idempotency_key(queue: str, key: str) -> str:
  return f"open_rl:q:{queue}:idem:{key}"


def future_key(request_id: str) -> str:
  return f"open_rl:f:{request_id}"


def compact_prompt(record: dict[str, Any]) -> dict[str, Any]:
  """Store prompt_token_ids as array('I') (~4B/tok vs ~36B as list[int]) when representable."""
  if isinstance(tokens := record.get("prompt_token_ids"), list):
    try:
      record["prompt_token_ids"] = array("I", tokens)
    except (OverflowError, TypeError):
      pass
  return record


def job_of(r: dict[str, Any]) -> str:
  return r.get("job") or r["model"]


def accepted_at(r: dict[str, Any]) -> float:
  return int(r["entry_id"].split("-")[0]) / 1000


def _job_rev(fields: dict[str, str]) -> tuple[str, str]:
  if "job" in fields:
    return fields["job"], fields["revision"]
  rec = json.loads(fields["record"], object_hook=compact_prompt)
  return job_of(rec), rec["revision"]


def _decode(entry_id: str, fields: dict[str, str], compact: bool = False) -> dict[str, Any]:
  return {**json.loads(fields["record"], object_hook=compact_prompt if compact else None), "entry_id": entry_id}


class StreamQueue:
  """One Redis Streams primitive for both training and sampling queues plus XREAD BLOCK futures."""

  def __init__(self, client: redis.Redis):
    self.redis = client

  async def enqueue(self, queue: str, record: dict[str, Any], *, idempotency_key: str | None = None, fields: dict[str, str] | None = None) -> str:
    entry, stream, req_id = {"record": json.dumps(record), **(fields or {})}, queue_key(queue), record.get("request_id", "")
    if not idempotency_key:
      await self.redis.xadd(stream, entry)
      return req_id
    ikey = f"{stream}:idem:{idempotency_key}"

    async def add(pipe) -> str:
      if existing := await pipe.get(ikey):
        return existing
      pipe.multi()
      pipe.set(ikey, req_id, ex=PENDING_TTL_SECONDS)
      pipe.xadd(stream, entry)
      return req_id

    return await self.redis.transaction(add, ikey, value_from_callable=True)

  async def pending(self, queue: str, limit: int | None = 1000, after: str = "0-0", *, compact: bool = False) -> list[dict[str, Any]]:
    entries = await self.redis.xrange(queue_key(queue), min=f"({after}", count=limit)
    return [_decode(eid, fields, compact) for eid, fields in entries]

  async def wait_for_new(self, queue: str, after: str = "0-0", timeout_ms: int = 5000) -> None:
    await self.redis.xread({queue_key(queue): after}, count=1, block=timeout_ms)

  async def complete(self, queue: str, entry_id: str, request_id: str, result: dict[str, Any], *, idempotency_key: str | None = None) -> bool:
    if result.get("status") == "pending":
      return False
    stream, future = queue_key(queue), future_key(request_id)

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

    return await self.redis.transaction(save, future, value_from_callable=True)

  async def ack(self, queue: str, entry_id: str) -> bool:
    return bool(await self.redis.xdel(queue_key(queue), entry_id))

  async def set_result(self, request_id: str, result: dict[str, Any]) -> None:
    if result.get("status") == "pending":
      return
    future = future_key(request_id)
    async with self.redis.pipeline(transaction=True) as pipe:
      pipe.xadd(future, {"data": json.dumps(result)}, maxlen=1)
      pipe.expire(future, RESULT_TTL_SECONDS)
      await pipe.execute()

  async def get_result(self, request_id: str, timeout: float) -> dict[str, Any]:
    future, deadline = future_key(request_id), time.monotonic() + timeout
    while True:
      remaining = deadline - time.monotonic()
      block = max(1, min(int(remaining * 1000), BLOCK_CHUNK_MS)) if remaining > 0 else None
      try:
        res = await self.redis.xread({future: "0-0"}, count=1, block=block)
      except RedisTimeoutError:
        res = None
      if res:
        payload = json.loads(res[0][1][0][1]["data"])
        await self.redis.expire(future, RESULT_TTL_SECONDS)
        return payload
      if remaining <= 0 or time.monotonic() >= deadline:
        return {"type": "try_again", "request_id": request_id, "queue_state": "active"}

  async def queued(self, queue: str) -> int:
    return await self.redis.xlen(queue_key(queue))

  async def delete_if_empty(self, queue: str) -> bool:
    stream = queue_key(queue)

    async def delete(pipe) -> bool:
      if await pipe.xlen(stream):
        return False
      pipe.multi()
      pipe.delete(stream)
      return True

    return await self.redis.transaction(delete, stream, value_from_callable=True)

  async def entries(self, queue: str, until: str = "+"):
    stream, after = queue_key(queue), "-"
    while page := await self.redis.xrange(stream, min=after, max=until, count=SWEEP_PAGE):
      for entry in page:
        yield entry
      after = f"({page[-1][0]}"

  async def answered(self, request_ids: list[str]) -> set[str]:
    if not request_ids:
      return set()
    async with self.redis.pipeline(transaction=False) as pipe:
      for rid in request_ids:
        pipe.exists(future_key(rid))
      found = await pipe.execute()
    return {rid for rid, exists in zip(request_ids, found) if exists}

  # ***** sampling pool helpers *****

  async def accept(self, record: dict[str, Any]) -> str:
    fields = {"job": job_of(record), "revision": record["revision"]}
    return await self.enqueue(sample_queue(record["pool"]), record, idempotency_key=record.get("idempotency_key"), fields=fields)

  async def complete_sample(self, record: dict[str, Any], result: dict[str, Any], started_at: float | None = None) -> bool:
    q = sample_queue(record["pool"])
    saved = await self.complete(q, record["entry_id"], record["request_id"], result, idempotency_key=record.get("idempotency_key"))
    if saved:
      from server import telemetry

      telemetry.sample_answered(job_of(record), record["pool"], record["request_id"], result, started_at or accepted_at(record))
    return saved

  async def active_pools(self) -> list[str]:
    prefix = queue_key(sample_queue(""))
    return [stream.removeprefix(prefix) async for stream in self.redis.scan_iter(match=f"{prefix}*", _type="stream")]

  async def pending_jobs(self, pool: str) -> set[str]:
    return {_job_rev(fields)[0] async for _, fields in self.entries(sample_queue(pool))}

  async def job_records(self, pool: str, job: str) -> list[dict[str, Any]]:
    return [_decode(eid, fields, True) async for eid, fields in self.entries(sample_queue(pool)) if _job_rev(fields)[0] == job]

  async def fail_job(self, pool: str, job: str, result: dict[str, Any]) -> int:
    return sum([await self.complete_sample(r, result) for r in await self.job_records(pool, job)])

  async def fail_older_than(self, seconds: float, result: dict[str, Any]) -> int:
    cutoff, failed = str(int((time.time() - seconds) * 1000)), 0
    for pool in await self.active_pools():
      async for eid, fields in self.entries(sample_queue(pool), cutoff):
        failed += await self.complete_sample(_decode(eid, fields, True), result)
    return failed

  async def pinned_revisions(self, pool: str) -> set[str]:
    return {_job_rev(fields)[1] async for _, fields in self.entries(sample_queue(pool))} - {""}
