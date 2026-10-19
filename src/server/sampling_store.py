"""Each sampling pool's pending requests are one Redis stream, read without removal.
Results land in the `open_rl:future:<id>` list that `retrieve_future` already reads."""

import json
from array import array
from typing import Any

import redis.asyncio as redis

RESULT_TTL_SECONDS = 300
# An accepted request may never be committed, so its idempotency key expires instead of staying until Redis is full.
PENDING_TTL_SECONDS = 3 * 24 * 3600


def stream_key(pool: str) -> str:
  return f"open_rl:sampling:{pool}"


def compact_prompt(record: dict[str, Any]) -> dict[str, Any]:
  """A json.loads object_hook that holds a prompt as array('I'), four bytes a token against about 36 as a list of ints,
  and drops each list as soon as its record is read. A prompt array('I') cannot hold stays the list, for vLLM to refuse."""
  if isinstance(tokens := record.get("prompt_token_ids"), list):
    try:
      record["prompt_token_ids"] = array("I", tokens)
    except (OverflowError, TypeError):
      pass
  return record


def idempotency_key(pool: str, key: str) -> str:
  return f"open_rl:sampling:{pool}:idem:{key}"


def future_key(request_id: str) -> str:
  return f"open_rl:future:{request_id}"


class SamplingStore:
  def __init__(self, client: redis.Redis):
    self.redis = client

  async def accept(self, record: dict[str, Any]) -> str:
    """The new request's id, or the id already stored under its idempotency key."""
    stream, fields = stream_key(record["pool"]), {"record": json.dumps(record)}
    if not record.get("idempotency_key"):
      await self.redis.xadd(stream, fields)
      return record["request_id"]
    key = idempotency_key(record["pool"], record["idempotency_key"])

    async def add(pipe) -> str:
      if existing := await pipe.get(key):
        return existing
      pipe.multi()
      pipe.set(key, record["request_id"], ex=PENDING_TTL_SECONDS)
      pipe.xadd(stream, fields)
      return record["request_id"]

    return await self.redis.transaction(add, key, value_from_callable=True)

  async def pending(self, pool: str, limit: int | None = 1000, after: str = "0-0") -> list[dict[str, Any]]:
    """Pending records in accept order. The dispatcher holds every one until it commits, so prompts come back compact."""
    entries = await self.redis.xrange(stream_key(pool), min=f"({after}", count=limit)
    return [{**json.loads(fields["record"], object_hook=compact_prompt), "entry_id": entry_id} for entry_id, fields in entries]

  async def wait_for_new(self, pool: str, after: str, timeout_ms: int) -> None:
    await self.redis.xread({stream_key(pool): after}, count=1, block=timeout_ms)

  async def commit(self, record: dict[str, Any], result: dict[str, Any]) -> bool:
    """Save the first final result and drop the pending entry. False if it was already committed."""
    stream, future, entry_id = stream_key(record["pool"]), future_key(record["request_id"]), record["entry_id"]

    async def save(pipe) -> bool:
      if not await pipe.xrange(stream, entry_id, entry_id):
        return False
      pipe.multi()
      pipe.rpush(future, json.dumps(result))
      pipe.expire(future, RESULT_TTL_SECONDS)
      pipe.xdel(stream, entry_id)
      if key := record.get("idempotency_key"):
        pipe.expire(idempotency_key(record["pool"], key), RESULT_TTL_SECONDS)
      return True

    # Every commit writes the result key, so watching it lets only one through.
    return await self.redis.transaction(save, future, value_from_callable=True)

  async def pinned_revisions(self, pool: str) -> set[str]:
    """Revisions that pending requests still name, so their weights must stay."""
    return {record["revision"] for record in await self.pending(pool, limit=None)} - {""}
