"""A sampler set's accepted requests, kept in one Redis Stream until a result is saved.

The API appends each request to its set's stream, and the entry's ID becomes the
request ID. The set's dispatcher reads entries through a consumer group, so an
entry stays pending until the dispatcher saves a result and acknowledges it. The
first result saved wins (SET NX); a late or repeated attempt changes nothing.

Only plain Redis commands: a request is accepted in a transaction that watches the
set's admission flag, and finished in one MULTI/EXEC. A request is finished when
its result key exists, and the stream's length is the set's unfinished count, since
finished entries are deleted. All of a set's keys share the hash tag {set}, so each
transaction stays on one node of a sharded Redis.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from redis.exceptions import ResponseError, WatchError

GROUP = "dispatch"
ID_PREFIX = "sampled"
# The label every sampler pod of a set carries, which the set's router selects by.
SAMPLER_SET_LABEL = "openrl.io/sampler-set"


def sampler_set(runtime: str) -> str:
  """A label-safe name for a runtime's samplers."""
  return "set-" + hashlib.sha256(runtime.encode()).hexdigest()[:16]


def stream_key(set_id: str) -> str:
  return f"openrl:{{{set_id}}}:stream"


def admission_key(set_id: str) -> str:
  return f"openrl:{{{set_id}}}:admission"


def result_key(set_id: str, entry_id: str) -> str:
  return f"openrl:{{{set_id}}}:result:{entry_id}"


def request_id(set_id: str, entry_id: str) -> str:
  return f"{ID_PREFIX}:{set_id}:{entry_id}"


def parse(request_id: str) -> tuple[str, str] | None:
  """The set and stream entry of a stream request ID, or None for any other ID."""
  parts = request_id.split(":")
  return (parts[1], parts[2]) if len(parts) == 3 and parts[0] == ID_PREFIX and parts[1] and parts[2] else None


def failed(message: str, category: str = "server") -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message, "category": category}


class SetClosed(RuntimeError):
  pass


class SetFull(RuntimeError):
  pass


@dataclass(frozen=True)
class Claim:
  request_id: str
  entry_id: str
  input: dict[str, Any]


def decode(value: Any) -> Any:
  return value.decode() if isinstance(value, bytes) else value


def claims(set_id: str, entries: list) -> list[Claim]:
  # XAUTOCLAIM can return an entry whose payload was deleted; it has no fields.
  found = []
  for entry, fields in entries:
    if fields:
      entry_id = decode(entry)
      values = {decode(k): decode(v) for k, v in fields.items()}
      found.append(Claim(request_id(set_id, entry_id), entry_id, json.loads(values["input"])))
  return found


class SamplingStreams:
  """The Redis side of stream sampling, shared by the API and the dispatcher."""

  def __init__(self, client: redis.Redis, *, result_ttl: float = 1800):
    self.client = client
    self.result_ttl = int(result_ttl)

  # *** API side ***

  async def open(self, set_id: str) -> None:
    await self.client.set(admission_key(set_id), "open")

  async def close(self, set_id: str) -> None:
    await self.client.set(admission_key(set_id), "closed")

  async def accept(self, set_id: str, request: dict[str, Any], *, limit: int, deadline_seconds: float) -> str:
    """Store a request and return its ID; raise if the set is closed or full.
    Closing the set changes the watched flag, so a racing accept retries and sees
    it closed. The limit is approximate: concurrent accepts may pass it by a few."""
    now = time.time()
    stored = json.dumps({**request, "accepted_at": now, "deadline": now + deadline_seconds})
    async with self.client.pipeline(transaction=True) as pipe:
      while True:
        try:
          await pipe.watch(admission_key(set_id))
          if decode(await pipe.get(admission_key(set_id))) != "open":
            raise SetClosed(f"Sampler set {set_id} is not accepting requests")
          if await pipe.xlen(stream_key(set_id)) >= limit:
            raise SetFull(f"Sampler set {set_id} has {limit} unfinished requests; try again later")
          pipe.multi()
          pipe.xadd(stream_key(set_id), {"input": stored})
          [entry_id] = await pipe.execute()
          return request_id(set_id, decode(entry_id))
        except WatchError:
          continue

  async def result(self, request_id: str) -> dict[str, Any]:
    """The saved result, pending while the entry is still in the stream, and a
    failure for an unknown or expired ID."""
    parsed = parse(request_id)
    if parsed is None:
      return failed("Not a stream sampling request", "user")
    set_id, entry_id = parsed
    raw = await self.client.get(result_key(set_id, entry_id))
    if raw is not None:
      return json.loads(raw)
    if await self.client.xrange(stream_key(set_id), entry_id, entry_id):
      return {"type": "try_again"}
    return failed("Sample request not found, or its result expired", "user")

  async def unfinished(self, set_id: str) -> int:
    return await self.client.xlen(stream_key(set_id))

  # *** dispatcher side ***

  async def ensure_group(self, set_id: str) -> None:
    # From the start of the stream, so requests accepted before startup are read.
    try:
      await self.client.xgroup_create(stream_key(set_id), GROUP, id="0", mkstream=True)
    except ResponseError as exc:
      if "BUSYGROUP" not in str(exc):
        raise

  async def read_new(self, set_id: str, consumer: str, count: int, block_ms: int) -> list[Claim]:
    if count <= 0:
      return []
    reply = await self.client.xreadgroup(GROUP, consumer, {stream_key(set_id): ">"}, count=count, block=block_ms)
    return [claim for _, entries in reply or [] for claim in claims(set_id, entries)]

  async def reclaim(self, set_id: str, consumer: str, count: int, min_idle: float) -> list[Claim]:
    """Take over entries no consumer has touched for min_idle seconds."""
    if count <= 0:
      return []
    reply = await self.client.xautoclaim(stream_key(set_id), GROUP, consumer, int(min_idle * 1000), start_id="0-0", count=count)
    return claims(set_id, reply[1])

  async def touch(self, set_id: str, consumer: str, entry_ids: list[str]) -> None:
    """Keep this consumer's entries from looking idle, so they are not reclaimed."""
    if entry_ids:
      await self.client.xclaim(stream_key(set_id), GROUP, consumer, 0, entry_ids, justid=True)

  async def finished(self, set_id: str, entry_id: str) -> bool:
    return bool(await self.client.exists(result_key(set_id, entry_id)))

  async def finish(self, set_id: str, entry_id: str, result: dict[str, Any]) -> bool:
    """Save the result unless one exists, and acknowledge and delete the entry.
    Returns whether this call's result was the one saved."""
    async with self.client.pipeline(transaction=True) as pipe:
      pipe.set(result_key(set_id, entry_id), json.dumps(result), nx=True, ex=self.result_ttl)
      pipe.xack(stream_key(set_id), GROUP, entry_id)
      pipe.xdel(stream_key(set_id), entry_id)
      saved, _, _ = await pipe.execute()
    return bool(saved)


# *** set lifecycle, from the worker manager's threads (a synchronous redis.Redis) ***


def open_set(client: Any, set_id: str) -> None:
  client.set(admission_key(set_id), "open")


def close_and_cancel(client: Any, set_id: str, reason: str, *, result_ttl: float = 1800) -> int:
  """Close the set to new requests, then finish every unfinished one as cancelled."""
  client.set(admission_key(set_id), "closed")
  cancelled = 0
  for claim in claims(set_id, client.xrange(stream_key(set_id))):
    with client.pipeline(transaction=True) as pipe:
      pipe.set(result_key(set_id, claim.entry_id), json.dumps(failed(reason, "user")), nx=True, ex=int(result_ttl))
      pipe.xack(stream_key(set_id), GROUP, claim.entry_id)
      pipe.xdel(stream_key(set_id), claim.entry_id)
      cancelled += bool(pipe.execute()[0])
  return cancelled
