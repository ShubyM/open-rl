"""Accepted sampling requests for one sampler set, kept in a Redis Stream until a
final outcome is saved.

The API appends a request and returns its ID. The set's dispatcher reads it
through a consumer group, so it stays pending until the dispatcher saves a final
result and acknowledges it in one script. Every key of a set shares the hash tag
{set}, which keeps those scripts on one Redis node in a sharded deployment.

A request's input is immutable. Its execution state (attempts, current attempt,
retry time) lives in a separate hash, and only the lease holder may change it.
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis

GROUP = "dispatch"
ID_PREFIX = "sampled"

PENDING = "pending"
FINAL = "final"

ACCEPT = """
if redis.call('GET', KEYS[1]) ~= 'open' then return {'closed'} end
if tonumber(redis.call('GET', KEYS[2]) or '0') >= tonumber(ARGV[1]) then return {'full'} end
local entry = redis.call('XADD', KEYS[3], '*', 'rid', ARGV[2], 'input', ARGV[3])
redis.call('INCR', KEYS[2])
redis.call('HSET', KEYS[4], 'stream_id', entry, 'status', 'pending', 'attempts', 0, 'next_retry_at', 0,
  'deadline', ARGV[4], 'accepted_at', ARGV[5])
return {'ok', entry}
"""

# Take or renew the set's dispatcher lease. A holder whose lease expired and was
# taken cannot renew, because the stored owner no longer matches.
LEASE = """
local holder = redis.call('GET', KEYS[1])
if not holder then
  redis.call('SET', KEYS[1], ARGV[1], 'PX', ARGV[2])
  return 1
end
if holder == ARGV[1] then
  redis.call('PEXPIRE', KEYS[1], ARGV[2])
  return 1
end
return 0
"""

RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""

# Reserve an attempt. The attempt counts even if the dispatcher stops before sending.
BEGIN = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return {'lease'} end
if redis.call('HGET', KEYS[2], 'status') ~= 'pending' then return {'final'} end
local attempts = tonumber(redis.call('HGET', KEYS[2], 'attempts') or '0')
if attempts >= tonumber(ARGV[3]) then return {'exhausted'} end
attempts = attempts + 1
redis.call('HSET', KEYS[2], 'attempts', attempts, 'attempt_id', ARGV[2], 'owner', ARGV[1], 'next_retry_at', 0, 'started_at', ARGV[4])
return {'ok', tostring(attempts)}
"""

RETRY = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 'lease' end
if redis.call('HGET', KEYS[2], 'status') ~= 'pending' then return 'final' end
if redis.call('HGET', KEYS[2], 'attempt_id') ~= ARGV[2] then return 'stale' end
redis.call('HSET', KEYS[2], 'next_retry_at', ARGV[3], 'last_error', ARGV[4])
return 'ok'
"""

# Save the one final outcome, acknowledge the entry and release its capacity.
# A late attempt, an old lease holder, or a second outcome changes nothing.
# KEYS: lease, request, result, stream, unfinished. An empty owner skips the lease
# check (cancellation); an empty attempt ID finalizes without an attempt (deadline).
FINALIZE = """
local state = redis.call('HMGET', KEYS[2], 'status', 'attempt_id', 'stream_id')
if ARGV[1] ~= '' and redis.call('GET', KEYS[1]) ~= ARGV[1] then return 'lease' end
if state[1] == false then return 'missing' end
if state[1] ~= 'pending' then
  redis.call('XACK', KEYS[4], ARGV[6], state[3])
  redis.call('XDEL', KEYS[4], state[3])
  return 'final'
end
if ARGV[2] ~= '' and state[2] ~= ARGV[2] then return 'stale' end
redis.call('SET', KEYS[3], ARGV[4], 'EX', ARGV[5])
redis.call('HSET', KEYS[2], 'status', 'final', 'outcome', ARGV[3], 'finished_at', ARGV[8])
redis.call('EXPIRE', KEYS[2], ARGV[7])
redis.call('XACK', KEYS[4], ARGV[6], state[3])
redis.call('XDEL', KEYS[4], state[3])
redis.call('DECR', KEYS[5])
return 'ok'
"""

# Keep claims alive only while holding the lease and still owning each entry.
RENEW_CLAIMS = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return -1 end
local renewed = 0
for i = 3, #ARGV do
  local pending = redis.call('XPENDING', KEYS[2], ARGV[2], ARGV[i], ARGV[i], 1)
  if pending[1] and pending[1][2] == ARGV[1] then
    redis.call('XCLAIM', KEYS[2], ARGV[2], ARGV[1], 0, ARGV[i], 'JUSTID')
    renewed = renewed + 1
  end
end
return renewed
"""


def sampler_set(runtime: str) -> str:
  """A label-safe name for a runtime's samplers."""
  return "set-" + hashlib.sha256(runtime.encode()).hexdigest()[:16]


def keys(set_id: str) -> dict[str, str]:
  tag = f"openrl:{{{set_id}}}"
  return {"admission": f"{tag}:admission", "unfinished": f"{tag}:unfinished", "stream": f"{tag}:stream", "lease": f"{tag}:lease"}


def request_key(set_id: str, request_id: str) -> str:
  return f"openrl:{{{set_id}}}:req:{request_id}"


def result_key(set_id: str, request_id: str) -> str:
  return f"openrl:{{{set_id}}}:result:{request_id}"


def new_request_id(set_id: str) -> str:
  return f"{ID_PREFIX}:{set_id}:{uuid.uuid4().hex}"


def set_of(request_id: str) -> str | None:
  """The set a stream request ID belongs to, or None for any other request ID."""
  prefix, _, rest = request_id.partition(":")
  set_id, sep, _ = rest.partition(":")
  return set_id if prefix == ID_PREFIX and sep and set_id else None


def failed(message: str, category: str = "server") -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message, "category": category}


class SetClosed(RuntimeError):
  pass


class SetFull(RuntimeError):
  pass


@dataclass
class Claim:
  stream_id: str
  request_id: str
  input: dict[str, Any]


class SamplingStreams:
  """The Redis side of stream sampling, shared by the API and the dispatcher."""

  def __init__(self, client: redis.Redis, *, result_ttl: float = 1800, guard_ttl: float = 7200):
    self.client = client
    self.result_ttl = int(result_ttl)
    # Completion guards outlive results so a late attempt still finds its request final.
    self.guard_ttl = int(max(guard_ttl, result_ttl))
    self.accept_script = client.register_script(ACCEPT)
    self.lease_script = client.register_script(LEASE)
    self.release_script = client.register_script(RELEASE)
    self.begin_script = client.register_script(BEGIN)
    self.retry_script = client.register_script(RETRY)
    self.finalize_script = client.register_script(FINALIZE)
    self.renew_script = client.register_script(RENEW_CLAIMS)

  # *** API side ***

  async def open(self, set_id: str) -> None:
    await self.client.set(keys(set_id)["admission"], "open")

  async def close(self, set_id: str) -> None:
    await self.client.set(keys(set_id)["admission"], "closed")

  async def accept(self, set_id: str, request: dict[str, Any], *, limit: int, deadline_seconds: float, request_id: str | None = None) -> str:
    """Store a request and return its ID; raise if the set is closed or full."""
    request_id = request_id or new_request_id(set_id)
    now = time.time()
    stored = {**request, "request_id": request_id, "set_id": set_id, "accepted_at": now, "deadline": now + deadline_seconds}
    k = keys(set_id)
    reply = await self.accept_script(
      keys=[k["admission"], k["unfinished"], k["stream"], request_key(set_id, request_id)],
      args=[limit, request_id, json.dumps(stored), stored["deadline"], now],
    )
    status = decode(reply[0])
    if status == "closed":
      raise SetClosed(f"Sampler set {set_id} is not accepting requests")
    if status == "full":
      raise SetFull(f"Sampler set {set_id} has {limit} unfinished requests; try again later")
    return request_id

  async def result(self, request_id: str) -> dict[str, Any]:
    """The saved final outcome, pending, or a failure for an unknown or expired ID."""
    set_id = set_of(request_id)
    if set_id is None:
      return failed("Not a stream sampling request", "user")
    raw = await self.client.get(result_key(set_id, request_id))
    if raw is not None:
      return json.loads(raw)
    status = decode(await self.client.hget(request_key(set_id, request_id), "status"))
    if status == PENDING:
      return {"type": "try_again"}
    if status == FINAL:
      return failed("Sample result expired after its retention window", "user")
    return failed("Sample request not found or expired", "user")

  async def unfinished(self, set_id: str) -> int:
    return int(await self.client.get(keys(set_id)["unfinished"]) or 0)

  async def cancel_unfinished(self, set_id: str, reason: str) -> int:
    """Finalize every unfinished request as cancelled; call after closing admission."""
    k = keys(set_id)
    cancelled = 0
    for claim in claims(await self.client.xrange(k["stream"])):
      request_id = claim.request_id
      if await self.finalize(set_id, request_id, owner="", attempt_id="", outcome="cancelled", result=failed(reason, "user")) == "ok":
        cancelled += 1
    return cancelled

  # *** dispatcher side ***

  async def ensure_group(self, set_id: str) -> None:
    # From the start of the stream, so requests accepted before startup are read.
    try:
      await self.client.xgroup_create(keys(set_id)["stream"], GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:
      if "BUSYGROUP" not in str(exc):
        raise

  async def lease(self, set_id: str, owner: str, ttl: float) -> bool:
    return bool(await self.lease_script(keys=[keys(set_id)["lease"]], args=[owner, int(ttl * 1000)]))

  async def release(self, set_id: str, owner: str) -> None:
    await self.release_script(keys=[keys(set_id)["lease"]], args=[owner])

  async def read_new(self, set_id: str, owner: str, count: int, block_ms: int) -> list[Claim]:
    if count <= 0:
      return []
    reply = await self.client.xreadgroup(GROUP, owner, {keys(set_id)["stream"]: ">"}, count=count, block=block_ms)
    return [claim for _, entries in reply or [] for claim in claims(entries)]

  async def reclaim(self, set_id: str, owner: str, count: int, min_idle: float) -> list[Claim]:
    """Take over entries whose owner stopped renewing them."""
    if count <= 0:
      return []
    reply = await self.client.xautoclaim(keys(set_id)["stream"], GROUP, owner, int(min_idle * 1000), start_id="0-0", count=count)
    return claims(reply[1])

  async def renew_claims(self, set_id: str, owner: str, stream_ids: list[str]) -> int:
    k = keys(set_id)
    return int(await self.renew_script(keys=[k["lease"], k["stream"]], args=[owner, GROUP, *stream_ids]))

  async def state(self, set_id: str, request_id: str) -> dict[str, str]:
    return {decode(key): decode(value) for key, value in (await self.client.hgetall(request_key(set_id, request_id))).items()}

  async def begin(self, set_id: str, request_id: str, owner: str, attempt_id: str, max_attempts: int) -> tuple[str, int]:
    reply = await self.begin_script(
      keys=[keys(set_id)["lease"], request_key(set_id, request_id)], args=[owner, attempt_id, max_attempts, time.time()]
    )
    return decode(reply[0]), int(decode(reply[1])) if len(reply) > 1 else 0

  async def retry(self, set_id: str, request_id: str, owner: str, attempt_id: str, at: float, error: str) -> str:
    return decode(await self.retry_script(keys=[keys(set_id)["lease"], request_key(set_id, request_id)], args=[owner, attempt_id, at, error[:500]]))

  async def finalize(self, set_id: str, request_id: str, *, owner: str, attempt_id: str, outcome: str, result: dict[str, Any]) -> str:
    k = keys(set_id)
    return decode(
      await self.finalize_script(
        keys=[k["lease"], request_key(set_id, request_id), result_key(set_id, request_id), k["stream"], k["unfinished"]],
        args=[owner, attempt_id, outcome, json.dumps(result), self.result_ttl, GROUP, self.guard_ttl, time.time()],
      )
    )


def claims(entries: list) -> list[Claim]:
  # XAUTOCLAIM can return an entry whose payload was deleted; it has no fields.
  found = []
  for sid, fields in entries:
    if fields:
      values = {decode(k): decode(v) for k, v in fields.items()}
      found.append(Claim(decode(sid), values["rid"], json.loads(values["input"])))
  return found


def decode(value: Any) -> Any:
  return value.decode() if isinstance(value, bytes) else value


# *** set lifecycle, from the worker manager's threads ***


def open_set(client: Any, set_id: str) -> None:
  client.set(keys(set_id)["admission"], "open")


def close_and_cancel(client: Any, set_id: str, reason: str, *, result_ttl: float = 1800, guard_ttl: float = 7200) -> int:
  """Close admission, then finalize every unfinished request as cancelled.
  `client` is a synchronous redis.Redis."""
  k = keys(set_id)
  client.set(k["admission"], "closed")
  finalize = client.register_script(FINALIZE)
  cancelled = 0
  for claim in claims(client.xrange(k["stream"])):
    status = finalize(
      keys=[k["lease"], request_key(set_id, claim.request_id), result_key(set_id, claim.request_id), k["stream"], k["unfinished"]],
      args=["", "", "cancelled", json.dumps(failed(reason, "user")), int(result_ttl), GROUP, int(max(guard_ttl, result_ttl)), time.time()],
    )
    cancelled += decode(status) == "ok"
  return cancelled
