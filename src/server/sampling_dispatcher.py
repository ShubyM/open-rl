"""The dispatcher for one sampler set: pulls requests from the set's Redis queue
and sends each through the llm-d router (Envoy + EPP) beside it on localhost.

Because each set runs one Recreate dispatcher pod at a time, crash recovery needs
no consumer groups or heartbeats: requests are moved atomically to an in-flight
list via BLMOVE, and on startup any stranded in-flight entries are pushed back to
the head of the queue. Results are written through the standard RequestStore.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import signal
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
import redis.asyncio as redis
from pydantic import BaseModel, ValidationError

from server.store import RedisStore, RequestStore

logger = logging.getLogger(__name__)

# The label every sampler pod of a set carries, which the set's router selects by.
SAMPLER_SET_LABEL = "openrl.io/sampler-set"


def sampler_set(runtime: str) -> str:
  """A label-safe name for a runtime's samplers."""
  return "set-" + hashlib.sha256(runtime.encode()).hexdigest()[:16]


def queue_key(set_id: str) -> str:
  return f"open_rl:sampler_queue:{set_id}"


def inflight_key(set_id: str) -> str:
  return f"open_rl:inflight:{set_id}"


def admission_key(set_id: str) -> str:
  return f"open_rl:admission:{set_id}"


def failed(message: str, category: str = "server") -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message, "category": category}


class SetClosed(RuntimeError):
  pass


class SetFull(RuntimeError):
  pass


async def unfinished_count(client: redis.Redis, set_id: str) -> int:
  return int(await client.llen(queue_key(set_id))) + int(await client.llen(inflight_key(set_id)))


async def enqueue_to_set(
  client: redis.Redis,
  store: RequestStore,
  set_id: str,
  request: dict[str, Any],
  *,
  limit: int,
  deadline_seconds: float,
) -> str:
  state = await client.get(admission_key(set_id))
  if (state.decode() if isinstance(state, bytes) else state) != "open":
    raise SetClosed(f"Sampler set {set_id} is not accepting requests")
  if await unfinished_count(client, set_id) >= limit:
    raise SetFull(f"Sampler set {set_id} has {limit} unfinished requests; try again later")
  queued = {**request, "model_id": set_id, "deadline": time.time() + deadline_seconds}
  await store.put_sampling_request(queued)
  return str(queued["request_id"])


def open_set(client: Any, set_id: str) -> None:
  client.set(admission_key(set_id), "open")


def close_and_cancel(client: Any, set_id: str, reason: str, *, ttl: int = 300) -> int:
  """Close the set to new requests and fail every queued or in-flight request."""
  client.set(admission_key(set_id), "closed")
  cancelled = 0
  payload = json.dumps(failed(reason, "user"))
  for key in (queue_key(set_id), inflight_key(set_id)):
    while raw := client.lpop(key):
      req_id = json.loads(raw)["request_id"]
      future_key = f"open_rl:future:{req_id}"
      if not client.lindex(future_key, -1):
        with client.pipeline(transaction=True) as pipe:
          pipe.rpush(future_key, payload)
          pipe.ltrim(future_key, -1, -1)
          pipe.expire(future_key, ttl)
          pipe.execute()
        cancelled += 1
  return cancelled


class SampleSequence(BaseModel):
  tokens: list[int]
  logprobs: list[float | None] | None = None
  stop_reason: str | None = None


class SampleResult(BaseModel):
  sequences: list[SampleSequence]
  prompt_logprobs: list[float | None] | None = None
  prompt_cache_hit_tokens: int = 0


@dataclass
class DispatchConfig:
  active: int = 256
  attempt_timeout: float = 1800
  backoff_base: float = 1
  backoff_cap: float = 60
  shutdown_grace: float = 60

  @classmethod
  def from_env(cls) -> "DispatchConfig":
    defaults = cls()
    return cls(**{name: type(value)(float(os.getenv(f"OPEN_RL_DISPATCH_{name.upper()}", value))) for name, value in vars(defaults).items()})


@dataclass
class Outcome:
  result: dict[str, Any] | None = None
  retry_after: float | None = None
  error: str = ""


class Dispatcher:
  def __init__(
    self,
    client: redis.Redis,
    set_id: str,
    router_url: str,
    http: httpx.AsyncClient,
    config: DispatchConfig | None = None,
    store: RequestStore | None = None,
  ) -> None:
    self.client = client
    self.store: RequestStore = store if store is not None else RedisStore(client)
    self.set_id = set_id
    self.router_url = router_url
    self.http = http
    self.config = config or DispatchConfig()
    self.tasks: dict[str, asyncio.Task] = {}
    self.stopping = asyncio.Event()

  async def run(self) -> None:
    while await self.client.lmove(inflight_key(self.set_id), queue_key(self.set_id), "RIGHT", "LEFT"):
      pass
    try:
      while not self.stopping.is_set():
        try:
          await self.fill()
        except redis.RedisError as exc:
          logger.warning("Cannot read queue for set %s: %s", self.set_id, exc)
          await self.wait(1)
    finally:
      await self.shutdown()

  async def fill(self) -> None:
    free = self.config.active - len(self.tasks)
    if free <= 0:
      await self.wait(0.05)
      return
    claimed = []
    for _ in range(free):
      item = await self.client.lmove(queue_key(self.set_id), inflight_key(self.set_id), "LEFT", "RIGHT")
      if item is None:
        break
      claimed.append(item)
    if not claimed:
      await self.wait(0.05)
      return
    for raw in claimed:
      token = raw.decode() if isinstance(raw, bytes) else raw
      req_id = json.loads(token)["request_id"]
      if req_id not in self.tasks:
        task = asyncio.create_task(self.handle(token))
        self.tasks[req_id] = task
        task.add_done_callback(lambda done, rid=req_id: self.finished(rid, done))

  def finished(self, req_id: str, task: asyncio.Task) -> None:
    self.tasks.pop(req_id, None)
    if not task.cancelled() and (error := task.exception()) is not None:
      logger.error("Dispatch of %s stopped: %s", req_id, error)

  async def handle(self, raw: str) -> None:
    req = json.loads(raw)
    req_id = str(req["request_id"])
    deadline = float(req.get("deadline", time.time() + self.config.attempt_timeout))
    attempt = 0
    while True:
      try:
        if await self.client.lindex(f"open_rl:future:{req_id}", -1):
          await self.client.lrem(inflight_key(self.set_id), 1, raw)
          return
        remaining = deadline - time.time()
        if remaining <= 0:
          await self.complete(req_id, raw, failed(f"Sampling deadline passed after {attempt} attempt(s)", "user"))
          return
        attempt += 1
        outcome = await self.attempt(req, min(self.config.attempt_timeout, remaining))
        if outcome.result is not None:
          await self.complete(req_id, raw, outcome.result)
          return
        delay = outcome.retry_after if outcome.retry_after is not None else self.backoff(attempt)
        logger.info("Retrying %s in %.1fs: %s", req_id, delay, outcome.error)
        await asyncio.sleep(min(delay, max(deadline - time.time(), 0)))
      except redis.RedisError as exc:
        logger.warning("Redis error while dispatching %s: %s", req_id, exc)
        await asyncio.sleep(1)

  async def complete(self, req_id: str, raw: str, result: dict[str, Any]) -> None:
    if not await self.client.lindex(f"open_rl:future:{req_id}", -1):
      await self.store.set_future(req_id, result)
    await self.client.lrem(inflight_key(self.set_id), 1, raw)

  def backoff(self, attempt: int) -> float:
    return random.uniform(0.5, 1) * min(self.config.backoff_cap, self.config.backoff_base * 2 ** (attempt - 1))

  async def attempt(self, req: dict[str, Any], timeout: float) -> Outcome:
    sampler_req = {**req, "request_id": f"{req['request_id']}#{uuid.uuid4().hex[:8]}", "deadline": time.time() + timeout}
    body = {
      "model": req.get("lora_id") or req["model_id"],
      "prompt": req["prompt_token_ids"],
      "max_tokens": req["max_tokens"],
      "openrl": sampler_req,
    }
    try:
      resp = await self.http.post(f"{self.router_url}/v1/completions", json=body, timeout=httpx.Timeout(timeout, connect=10, pool=10))
    except httpx.TimeoutException:
      return Outcome(error=f"timed out after {timeout:.0f}s")
    except httpx.TransportError as exc:
      return Outcome(error=f"{type(exc).__name__}: {exc}")
    return self.judge(resp, req["num_samples"])

  def judge(self, resp: httpx.Response, num_samples: int) -> Outcome:
    detail = resp.text[:500]
    if resp.status_code in (429, 503):
      after = resp.headers.get("retry-after")
      delay = min(float(after), self.config.backoff_cap) if after and after.replace(".", "", 1).isdigit() else None
      return Outcome(retry_after=delay, error=f"{resp.status_code}: {detail}")
    if resp.status_code >= 500:
      return Outcome(error=f"{resp.status_code}: {detail}")
    if resp.status_code != 200:
      return Outcome(result=failed(f"Sampler rejected the request ({resp.status_code}): {detail}", "user"))
    try:
      result = resp.json()
      if result.get("type") == "RequestFailedResponse":
        return Outcome(result=failed(str(result.get("error_message")), result.get("category", "server")))
      sample = SampleResult.model_validate(result, strict=True)
      if result.get("type") != "sample" or len(sample.sequences) != num_samples:
        raise ValueError(f"expected type sample with {num_samples} sequences")
    except (ValueError, ValidationError, AttributeError) as exc:
      return Outcome(result=failed(f"Sampler response broke the contract: {exc}"))
    return Outcome(result=result)

  def stop(self) -> None:
    self.stopping.set()

  async def wait(self, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
      await asyncio.wait_for(self.stopping.wait(), seconds)

  async def shutdown(self) -> None:
    if self.tasks:
      await asyncio.wait(list(self.tasks.values()), timeout=self.config.shutdown_grace)
    for task in list(self.tasks.values()):
      task.cancel()
    await asyncio.gather(*self.tasks.values(), return_exceptions=True)


async def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  logging.getLogger("httpx").setLevel(logging.WARNING)
  config = DispatchConfig.from_env()
  client = redis.from_url(os.environ["REDIS_URL"])
  http = httpx.AsyncClient(limits=httpx.Limits(max_connections=config.active))
  dispatcher = Dispatcher(client, os.environ["OPEN_RL_SAMPLER_SET"], os.getenv("OPEN_RL_ROUTER_URL", "http://127.0.0.1:8081"), http, config)
  loop = asyncio.get_running_loop()
  for sig in (signal.SIGTERM, signal.SIGINT):
    loop.add_signal_handler(sig, dispatcher.stop)
  try:
    await dispatcher.run()
  finally:
    await http.aclose()
    await client.aclose()


if __name__ == "__main__":
  asyncio.run(main())
