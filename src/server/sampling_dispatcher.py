"""The dispatcher for one sampler set: pulls accepted requests from the set's stream
and sends each through the llm-d router beside it, which picks the sampler replica.
It runs in the set's CPU pod with Envoy and the endpoint picker.

It works on at most `active` requests at once, counting those waiting to retry, and
reads only enough entries to fill free slots. Each request is retried until it
succeeds, fails for good, or reaches its deadline; then its result is saved and its
entry acknowledged. A dispatcher that stops leaves unfinished entries in the stream,
and the next one reclaims them.
"""

import asyncio
import contextlib
import logging
import os
import random
import signal
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import redis.asyncio as redis
from pydantic import BaseModel, ValidationError

from server.sampling_streams import Claim, SamplingStreams, failed

logger = logging.getLogger(__name__)


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
  touch_interval: float = 30
  reclaim_idle: float = 120
  shutdown_grace: float = 60

  @classmethod
  def from_env(cls) -> "DispatchConfig":
    defaults = cls()
    config = cls(**{name: type(value)(float(os.getenv(f"OPEN_RL_DISPATCH_{name.upper()}", value))) for name, value in vars(defaults).items()})
    if config.reclaim_idle < 2 * config.touch_interval:
      raise ValueError("Reclaim only entries untouched for at least two touch intervals")
    return config


@dataclass
class Outcome:
  """What one attempt's response means: a result to save, or a retry after a delay."""

  result: dict[str, Any] | None = None
  retry_after: float | None = None
  error: str = ""


@dataclass
class Dispatcher:
  streams: SamplingStreams
  set_id: str
  router_url: str
  client: httpx.AsyncClient
  config: DispatchConfig = field(default_factory=DispatchConfig)
  consumer: str = field(default_factory=lambda: f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}")

  def __post_init__(self) -> None:
    self.tasks: dict[str, asyncio.Task] = {}
    self.stopping = asyncio.Event()

  async def run(self) -> None:
    await self.streams.ensure_group(self.set_id)
    toucher = asyncio.create_task(self.keep_claims())
    try:
      while not self.stopping.is_set():
        try:
          await self.fill()
        except redis.RedisError as exc:
          logger.warning("Cannot read set %s: %s", self.set_id, exc)
          await self.wait(1)
    finally:
      toucher.cancel()
      await asyncio.gather(toucher, return_exceptions=True)
      await self.shutdown()

  async def fill(self) -> None:
    free = self.config.active - len(self.tasks)
    if free <= 0:
      await self.wait(0.2)
      return
    recovered = await self.streams.reclaim(self.set_id, self.consumer, free, self.config.reclaim_idle)
    fresh = await self.streams.read_new(self.set_id, self.consumer, free - len(recovered), block_ms=1000)
    for claim in [*recovered, *fresh]:
      if claim.entry_id not in self.tasks:
        task = asyncio.create_task(self.handle(claim))
        self.tasks[claim.entry_id] = task
        task.add_done_callback(lambda done, entry=claim.entry_id: self.finished(entry, done))

  def finished(self, entry_id: str, task: asyncio.Task) -> None:
    self.tasks.pop(entry_id, None)
    if not task.cancelled() and (error := task.exception()) is not None:
      logger.error("Dispatch of %s stopped: %s", entry_id, error)

  async def keep_claims(self) -> None:
    """Touch the entries being worked on, so no other dispatcher reclaims them."""
    while not self.stopping.is_set():
      await self.wait(self.config.touch_interval)
      try:
        await self.streams.touch(self.set_id, self.consumer, list(self.tasks))
      except redis.RedisError as exc:
        logger.warning("Cannot touch claims in set %s: %s", self.set_id, exc)

  async def handle(self, claim: Claim) -> None:
    deadline = float(claim.input["deadline"])
    attempt = 0
    while True:
      try:
        # Finished elsewhere, or by a save whose reply was lost: just clean up.
        if await self.streams.finished(self.set_id, claim.entry_id):
          await self.streams.finish(self.set_id, claim.entry_id, failed("Request already finished"))
          return
        remaining = deadline - time.time()
        if remaining <= 0:
          await self.streams.finish(self.set_id, claim.entry_id, failed(f"Sampling deadline passed after {attempt} attempt(s)", "user"))
          return
        attempt += 1
        outcome = await self.attempt(claim, min(self.config.attempt_timeout, remaining))
        if outcome.result is not None:
          await self.streams.finish(self.set_id, claim.entry_id, outcome.result)
          return
        delay = outcome.retry_after if outcome.retry_after is not None else self.backoff(attempt)
        logger.info("Retrying %s in %.1fs: %s", claim.request_id, delay, outcome.error)
        await asyncio.sleep(min(delay, max(deadline - time.time(), 0)))
      except redis.RedisError as exc:
        logger.warning("Redis error while dispatching %s: %s", claim.request_id, exc)
        await asyncio.sleep(1)

  def backoff(self, attempt: int) -> float:
    return random.uniform(0.5, 1) * min(self.config.backoff_cap, self.config.backoff_base * 2 ** (attempt - 1))

  async def attempt(self, claim: Claim, timeout: float) -> Outcome:
    request = claim.input
    # Each attempt has its own engine request ID, and the sampler stops at the deadline.
    sampler_request = {**request, "request_id": f"{claim.request_id}#{uuid.uuid4().hex[:8]}", "deadline": time.time() + timeout}
    body = {
      "model": request.get("lora_id") or request["model_id"],
      "prompt": request["prompt_token_ids"],
      "max_tokens": request["max_tokens"],
      "openrl": sampler_request,
    }
    try:
      response = await self.client.post(f"{self.router_url}/v1/completions", json=body, timeout=httpx.Timeout(timeout, connect=10, pool=10))
    except httpx.TimeoutException:
      return Outcome(error=f"timed out after {timeout:.0f}s")
    except httpx.TransportError as exc:
      return Outcome(error=f"{type(exc).__name__}: {exc}")
    return self.judge(response, request["num_samples"])

  def judge(self, response: httpx.Response, num_samples: int) -> Outcome:
    detail = response.text[:500]
    if response.status_code in (429, 503):
      after = response.headers.get("retry-after")
      delay = min(float(after), self.config.backoff_cap) if after and after.replace(".", "", 1).isdigit() else None
      return Outcome(retry_after=delay, error=f"{response.status_code}: {detail}")
    if response.status_code >= 500:
      return Outcome(error=f"{response.status_code}: {detail}")
    if response.status_code != 200:
      # The sampler refused the request itself: retrying cannot help.
      return Outcome(result=failed(f"Sampler rejected the request ({response.status_code}): {detail}", "user"))
    try:
      result = response.json()
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
    """Let active calls finish within the grace period; the rest stay pending."""
    if self.tasks:
      await asyncio.wait(list(self.tasks.values()), timeout=self.config.shutdown_grace)
    for task in list(self.tasks.values()):
      task.cancel()
    await asyncio.gather(*self.tasks.values(), return_exceptions=True)


async def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  # One line per sample request is noise; failures and retries are logged above.
  logging.getLogger("httpx").setLevel(logging.WARNING)
  config = DispatchConfig.from_env()
  client = redis.from_url(os.environ["REDIS_URL"])
  http = httpx.AsyncClient(limits=httpx.Limits(max_connections=config.active))
  streams = SamplingStreams(client, result_ttl=float(os.getenv("OPEN_RL_SAMPLE_RESULT_TTL", 1800)))
  dispatcher = Dispatcher(streams, os.environ["OPEN_RL_SAMPLER_SET"], os.getenv("OPEN_RL_ROUTER_URL", "http://127.0.0.1:8081"), http, config)
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
