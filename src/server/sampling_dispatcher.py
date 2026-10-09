"""The dispatcher for one sampler set: pulls accepted requests from the set's
stream and sends each through the llm-d router beside it, which picks the
sampler replica. It runs in the set's CPU pod with Envoy and the endpoint picker.

Only the holder of the set's Redis lease dispatches. It owns at most `active`
requests at once, counting those waiting to retry, and reads only enough new
entries to fill free slots. A request stays pending in the stream until one
final outcome is saved; a dispatcher that stops leaves its requests for the next
lease holder to recover.
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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pydantic import BaseModel, ValidationError

from server.sampling_streams import Claim, SamplingStreams, failed

logger = logging.getLogger(__name__)

SECONDS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800, 3600)
FINALIZED = Counter("openrl_dispatch_finalized_total", "Requests this dispatcher finalized", ["outcome"])
ATTEMPTS = Counter("openrl_dispatch_attempts_total", "Attempts started")
RETRIES = Counter("openrl_dispatch_retries_total", "Attempts that ended in a scheduled retry")
RECLAIMED = Counter("openrl_dispatch_reclaimed_total", "Stale claims taken over from another dispatcher")
OWNED = Gauge("openrl_dispatch_owned", "Requests owned, including those waiting to retry")
ACTIVE = Gauge("openrl_dispatch_active", "Attempts waiting on a sampler")
LEADER = Gauge("openrl_dispatch_leader", "1 while this dispatcher holds the set's lease")
QUEUE_WAIT = Histogram("openrl_dispatch_queue_wait_seconds", "From acceptance to the first attempt", buckets=SECONDS)
ATTEMPT_TIME = Histogram("openrl_dispatch_attempt_seconds", "Duration of one attempt", buckets=SECONDS)


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
  max_attempts: int = 4
  attempt_timeout: float = 1800
  backoff_base: float = 1
  backoff_cap: float = 60
  lease_ttl: float = 30
  renew_interval: float = 10
  reclaim_idle: float = 60
  shutdown_grace: float = 60
  ready_poll: float = 2

  @classmethod
  def from_env(cls) -> "DispatchConfig":
    def number(name: str, default: float) -> float:
      return float(os.getenv(f"OPEN_RL_DISPATCH_{name.upper()}", default))

    defaults = cls()
    config = cls(**{name: type(value)(number(name, value)) for name, value in vars(defaults).items()})
    if config.renew_interval * 2 >= config.lease_ttl or config.lease_ttl >= config.reclaim_idle:
      raise ValueError("Renew well within the lease, and keep the lease shorter than the reclaim delay")
    return config


@dataclass
class Outcome:
  """What one attempt's response means: a final result, or a retry after a delay."""

  final: dict[str, Any] | None = None
  outcome: str = "success"
  retry_after: float | None = None
  error: str = ""


@dataclass
class Dispatcher:
  streams: SamplingStreams
  set_id: str
  router_url: str
  client: httpx.AsyncClient
  ready: Callable[[], Awaitable[bool]]
  config: DispatchConfig = field(default_factory=DispatchConfig)
  owner: str = field(default_factory=lambda: f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}")

  def __post_init__(self) -> None:
    self.tasks: dict[str, asyncio.Task] = {}
    self.claims: dict[str, str] = {}
    self.leader = False
    self.leading = asyncio.Event()
    self.stopping = asyncio.Event()
    self.confirmed_at = 0.0

  # *** leadership ***

  async def keep_lease(self) -> None:
    while not self.stopping.is_set():
      try:
        held = await self.streams.lease(self.set_id, self.owner, self.config.lease_ttl)
        if held:
          self.confirmed_at = time.monotonic()
          if not self.leader:
            logger.info("Dispatcher %s leads set %s", self.owner, self.set_id)
          self.leader = True
          self.leading.set()
          LEADER.set(1)
          if self.claims and await self.streams.renew_claims(self.set_id, self.owner, list(self.claims.values())) < 0:
            held = False
          await self.drop_finished()
        if not held and self.leader:
          self.lose("lease taken by another dispatcher")
      except redis.RedisError as exc:
        # Ownership that cannot be confirmed before the lease could lapse is lost.
        if self.leader and time.monotonic() - self.confirmed_at > self.config.lease_ttl - self.config.renew_interval:
          self.lose(f"cannot confirm the lease: {exc}")
      await self.wait(self.config.renew_interval)

  def lose(self, reason: str) -> None:
    logger.warning("Dispatcher %s stops dispatching set %s: %s", self.owner, self.set_id, reason)
    self.leader = False
    self.leading.clear()
    LEADER.set(0)
    for task in list(self.tasks.values()):
      task.cancel()

  async def drop_finished(self) -> None:
    """Stop attempts for requests finalized elsewhere, such as cancelled ones."""
    for request_id in list(self.tasks):
      if (await self.streams.state(self.set_id, request_id)).get("status") != "pending" and request_id in self.tasks:
        self.tasks[request_id].cancel()

  # *** intake ***

  async def run(self) -> None:
    await self.streams.ensure_group(self.set_id)
    keeper = asyncio.create_task(self.keep_lease())
    try:
      while not self.stopping.is_set():
        if not self.leader:
          with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self.leading.wait(), 0.5)
          continue
        try:
          await self.fill()
        except redis.RedisError as exc:
          logger.warning("Dispatcher cannot read set %s: %s", self.set_id, exc)
          await self.wait(1)
    finally:
      await self.shutdown()
      keeper.cancel()
      await asyncio.gather(keeper, return_exceptions=True)

  async def fill(self) -> None:
    free = self.config.active - len(self.tasks)
    if free <= 0:
      await self.wait(0.2)
      return
    recovered = await self.streams.reclaim(self.set_id, self.owner, free, self.config.reclaim_idle)
    RECLAIMED.inc(len(recovered))
    fresh = await self.streams.read_new(self.set_id, self.owner, free - len(recovered), block_ms=1000)
    for claim in [*recovered, *fresh]:
      if claim.request_id not in self.tasks and self.leader:
        self.claims[claim.request_id] = claim.stream_id
        task = asyncio.create_task(self.handle(claim))
        self.tasks[claim.request_id] = task
        task.add_done_callback(lambda done, request_id=claim.request_id: self.finished(request_id, done))
    OWNED.set(len(self.tasks))

  def finished(self, request_id: str, task: asyncio.Task) -> None:
    self.tasks.pop(request_id, None)
    self.claims.pop(request_id, None)
    OWNED.set(len(self.tasks))
    if not task.cancelled() and (error := task.exception()) is not None:
      logger.error("Dispatch of %s stopped: %s", request_id, error)

  # *** one request ***

  async def handle(self, claim: Claim) -> None:
    request_id = claim.request_id
    while True:
      try:
        state = await self.streams.state(self.set_id, request_id)
        if state.get("status") != "pending":
          # Already final, perhaps by a completion whose reply was lost: clean up only.
          await self.finalize(request_id, "", "failure", failed("Request already finalized"))
          return
        now = time.time()
        deadline = float(state["deadline"])
        attempts = int(state.get("attempts", 0))
        if now >= deadline:
          await self.finalize(request_id, "", "expired", failed(f"Sampling deadline passed after {attempts} attempt(s)", "user"))
          return
        if attempts >= self.config.max_attempts:
          await self.finalize(request_id, "", "failure", failed(f"Sampling failed after {attempts} attempts: {state.get('last_error', '')}"))
          return
        retry_at = float(state.get("next_retry_at") or 0)
        if retry_at > now:
          await asyncio.sleep(min(retry_at, deadline) - now)
          continue
        if not await self.ready():
          # Waiting for a sampler does not use an attempt.
          await asyncio.sleep(min(self.config.ready_poll, max(deadline - now, 0)))
          continue
        attempt_id = uuid.uuid4().hex
        status, attempt = await self.streams.begin(self.set_id, request_id, self.owner, attempt_id, self.config.max_attempts)
        if status == "lease":
          return
        if status != "ok":
          continue
        ATTEMPTS.inc()
        if attempt == 1:
          QUEUE_WAIT.observe(max(time.time() - float(state["accepted_at"]), 0))
        result = await self.attempt(claim, attempt, min(self.config.attempt_timeout, deadline - time.time()))
        if result.final is not None:
          await self.finalize(request_id, attempt_id, result.outcome, result.final)
          return
        delay = result.retry_after if result.retry_after is not None else self.backoff(attempt)
        RETRIES.inc()
        await self.streams.retry(self.set_id, request_id, self.owner, attempt_id, time.time() + delay, result.error)
      except redis.RedisError as exc:
        # The outcome of the last write is unknown; the next pass reads it back first.
        logger.warning("Redis error while dispatching %s: %s", request_id, exc)
        await asyncio.sleep(1)

  async def finalize(self, request_id: str, attempt_id: str, outcome: str, result: dict[str, Any]) -> None:
    status = await self.streams.finalize(self.set_id, request_id, owner=self.owner, attempt_id=attempt_id, outcome=outcome, result=result)
    if status == "ok":
      FINALIZED.labels(outcome).inc()
    if status not in ("ok", "final", "stale", "lease"):
      logger.warning("Could not finalize %s: %s", request_id, status)

  def backoff(self, attempt: int) -> float:
    return random.uniform(0.5, 1) * min(self.config.backoff_cap, self.config.backoff_base * 2 ** (attempt - 1))

  async def attempt(self, claim: Claim, attempt: int, timeout: float) -> Outcome:
    request = claim.input
    # Each attempt gets its own engine request ID, and the sampler stops at the deadline.
    sampler_request = {**request, "request_id": f"{claim.request_id}#{attempt}", "deadline": time.time() + timeout}
    body = {
      "model": request.get("lora_id") or request["model_id"],
      "prompt": request["prompt_token_ids"],
      "max_tokens": request["max_tokens"],
      "openrl": sampler_request,
    }
    started = time.monotonic()
    ACTIVE.inc()
    try:
      response = await self.client.post(f"{self.router_url}/v1/completions", json=body, timeout=httpx.Timeout(timeout, connect=10, pool=10))
    except httpx.TimeoutException:
      return Outcome(error=f"attempt {attempt} timed out after {timeout:.0f}s")
    except httpx.TransportError as exc:
      return Outcome(error=f"attempt {attempt}: {type(exc).__name__}: {exc}")
    finally:
      ACTIVE.dec()
      ATTEMPT_TIME.observe(time.monotonic() - started)
    return self.judge(response, request["num_samples"])

  def judge(self, response: httpx.Response, num_samples: int) -> Outcome:
    detail = response.text[:500]
    if response.status_code in (429, 503):
      after = response.headers.get("retry-after")
      delay = min(float(after), self.config.backoff_cap) if after and after.replace(".", "", 1).isdigit() else None
      return Outcome(retry_after=delay, error=f"router returned {response.status_code}: {detail}")
    if response.status_code >= 500:
      return Outcome(error=f"router returned {response.status_code}: {detail}")
    if response.status_code != 200:
      # The sampler refused the request itself: retrying cannot help.
      return Outcome(final=failed(f"Sampler rejected the request ({response.status_code}): {detail}", "user"), outcome="failure")
    try:
      result = response.json()
      if result.get("type") == "RequestFailedResponse":
        return Outcome(final=failed(str(result.get("error_message")), result.get("category", "server")), outcome="failure")
      sample = SampleResult.model_validate(result, strict=True)
      if result.get("type") != "sample" or len(sample.sequences) != num_samples:
        raise ValueError(f"expected type sample with {num_samples} sequences")
    except (ValueError, ValidationError, AttributeError) as exc:
      return Outcome(final=failed(f"Sampler response broke the contract: {exc}"), outcome="failure")
    return Outcome(final=result)

  # *** shutdown ***

  def stop(self) -> None:
    self.stopping.set()

  async def wait(self, seconds: float) -> None:
    try:
      await asyncio.wait_for(self.stopping.wait(), seconds)
    except TimeoutError:
      pass

  async def shutdown(self) -> None:
    """Stop claiming, let active calls finish within the grace period, and leave
    the rest pending for the next lease holder."""
    if self.tasks:
      await asyncio.wait(list(self.tasks.values()), timeout=self.config.shutdown_grace)
    for task in list(self.tasks.values()):
      task.cancel()
    await asyncio.gather(*self.tasks.values(), return_exceptions=True)
    if self.leader:
      try:
        await self.streams.release(self.set_id, self.owner)
      except redis.RedisError:
        pass
    self.leader = False


def samplers_ready(set_id: str, namespace: str, label: str) -> Callable[[], Awaitable[bool]]:
  """Whether any of the set's sampler pods is ready, cached for a few seconds."""
  from kubernetes import client, config

  config.load_incluster_config()
  core = client.CoreV1Api()
  cache = {"at": 0.0, "ready": False}

  def check() -> bool:
    pods = core.list_namespaced_pod(namespace, label_selector=f"{label}={set_id}", _request_timeout=10).items
    return any(pod.status.phase == "Running" and any(c.type == "Ready" and c.status == "True" for c in pod.status.conditions or []) for pod in pods)

  async def ready() -> bool:
    if time.monotonic() - cache["at"] > 3:
      try:
        cache["ready"] = await asyncio.to_thread(check)
      except Exception as exc:
        logger.warning("Cannot list samplers of %s: %s", set_id, exc)
        cache["ready"] = False
      cache["at"] = time.monotonic()
    return cache["ready"]

  return ready


async def main() -> None:
  from server.scheduler_worker_manager import SAMPLER_SET_LABEL

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  # One line per sample request is noise; failures are logged by the dispatcher.
  logging.getLogger("httpx").setLevel(logging.WARNING)
  set_id = os.environ["OPEN_RL_SAMPLER_SET"]
  config = DispatchConfig.from_env()
  client = redis.from_url(os.environ["REDIS_URL"])
  streams = SamplingStreams(
    client, result_ttl=float(os.getenv("OPEN_RL_SAMPLE_RESULT_TTL", 1800)), guard_ttl=float(os.getenv("OPEN_RL_SAMPLE_GUARD_TTL", 7200))
  )
  namespace = os.getenv("POD_NAMESPACE") or open("/var/run/secrets/kubernetes.io/serviceaccount/namespace").read().strip()
  http = httpx.AsyncClient(limits=httpx.Limits(max_connections=config.active))
  dispatcher = Dispatcher(
    streams, set_id, os.getenv("OPEN_RL_ROUTER_URL", "http://127.0.0.1:8081"), http, samplers_ready(set_id, namespace, SAMPLER_SET_LABEL), config
  )
  start_http_server(int(os.getenv("OPEN_RL_DISPATCH_METRICS_PORT", "9100")))
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
