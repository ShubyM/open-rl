"""Serves one sampling pool's pending records with a sampler and saves the first result for each.

The sampler is HttpSampler, which sends through the pool's router, Envoy in front of the llm-d EPP, to its vllm serve
replicas."""

import asyncio
import contextlib
import logging
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as redis

from server.sampling_store import SamplingStore

logger = logging.getLogger(__name__)

# The dispatcher wakes for each new request, or after this long to retry a failed send and look at its replicas.
RETRY_MS = 1000
# A request no replica has answered for this long fails, so its client stops waiting.
SAMPLE_DEADLINE_SECONDS = 3600


def tmp_root() -> str:
  """OPEN_RL_TMP_DIR, or the documented default. The API server renders the readiness probe's path
  from it and hands the same value to the dispatcher and the replicas, so they all agree."""
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def revision_file(pool: str) -> Path:
  """The readiness probe compares vLLM's weight version with the revision in this file."""
  return Path(tmp_root(), "sampling", pool, "revision")


def accepted_at(r: dict[str, Any]) -> float:
  """When the record's request was accepted. A stream id starts with that time in milliseconds."""
  return int(r["entry_id"].split("-")[0]) / 1000


def failed(message: str) -> dict[str, str]:
  return {"type": "RequestFailedResponse", "error_message": message}


def to_result(body: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
  """The old sampler's result shape, from a vLLM /inference/v1/generate response."""
  logprobs = [[e["logprob"] for e in c["logprobs"]["content"]] for c in body["choices"]]
  sequences = [{"tokens": c["token_ids"], "logprobs": lp, "stop_reason": c["finish_reason"]} for c, lp in zip(body["choices"], logprobs)]
  result: dict[str, Any] = {"type": "sample", "sequences": sequences}
  if record["include_prompt_logprobs"]:
    pairs = zip(record["prompt_token_ids"], body["prompt_logprobs"], strict=True)
    result["prompt_logprobs"] = [None if e is None else e[str(token)]["logprob"] for token, e in pairs]
  return result


class Dispatcher:
  """Serves `pool` with `sampler`, which loads a revision with ensure_weights and answers a record with generate."""

  def __init__(self, store: SamplingStore, pool: str, sampler: Any, concurrency: int) -> None:
    self.store, self.pool, self.sampler, self.concurrency = store, pool, sampler, concurrency
    self.revision = ""
    # This dispatcher is the pool's only reader, so it reads each record once and drops it when its result is saved.
    self.records: dict[str, dict[str, Any]] = {}
    self.newest = "0-0"
    self.in_flight: dict[str, asyncio.Task] = {}

  async def run(self) -> None:
    try:
      while True:
        try:
          await self.dispatch()
          await self.store.wait_for_new(self.pool, self.newest, RETRY_MS)
        except Exception:
          # A Redis or Kubernetes hiccup must not end the pool's dispatching.
          logger.exception("Dispatching pool %s failed; retrying", self.pool)
          await asyncio.sleep(RETRY_MS / 1000)
    finally:
      for task in self.in_flight.values():
        task.cancel()

  async def dispatch(self) -> None:
    await self.read()
    if await self.sampler.ensure_weights(self.revision):
      # Read after the await above, so a request whose result was saved meanwhile is not sent again.
      for r in self.to_send()[: self.concurrency - len(self.in_flight)]:
        self.in_flight[r["request_id"]] = task = asyncio.create_task(self.send(r))
        task.add_done_callback(lambda _, request_id=r["request_id"]: self.in_flight.pop(request_id))

  async def read(self) -> None:
    """Reads new records, fails those past their deadline, and switches revision once nothing pending names the
    current one."""
    for r in await self.store.pending(self.pool, after=self.newest):
      self.records[r["entry_id"]] = r
      self.newest = r["entry_id"]
    for r in list(self.records.values()):
      if (waited := time.time() - accepted_at(r)) > SAMPLE_DEADLINE_SECONDS:
        logger.warning("Failing %s after %.0fs without a result", r["request_id"], waited)
        await self.set_result(r, failed(f"No sampler answered in {waited:.0f}s; pool {self.pool} serves {self.revision or 'base weights'}"))
        # A send still waiting on a replica that never answers would keep its slot forever.
        if task := self.in_flight.get(r["request_id"]):
          task.cancel()
    records = list(self.records.values())
    # Nothing pending names the current revision, so nothing is in flight on it.
    if records and all(r["revision"] != self.revision for r in records):
      revision = records[0]["revision"]
      logger.info("Pool %s switching from %r to %r", self.pool, self.revision, revision)
      self.revision = revision

  def to_send(self) -> list[dict[str, Any]]:
    return [r for r in self.records.values() if r["revision"] == self.revision and r["request_id"] not in self.in_flight]

  async def set_result(self, r: dict[str, Any], result: dict[str, Any]) -> None:
    await self.store.set_result(r, result)
    self.records.pop(r["entry_id"], None)

  async def send(self, r: dict[str, Any]) -> None:
    """Saves the sampler's answer. None, or an error, leaves the record pending to be sent again."""
    try:
      # A sampler that never answers must not hold the send past its record's deadline.
      timeout = max(0.0, accepted_at(r) + SAMPLE_DEADLINE_SECONDS - time.time())
      result = await asyncio.wait_for(self.sampler.generate(r), timeout)
    except Exception as exc:
      logger.info("Retrying %s after %r", r["request_id"], exc)
      return
    if result is not None:
      await self.set_result(r, result)


class HttpSampler:
  """Sends a pool's records through `router` to its vllm serve replicas, and keeps each replica on the pool's revision.
  `urls`, when given, lists the replicas instead of finding them by pod label."""

  def __init__(self, http: httpx.AsyncClient, pool: str, router: str, urls: list[str]) -> None:
    self.http, self.pool, self.router, self.urls = http, pool, router, urls
    self.revision: str | None = None
    self.k8s = None

  async def ensure_weights(self, revision: str) -> bool:
    """True once one replica serves `revision`."""
    if revision != self.revision:
      self.revision = revision
      self.write_revision()
    target = revision or "default"
    urls = await self.list_replicas()
    versions = await asyncio.gather(*(self.weight_version(url) for url in urls), return_exceptions=True)
    return target in versions

  async def generate(self, r: dict[str, Any]) -> dict[str, Any] | None:
    """The result, or None when the answer did not come from the record's revision."""
    target = r["revision"] or "default"
    params = {**r["sampling_params"], "logprobs": 0, "prompt_logprobs": 0 if r["include_prompt_logprobs"] else None}
    attempt_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    # The store hands prompts back as array('I'), which JSON cannot encode.
    body = {"request_id": attempt_id, "token_ids": list(r["prompt_token_ids"]), "model": r["model"], "sampling_params": params}
    url = self.router
    response = await self.http.post(f"{url}/inference/v1/generate", json=body, timeout=None)
    # Envoy names the replica that answered. Its weights must still be the target.
    replica = f"http://{name}" if (name := response.headers.get("x-openrl-upstream")) else url
    if response.status_code in (400, 404, 422):
      return failed(f"Sampler rejected the request ({response.status_code}): {response.text[:500]}")
    if response.status_code != 200 or await self.weight_version(replica) != target:
      return None
    try:
      return to_result(response.json(), r)
    except (KeyError, TypeError, ValueError) as exc:
      return failed(f"Sampler response broke the contract: {exc!r}")

  async def list_replicas(self) -> list[str]:
    if self.urls:
      return self.urls
    if self.k8s is None:
      from kubernetes import client, config

      config.load_incluster_config()
      self.k8s = client.CoreV1Api()
    selector = f"app.kubernetes.io/component=sampler,app.kubernetes.io/instance={self.pool}"
    namespace = os.getenv("OPEN_RL_WORKER_NAMESPACE", "openrl-system")
    pods = await asyncio.to_thread(self.k8s.list_namespaced_pod, namespace, label_selector=selector, _request_timeout=10)
    return [f"http://{p.status.pod_ip}:8000" for p in pods.items if p.status.pod_ip and p.status.phase == "Running"]

  async def weight_version(self, url: str) -> str:
    return (await self.http.get(f"{url}/weight_info", timeout=5)).raise_for_status().json()["weight_version"]

  def write_revision(self) -> None:
    path = revision_file(self.pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(self.revision or "default")
    path.with_suffix(".tmp").rename(path)


def redis_client(url: str) -> redis.Redis:
  """When replicas come up, every in-flight send saves its result at once. Past max_connections a
  save waits for a free connection; redis-py's default pool of 100 raises instead."""
  return redis.Redis.from_pool(redis.BlockingConnectionPool.from_url(url, decode_responses=True, max_connections=64, timeout=30))


def http_client() -> httpx.AsyncClient:
  """Sends have no connection limit, so a burst goes out at once. httpx scans its whole pool for each idle
  connection whenever a request starts or ends, so the hundreds a burst leaves idle stall this loop for
  seconds. Few are kept."""
  return httpx.AsyncClient(limits=httpx.Limits(max_connections=None, max_keepalive_connections=32))


async def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  logging.getLogger("httpx").setLevel(logging.WARNING)
  router = os.getenv("OPEN_RL_SAMPLER_ROUTER", "http://127.0.0.1:8081")
  store, pool = SamplingStore(redis_client(os.environ["REDIS_URL"])), os.environ["OPEN_RL_SAMPLER_POOL"]
  async with http_client() as http:
    sampler = HttpSampler(http, pool, router, [])
    task = asyncio.create_task(Dispatcher(store, pool, sampler, int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256"))).run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await task


if __name__ == "__main__":
  asyncio.run(main())
