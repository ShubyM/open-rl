"""Serves one sampling pool's pending records with a sampler and saves the first result for each.

The sampler is either vLLM in this process (vllm_sampler.Sampler), for a pool of one replica, or HttpSampler, which
sends through the pool's router, Envoy in front of the llm-d EPP, to its vllm serve replicas. A replica that shares
its GPU with its trainer takes a time-slicer turn around each batch it runs."""

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as redis

from accel_timeslicer.time_slicer import TimeSlicerClient
from accel_timeslicer.workload import WorkloadRef
from server.sampling_store import SamplingStore

logger = logging.getLogger(__name__)

# The dispatcher wakes for each new request, or after this long to retry a failed send and look at its replicas.
RETRY_MS = 1000
# A request no replica has answered for this long fails, so its client stops waiting.
SAMPLE_DEADLINE_SECONDS = 3600
# The version a replica reports once its weights may not match any revision. Never a path, so no pool revision equals it.
UNKNOWN_WEIGHTS = "unknown"


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


def delta_applies(revision: str, current: str) -> bool:
  """Whether weights at `current` can load `revision`. A delta applies on any save at the weights it starts from or
  reaches. Deltas written before the trainer listed them name one parent."""
  metadata = json.loads(Path(revision, "metadata.json").read_text())
  bases = [base or "default" for base in metadata.get("bases", [metadata.get("parent")])]
  return metadata.get("format") != "sparse_delta" or current in bases


@asynccontextmanager
async def holding_gpu(time_slicer: TimeSlicerClient | None, workload: WorkloadRef | None, sampler: Any = None):
  """Hold the time-slicer turn for the body, waking the sampler inside it. Without a slicer the GPU is ours."""
  if time_slicer is None:
    yield
    return
  async with time_slicer.acquire(workload):
    if sampler is not None:
      await sampler.wake()
    try:
      yield
    finally:
      if sampler is not None:
        await sampler.sleep()


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
  """Serves `pool` with `sampler`, which loads a revision with ensure_weights and answers a record with generate. With a
  time slicer every batch runs inside one GPU turn, so the sampler shares its GPU with the trainer."""

  def __init__(
    self,
    store: SamplingStore,
    pool: str,
    sampler: Any,
    concurrency: int,
    time_slicer: TimeSlicerClient | None = None,
    workload: WorkloadRef | None = None,
  ) -> None:
    self.store, self.pool, self.sampler, self.concurrency = store, pool, sampler, concurrency
    self.time_slicer, self.workload = time_slicer, workload
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
    if self.time_slicer is not None:
      if self.to_send():
        async with holding_gpu(self.time_slicer, self.workload, self.sampler):
          await self.read()  # requests that arrived while the turn was pending join it
          if await self.sampler.ensure_weights(self.revision):
            await asyncio.gather(*(self.send(r) for r in self.to_send()[: self.concurrency]))
      return
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
      if revision and not Path(revision, "metadata.json").is_file():
        # Retention can delete a revision between the API server's check and its pin.
        for r in [r for r in records if r["revision"] == revision]:
          await self.set_result(r, failed(f"Sampler weights {revision} no longer exist"))
        return
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
  A run on one machine has no router and lists its replicas in `urls`."""

  def __init__(self, http: httpx.AsyncClient, pool: str, router: str | None, urls: list[str]) -> None:
    self.http, self.pool, self.router, self.urls = http, pool, router, urls
    self.revision: str | None = None
    self.k8s = None
    self.loading: dict[str, asyncio.Task] = {}
    self.paused: dict[str, float] = {}  # when each replica last paused to load weights
    self.checked: set[str] = set()  # replicas checked for a pause or an open update, checked again after a load of ours breaks off

  async def ensure_weights(self, revision: str) -> bool:
    """Starts a load on every replica not at `revision`. True once one replica serves it."""
    if revision != self.revision:
      self.revision = revision
      self.write_revision()
    target = revision or "default"
    urls = await self.list_replicas()
    versions = await asyncio.gather(*(self.inspect(url) for url in urls), return_exceptions=True)
    for url, version in zip(urls, versions):
      # A replica still starting does not answer yet, and a pool on base weights never loads any.
      if revision and isinstance(version, str) and version != target and url not in self.loading:
        self.loading[url] = asyncio.create_task(self.load_weights(url, revision))
    return target in versions

  async def generate(self, r: dict[str, Any]) -> dict[str, Any] | None:
    """The result, or None when the answer did not come from the record's revision."""
    target, sent = r["revision"] or "default", time.monotonic()
    params = {**r["sampling_params"], "logprobs": 0, "prompt_logprobs": 0 if r["include_prompt_logprobs"] else None}
    attempt_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    # The store hands prompts back as array('I'), which JSON cannot encode.
    body = {"request_id": attempt_id, "token_ids": list(r["prompt_token_ids"]), "model": r["model"], "sampling_params": params}
    url = self.router or random.choice(self.urls)
    response = await self.http.post(f"{url}/inference/v1/generate", json=body, timeout=None)
    # Envoy names the replica that answered. Its weights must still be the target, with no pause to load others since this was sent.
    replica = f"http://{name}" if (name := response.headers.get("x-openrl-upstream")) else url
    if response.status_code in (400, 404, 422):
      return failed(f"Sampler rejected the request ({response.status_code}): {response.text[:500]}")
    if response.status_code != 200 or await self.weight_version(replica) != target or self.paused.get(replica, 0.0) >= sent:
      return None
    try:
      return to_result(response.json(), r)
    except (KeyError, TypeError, ValueError) as exc:
      return failed(f"Sampler response broke the contract: {exc!r}")

  async def close(self) -> None:
    """A load cut off midway leaves its replica paused, so loads get a moment to finish and resume before the HTTP client closes."""
    if self.loading:
      _, pending = await asyncio.wait(list(self.loading.values()), timeout=5)
      for task in pending:
        task.cancel()

  async def load_weights(self, url: str, revision: str) -> None:
    try:
      if not Path(revision, "metadata.json").is_file():
        return  # the dispatcher fails the revision's records
      if not delta_applies(revision, current := await self.weight_version(url)):
        raise RuntimeError(f"{url} holds {current!r}, not a revision delta {revision!r} applies on")
      self.paused[url] = time.monotonic()
      # A load that breaks off may leave the replica paused or inside its update, so it is checked again unless this one ends cleanly.
      self.checked.discard(url)
      await self.post(url, "/pause?mode=wait&clear_cache=false")
      updating = True
      try:
        await self.post(url, "/start_weight_update")
        await self.post(url, "/update_weights", {"update_info": {"target_weights_path": revision}})
        await self.post(url, "/finish_weight_update", {"weight_version": revision})
        updating = False
      except Exception:
        await self.forget_weights(url)
        updating = False
        raise
      else:
        if not (await self.post(url, "/reset_prefix_cache")).json()["success"]:
          raise RuntimeError("vLLM did not reset the prefix cache")
      finally:
        # A load cut off inside its update leaves the replica paused, so nothing is served from part of a revision
        # before a dispatcher finds the open update and forgets the weights.
        if not updating:
          await self.post(url, "/resume")
      self.checked.add(url)
    except Exception as exc:
      logger.warning("Loading %r on %s failed: %r", revision, url, exc)  # the next dispatch tries again
    finally:
      self.loading.pop(url)

  async def inspect(self, url: str) -> str:
    """The version a replica holds. On first sight any weight update an earlier load left open is closed and its weights
    forgotten, paused or not, since a dispatcher stopped mid-update used to resume the replica. A dispatcher killed
    mid-load, or a failed /resume, leaves a replica paused, where the router would queue requests forever, so one found
    paused outside this dispatcher's loads is resumed."""
    if url not in self.checked and url not in self.loading:
      await self.forget_weights(url, if_updating=True)
      if (await self.http.get(f"{url}/is_paused", timeout=5)).raise_for_status().json()["is_paused"]:
        logger.info("Resuming %s, left paused by an earlier load", url)
        await self.post(url, "/resume")
      self.checked.add(url)
    return await self.weight_version(url)

  async def forget_weights(self, url: str, if_updating: bool = False) -> None:
    """Closes an update a cut-off load left active, which refuses every later start, and has the replica report
    UNKNOWN_WEIGHTS until a full load, since it may hold part of a revision. vLLM has no abort, but a bare finish closes
    an active update and is refused when none is, so with `if_updating` a replica that was not updating keeps its version."""
    # It waits behind an update a dead dispatcher left running, but a wedged replica must not hold up the pool.
    response = await self.http.post(f"{url}/finish_weight_update", timeout=300)
    if if_updating and response.status_code != 200:
      return
    logger.warning("Weights on %s may be partial, so it reports %r until a full load replaces them", url, UNKNOWN_WEIGHTS)
    await self.post(url, "/update_weight_version", {"new_version": UNKNOWN_WEIGHTS})

  async def post(self, url: str, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
    return (await self.http.post(url + path, json=body, timeout=None)).raise_for_status()

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
  urls = [url for url in os.getenv("OPEN_RL_SAMPLER_ENDPOINTS", "").split(",") if url]
  router = None if urls else os.getenv("OPEN_RL_SAMPLER_ROUTER", "http://127.0.0.1:8081")
  store, pool = SamplingStore(redis_client(os.environ["REDIS_URL"])), os.environ["OPEN_RL_SAMPLER_POOL"]
  async with http_client() as http:
    sampler = HttpSampler(http, pool, router, urls)
    task = asyncio.create_task(Dispatcher(store, pool, sampler, int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256"))).run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    try:
      with contextlib.suppress(asyncio.CancelledError):
        await task
    finally:
      await sampler.close()


if __name__ == "__main__":
  asyncio.run(main())
