"""Sends one sampling pool's pending records through its router, Envoy in front of the llm-d EPP, and commits the first result.

A replica that shares its GPU with its trainer is off the GPU between turns, checkpointed by the node's time slicer, so
the dispatcher holds its lease for every call it makes to it. It acquires the workload, wakes vLLM, inspects it, loads
any weights, sends to that replica directly, puts vLLM to sleep and releases. Nothing goes through the router to such a
replica, since the router cannot tell whose turn it is."""

import asyncio
import contextlib
import json
import logging
import os
import random
import signal
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import redis.asyncio as redis

from accel_timeslicer.time_slicer import DEFAULT_TCP_PORT, SocketTimeSlicerClient, TimeSlicerClient, time_slicer_client_from_env, workload_from_env
from accel_timeslicer.workload import SAMPLER_CLAIM, WorkloadRef
from server.sampling_store import SamplingStore

logger = logging.getLogger(__name__)

# The dispatcher wakes for each new request, or after this long to retry a failed send and look at its replicas.
RETRY_MS = 1000
# A request no replica has answered for this long fails, so its client stops waiting.
SAMPLE_DEADLINE_SECONDS = 3600
# The version a replica reports once its weights may not match any revision. Never a path, so no pool revision equals it.
UNKNOWN_WEIGHTS = "unknown"
# A replica that shares its GPU gives it back after this long with requests still pending, so its trainer gets a turn.
MAX_TURN_SECONDS = float(os.getenv("OPEN_RL_SAMPLER_TURN_SECONDS", "60"))
# A turn ends once the replica has had nothing to send for this long.
IDLE_SECONDS = 1.0
# Requests still running when a turn ends get this long to answer. The rest are cancelled and sent again in a later turn.
TURN_GRACE_SECONDS = 10.0
# A call that wakes, prepares or sleeps a replica gives up after this long, so a wedged replica cannot keep its lease,
# and with it the trainer's GPU, for good. A weight update reads a whole checkpoint from shared storage, so it gets longer.
CALL_TIMEOUT_SECONDS = 300.0
UPDATE_TIMEOUT_SECONDS = 1800.0
# The worker manager sets this in the env of a replica that shares its GPU, and of its dispatcher on one machine.
TIME_SLICED_SAMPLER = "OPEN_RL_TIME_SLICED_SAMPLER"
# A time-sliced replica is leased with its workload and a client to the time slicer on its node.
Lease = tuple[WorkloadRef, TimeSlicerClient]


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
  """Sends through `router`. A run on one machine has no router and lists its replicas in `urls`."""

  def __init__(self, store: SamplingStore, pool: str, http: httpx.AsyncClient, router: str | None, urls: list[str], concurrency: int) -> None:
    self.store, self.pool, self.http, self.router, self.urls, self.concurrency = store, pool, http, router, urls, concurrency
    self.revision, self.k8s = "", None
    # This dispatcher is the pool's only committer, so it reads each record once and drops it on commit.
    self.records: dict[str, dict[str, Any]] = {}
    self.newest = "0-0"
    self.in_flight: dict[str, asyncio.Task] = {}
    self.loading: dict[str, asyncio.Task] = {}
    self.paused: dict[str, float] = {}  # when each replica last paused to load weights
    self.checked: set[str] = set()  # replicas checked for a pause or an open update, checked again after a load of ours breaks off
    # The replicas that share their GPU, each with the turn in progress on it.
    self.leases: dict[str, Lease] = {}
    self.turns: dict[str, asyncio.Task] = {}
    self.stopping = False

  async def run(self) -> None:
    self.write_revision()
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
      self.stopping = True
      for task in self.in_flight.values():
        task.cancel()
      # A load cut off midway leaves its replica paused, and a turn cut off leaves it awake on the GPU, so both get a
      # moment to finish, resume, sleep and release before the HTTP client closes.
      if pending := [*self.loading.values(), *self.turns.values()]:
        _, pending = await asyncio.wait(pending, timeout=5)
        for task in pending:
          task.cancel()

  async def dispatch(self) -> None:
    for r in await self.store.pending(self.pool, after=self.newest):
      self.records[r["entry_id"]] = r
      self.newest = r["entry_id"]
    for r in list(self.records.values()):
      if (waited := time.time() - accepted_at(r)) > SAMPLE_DEADLINE_SECONDS:
        logger.warning("Failing %s after %.0fs without a result", r["request_id"], waited)
        await self.commit(r, failed(f"No sampler answered in {waited:.0f}s; pool {self.pool} serves {self.revision or 'base weights'}"))
        # A send still waiting on a replica that never answers would keep its slot forever.
        if task := self.in_flight.get(r["request_id"]):
          task.cancel()
    records = list(self.records.values())
    # Nothing pending names the current revision, so nothing is in flight on it.
    if records and all(r["revision"] != self.revision for r in records):
      logger.info("Pool %s switching from %r to %r", self.pool, self.revision, records[0]["revision"])
      self.revision = records[0]["revision"]
      self.write_revision()
    target = self.revision or "default"
    replicas = await self.list_replicas()
    # A time-sliced replica is off the GPU between turns, so it is only looked at inside one.
    urls, leased = [u for u in replicas if u not in self.leases], [u for u in replicas if u in self.leases]
    versions = await asyncio.gather(*(self.inspect(url) for url in urls), return_exceptions=True)
    for url, version in zip(urls, versions):
      # A replica still starting does not answer yet, and a pool on base weights never loads any.
      if self.revision and isinstance(version, str) and version != target and url not in self.loading:
        self.loading[url] = asyncio.create_task(self.load_weights(url, self.revision))
    if target in versions:
      # Read after the awaits above, so a request that committed meanwhile is not sent again.
      for r in self.to_send()[: self.concurrency - len(self.in_flight)]:
        self.start_send(r)
    if self.to_send():
      for url in leased:
        if url not in self.turns and not self.leases[url][1].faulted:
          self.turns[url] = task = asyncio.create_task(self.turn(url))
          task.add_done_callback(lambda _, url=url: self.turns.pop(url))

  def to_send(self) -> list[dict[str, Any]]:
    return [r for r in self.records.values() if r["revision"] == self.revision and r["request_id"] not in self.in_flight]

  def start_send(self, r: dict[str, Any], url: str | None = None) -> asyncio.Task:
    self.in_flight[r["request_id"]] = task = asyncio.create_task(self.send(r, url))
    task.add_done_callback(lambda _, request_id=r["request_id"]: self.in_flight.pop(request_id))
    return task

  async def turn(self, url: str) -> None:
    """Turns on a replica that shares its GPU, one after another while requests for it keep coming. A turn that
    ends on the time bound with requests still pending is followed by the next at once; the slicer serves the
    trainer first if it is waiting."""
    workload, slicer = self.leases[url]
    while not self.stopping:
      try:
        backlog = await self.hold(url, workload, slicer)
        if slicer.faulted:
          raise RuntimeError(f"the time slicer could not park the replica, which still holds its GPU: {slicer.faulted}")
      except Exception as exc:
        logger.warning("Turn on %s failed: %r", url, exc)
        return
      if not backlog:
        return

  async def hold(self, url: str, workload: WorkloadRef, slicer: TimeSlicerClient) -> bool:
    """One turn. It holds the replica's lease throughout, with vLLM woken and inspected, any weight load done as a whole,
    the pending requests sent to the replica directly, then vLLM asleep before the lease is released. The turn ends once
    the replica has been idle for IDLE_SECONDS, when the dispatcher stops, or after MAX_TURN_SECONDS so the trainer gets
    its turn, which is the one case it returns True, for requests still pending. Requests in flight get TURN_GRACE_SECONDS
    to finish, and any still running then are cancelled, to be sent again in a later turn."""
    started, idle_since, sends = time.monotonic(), None, set()
    async with slicer.acquire(workload):
      await self.post(url, "/wake_up")
      try:
        # A replica an earlier dispatcher left paused or inside a weight update is put right here, inside its turn.
        version = await self.inspect(url)
        while not self.stopping:
          target = self.revision or "default"
          if self.revision and version != target:
            self.loading[url] = load = asyncio.create_task(self.load_weights(url, self.revision))
            await load
            version = await self.weight_version(url)
            if version != target:
              return False  # the load failed and is logged; the next dispatch plans again
          if (room := self.concurrency - len(self.in_flight)) > 0 and (to_send := self.to_send()[:room]):
            sends |= {self.start_send(r, url) for r in to_send}
            idle_since = None
          sends = {task for task in sends if not task.done()}
          if not sends and not self.to_send():
            idle_since = idle_since or time.monotonic()
            if time.monotonic() - idle_since >= IDLE_SECONDS:
              return False
          if time.monotonic() - started >= MAX_TURN_SECONDS:
            break
          await asyncio.sleep(0.02)
        return bool(self.to_send()) and not self.stopping
      finally:
        if sends:
          # A generate that has not answered by now would keep the trainer off the GPU. Cancelled, its record stays
          # pending, and the sleep that follows aborts it inside vLLM.
          _, late = await asyncio.wait(sends, timeout=TURN_GRACE_SECONDS)
          for task in late:
            task.cancel()
          if late:
            await asyncio.wait(late)
        await self.post(url, "/sleep?level=1&mode=abort")

  async def commit(self, r: dict[str, Any], result: dict[str, Any]) -> None:
    await self.store.commit(r, result)
    self.records.pop(r["entry_id"], None)

  async def send(self, r: dict[str, Any], url: str | None = None) -> None:
    """Sends through the router, or to `url`, the replica whose turn it is."""
    target, sent = self.revision or "default", time.monotonic()
    params = {**r["sampling_params"], "logprobs": 0, "prompt_logprobs": 0 if r["include_prompt_logprobs"] else None}
    attempt_id = f"{r['request_id']}-{uuid.uuid4().hex[:8]}"
    # The store hands prompts back as array('I'), which JSON cannot encode.
    body = {"request_id": attempt_id, "token_ids": list(r["prompt_token_ids"]), "model": r["model"], "sampling_params": params}
    url = url or self.router or random.choice(self.urls)
    try:
      # A replica that never answers must not hold the send past its record's deadline.
      timeout = max(0.0, accepted_at(r) + SAMPLE_DEADLINE_SECONDS - time.time())
      response = await self.http.post(f"{url}/inference/v1/generate", json=body, timeout=timeout)
      # Envoy names the replica that answered. Its weights must still be the target, with no pause to load others since this was sent.
      replica = f"http://{name}" if (name := response.headers.get("x-openrl-upstream")) else url
      if response.status_code in (400, 404, 422):
        await self.commit(r, failed(f"Sampler rejected the request ({response.status_code}): {response.text[:500]}"))
      elif response.status_code == 200 and await self.weight_version(replica) == target and self.paused.get(replica, 0.0) < sent:
        try:
          result = to_result(response.json(), r)
        except (KeyError, TypeError, ValueError) as exc:
          result = failed(f"Sampler response broke the contract: {exc!r}")
        await self.commit(r, result)
    except httpx.HTTPError as exc:
      logger.info("Retrying %s after %r", r["request_id"], exc)

  async def load_weights(self, url: str, revision: str) -> None:
    try:
      if not (path := Path(revision, "metadata.json")).is_file():
        # Retention can delete a revision between the API server's check and its pin.
        for r in [r for r in self.records.values() if r["revision"] == revision]:
          await self.commit(r, failed(f"Sampler weights {revision} no longer exist"))
        return
      metadata, current = json.loads(path.read_text()), await self.weight_version(url)
      # A delta applies on any save at the weights it starts from or reaches. Deltas written before the trainer listed them name one parent.
      bases = [base or "default" for base in metadata.get("bases", [metadata.get("parent")])]
      if metadata.get("format") == "sparse_delta" and current not in bases:
        raise RuntimeError(f"{url} holds {current!r}, not a revision delta {revision!r} applies on")
      self.paused[url] = time.monotonic()
      # A load that breaks off may leave the replica paused or inside its update, so it is checked again unless this one ends cleanly.
      self.checked.discard(url)
      await self.post(url, "/pause?mode=wait&clear_cache=false")
      updating = True
      try:
        await self.post(url, "/start_weight_update")
        await self.post(url, "/update_weights", {"update_info": {"target_weights_path": revision}}, UPDATE_TIMEOUT_SECONDS)
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

  async def post(self, url: str, path: str, body: dict[str, Any] | None = None, timeout: float = CALL_TIMEOUT_SECONDS) -> httpx.Response:
    return (await self.http.post(url + path, json=body, timeout=timeout)).raise_for_status()

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
    running = [p for p in pods.items if p.status.pod_ip and p.status.phase == "Running"]
    # A time-sliced replica's container counts as started once its launcher has registered it and put vLLM to
    # sleep, so no turn is taken on one still booting.
    started = [p for p in running if all(s.started for s in p.status.container_statuses or [])]
    urls = [f"http://{p.status.pod_ip}:8000" for p in started]
    for pod, url in zip(started, urls):
      if url not in self.leases and (lease := pod_lease(pod)):
        self.leases[url] = lease
    # A pod that is gone takes its slicer connection with it.
    for url in [u for u in self.leases if u not in urls and u not in self.turns]:
      await self.leases.pop(url)[1].close()
    return urls

  async def weight_version(self, url: str) -> str:
    return (await self.http.get(f"{url}/weight_info", timeout=5)).raise_for_status().json()["weight_version"]

  def write_revision(self) -> None:
    path = revision_file(self.pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(self.revision or "default")
    path.with_suffix(".tmp").rename(path)


def pod_lease(pod: Any) -> Lease | None:
  """For a time-sliced replica, the workload its launcher registered, read from the same env it read, and
  a client to the time slicer on the replica's node, which listens on the host network."""
  env = {e.name: e.value for c in pod.spec.containers if c.name == "worker" for e in c.env or []}
  if env.get(TIME_SLICED_SAMPLER) != "true":
    return None
  name = env.get("OPEN_RL_WORKLOAD_ID") or env.get("OPEN_RL_TIME_SLICE_JOB_ID") or pod.metadata.name
  claim = env.get("OPEN_RL_CLAIM") or env.get("OPEN_RL_TIME_SLICE_GROUP") or SAMPLER_CLAIM
  port = int(env.get("OPEN_RL_ACCEL_TIMESLICER_PORT") or DEFAULT_TCP_PORT)
  return WorkloadRef(name, claim), SocketTimeSlicerClient(host=pod.status.host_ip, port=port)


def redis_client(url: str) -> redis.Redis:
  """When replicas come up, every in-flight send commits at once. Past max_connections a
  commit waits for a free connection; redis-py's default pool of 100 raises instead."""
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
  store = SamplingStore(redis_client(os.environ["REDIS_URL"]))
  async with http_client() as http:
    dispatcher = Dispatcher(store, os.environ["OPEN_RL_SAMPLER_POOL"], http, router, urls, int(os.getenv("OPEN_RL_DISPATCH_CONCURRENCY", "256")))
    if os.getenv(TIME_SLICED_SAMPLER) == "true":
      # The worker manager marks a local replica's dispatcher time-sliced too, with the replica's tags, when the replica shares its GPU.
      dispatcher.leases = {url: (workload_from_env(claim=SAMPLER_CLAIM), time_slicer_client_from_env()) for url in urls}
    task = asyncio.create_task(dispatcher.run())
    for sig in (signal.SIGTERM, signal.SIGINT):
      asyncio.get_running_loop().add_signal_handler(sig, task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
      await task


if __name__ == "__main__":
  asyncio.run(main())
