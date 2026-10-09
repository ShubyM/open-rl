"""The dispatcher against real Redis and the real sampler HTTP app; only the
engine call is replaced. The dispatcher talks to the app in-process, standing
in for the set's Envoy and endpoint picker."""

import asyncio
import contextlib
import unittest
from types import SimpleNamespace

import httpx
import redis.asyncio as redis

from server import sampler_http
from server.sampling_dispatcher import DispatchConfig, Dispatcher
from server.sampling_streams import SamplingStreams
from tests.redis_server import RedisServer, needs_redis

SET = "set-dispatch"
FAST = DispatchConfig(
  active=8,
  max_attempts=3,
  attempt_timeout=5,
  backoff_base=0.05,
  backoff_cap=0.2,
  lease_ttl=1,
  renew_interval=0.2,
  reclaim_idle=1.5,
  shutdown_grace=0,
  ready_poll=0.1,
)


class Engine:
  """Stands in for vLLM's generate; everything around it is the real sampler app."""

  def __init__(self, fail_first: int = 0, error: Exception | None = None, block: bool = False) -> None:
    self.engine = SimpleNamespace(errored=False)
    self.calls: list[dict] = []
    self.fail_first = fail_first
    self.error = error
    self.release = asyncio.Event()
    if not block:
      self.release.set()
    self.running = 0
    self.most_running = 0
    self.cancelled = 0

  async def generate(self, request: dict) -> dict:
    self.calls.append(request)
    self.running += 1
    self.most_running = max(self.most_running, self.running)
    try:
      await self.release.wait()
      if len(self.calls) <= self.fail_first:
        raise RuntimeError("engine hiccup")
      if self.error is not None:
        raise self.error
      return {"sequences": [{"tokens": request["prompt_token_ids"][-1:], "logprobs": [-0.5], "stop_reason": "length"}]}
    except asyncio.CancelledError:
      self.cancelled += 1
      raise
    finally:
      self.running -= 1


@needs_redis
class DispatcherTest(unittest.IsolatedAsyncioTestCase):
  server: RedisServer

  @classmethod
  def setUpClass(cls) -> None:
    cls.server = RedisServer()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  async def asyncSetUp(self) -> None:
    self.client = redis.from_url(self.server.url)
    await self.client.flushdb()
    self.streams = SamplingStreams(self.client, result_ttl=60, guard_ttl=120)
    await self.streams.open(SET)
    self.is_ready = True
    self.running: list[tuple[Dispatcher, asyncio.Task]] = []

  async def asyncTearDown(self) -> None:
    for dispatcher, task in self.running:
      dispatcher.stop()
      with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    await self.client.aclose()

  async def ready(self) -> bool:
    return self.is_ready

  def start(self, engine: Engine, config: DispatchConfig = FAST) -> Dispatcher:
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=sampler_http.http_app(engine)))
    dispatcher = Dispatcher(self.streams, SET, "http://router", client, self.ready, config)
    self.running.append((dispatcher, asyncio.create_task(dispatcher.run())))
    return dispatcher

  async def accept(self, n: int = 1, deadline: float = 30) -> list[str]:
    request = {"prompt_token_ids": [1, 2, 3], "max_tokens": 2, "num_samples": 1, "lora_id": None, "model_id": "m"}
    return [await self.streams.accept(SET, request, limit=100, deadline_seconds=deadline) for _ in range(n)]

  async def until(self, condition, timeout: float = 5) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
      self.assertLess(loop.time(), end, "condition not reached in time")
      await asyncio.sleep(0.02)

  async def results(self, request_ids: list[str], timeout: float = 10) -> list[dict]:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while True:
      found = [await self.streams.result(r) for r in request_ids]
      if all(f["type"] != "try_again" for f in found) or loop.time() > end:
        return found

      await asyncio.sleep(0.05)

  async def test_accepted_requests_get_one_saved_result_each(self) -> None:
    engine = Engine()
    self.start(engine)
    found = await self.results(await self.accept(3))
    self.assertEqual([f["type"] for f in found], ["sample"] * 3)
    self.assertEqual(found[0]["sequences"][0]["tokens"], [3])
    self.assertEqual(await self.streams.unfinished(SET), 0)

  async def test_a_temporary_failure_is_retried_with_a_new_engine_request(self) -> None:
    engine = Engine(fail_first=1)
    self.start(engine)
    [request_id] = await self.accept()
    [found] = await self.results([request_id])
    self.assertEqual(found["type"], "sample")
    self.assertEqual([c["request_id"] for c in engine.calls], [f"{request_id}#1", f"{request_id}#2"])
    self.assertEqual((await self.streams.state(SET, request_id))["attempts"], "2")

  async def test_a_request_the_sampler_cannot_serve_fails_without_retrying(self) -> None:
    engine = Engine(error=ValueError("prompt too long"))
    self.start(engine)
    [found] = await self.results(await self.accept())
    self.assertEqual(found["type"], "RequestFailedResponse")
    self.assertIn("prompt too long", found["error_message"])
    self.assertEqual(len(engine.calls), 1)

  async def test_attempts_are_bounded(self) -> None:
    engine = Engine(fail_first=10)
    self.start(engine)
    [found] = await self.results(await self.accept())
    self.assertIn("after 3 attempts", found["error_message"])
    self.assertEqual(len(engine.calls), 3)

  async def test_waiting_for_a_ready_sampler_uses_no_attempt(self) -> None:
    self.is_ready = False
    engine = Engine()
    self.start(engine)
    [request_id] = await self.accept()
    await asyncio.sleep(0.5)
    self.assertEqual(engine.calls, [])
    self.assertEqual((await self.streams.state(SET, request_id))["attempts"], "0")
    self.is_ready = True
    [found] = await self.results([request_id])
    self.assertEqual(found["type"], "sample")

  async def test_the_deadline_finalizes_work_even_with_no_sampler_ready(self) -> None:
    self.is_ready = False
    self.start(Engine())
    [found] = await self.results(await self.accept(deadline=0.5))
    self.assertIn("deadline passed", found["error_message"])
    self.assertEqual(await self.streams.unfinished(SET), 0)

  async def test_active_work_is_bounded(self) -> None:
    engine = Engine(block=True)
    self.start(engine, DispatchConfig(**{**vars(FAST), "active": 2}))
    request_ids = await self.accept(5)
    await self.until(lambda: engine.running == 2)
    await asyncio.sleep(0.3)
    self.assertEqual(engine.running, 2)
    engine.release.set()
    found = await self.results(request_ids)
    self.assertEqual([f["type"] for f in found], ["sample"] * 5)
    self.assertEqual(engine.most_running, 2)

  async def test_cancelling_accepted_work_stops_its_active_attempt(self) -> None:
    engine = Engine(block=True)
    self.start(engine)
    [request_id] = await self.accept()
    await self.until(lambda: engine.running == 1)
    await self.streams.close(SET)
    await self.streams.cancel_unfinished(SET, "set removed")
    await self.until(lambda: engine.cancelled == 1)
    self.assertEqual(engine.running, 0)
    self.assertEqual((await self.streams.result(request_id))["error_message"], "set removed")

  async def test_a_new_dispatcher_recovers_work_from_one_that_stopped(self) -> None:
    stuck = Engine(block=True)
    first = self.start(stuck)
    [request_id] = await self.accept()
    await self.until(lambda: stuck.running == 1)
    # The first dispatcher loses its lease and its claim goes stale.
    first.lose("test")
    first.stop()
    await asyncio.sleep(0.1)
    await self.client.delete(f"openrl:{{{SET}}}:lease")
    healthy = Engine()
    second = self.start(healthy)
    [found] = await self.results([request_id], timeout=10)
    self.assertEqual(found["type"], "sample")
    self.assertNotEqual(first.owner, second.owner)
    self.assertEqual([c["request_id"] for c in healthy.calls], [f"{request_id}#2"])
    self.assertEqual(await self.streams.unfinished(SET), 0)


if __name__ == "__main__":
  unittest.main()
