"""Routed sampler sets: the pods each set gets, the set's lifecycle, and the API
path from asample through the set's stream and dispatcher to retrieve_future.
Redis is real; only vLLM's generate is replaced."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import redis as sync_redis
import redis.asyncio as redis
from fastapi.testclient import TestClient

from server import api_server, sampler_http, sampling_streams
from server.lora_snapshots import snapshot_path
from server.sampling_dispatcher import DispatchConfig, Dispatcher
from server.sampling_streams import SamplingStreams, sampler_set
from server.scheduler_worker_manager import DISPATCH_SET_LABEL, DISPATCHER_PREFIX, SAMPLER_SET_LABEL, SchedulerWorkerManager
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import LocalWorkerManager
from tests.redis_server import RedisServer, needs_redis
from tests.test_scheduler_worker_manager import FakeAppsApi, FakeCustomObjectsApi

ROUTED = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 2, "sampler_router": "llmd"}
SAVED = "tinker://job/sampler_weights/000003"
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


def state_with(model_id: str, meta: dict) -> InMemoryStateStore:
  state = InMemoryStateStore()
  state.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
  return state


def snapshot_in(tmp: str, ref: str) -> Path:
  with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}):
    return snapshot_path(ref)


class EchoSampler:
  """Stands in for the engine call only; everything around it is the real app."""

  engine = SimpleNamespace(errored=False)

  def __init__(self) -> None:
    self.requests: list[dict] = []

  async def generate(self, request: dict) -> dict:
    self.requests.append(request)
    return {"sequences": [{"tokens": request["prompt_token_ids"][-2:], "logprobs": [-0.5, -0.25], "stop_reason": "length"}]}


class SetTemplateTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}))
    self.api = FakeCustomObjectsApi()
    self.apps = FakeAppsApi()
    self.opened: list[str] = []
    admission = SimpleNamespace(set=lambda key, value: self.opened.append(key))
    self.manager = SchedulerWorkerManager(custom_api=self.api, apps_api=self.apps, redis_client=admission)

  def templates(self, meta: dict) -> dict[str, dict]:
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", meta)):
      self.manager.ensure("job", "trainer")
      self.manager.ensure("job", "sampler")
    return {body["metadata"]["name"]: body["spec"]["template"] for body in self.api.created}

  def test_every_routed_sampler_only_joins_its_set(self) -> None:
    templates = self.templates(ROUTED)
    name = sampler_set("job")
    for sampler in ("lora-job-0-sampler", "lora-job-1-sampler"):
      template = templates[sampler]
      self.assertEqual(template["metadata"], {"labels": {SAMPLER_SET_LABEL: name}})
      self.assertEqual([c["name"] for c in template["spec"]["containers"]], ["worker"])
      self.assertEqual(template["spec"]["containers"][0]["readinessProbe"]["httpGet"]["path"], "/health")
    self.assertNotIn("metadata", templates["lora-job-0-trainer"])

  def test_the_set_gets_one_dispatcher_pod_that_is_not_a_sampler(self) -> None:
    self.templates(ROUTED)
    [deployment] = self.apps.existing.values()
    name = sampler_set("job")
    self.assertEqual(deployment["metadata"]["name"], DISPATCHER_PREFIX + name)
    self.assertEqual(self.opened, [sampling_streams.keys(name)["admission"]])
    pod = deployment["spec"]["template"]
    self.assertNotIn(SAMPLER_SET_LABEL, pod["metadata"]["labels"])
    self.assertEqual(pod["metadata"]["labels"][DISPATCH_SET_LABEL], name)
    containers = {c["name"]: c for c in pod["spec"]["containers"]}
    self.assertEqual(list(containers), ["dispatcher", "router-proxy", "router-picker"])
    env = {e["name"]: e.get("value") for e in containers["dispatcher"]["env"]}
    self.assertEqual((env["OPEN_RL_SAMPLER_SET"], env["OPEN_RL_ROUTER_URL"]), (name, "http://127.0.0.1:8081"))
    self.assertIn(f"{SAMPLER_SET_LABEL}={name}", containers["router-picker"]["args"])

  def test_unrouted_jobs_get_no_set(self) -> None:
    templates = self.templates({**ROUTED, "sampler_router": None})
    for template in templates.values():
      self.assertNotIn("metadata", template)
    self.assertEqual(self.apps.existing, {})


@needs_redis
class SetLifecycleTest(unittest.IsolatedAsyncioTestCase):
  server: RedisServer

  @classmethod
  def setUpClass(cls) -> None:
    cls.server = RedisServer()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  async def asyncSetUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": self.server.url}))
    self.client = redis.from_url(self.server.url)
    await self.client.flushdb()
    self.streams = SamplingStreams(self.client)
    self.apps = FakeAppsApi()
    self.sync = sync_redis.Redis.from_url(self.server.url)
    self.manager = SchedulerWorkerManager(custom_api=FakeCustomObjectsApi(), apps_api=self.apps, redis_client=self.sync)

  async def asyncTearDown(self) -> None:
    await self.client.aclose()
    self.sync.close()

  async def test_a_set_opens_with_its_dispatcher_and_drains_before_removal(self) -> None:
    name = sampler_set("job")
    with self.assertRaises(sampling_streams.SetClosed):
      await self.streams.accept(name, {"num_samples": 1}, limit=10, deadline_seconds=60)
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", ROUTED)):
      await asyncio.to_thread(self.manager.ensure, "job", "sampler")
      request_id = await self.streams.accept(name, {"num_samples": 1}, limit=10, deadline_seconds=60)
      await asyncio.to_thread(self.manager.release_owner, "job")
    self.assertEqual(self.apps.deleted, [DISPATCHER_PREFIX + name])
    self.assertIn("was removed", (await self.streams.result(request_id))["error_message"])
    self.assertEqual(await self.streams.unfinished(name), 0)
    with self.assertRaises(sampling_streams.SetClosed):
      await self.streams.accept(name, {"num_samples": 1}, limit=10, deadline_seconds=60)


@needs_redis
class StreamSamplingApiTest(unittest.IsolatedAsyncioTestCase):
  server: RedisServer

  @classmethod
  def setUpClass(cls) -> None:
    cls.server = RedisServer()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  async def asyncSetUp(self) -> None:
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp, "VLLM_MAX_MODEL_LEN": "64"}))
    snapshot = snapshot_in(self.tmp, SAVED)
    snapshot.mkdir(parents=True)
    (snapshot / "adapter_config.json").write_text("{}")
    self.client = redis.from_url(self.server.url)
    await self.client.flushdb()
    self.set_id = sampler_set("job")
    self.api_streams = SamplingStreams(self.client)
    await self.api_streams.open(self.set_id)
    self.enterContext(patch.object(api_server, "streams", self.api_streams))
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", state_with("job", ROUTED)))
    self.enterContext(patch.object(api_server, "get_sampler_backend", return_value="vllm"))
    self.enterContext(patch.object(api_server, "TMP_DIR", self.tmp))
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app), base_url="http://api")
    self.dispatchers: list[tuple[Dispatcher, asyncio.Task]] = []

  async def asyncTearDown(self) -> None:
    for dispatcher, task in self.dispatchers:
      dispatcher.stop()
      await asyncio.wait_for(task, 5)
      await dispatcher.streams.client.aclose()
    await self.http.aclose()
    await self.client.aclose()

  async def ready(self) -> bool:
    return True

  def start_dispatcher(self, sampler: EchoSampler) -> None:
    app = httpx.AsyncClient(transport=httpx.ASGITransport(app=sampler_http.http_app(sampler)))
    dispatcher = Dispatcher(SamplingStreams(redis.from_url(self.server.url)), self.set_id, "http://router", app, self.ready, FAST)
    self.dispatchers.append((dispatcher, asyncio.create_task(dispatcher.run())))

  async def sample(self, model_id: str = SAVED, **overrides) -> httpx.Response:
    body = {"model_id": model_id, "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}, "num_samples": 1, **overrides}
    return await self.http.post("/api/v1/asample", json=body)

  async def retrieve(self, request_id: str) -> httpx.Response:
    return await self.http.post("/api/v1/retrieve_future", json={"request_id": request_id})

  async def test_a_sample_goes_through_the_stream_and_dispatcher(self) -> None:
    sampler = EchoSampler()
    self.start_dispatcher(sampler)
    accepted = (await self.sample()).json()
    self.assertEqual(sampling_streams.set_of(accepted["request_id"]), self.set_id)
    response = await self.retrieve(accepted["request_id"])
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["sequences"][0]["tokens"], [2, 3])
    [sent] = sampler.requests
    self.assertEqual(sent["lora_id"], SAVED)
    self.assertEqual(sent["lora_path"], str(snapshot_in(self.tmp, SAVED)))
    self.assertEqual(await self.api_streams.unfinished(self.set_id), 0)

  async def test_accepted_work_outlives_the_api_process(self) -> None:
    accepted = (await self.sample()).json()
    # A new API process has its own connection; the dispatcher starts later still.
    restarted = SamplingStreams(redis.from_url(self.server.url))
    with patch.object(api_server, "streams", restarted):
      self.start_dispatcher(EchoSampler())
      response = await self.retrieve(accepted["request_id"])
    await restarted.client.aclose()
    self.assertEqual(response.json()["sequences"][0]["tokens"], [2, 3])

  async def test_requests_beyond_the_limits_are_refused_before_acceptance(self) -> None:
    too_many = await self.sample(num_samples=1000)
    too_long = await self.sample(sampling_params={"max_tokens": 100})
    self.assertEqual((too_many.status_code, too_long.status_code), (400, 400))
    self.assertIn("num_samples", too_many.text)
    self.assertIn("window", too_long.text)
    self.assertEqual(await self.api_streams.unfinished(self.set_id), 0)

  async def test_a_live_adapter_is_refused_so_retries_keep_fixed_weights(self) -> None:
    live = Path(self.tmp, "peft", "job", "job")
    live.mkdir(parents=True)
    (live / "adapter_config.json").write_text("{}")
    response = await self.sample(model_id="job")
    self.assertEqual(response.status_code, 400)
    self.assertIn("live adapter", response.text)

  async def test_a_closed_or_full_set_refuses_new_work(self) -> None:
    with patch.object(api_server, "SAMPLE_UNFINISHED_LIMIT", 1):
      self.assertEqual((await self.sample()).status_code, 200)
      self.assertEqual((await self.sample()).status_code, 429)
    await self.api_streams.close(self.set_id)
    self.assertEqual((await self.sample()).status_code, 503)


class SamplerRouterSettingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.object(api_server, "store", InMemoryStore()))
    self.enterContext(patch.object(api_server, "state", InMemoryStateStore()))
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}))
    # No lifespan: with FFT enabled it would build a worker manager of its own.
    self.client = TestClient(api_server.app)

  def create(self, metadata: dict) -> httpx.Response:
    return self.client.post("/api/v1/create_model", json={"base_model": "m", "user_metadata": metadata})

  def test_full_fine_tuning_is_refused(self) -> None:
    with patch.object(api_server, "worker_manager", SimpleNamespace()):
      response = self.create({"openrl.sampler_router": "llmd", "openrl.fine_tuning_type": "full"})
    self.assertEqual(response.status_code, 400)
    self.assertIn("supports LoRA only", response.json()["error"])

  def test_local_workers_are_refused(self) -> None:
    with patch.object(api_server, "worker_manager", LocalWorkerManager.__new__(LocalWorkerManager)):
      response = self.create({"openrl.sampler_router": "llmd"})
    self.assertEqual(response.status_code, 400)
    self.assertIn("launches workers as pods", response.json()["error"])


if __name__ == "__main__":
  unittest.main()
