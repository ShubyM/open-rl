"""Routed sampling: the pods each set gets, the stream and its dispatcher, and the
API path from asample to retrieve_future in both modes. Redis is real. Only
`vllm serve` is replaced, by an app that answers in its shape, and the conversion
is also checked against a response recorded from vLLM 0.30."""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import redis as sync_redis
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from server import api_server, sampler_routing
from server.lora_snapshots import snapshot_path
from server.sampler_routing import ROUTER_PREFIX, SAMPLER_SET_LABEL, sampler_set
from server.scheduler_worker_manager import ROUTER_SET_LABEL, SchedulerWorkerManager
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import LocalWorkerManager
from tests.redis_server import RedisServer, needs_redis
from tests.test_scheduler_worker_manager import FakeAppsApi, FakeCoreApi, FakeCustomObjectsApi

ROUTED = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 2, "sampler_router": "llmd"}
STREAMED = {**ROUTED, "sampler_router": "llmd-stream"}
SAVED = "tinker://job/sampler_weights/000003"
SET = "set-test"
REQUEST = {"model": "m", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "num_samples": 1}

# vLLM 0.30's answer to a token prompt with n=2, return_token_ids, logprobs=0 and
# prompt_logprobs=0 (Qwen3.5-9B), trimmed to what the conversion reads.
RECORDED = {
  "choices": [
    {"index": 1, "token_ids": [71027, 61], "logprobs": {"token_logprobs": [-6.68, -3.89]}, "finish_reason": "stop", "prompt_logprobs": None},
    {
      "index": 0,
      "token_ids": [6865, 3633],
      "logprobs": {"token_logprobs": [-10.44, -6.18]},
      "finish_reason": "length",
      "prompt_logprobs": [None, {"6722": {"logprob": -13.38, "rank": 12285}}, {"315": {"logprob": -7.11, "rank": 162}}],
    },
  ]
}


class FakeVllm:
  """Answers /v1/completions in vLLM's shape: the last prompt token back, or a
  failure. Everything around it is real."""

  def __init__(self, fail_first: int = 0, reject: bool = False, block: bool = False, token: int | None = None) -> None:
    self.requests: list[dict] = []
    self.fail_first = fail_first
    self.reject = reject
    self.token = token
    self.release = asyncio.Event()
    if not block:
      self.release.set()
    self.running = 0
    self.most_running = 0
    self.app = FastAPI()
    self.app.post("/v1/completions")(self.completions)

  async def completions(self, request: Request) -> JSONResponse:
    body = await request.json()
    self.requests.append(body)
    self.running += 1
    self.most_running = max(self.most_running, self.running)
    try:
      await self.release.wait()
    finally:
      self.running -= 1
    if len(self.requests) <= self.fail_first:
      return JSONResponse({"error": {"message": "engine hiccup"}}, status_code=500)
    if self.reject:
      return JSONResponse({"error": {"message": "prompt too long"}}, status_code=400)
    prompt = body["prompt"]
    prompt_logprobs = [None] + [{str(t): {"logprob": -1.0}} for t in prompt[1:]] if body.get("prompt_logprobs") is not None else None
    tokens = [self.token] if self.token is not None else prompt[-2:]
    logprobs = {"token_logprobs": [-0.5] * len(tokens)}
    choice = {"token_ids": tokens, "logprobs": logprobs, "finish_reason": "length", "prompt_logprobs": prompt_logprobs}
    return JSONResponse({"choices": [{"index": i, **choice} for i in range(body["n"])]})

  def client(self) -> httpx.AsyncClient:
    """Stands in for the set's Envoy and picker: every call reaches this sampler."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app))


def state_with(model_id: str, meta: dict) -> InMemoryStateStore:
  state = InMemoryStateStore()
  state.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
  return state


def snapshot_in(tmp: str, ref: str) -> Path:
  with patch.dict(os.environ, {"OPEN_RL_TMP_DIR": tmp}):
    return snapshot_path(ref)


class RouteTest(unittest.IsolatedAsyncioTestCase):
  def test_a_vllm_completion_becomes_a_sample(self) -> None:
    sample = sampler_routing.sample(RECORDED, [785, 6722, 315])
    self.assertEqual(sample["type"], "sample")
    self.assertEqual(
      sample["sequences"],
      [
        {"tokens": [6865, 3633], "logprobs": [-10.44, -6.18], "stop_reason": "length"},
        {"tokens": [71027, 61], "logprobs": [-6.68, -3.89], "stop_reason": "stop"},
      ],
    )
    self.assertEqual(sample["prompt_logprobs"], [None, -13.38, -7.11])


# A live `vllm serve` for the real-engine checks, e.g. one started with the same
# flags as a routed sampler. VLLM_TEST_LORA names an adapter directory under its
# VLLM_LORA_RESOLVER_CACHE_DIR.
VLLM_URL = os.getenv("OPEN_RL_TEST_VLLM_URL")
VLLM_MODEL = os.getenv("OPEN_RL_TEST_VLLM_MODEL", "Qwen/Qwen3.5-9B")
VLLM_LORA = os.getenv("OPEN_RL_TEST_VLLM_LORA")


@unittest.skipUnless(VLLM_URL, "needs OPEN_RL_TEST_VLLM_URL, a live vllm serve")
class LiveVllmTest(unittest.IsolatedAsyncioTestCase):
  """route against real vLLM: the request it builds is accepted and read as meant."""

  async def sample(self, **request) -> dict:
    async with httpx.AsyncClient() as http:
      prompt = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374]
      full = {"model": VLLM_MODEL, "prompt_token_ids": prompt, "max_tokens": 8, "num_samples": 2, "temperature": 1.0, **request}
      return await sampler_routing.route(http, VLLM_URL, "live", full, time.time() + 300)

  async def test_a_sample_has_tokens_logprobs_and_prompt_logprobs(self) -> None:
    sample = await self.sample(include_prompt_logprobs=True)
    self.assertEqual(sample["type"], "sample", sample)
    self.assertEqual(len(sample["sequences"]), 2)
    for sequence in sample["sequences"]:
      self.assertEqual(len(sequence["tokens"]), 8)
      self.assertEqual(len(sequence["logprobs"]), 8)
      self.assertTrue(all(lp <= 0 for lp in sequence["logprobs"]))
      self.assertEqual(sequence["stop_reason"], "length")
    prompt_logprobs = sample["prompt_logprobs"]
    self.assertEqual(len(prompt_logprobs), 12)
    self.assertIsNone(prompt_logprobs[0])
    self.assertTrue(all(lp is not None and lp <= 0 for lp in prompt_logprobs[1:]))

  async def test_a_stop_token_ends_a_sequence(self) -> None:
    greedy = {"temperature": 0.0, "num_samples": 1, "max_tokens": 20}
    [free] = (await self.sample(**greedy))["sequences"]
    stop = free["tokens"][4]
    [stopped] = (await self.sample(**greedy, stop=[stop]))["sequences"]
    self.assertEqual((stopped["stop_reason"], stopped["tokens"][-1]), ("stop", stop))
    self.assertEqual(stopped["tokens"], free["tokens"][: free["tokens"].index(stop) + 1])

  async def test_greedy_sampling_with_several_samples_is_refused_at_once(self) -> None:
    # As vLLM refuses it on the queue path too.
    start = time.monotonic()
    sample = await self.sample(temperature=0.0)
    self.assertIn("greedy", sample["error_message"])
    self.assertLess(time.monotonic() - start, 10)

  @unittest.skipUnless(VLLM_LORA, "needs OPEN_RL_TEST_VLLM_LORA")
  async def test_a_snapshot_name_loads_the_adapter(self) -> None:
    self.assertEqual((await self.sample(model=VLLM_LORA))["type"], "sample")

  async def test_an_unknown_adapter_fails_without_retrying(self) -> None:
    start = time.monotonic()
    sample = await self.sample(model="no-such-snapshot")
    self.assertIn("does not exist", sample["error_message"])
    self.assertLess(time.monotonic() - start, 10)


class SetTemplateTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379", "VLLM_MAX_MODEL_LEN": "196608"}))
    self.api = FakeCustomObjectsApi()
    self.apps = FakeAppsApi()
    self.core = FakeCoreApi()
    self.created_sets: list[str] = []
    stream_client = SimpleNamespace(xgroup_create=lambda key, *args, **kwargs: self.created_sets.append(key))
    self.manager = SchedulerWorkerManager(custom_api=self.api, apps_api=self.apps, core_api=self.core, redis_client=stream_client)

  def templates(self, meta: dict) -> dict[str, dict]:
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", meta)):
      self.manager.ensure("job", "trainer")
      self.manager.ensure("job", "sampler")
    return {body["metadata"]["name"]: body["spec"]["template"] for body in self.api.created}

  def test_every_routed_sampler_is_stock_vllm_in_its_set(self) -> None:
    templates = self.templates(ROUTED)
    for sampler in ("lora-job-0-sampler", "lora-job-1-sampler"):
      template = templates[sampler]
      self.assertEqual(template["metadata"], {"labels": {SAMPLER_SET_LABEL: sampler_set("job")}})
      [container] = template["spec"]["containers"]
      self.assertTrue(container["image"].startswith("vllm/vllm-openai:"))
      self.assertEqual(container["command"], ["vllm", "serve"])
      self.assertEqual(container["args"][:2], ["Qwen/Qwen3-0.6B", "--port=8000"])
      self.assertIn("--max-model-len=196608", container["args"])
      env = {e["name"]: e.get("value") for e in container["env"]}
      self.assertEqual(env["VLLM_LORA_RESOLVER_CACHE_DIR"], "/mnt/shared/open-rl/sampler_lora")
      self.assertEqual(container["readinessProbe"]["httpGet"]["path"], "/health")
    self.assertNotIn("metadata", templates["lora-job-0-trainer"])

  def test_the_set_gets_one_router_pod_that_is_not_a_sampler_and_its_service(self) -> None:
    self.templates(ROUTED)
    name = sampler_set("job")
    [deployment] = self.apps.existing.values()
    self.assertEqual(deployment["metadata"]["name"], ROUTER_PREFIX + name)
    pod = deployment["spec"]["template"]
    self.assertNotIn(SAMPLER_SET_LABEL, pod["metadata"]["labels"])
    self.assertEqual(pod["metadata"]["labels"][ROUTER_SET_LABEL], name)
    containers = {c["name"]: c for c in pod["spec"]["containers"]}
    self.assertEqual(list(containers), ["router-proxy", "router-picker"])
    self.assertIn(f"{SAMPLER_SET_LABEL}={name}", containers["router-picker"]["args"])
    [service] = self.core.existing.values()
    self.assertEqual(service["spec"]["selector"], {ROUTER_SET_LABEL: name})
    self.assertEqual(self.created_sets, [])

  def test_a_streamed_set_also_gets_its_stream_and_dispatcher(self) -> None:
    self.templates(STREAMED)
    name = sampler_set("job")
    [deployment] = self.apps.existing.values()
    containers = {c["name"]: c for c in deployment["spec"]["template"]["spec"]["containers"]}
    self.assertEqual(list(containers), ["dispatcher", "router-proxy", "router-picker"])
    env = {e["name"]: e.get("value") for e in containers["dispatcher"]["env"]}
    self.assertEqual((env["OPEN_RL_SAMPLER_SET"], env["OPEN_RL_ROUTER_URL"]), (name, "http://127.0.0.1:8081"))
    self.assertEqual(self.created_sets, [sampler_routing.stream_key(name)])

  def test_unrouted_jobs_get_no_set(self) -> None:
    templates = self.templates({**ROUTED, "sampler_router": None})
    for template in templates.values():
      self.assertNotIn("metadata", template)
    self.assertEqual((self.apps.existing, self.core.existing), ({}, {}))


class RedisTest(unittest.IsolatedAsyncioTestCase):
  server: RedisServer

  @classmethod
  def setUpClass(cls) -> None:
    cls.server = RedisServer()

  @classmethod
  def tearDownClass(cls) -> None:
    cls.server.stop()

  async def asyncSetUp(self) -> None:
    self.client = redis.from_url(self.server.url, decode_responses=True)
    await self.client.flushdb()
    self.sync = sync_redis.Redis.from_url(self.server.url)
    # Fast retries: the first wait is 0.5-1 s otherwise.
    self.enterContext(patch.object(sampler_routing, "BACKOFF_CAP", 0.05))
    self.dispatchers: list[asyncio.Task] = []

  async def asyncTearDown(self) -> None:
    for task in self.dispatchers:
      await self.stop(task)
    await self.client.aclose()
    self.sync.close()

  def start(self, vllm: FakeVllm, set_id: str = SET, active: int = 8) -> asyncio.Task:
    run = sampler_routing.dispatch(redis.from_url(self.server.url, decode_responses=True), set_id, "http://router", vllm.client(), active)
    self.dispatchers.append(task := asyncio.create_task(run))
    return task

  async def stop(self, task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@needs_redis
class StreamTest(RedisTest):
  async def asyncSetUp(self) -> None:
    await super().asyncSetUp()
    sampler_routing.create_set(self.sync, SET)

  async def accept(self, n: int = 1, deadline: float = 30) -> list[str]:
    return [await sampler_routing.accept(self.client, SET, REQUEST, deadline) for _ in range(n)]

  async def until(self, condition, timeout: float = 5) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
      self.assertLess(loop.time(), end, "condition not reached in time")
      await asyncio.sleep(0.02)

  async def results(self, request_ids: list[str]) -> list[dict]:
    return [await sampler_routing.result(self.client, r, wait=10) for r in request_ids]

  async def unfinished(self) -> int:
    return await self.client.xlen(sampler_routing.stream_key(SET))

  async def test_each_accepted_request_gets_one_saved_result(self) -> None:
    request_ids = await self.accept(3)
    self.assertEqual(await sampler_routing.result(self.client, request_ids[0], wait=0), {"type": "try_again"})
    self.start(FakeVllm())
    found = await self.results(request_ids)
    self.assertEqual([f["sequences"][0]["tokens"] for f in found], [[2, 3]] * 3)
    self.assertEqual(await self.unfinished(), 0)

  async def test_only_the_first_result_is_saved(self) -> None:
    [request_id] = await self.accept()
    _, entry_id = sampler_routing.parse(request_id)
    for token in (7, 9):
      await sampler_routing.send(self.client, SET, entry_id, {**REQUEST, "deadline": 1e12}, "http://router", FakeVllm(token=token).client())
    self.assertEqual((await sampler_routing.result(self.client, request_id))["sequences"][0]["tokens"], [7])

  async def test_a_temporary_failure_is_retried(self) -> None:
    vllm = FakeVllm(fail_first=1)
    self.start(vllm)
    [found] = await self.results(await self.accept())
    self.assertEqual(found["type"], "sample")
    self.assertEqual(len(vllm.requests), 2)

  async def test_a_request_the_sampler_cannot_serve_fails_without_retrying(self) -> None:
    vllm = FakeVllm(reject=True)
    self.start(vllm)
    [found] = await self.results(await self.accept())
    self.assertIn("prompt too long", found["error_message"])
    self.assertEqual(len(vllm.requests), 1)

  async def test_the_deadline_ends_retries(self) -> None:
    vllm = FakeVllm(fail_first=1000)
    self.start(vllm)
    [found] = await self.results(await self.accept(deadline=0.8))
    self.assertIn("deadline passed", found["error_message"])
    self.assertGreater(len(vllm.requests), 1)
    self.assertEqual(await self.unfinished(), 0)

  async def test_active_work_is_bounded(self) -> None:
    vllm = FakeVllm(block=True)
    self.start(vllm, active=2)
    request_ids = await self.accept(5)
    await self.until(lambda: vllm.running == 2)
    await asyncio.sleep(0.3)
    self.assertEqual(vllm.running, 2)
    vllm.release.set()
    found = await self.results(request_ids)
    self.assertEqual([f["type"] for f in found], ["sample"] * 5)
    self.assertEqual(vllm.most_running, 2)

  async def test_a_restarted_dispatcher_finishes_what_the_last_one_took(self) -> None:
    stuck = FakeVllm(block=True)
    first = self.start(stuck)
    request_ids = await self.accept(2)
    await self.until(lambda: stuck.running == 2)
    await self.stop(first)
    healthy = FakeVllm()
    self.start(healthy)
    found = await self.results(request_ids)
    self.assertEqual([f["type"] for f in found], ["sample"] * 2)
    self.assertEqual(len(healthy.requests), 2)
    self.assertEqual(await self.unfinished(), 0)

  async def test_a_removed_set_refuses_new_requests_and_drops_unfinished_ones(self) -> None:
    [request_id] = await self.accept()
    sampler_routing.remove_set(self.sync, SET)
    self.assertIn("not found", (await sampler_routing.result(self.client, request_id))["error_message"])
    self.assertIsNone(await sampler_routing.accept(self.client, SET, REQUEST, 30))

  async def test_an_expired_result_reads_as_not_found(self) -> None:
    [request_id] = await self.accept()
    self.start(FakeVllm())
    await self.results([request_id])
    await self.client.delete(sampler_routing.result_key(SET, sampler_routing.parse(request_id)[1]))
    self.assertIn("not found", (await sampler_routing.result(self.client, request_id))["error_message"])
    self.assertIsNone(sampler_routing.parse("routed-123"))

  async def test_a_stream_lives_and_dies_with_its_set(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": self.server.url}))
    apps, core = FakeAppsApi(), FakeCoreApi()
    manager = SchedulerWorkerManager(custom_api=FakeCustomObjectsApi(), apps_api=apps, core_api=core, redis_client=self.sync)
    name = sampler_set("job")
    self.assertIsNone(await sampler_routing.accept(self.client, name, REQUEST, 60))
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", STREAMED)):
      await asyncio.to_thread(manager.ensure, "job", "sampler")
      request_id = await sampler_routing.accept(self.client, name, REQUEST, 60)
      await asyncio.to_thread(manager.release_owner, "job")
    self.assertEqual((apps.deleted, core.deleted), ([ROUTER_PREFIX + name], [ROUTER_PREFIX + name]))
    self.assertIn("not found", (await sampler_routing.result(self.client, request_id))["error_message"])
    self.assertIsNone(await sampler_routing.accept(self.client, name, REQUEST, 60))


class ApiTest(unittest.IsolatedAsyncioTestCase):
  meta = ROUTED

  async def asyncSetUp(self) -> None:
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp}))
    snapshot = snapshot_in(self.tmp, SAVED)
    snapshot.mkdir(parents=True)
    (snapshot / "adapter_config.json").write_text("{}")
    self.store = InMemoryStore()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", state_with("job", self.meta)))
    self.enterContext(patch.object(api_server, "get_sampler_backend", return_value="vllm"))
    self.enterContext(patch.object(api_server, "TMP_DIR", self.tmp))
    self.enterContext(patch.object(sampler_routing, "BACKOFF_CAP", 0.05))
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app), base_url="http://api")

  async def asyncTearDown(self) -> None:
    await self.http.aclose()

  async def sample(self, model_id: str = SAVED, **overrides) -> httpx.Response:
    body = {"model_id": model_id, "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}, "num_samples": 1, **overrides}
    return await self.http.post("/api/v1/asample", json=body)

  async def retrieve(self, request_id: str) -> httpx.Response:
    return await self.http.post("/api/v1/retrieve_future", json={"request_id": request_id})

  async def check_fixed_weights(self) -> None:
    """A saved reference names its snapshot; a live adapter is refused."""
    live = Path(self.tmp, "peft", "job", "job")
    live.mkdir(parents=True)
    (live / "adapter_config.json").write_text("{}")
    response = await self.sample(model_id="job")
    self.assertEqual(response.status_code, 400)
    self.assertIn("live adapter", response.text)


class RoutedSamplingApiTest(ApiTest):
  """openrl.sampler_router=llmd: the API server sends each sample itself."""

  def use(self, vllm: FakeVllm) -> None:
    self.enterContext(patch.object(api_server, "router_http", vllm.client()))

  async def restart(self, vllm: FakeVllm) -> None:
    """A new API process: what the old one was running is gone, saved results stay."""
    for task in api_server.routed.values():
      task.cancel()
    await asyncio.gather(*api_server.routed.values(), return_exceptions=True)
    self.use(vllm)

  async def asyncTearDown(self) -> None:
    await self.restart(FakeVllm())
    await super().asyncTearDown()

  async def test_a_sample_goes_through_the_router_and_names_its_snapshot(self) -> None:
    vllm = FakeVllm(fail_first=1)
    self.use(vllm)
    accepted = (await self.sample()).json()
    response = await self.retrieve(accepted["request_id"])
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["sequences"][0]["tokens"], [2, 3])
    self.assertEqual(len(vllm.requests), 2)
    self.assertEqual(vllm.requests[0]["model"], snapshot_in(self.tmp, SAVED).name)

  async def test_a_live_adapter_is_refused(self) -> None:
    await self.check_fixed_weights()

  async def test_a_sample_lost_to_a_restart_answers_410_so_the_client_resends_it(self) -> None:
    self.use(FakeVllm(block=True))
    lost = (await self.sample()).json()["request_id"]
    await self.restart(FakeVllm())
    self.assertEqual((await self.retrieve(lost)).status_code, 410)
    # The client resends it, and the new process serves it.
    resent = (await self.sample()).json()["request_id"]
    self.assertEqual((await self.retrieve(resent)).json()["sequences"][0]["tokens"], [2, 3])

  async def test_a_sample_finished_before_a_restart_keeps_its_result(self) -> None:
    self.use(FakeVllm())
    request_id = (await self.sample()).json()["request_id"]
    self.assertEqual((await self.retrieve(request_id)).status_code, 200)
    await self.restart(FakeVllm())
    self.assertEqual((await self.retrieve(request_id)).json()["sequences"][0]["tokens"], [2, 3])

  async def test_stopping_lets_running_samples_finish(self) -> None:
    vllm = FakeVllm(block=True)
    self.use(vllm)
    async with api_server.lifespan(api_server.app):
      request_id = (await self.sample()).json()["request_id"]
      asyncio.get_running_loop().call_later(0.1, vllm.release.set)
    self.assertEqual((await self.store.get_future(request_id, timeout=1))["type"], "sample")

  async def test_too_many_unfinished_samples_are_refused(self) -> None:
    self.use(FakeVllm(block=True))
    with patch.object(api_server, "SAMPLE_UNFINISHED_LIMIT", 1):
      self.assertEqual((await self.sample()).status_code, 200)
      self.assertEqual((await self.sample()).status_code, 429)


@needs_redis
class StreamSamplingApiTest(ApiTest, RedisTest):
  """openrl.sampler_router=llmd-stream: requests wait in the set's stream."""

  meta = STREAMED

  async def asyncSetUp(self) -> None:
    await RedisTest.asyncSetUp(self)
    await ApiTest.asyncSetUp(self)
    self.set_id = sampler_set("job")
    sampler_routing.create_set(self.sync, self.set_id)
    self.enterContext(patch.object(api_server, "stream_redis", self.client))

  async def asyncTearDown(self) -> None:
    await ApiTest.asyncTearDown(self)
    await RedisTest.asyncTearDown(self)

  async def test_a_sample_goes_through_the_stream_and_dispatcher(self) -> None:
    vllm = FakeVllm()
    self.start(vllm, self.set_id)
    accepted = (await self.sample()).json()
    self.assertEqual(sampler_routing.parse(accepted["request_id"])[0], self.set_id)
    response = await self.retrieve(accepted["request_id"])
    self.assertEqual(response.json()["sequences"][0]["tokens"], [2, 3])
    self.assertEqual(vllm.requests[0]["model"], snapshot_in(self.tmp, SAVED).name)
    self.assertEqual(await self.client.xlen(sampler_routing.stream_key(self.set_id)), 0)

  async def test_accepted_work_outlives_the_api_process(self) -> None:
    accepted = (await self.sample()).json()
    # A new API process has its own connection; the dispatcher starts later still.
    restarted = redis.from_url(self.server.url, decode_responses=True)
    with patch.object(api_server, "stream_redis", restarted):
      self.start(FakeVllm(), self.set_id)
      response = await self.retrieve(accepted["request_id"])
    await restarted.aclose()
    self.assertEqual(response.json()["sequences"][0]["tokens"], [2, 3])

  async def test_a_live_adapter_is_refused(self) -> None:
    await self.check_fixed_weights()

  async def test_a_removed_set_refuses_new_work(self) -> None:
    sampler_routing.remove_set(self.sync, self.set_id)
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
      response = self.create({"openrl.sampler_router": "llmd-stream", "openrl.fine_tuning_type": "full"})
    self.assertEqual(response.status_code, 400)
    self.assertIn("supports LoRA only", response.json()["error"])

  def test_local_workers_are_refused(self) -> None:
    with patch.object(api_server, "worker_manager", LocalWorkerManager.__new__(LocalWorkerManager)):
      response = self.create({"openrl.sampler_router": "llmd"})
    self.assertEqual(response.status_code, 400)
    self.assertIn("launches workers as pods", response.json()["error"])


if __name__ == "__main__":
  unittest.main()
