"""openrl.sampler_router=llmd: the pods a routed set gets, and the API path from
asample to retrieve_future. Only `vllm serve` is replaced, by an app that answers
in its shape; the conversion is checked against a response recorded from vLLM
0.30, and LiveVllmTest runs against a real server when one is given."""

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
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from server import api_server, sampler_routing
from server.lora_snapshots import snapshot_path
from server.sampler_routing import SAMPLER_SET_LABEL, sampler_set
from server.scheduler_worker_manager import SchedulerWorkerManager
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import LocalWorkerManager
from tests.test_scheduler_worker_manager import FakeAppsApi, FakeCoreApi, FakeCustomObjectsApi

ROUTED = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 2, "sampler_router": "llmd"}
SAVED = "tinker://job/sampler_weights/000003"

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
  """Answers /v1/completions in vLLM's shape with the prompt's last two tokens."""

  def __init__(self, fail_first: int = 0, block: bool = False) -> None:
    self.requests: list[dict] = []
    self.fail_first = fail_first
    self.release = asyncio.Event()
    if not block:
      self.release.set()
    self.app = FastAPI()
    self.app.post("/v1/completions")(self.completions)

  async def completions(self, request: Request) -> JSONResponse:
    body = await request.json()
    self.requests.append(body)
    await self.release.wait()
    if len(self.requests) <= self.fail_first:
      return JSONResponse({"error": {"message": "engine hiccup"}}, status_code=500)
    choice = {"token_ids": body["prompt"][-2:], "logprobs": {"token_logprobs": [-0.5, -0.5]}, "finish_reason": "length"}
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


class SampleTest(unittest.TestCase):
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


# A live `vllm serve` started with a routed sampler's flags. OPEN_RL_TEST_VLLM_LORA
# names an adapter directory under its VLLM_LORA_RESOLVER_CACHE_DIR.
VLLM_URL = os.getenv("OPEN_RL_TEST_VLLM_URL")
VLLM_MODEL = os.getenv("OPEN_RL_TEST_VLLM_MODEL", "Qwen/Qwen3.5-9B")
VLLM_LORA = os.getenv("OPEN_RL_TEST_VLLM_LORA")


@unittest.skipUnless(VLLM_URL, "needs OPEN_RL_TEST_VLLM_URL, a live vllm serve")
class LiveVllmTest(unittest.IsolatedAsyncioTestCase):
  """route against real vLLM: the request it builds is accepted and read as meant."""

  async def sample(self, **request) -> dict:
    prompt = [785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374]
    full = {"model": VLLM_MODEL, "prompt_token_ids": prompt, "max_tokens": 8, "num_samples": 2, "temperature": 1.0, **request}
    async with httpx.AsyncClient() as http:
      return await sampler_routing.route(http, VLLM_URL, "live", full, time.time() + 300)

  async def test_a_sample_has_tokens_logprobs_and_prompt_logprobs(self) -> None:
    sample = await self.sample(include_prompt_logprobs=True)
    self.assertEqual(len(sample["sequences"]), 2, sample)
    for sequence in sample["sequences"]:
      self.assertEqual((len(sequence["tokens"]), len(sequence["logprobs"]), sequence["stop_reason"]), (8, 8, "length"))
    self.assertEqual(len(sample["prompt_logprobs"]), 12)
    self.assertIsNone(sample["prompt_logprobs"][0])
    self.assertTrue(all(lp is not None and lp <= 0 for lp in sample["prompt_logprobs"][1:]))

  async def test_a_stop_token_ends_a_sequence(self) -> None:
    greedy = {"temperature": 0.0, "num_samples": 1, "max_tokens": 20}
    [free] = (await self.sample(**greedy))["sequences"]
    stop = free["tokens"][4]
    [stopped] = (await self.sample(**greedy, stop=[stop]))["sequences"]
    self.assertEqual(stopped["tokens"], free["tokens"][: free["tokens"].index(stop) + 1])
    self.assertEqual(stopped["stop_reason"], "stop")

  async def test_refused_requests_fail_at_once(self) -> None:
    start = time.monotonic()
    self.assertIn("does not exist", (await self.sample(model="no-such-snapshot"))["error_message"])
    # As vLLM refuses it on the queue path too.
    self.assertIn("greedy", (await self.sample(temperature=0.0))["error_message"])
    self.assertLess(time.monotonic() - start, 10)

  @unittest.skipUnless(VLLM_LORA, "needs OPEN_RL_TEST_VLLM_LORA")
  async def test_a_snapshot_name_loads_the_adapter(self) -> None:
    self.assertEqual((await self.sample(model=VLLM_LORA))["type"], "sample")


class SetTemplateTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379", "VLLM_MAX_MODEL_LEN": "196608"}))
    self.api, self.apps, self.core = FakeCustomObjectsApi(), FakeAppsApi(), FakeCoreApi()
    self.manager = SchedulerWorkerManager(custom_api=self.api, apps_api=self.apps, core_api=self.core)

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
      self.assertEqual((container["command"], container["args"][:2]), (["vllm", "serve"], ["Qwen/Qwen3-0.6B", "--port=8000"]))
      self.assertIn("--max-model-len=196608", container["args"])
      env = {e["name"]: e.get("value") for e in container["env"]}
      self.assertEqual(env["VLLM_LORA_RESOLVER_CACHE_DIR"], "/mnt/shared/open-rl/sampler_lora")
    self.assertNotIn("metadata", templates["lora-job-0-trainer"])

  def test_the_set_gets_a_router_and_service_that_go_with_its_owner(self) -> None:
    self.templates(ROUTED)
    name = sampler_set("job")
    [deployment] = self.apps.existing.values()
    pod = deployment["spec"]["template"]
    self.assertNotIn(SAMPLER_SET_LABEL, pod["metadata"]["labels"])
    picker = next(c for c in pod["spec"]["containers"] if c["name"] == "picker")
    self.assertIn(f"{SAMPLER_SET_LABEL}={name}", picker["args"])
    [service] = self.core.existing.values()
    self.assertEqual(service["metadata"]["name"], deployment["metadata"]["name"])
    self.manager.release_owner("job")
    self.assertEqual((self.apps.existing, self.core.existing), ({}, {}))

  def test_unrouted_jobs_get_no_set(self) -> None:
    for template in self.templates({**ROUTED, "sampler_router": None}).values():
      self.assertNotIn("metadata", template)
    self.assertEqual((self.apps.existing, self.core.existing), ({}, {}))


class RoutedSamplingApiTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.tmp = self.enterContext(tempfile.TemporaryDirectory())
    self.enterContext(patch.dict(os.environ, {"OPEN_RL_TMP_DIR": self.tmp}))
    snapshot = snapshot_in(self.tmp, SAVED)
    snapshot.mkdir(parents=True)
    (snapshot / "adapter_config.json").write_text("{}")
    self.store = InMemoryStore()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", state_with("job", ROUTED)))
    self.enterContext(patch.object(api_server, "get_sampler_backend", return_value="vllm"))
    self.enterContext(patch.object(api_server, "TMP_DIR", self.tmp))
    self.enterContext(patch.object(sampler_routing, "BACKOFF_CAP", 0.05))
    self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app), base_url="http://api")

  async def asyncTearDown(self) -> None:
    await self.restart(FakeVllm())
    await self.http.aclose()

  def use(self, vllm: FakeVllm) -> None:
    self.enterContext(patch.object(api_server, "router_http", vllm.client()))

  async def restart(self, vllm: FakeVllm) -> None:
    """A new API process: what the old one was running is gone, saved results stay."""
    for task in api_server.routed.values():
      task.cancel()
    await asyncio.gather(*api_server.routed.values(), return_exceptions=True)
    self.use(vllm)

  async def sample(self, model_id: str = SAVED) -> httpx.Response:
    body = {"model_id": model_id, "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}, "num_samples": 1}
    return await self.http.post("/api/v1/asample", json=body)

  async def retrieve(self, request_id: str) -> httpx.Response:
    return await self.http.post("/api/v1/retrieve_future", json={"request_id": request_id})

  async def test_a_sample_goes_through_the_router_and_names_its_snapshot(self) -> None:
    vllm = FakeVllm(fail_first=1)
    self.use(vllm)
    response = await self.retrieve((await self.sample()).json()["request_id"])
    self.assertEqual(response.json()["sequences"][0]["tokens"], [2, 3])
    # The first attempt failed and was retried.
    self.assertEqual([r["model"] for r in vllm.requests], [snapshot_in(self.tmp, SAVED).name] * 2)

  async def test_a_sample_lost_to_a_restart_answers_410_so_the_client_resends_it(self) -> None:
    self.use(FakeVllm(block=True))
    lost = (await self.sample()).json()["request_id"]
    await self.restart(FakeVllm())
    self.assertEqual((await self.retrieve(lost)).status_code, 410)
    resent = (await self.sample()).json()["request_id"]
    self.assertEqual((await self.retrieve(resent)).json()["sequences"][0]["tokens"], [2, 3])

  async def test_a_sample_finished_before_a_restart_keeps_its_result(self) -> None:
    self.use(FakeVllm())
    request_id = (await self.sample()).json()["request_id"]
    self.assertEqual((await self.retrieve(request_id)).status_code, 200)
    await self.restart(FakeVllm())
    self.assertEqual((await self.retrieve(request_id)).json()["sequences"][0]["tokens"], [2, 3])

  async def test_a_live_adapter_is_refused_so_retries_keep_fixed_weights(self) -> None:
    live = Path(self.tmp, "peft", "job", "job")
    live.mkdir(parents=True)
    (live / "adapter_config.json").write_text("{}")
    response = await self.sample(model_id="job")
    self.assertEqual(response.status_code, 400)
    self.assertIn("live adapter", response.text)

  async def test_too_many_unfinished_samples_are_refused(self) -> None:
    self.use(FakeVllm(block=True))
    with patch.object(api_server, "SAMPLE_UNFINISHED_LIMIT", 1):
      self.assertEqual((await self.sample()).status_code, 200)
      self.assertEqual((await self.sample()).status_code, 429)


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
