import asyncio
import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from server import api_server, sampler_http
from server.scheduler_worker_manager import ROUTER_LABEL, ROUTER_PORT, SAMPLER_SET_LABEL, SchedulerWorkerManager, sampler_set
from server.store import InMemoryStateStore, InMemoryStore
from server.worker_manager import LocalWorkerManager
from tests.test_scheduler_worker_manager import FakeCustomObjectsApi

ROUTED = {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "lora", "exclusive": True, "sampler_replicas": 2, "sampler_router": "llmd"}


def state_with(model_id: str, meta: dict) -> InMemoryStateStore:
  state = InMemoryStateStore()
  state.kv_store[f"open_rl:model_meta:{model_id}"] = json.dumps(meta)
  return state


class RoutedSamplerTemplateTest(unittest.TestCase):
  def setUp(self) -> None:
    self.enterContext(patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}))
    self.api = FakeCustomObjectsApi()
    self.manager = SchedulerWorkerManager(custom_api=self.api)

  def workloads(self, meta: dict) -> dict[str, dict]:
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", meta)):
      self.manager.ensure("job", "trainer")
      self.manager.ensure("job", "sampler")
    return {body["metadata"]["name"]: body["spec"]["template"] for body in self.api.created}

  def test_the_first_sampler_carries_the_router_for_its_set(self) -> None:
    templates = self.workloads(ROUTED)
    first, second = templates["lora-job-0-sampler"], templates["lora-job-1-sampler"]
    name = sampler_set("job")

    self.assertEqual([c["name"] for c in first["spec"]["containers"]], ["worker", "router-proxy", "router-picker"])
    self.assertEqual(first["metadata"]["labels"], {SAMPLER_SET_LABEL: name, ROUTER_LABEL: name})
    self.assertEqual(first["spec"]["serviceAccountName"], "openrl-llmd-router")
    picker = first["spec"]["containers"][2]
    self.assertIn(f"{SAMPLER_SET_LABEL}={name}", picker["args"])

    self.assertEqual([c["name"] for c in second["spec"]["containers"]], ["worker"])
    self.assertEqual(second["metadata"]["labels"], {SAMPLER_SET_LABEL: name})
    for template in (first, second):
      env = {e["name"]: e.get("value") for e in template["spec"]["containers"][0]["env"]}
      self.assertEqual(env["OPEN_RL_SAMPLER_HTTP_PORT"], "8000")

  def test_trainers_and_unrouted_samplers_are_unchanged(self) -> None:
    templates = self.workloads({**ROUTED, "sampler_router": None})
    for template in templates.values():
      self.assertNotIn("metadata", template)
      self.assertEqual([c["name"] for c in template["spec"]["containers"]], ["worker"])
    self.api.created.clear()
    self.api.existing.clear()
    trainer = self.workloads(ROUTED)["lora-job-0-trainer"]
    self.assertNotIn("metadata", trainer)


class RouterLookupTest(unittest.TestCase):
  def test_the_running_router_pod_is_found_by_its_set(self) -> None:
    def pod(phase: str, ip: str, deleting: bool = False) -> SimpleNamespace:
      return SimpleNamespace(status=SimpleNamespace(phase=phase, pod_ip=ip), metadata=SimpleNamespace(deletion_timestamp="now" if deleting else None))

    selectors = []

    class Pods:
      def list_namespaced_pod(self, namespace: str, label_selector: str) -> SimpleNamespace:
        selectors.append(label_selector)
        return SimpleNamespace(items=[pod("Running", "10.0.0.1", deleting=True), pod("Pending", ""), pod("Running", "10.0.0.2")])

    with patch.dict(os.environ, {"REDIS_URL": "redis://localhost:6379"}):
      manager = SchedulerWorkerManager(custom_api=FakeCustomObjectsApi(), core_api=Pods())
    with patch("server.worker_manager.get_state_store", return_value=state_with("job", ROUTED)):
      self.assertEqual(manager.router_url("job"), f"http://10.0.0.2:{ROUTER_PORT}")
    self.assertEqual(selectors, [f"{ROUTER_LABEL}={sampler_set('job')}"])


class EchoSampler:
  """Stands in for the engine call only; everything around it is the real app."""

  engine = SimpleNamespace(errored=False)

  def __init__(self) -> None:
    self.requests: list[dict] = []

  async def generate(self, request: dict) -> dict:
    self.requests.append(request)
    return {"sequences": [{"tokens": request["prompt_token_ids"][-2:], "logprobs": [-0.5, -0.25], "stop_reason": "length"}]}


class Routers:
  def __init__(self, url: str | None) -> None:
    self.url = url

  def router_url(self, model_id: str) -> str | None:
    return self.url


class GatewayRoutingTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    self.sampler = EchoSampler()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", state_with("job", ROUTED)))
    self.enterContext(patch.object(api_server, "get_sampler_backend", return_value="vllm"))
    self.enterContext(patch.object(api_server, "router_urls", {}))
    transport = httpx.ASGITransport(app=sampler_http.http_app(self.sampler))
    self.enterContext(patch.object(api_server, "router_client", httpx.AsyncClient(transport=transport)))
    self.client = self.enterContext(TestClient(api_server.app))

  def sample(self) -> dict:
    body = {"model_id": "tinker://job/sampler_weights/000003", "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}}
    promise = self.client.post("/api/v1/asample", json=body).json()
    return self.client.post("/api/v1/retrieve_future", json={"request_id": promise["request_id"]}).json()

  def test_a_routed_model_samples_through_its_router(self) -> None:
    with patch.object(api_server, "worker_manager", Routers("http://router")):
      result = self.sample()
    self.assertEqual(result["sequences"][0]["tokens"], [2, 3])
    self.assertEqual(self.sampler.requests[0]["lora_id"], "tinker://job/sampler_weights/000003")
    self.assertEqual(asyncio.run(self.store.get_sampling_requests_for_model("job")), [])

  def test_an_unreachable_router_falls_back_to_the_queue(self) -> None:
    with patch.object(api_server, "worker_manager", Routers(None)), patch.object(api_server, "ROUTER_ATTEMPTS", 1):
      body = {"model_id": "job", "prompt": {"chunks": [{"tokens": [1, 2, 3]}]}, "sampling_params": {"max_tokens": 2}}
      self.client.post("/api/v1/asample", json=body)
      # One attempt, a one-second backoff, then the queue.
      deadline = time.monotonic() + 5
      while not self.store.sampling_queues and time.monotonic() < deadline:
        time.sleep(0.1)
    queued = asyncio.run(self.store.get_sampling_requests_for_model("job"))
    self.assertEqual([request["prompt_token_ids"] for request in queued], [[1, 2, 3]])


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
    with patch.object(api_server, "worker_manager", Routers("http://router")):
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
