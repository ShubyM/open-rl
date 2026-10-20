"""The dashboard reads discovered gauges without per-metric schemas or charts."""

import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from server.dashboard import collector, router
from server.dashboard.__main__ import app
from server.store import InMemoryStateStore
from server.telemetry import http
from server.telemetry.prometheus import PromQL


class CollectorMetricsTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self):
    self.store = InMemoryStateStore()
    await self.store.set_value("open_rl:model_meta:model-a", json.dumps({"base_model": "base", "session_ids": ["session-a"]}))
    self.enterContext(patch.object(router, "get_state_store", return_value=self.store))
    self.enterContext(patch.object(router.gke, "discover", AsyncMock()))
    backend = PromQL("prometheus", "http://prometheus", None, None, "openrl-system")
    self.enterContext(patch.object(collector, "hardware", return_value=(backend, None)))
    self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://dashboard")
    self.addAsyncCleanup(self.client.aclose)
    self.params = {"run_id": "model-a", "since": "1970-01-01T00:02:00Z", "until": "1970-01-01T00:03:10Z"}

  async def test_discovered_metrics_keep_phase_producer_and_scope_and_escape_search(self):
    rows = [
      {"metric": {"__name__": "openrl_client_metric", "metric": "reward", **labels}, "values": [[120, "2.5"], [135, "NaN"], [150, "3"]]}
      for labels in (
        {"openrl_model_id": "model-a", "session_id": "session-a", "openrl_phase": "train", "service_instance_id": "one"},
        {"openrl_model_id": "model-a", "session_id": "session-a", "openrl_phase": "train", "service_instance_id": "two"},
        {"session_id": "session-a", "openrl_phase": "eval"},
        {"openrl_runtime_id": "base"},
      )
    ]
    search = 'reward.*"}'
    with patch.object(http, "get_json", AsyncMock(return_value={"status": "success", "data": {"result": rows}})) as query:
      response = await self.client.get("/api/v1/inspect/telemetry", params={**self.params, "metric": search})
    self.assertEqual(response.status_code, 200)
    body = response.json()
    self.assertTrue(body["available"])
    self.assertFalse(body["truncated"])
    series = body["series"]
    self.assertEqual(len({item["id"] for item in series}), 4)
    self.assertEqual(sorted(item["scope"] for item in series), ["model", "model", "runtime", "session"])
    self.assertTrue(all(item["points"] == [[120, 2.5], [150, 3]] for item in series))
    expression = query.call_args.kwargs["params"]["query"]
    self.assertIn('openrl_model_id="model-a"', expression)
    self.assertIn('openrl_model_id="",session_id=~' + json.dumps(r"session\-a"), expression)
    self.assertIn('openrl_model_id="",session_id="",openrl_runtime_id=~', expression)
    self.assertIn('metric=~"(?i).*reward\\\\.\\\\*\\"\\\\}.*"', expression)

  async def test_visible_series_cap_search_and_empty_or_failed_backend(self):
    rows = [
      {"metric": {"__name__": "openrl_server_metric", "metric": f"metric-{i:03}", "k8s_node_name": "node-a"}, "values": [[120, "1"]]}
      for i in range(collector.SERIES_LIMIT + 1)
    ]
    params = {**self.params, "node_id": "node-a"}
    del params["run_id"]
    with patch.object(http, "get_json", AsyncMock(return_value={"status": "success", "data": {"result": rows}})):
      response = await self.client.get("/api/v1/inspect/telemetry", params=params)
    self.assertEqual(response.status_code, 200)
    self.assertEqual(len(response.json()["series"]), collector.SERIES_LIMIT)
    self.assertTrue(response.json()["truncated"])
    self.assertTrue(all(item["scope"] == "node" for item in response.json()["series"]))
    with patch.object(http, "get_json", AsyncMock(return_value={"status": "success", "data": {"result": []}})):
      empty = await self.client.get("/api/v1/inspect/telemetry", params={**params, "metric": "not recorded"})
    self.assertTrue(empty.json()["available"])
    self.assertEqual(empty.json()["series"], [])
    with patch.object(http, "get_json", AsyncMock(side_effect=RuntimeError("unavailable"))):
      failed = await self.client.get("/api/v1/inspect/telemetry", params=params)
    self.assertFalse(failed.json()["available"])
    self.assertEqual(failed.json()["error"], "Collector metrics query unavailable")
    for invalid in ({}, {"run_id": "model-a", "node_id": "node-a"}):
      self.assertEqual((await self.client.get("/api/v1/inspect/telemetry", params=invalid)).status_code, 400)
    self.assertEqual((await self.client.get("/api/v1/inspect/telemetry", params={"run_id": "missing"})).status_code, 404)

    # Legacy restored models can have no runtime. An empty regex alternative
    # would match absent labels and incorrectly include unrelated process data.
    await self.store.set_value("open_rl:model_meta:model-a", json.dumps({"fine_tuning_type": "restored", "session_ids": [""]}))
    with patch.object(http, "get_json", AsyncMock(return_value={"status": "success", "data": {"result": []}})) as query:
      restored = await self.client.get("/api/v1/inspect/telemetry", params=self.params)
    self.assertTrue(restored.json()["available"])
    expression = query.call_args.kwargs["params"]["query"]
    self.assertNotIn(" or ", expression)
    self.assertNotIn("session_id=~", expression)
    self.assertNotIn("openrl_runtime_id=~", expression)


if __name__ == "__main__":
  unittest.main()
