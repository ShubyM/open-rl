import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from server import api_server
from server.dashboard import router
from server.dashboard.__main__ import app as dashboard_app
from server.scripts import forward_metrics
from server.store import InMemoryStateStore
from tests.api_client import asgi_client


class RunMetricsTest(unittest.IsolatedAsyncioTestCase):
  """A client's per-step metrics go to the gateway and come back from the dashboard API,
  through one store."""

  async def asyncSetUp(self) -> None:
    self.state = InMemoryStateStore()
    self.enterContext(patch.object(api_server, "state", self.state))
    self.enterContext(patch.object(router, "get_state_store", return_value=self.state))
    self.dashboard = httpx.AsyncClient(transport=httpx.ASGITransport(app=dashboard_app), base_url="http://dashboard")
    self.addAsyncCleanup(self.dashboard.aclose)

  async def rows(self, run_id: str) -> list[dict]:
    response = await self.dashboard.get("/api/v1/inspect/run_metrics", params={"run_id": run_id})
    self.assertEqual(response.status_code, 200)
    self.assertTrue(response.json()["available"])
    return response.json()["rows"]

  async def test_a_malformed_run_id_or_row_is_rejected(self) -> None:
    async with asgi_client() as gateway:
      self.assertEqual((await gateway.post("/api/v1/runs/a b/metrics", json={"rows": []})).status_code, 400)
      for step in (-1, 2**53, True):
        self.assertEqual((await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": [{"step": step, "metrics": {}}]})).status_code, 422)
    self.assertEqual(await self.rows("run-1"), [])

  async def test_ingestion_serves_step_history_and_latest_values_with_escaped_prometheus_labels(self) -> None:
    metric = 'reward"\\\nnext'
    async with asgi_client() as gateway:
      response = await gateway.post(
        "/api/v1/runs/run:latest/metrics", json={"rows": [{"step": 1, "metrics": {metric: 0.7, "note": "text", "done": True}}]}
      )
      self.assertEqual(response.json(), {"accepted": 1})
      await gateway.post("/api/v1/runs/run:latest/metrics", json={"rows": [{"step": 0, "metrics": {metric: 0.1}}]})
    rows = await self.rows("run:latest")
    self.assertEqual([(row["step"], row["metrics"]) for row in rows], [(0, {metric: 0.1}), (1, {metric: 0.7})])
    scrape = await self.dashboard.get("/metrics")
    self.assertEqual(scrape.status_code, 200)
    self.assertIn('openrl_run_metric{run_id="run:latest",metric="reward\\"\\\\\\nnext"} 0.7\n', scrape.text)
    self.assertIn('openrl_run_step{run_id="run:latest"} 1\n', scrape.text)
    self.assertNotIn('step="', scrape.text)

  async def test_unavailable_metric_store_returns_a_failed_scrape(self) -> None:
    with patch.object(self.state, "scan_values", side_effect=RuntimeError("store unavailable")):
      response = await self.dashboard.get("/metrics")
    self.assertEqual(response.status_code, 503)

  async def test_conditional_history_reuses_unchanged_rows_and_invalidates_old_step_corrections(self) -> None:
    url = "/api/v1/inspect/run_metrics?run_id=run-1"
    async with asgi_client() as gateway:
      await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": [{"step": step, "metrics": {"reward": step}} for step in (1, 2)]})
      first = await self.dashboard.get(url)
      etag = first.headers["etag"]
      self.assertEqual(first.headers["cache-control"], "private, no-cache")
      await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": []})
      for validator in (etag, etag.removeprefix("W/"), f'"older", {etag}', "*"):
        with self.subTest(validator=validator), patch("server.store.json.loads", side_effect=AssertionError("304 must not decode history")):
          unchanged = await self.dashboard.get(url, headers={"If-None-Match": validator})
        self.assertEqual((unchanged.status_code, unchanged.content, unchanged.headers["etag"]), (304, b"", etag))
      await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": [{"step": 1, "metrics": {"reward": 10}}]})
    changed = await self.dashboard.get(url, headers={"If-None-Match": etag})
    self.assertEqual(changed.status_code, 200)
    self.assertNotEqual(changed.headers["etag"], etag)
    self.assertEqual([row["metrics"]["reward"] for row in changed.json()["rows"]], [10, 2])
    with patch.object(self.state, "read_steps", side_effect=RuntimeError("store unavailable")):
      failed = await self.dashboard.get(url, headers={"If-None-Match": changed.headers["etag"]})
    self.assertEqual(failed.status_code, 200)
    self.assertFalse(failed.json()["available"])
    self.assertNotIn("etag", failed.headers)
    await self.state.put_steps("open_rl:run_steps:run-1", [{"step": 1, "metrics": {"reward": 1}}], 10, 0)
    expired = await self.dashboard.get(url, headers={"If-None-Match": changed.headers["etag"]})
    self.assertEqual((expired.status_code, expired.json()["rows"]), (200, []))
    self.assertNotIn("etag", expired.headers)


class ReadRowsTest(unittest.TestCase):
  def test_nonfinite_and_invalid_rows_do_not_stop_later_steps(self) -> None:
    path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "metrics.jsonl"
    path.write_text('[1, 2]\n{"step": -1, "reward": 1}\n{"step": 0, "loss": NaN, "reward": 0.3}\n{"step": 1, "reward": 0.5}\n')
    rows, _ = forward_metrics.read_rows(path, 0)
    self.assertEqual(rows, [{"step": 0, "metrics": {"reward": 0.3}}, {"step": 1, "metrics": {"reward": 0.5}}])
    (path.parent / "checkpoints.jsonl").write_text('{"state_path": "tinker://run-7/weights/000010"}\n')
    self.assertEqual(forward_metrics.run_id_of(path.parent), "run-7")

  def test_failed_batch_retries_do_not_resend_acknowledged_rows(self) -> None:
    seen = []

    def respond(request):
      seen.append([row["step"] for row in json.loads(request.content)["rows"]])
      return httpx.Response(503 if len(seen) == 2 else 200, json={"accepted": len(seen[-1])})

    pending = [{"step": n, "metrics": {"reward": n}} for n in range(5)]
    with httpx.Client(transport=httpx.MockTransport(respond)) as http, patch.object(forward_metrics, "BATCH", 2):
      with self.assertRaises(httpx.HTTPStatusError):
        forward_metrics.send(http, "http://gateway", "run", pending)
      self.assertEqual([row["step"] for row in pending], [2, 3, 4])
      self.assertEqual(forward_metrics.send(http, "http://gateway", "run", pending), 3)
    self.assertEqual(seen, [[0, 1], [2, 3], [2, 3], [4]])
    self.assertEqual(pending, [])

  def test_a_partial_last_line_waits_for_the_next_read(self) -> None:
    path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "metrics.jsonl"
    path.write_text('{"step": 0, "a": 1}\n{"progress/batch": 1, "a": 2}\n{"step": 2, "a"')
    rows, offset = forward_metrics.read_rows(path, 0)
    self.assertEqual(rows, [{"step": 0, "metrics": {"a": 1}}, {"step": 1, "metrics": {"progress/batch": 1, "a": 2}}])
    with path.open("a") as f:
      f.write(': 3}\n{"no step": true}\n')
    rows, _ = forward_metrics.read_rows(path, offset)
    self.assertEqual(rows, [{"step": 2, "metrics": {"a": 3}}])


if __name__ == "__main__":
  unittest.main()
