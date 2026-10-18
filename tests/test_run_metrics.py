import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn

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

  async def test_the_dashboard_serves_the_newest_row_per_step_with_only_finite_numbers(self) -> None:
    async with asgi_client() as gateway:
      first = {"step": 1, "metrics": {"env/all/reward/total": 0.2, "note": "text", "done": True, "lr": 2e-4}}
      response = await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": [first, {"step": 0, "metrics": {"env/all/reward/total": 0.1}}]})
      self.assertEqual(response.json(), {"accepted": 2})
      await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": [{"step": 1, "metrics": {"env/all/reward/total": 0.3}}]})
    rows = await self.rows("run-1")
    self.assertEqual([(r["step"], r["metrics"]) for r in rows], [(0, {"env/all/reward/total": 0.1}), (1, {"env/all/reward/total": 0.3})])

  async def test_a_malformed_run_id_or_row_is_rejected(self) -> None:
    async with asgi_client() as gateway:
      self.assertEqual((await gateway.post("/api/v1/runs/a b/metrics", json={"rows": []})).status_code, 400)
      self.assertEqual((await gateway.post("/api/v1/runs/run-1/metrics", json={"rows": [{"step": -1, "metrics": {}}]})).status_code, 422)
    self.assertEqual(await self.rows("run-1"), [])

  async def test_the_exporter_sends_complete_lines_to_the_run_its_checkpoints_name(self) -> None:
    server = uvicorn.Server(uvicorn.Config(api_server.app, port=0, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    self.addAsyncCleanup(lambda: (setattr(server, "should_exit", True), serving)[1])
    while not server.started:
      await asyncio.sleep(0.01)
    base_url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    log = Path(self.enterContext(tempfile.TemporaryDirectory()))
    lines = [
      json.dumps({"step": 0, "env/all/reward/total": 0.17, "test/env/all/reward/total": 0.31}),
      json.dumps({"step": 1, "env/all/reward/total": 0.41}),
    ]
    # The cookbook is partway through writing step 2.
    (log / "metrics.jsonl").write_text("\n".join(lines) + '\n{"step": 2, "env/all/rew')
    (log / "checkpoints.jsonl").write_text(json.dumps({"name": "000010", "state_path": "tinker://run-7/weights/000010"}) + "\n")
    argv = ["forward_metrics", str(log), "--base-url", base_url]
    with patch("sys.argv", argv), patch.dict(os.environ):
      await asyncio.to_thread(forward_metrics.main)
    rows = await self.rows("run-7")
    self.assertEqual([r["step"] for r in rows], [0, 1])
    self.assertEqual(rows[0]["metrics"], {"env/all/reward/total": 0.17, "test/env/all/reward/total": 0.31})


class ReadRowsTest(unittest.TestCase):
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
