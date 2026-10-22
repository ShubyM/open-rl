import asyncio
import time
import unittest
from datetime import datetime
from unittest import mock

from fastapi.testclient import TestClient

from server.dashboard import router
from server.dashboard.__main__ import app
from tests.test_dashboard_gke_logs import CONFIG
from tests.test_dashboard_inspection import PROCESS, RECORD, RUN, operation, state

WINDOW = {"since": "1970-01-01T00:02:00Z", "until": "1970-01-01T00:03:10Z"}


class InspectionApiTest(unittest.TestCase):
  def setUp(self):
    router.operations.clear()
    self.client = TestClient(app)
    self.state = state()
    self.state["runs"][0].update(shared_runtime=False)
    self.store = mock.Mock(read_samples=mock.AsyncMock(return_value=[]))
    patches = [
      mock.patch.object(router.snapshot, "current", mock.AsyncMock(return_value=self.state)),
      mock.patch.object(router.sources, "discover", mock.AsyncMock()),
      mock.patch.object(router, "get_state_store", return_value=self.store),
    ]
    for patch in patches:
      patch.start()
      self.addCleanup(patch.stop)

  def test_inventory_is_the_snapshot_without_history_and_the_api_is_read_only(self):
    self.state["cluster"]["pods"] = [{"name": "trainer-0", "problem": None}, {"name": "stuck", "problem": "Pending"}]
    response = self.client.get("/api/v1/inspect/inventory")
    self.assertEqual(response.status_code, 200)
    inventory = response.json()
    self.assertEqual(inventory["placements"][0]["devices"], RECORD["gpu_uuids"])
    self.assertEqual([pod["name"] for pod in inventory["cluster"]["pods"]], ["stuck"])
    self.assertNotIn("processes", inventory)
    self.assertNotIn("placement_records", inventory)
    paths = self.client.get("/openapi.json").json()["paths"]
    self.assertIn("/api/v1/inspect/activity", paths)
    self.assertTrue(all(set(methods) == {"get"} for methods in paths.values()))

  def test_activity_reads_each_runs_ops_once_and_keeps_per_lane_errors(self):
    self.state.update(placement_records=[{**RECORD, "model_id": "base"}], processes=[{**PROCESS, "runtime_id": "base"}])
    self.state["runs"] = [{**RUN, "run_id": identity, "runtime_id": "base", "shared_runtime": True} for identity in ("r1", "r2")]

    async def read_samples(key):
      if key == router.telemetry.ops_key("r2"):
        raise RuntimeError("offline")
      return [operation(110, 130)]

    self.store.read_samples.side_effect = read_samples
    response = self.client.get("/api/v1/inspect/activity", params=WINDOW)
    self.assertEqual(response.status_code, 200)
    keys = sorted(call.args[0] for call in self.store.read_samples.await_args_list)
    self.assertEqual(keys, sorted(key(k) for k in ("r1", "r2", "base") for key in (router.telemetry.ops_key, router.telemetry.samples_key)))
    lanes = {p["id"]: p for p in response.json()["placements"]}
    self.assertEqual(set(lanes), {"trainer@90:r1", "trainer@90:r2"})
    self.assertEqual((lanes["trainer@90:r1"]["intervals"], lanes["trainer@90:r1"].get("error")), ([[120, 130]], None))
    self.assertEqual((lanes["trainer@90:r2"]["intervals"], lanes["trainer@90:r2"]["error"]), ([], "Operation records unavailable"))

  def test_queries_validate_windows_without_touching_sources(self):
    for path in ("activity", "series", "logs"):
      for bound, value in (("since", "invalid"), ("until", "0001-01-01T00:00:00Z"), ("since", "9999-12-31T23:59:59-23:59")):
        response = self.client.get(f"/api/v1/inspect/{path}", params={"run_id": "r1", bound: value})
        self.assertEqual(response.status_code, 400)
    self.store.read_samples.assert_not_awaited()

  def test_run_activity_keeps_metrics_and_incidents_without_placement_history(self):
    self.state.update(placement_records=[], processes=[])
    self.state["runs"][0]["pods"] = [{"name": "p", "events": [{"reason": "OOMKilled", "last_seen_at": "1970-01-01T00:02:30Z", "message": "OOM"}]}]
    self.store.read_samples.side_effect = lambda key: [operation(125, 140), operation(150, 200)] if key == router.telemetry.ops_key("r1") else []
    response = self.client.get("/api/v1/inspect/activity", params={**WINDOW, "run_id": "r1"})
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["run"], self.client.get("/api/v1/inspect/inventory").json()["runs"][0])
    self.assertEqual(data["placements"], [])
    self.assertEqual([s["at"] for s in data["samples"]], [140])
    self.assertEqual(data["incidents"][0]["reason"], "OOMKilled")
    self.assertEqual(
      sorted(c.args[0] for c in self.store.read_samples.await_args_list), [router.telemetry.ops_key("r1"), router.telemetry.samples_key("r1")]
    )

  def test_series_is_independently_callable_and_keeps_partial_data(self):
    backend = mock.Mock(name="hardware", slots={})
    backend.name = "fake"
    backend.devices = mock.AsyncMock(return_value=({}, {"utilization": "GPU metrics query unavailable"}))
    backend.workers_series = mock.AsyncMock(return_value=({"cpu_cores": {"trainer-0": [[150, 2]]}}, {}))
    with mock.patch.object(router.metrics, "hardware", return_value=(backend, None)):
      response = self.client.get("/api/v1/inspect/series", params={**WINDOW, "node_id": "n1"})
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertTrue(data["available"])
    self.assertEqual(data["workers"][0]["cpu_cores"], [[150, 2]])
    self.assertIn("unavailable", data["reason"])
    self.assertEqual(data["devices"][0]["errors"], {"utilization": "GPU metrics query unavailable"})
    router.sources.discover.assert_awaited_once()

  def test_logs_filter_by_the_owner_label_and_page_with_cloud_loggings_token(self):
    with (
      mock.patch.object(router.sources, "configuration", return_value=CONFIG),
      mock.patch.object(router.sources, "gcp_request", mock.AsyncMock(side_effect=[{"nextPageToken": "page-2"}, {}])) as request,
    ):
      first = self.client.get("/api/v1/inspect/logs", params={"run_id": "r1", **WINDOW}).json()
      second = self.client.get("/api/v1/inspect/logs", params={"run_id": "r1", "cursor": first["next_cursor"], **WINDOW})
    self.assertEqual((first["next_cursor"], second.status_code), ("page-2", 200))
    self.assertEqual(first["sources"], [{"pod": "trainer-0", "node": "n1", "role": "trainer"}])
    before, after = [call.kwargs["json"] for call in request.await_args_list]
    self.assertIn('labels."k8s-pod/timeslice_io/owner"="r1"', before["filter"])
    self.assertNotIn("pod_name", before["filter"])
    self.assertEqual(after, {**before, "pageToken": "page-2"})

  def test_sampler_queries_cover_the_runs_released_placements(self):
    sampler = {**RECORD, "workload": "sampler", "role": "sampler", "pod": "deleted-sampler", "placed_at": 120, "released_at": 170}
    self.state.update(placement_records=[RECORD, sampler], processes=[])
    backend = mock.Mock(name="prometheus", slots={})
    backend.name = "prometheus"
    backend.scope.return_value = ""
    backend.range_rows = mock.AsyncMock(return_value=[{"metric": {}, "values": [[150, "2"]]}])
    with mock.patch.object(router.queries, "hardware", return_value=(backend, None)):
      running = self.client.get("/api/v1/inspect/query", params={"name": "vllm_requests_running", "run_id": "r1", **WINDOW})
      busy = self.client.get("/api/v1/inspect/query", params={"name": "gpu_busy", "run_id": "r1", **WINDOW})
    self.assertEqual(running.status_code, 200, running.text)
    self.assertEqual(running.json()["series"][0]["points"], [[150, 2]])
    self.assertIn("deleted\\\\-sampler", backend.range_rows.await_args_list[0].args[0])
    self.assertIn("GPU\\\\-1", busy.json()["query"])
    self.store.read_samples.assert_not_awaited()

  def test_request_deadline_cancels_pending_source_work(self):
    cancelled = []

    async def stalled_snapshot(_):
      try:
        await asyncio.Future()
      finally:
        cancelled.append(True)

    router.snapshot.current.side_effect = stalled_snapshot
    with mock.patch.object(router, "REQUEST_SECONDS", 0.02):
      response = self.client.get("/api/v1/inspect/inventory")
    self.assertEqual(response.status_code, 504)
    self.assertEqual(cancelled, [True])
    self.assertIn("Inspection timed out", response.json()["detail"])

  def test_omitted_bounds_default_to_the_30_minutes_ending_now(self):
    before = time.time()
    data = self.client.get("/api/v1/inspect/activity", params={"run_id": "r1"}).json()
    since, until = (datetime.fromisoformat(data[k]).timestamp() for k in ("since", "until"))
    self.assertTrue(before <= until <= time.time())
    self.assertAlmostEqual(until - since, 1800, places=3)
    self.assertEqual(data["observed_at"], self.state["observed_at"])
