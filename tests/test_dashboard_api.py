import asyncio
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server.dashboard import router
from server.dashboard.__main__ import app
from tests.test_dashboard_gke_logs import CONFIG
from tests.test_dashboard_inspection import PLACEMENT, PROCESS, RUN, operation, state

WINDOW = {"since": "1970-01-01T00:02:00Z", "until": "1970-01-01T00:03:10Z"}


class InspectionApiTest(unittest.TestCase):
  def setUp(self):
    self.client = TestClient(app)
    self.state = state()
    self.state["runs"][0].update(shared_runtime=False)
    self.store = mock.Mock(read_samples=mock.AsyncMock(return_value=[]), read_steps=mock.AsyncMock(return_value=(None, [])))
    patches = [
      mock.patch.object(router.snapshot, "current", mock.AsyncMock(return_value=self.state)),
      mock.patch.object(router.gke, "discover", mock.AsyncMock()),
      mock.patch.object(router, "get_state_store", return_value=self.store),
    ]
    for patch in patches:
      patch.start()
      self.addCleanup(patch.stop)

  def test_inventory_is_the_snapshot_without_history_and_the_api_is_read_only(self):
    response = self.client.get("/api/v1/inspect/inventory")
    self.assertEqual(response.status_code, 200)
    inventory = response.json()
    self.assertEqual(inventory["placements"][0]["devices"], PLACEMENT["devices"])
    self.assertNotIn("processes", inventory)
    paths = self.client.get("/openapi.json").json()["paths"]
    self.assertIn("/api/v1/inspect/activity", paths)
    self.assertTrue(all(set(methods) == {"get"} for methods in paths.values()))

  def test_activity_reads_each_runs_ops_once_and_keeps_per_lane_errors(self):
    self.state.update(
      placements=[], processes=[{**PROCESS, "runtime_id": "base"}, {**PROCESS, "id": "old", "pod_uid": "old", "started_at": 1, "heartbeat_at": 10}]
    )
    self.state["runs"] = [{**RUN, "run_id": identity, "runtime_id": "base", "shared_runtime": True} for identity in ("r1", "r2")]

    async def read_samples(key):
      if key == router.ops.key("r2"):
        raise RuntimeError("offline")
      return [operation(110, 130)]

    self.store.read_samples.side_effect = read_samples
    response = self.client.get("/api/v1/inspect/activity", params=WINDOW)
    self.assertEqual(response.status_code, 200)
    keys = sorted(call.args[0] for call in self.store.read_samples.await_args_list)
    self.assertEqual(keys, sorted(router.ops.key(k) for k in ("r1", "r2", "base")))
    lanes = {p["id"]: p for p in response.json()["placements"]}
    self.assertEqual(set(lanes), {"p1:r1", "p1:r2"})
    self.assertEqual((lanes["p1:r1"]["intervals"], lanes["p1:r1"].get("error")), ([[120, 130]], None))
    self.assertEqual((lanes["p1:r2"]["intervals"], lanes["p1:r2"]["error"]), ([], "Operation records unavailable"))

  def test_queries_validate_windows_without_touching_sources(self):
    for path in ("activity", "series", "logs"):
      for bound, value in (("since", "invalid"), ("until", "0001-01-01T00:00:00Z"), ("since", "9999-12-31T23:59:59-23:59")):
        response = self.client.get(f"/api/v1/inspect/{path}", params={"run_id": "r1", bound: value})
        self.assertEqual(response.status_code, 400)
    self.store.read_samples.assert_not_awaited()

  def test_run_activity_keeps_metrics_and_incidents_without_placement_history(self):
    self.state.update(placements=[], processes=[])
    self.state["runs"][0]["pods"] = [{"name": "p", "events": [{"reason": "OOMKilled", "last_seen_at": "1970-01-01T00:02:30Z", "message": "OOM"}]}]
    self.store.read_samples.return_value = [operation(125, 140), operation(150, 200)]
    response = self.client.get("/api/v1/inspect/activity", params={**WINDOW, "run_id": "r1"})
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["run"], self.state["runs"][0])
    self.assertEqual(data["placements"], [])
    self.assertEqual([s["at"] for s in data["samples"]], [140])
    self.assertEqual(data["incidents"][0]["reason"], "OOMKilled")
    self.store.read_samples.assert_awaited_once_with(router.ops.key("r1"))

  def test_series_is_independently_callable_and_keeps_partial_data(self):
    backend = mock.Mock(name="hardware")
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
    router.gke.discover.assert_awaited_once()

  def test_log_cursor_continuation_uses_no_inventory_or_store_reads(self):
    self.state["runs"][0]["shared_runtime"] = True
    with (
      mock.patch.object(router.gke, "configuration", return_value=CONFIG),
      mock.patch.object(router.gke, "request", mock.AsyncMock(side_effect=[{"nextPageToken": "next-provider-page"}, {}])) as request,
    ):
      first = self.client.get("/api/v1/inspect/logs", params={"run_id": "r1", **WINDOW}).json()
      router.snapshot.current.reset_mock()
      router.snapshot.current.side_effect = RuntimeError("inventory offline")
      router.get_state_store.reset_mock()
      self.store.read_samples.reset_mock()
      second = self.client.get("/api/v1/inspect/logs", params={"run_id": "r1", "cursor": first["next_cursor"], **WINDOW})
    self.assertEqual(second.status_code, 200)
    self.assertTrue(second.json()["shared_runtime"])
    router.snapshot.current.assert_not_awaited()
    router.get_state_store.assert_not_called()
    self.store.read_samples.assert_not_awaited()
    before, after = [call.kwargs["json"] for call in request.await_args_list]
    self.assertEqual(after, {**before, "pageToken": "next-provider-page"})

  def test_historical_sampler_operations_feed_both_logs_and_series(self):
    self.state.update(placements=[], processes=[])
    self.store.read_samples.return_value = [operation(125, 170, role="sampler", pod="deleted-sampler", operation="sample")]
    backend = mock.Mock(name="prometheus")
    backend.name = "prometheus"
    backend.samplers = mock.AsyncMock(return_value=({"requests_running": [[150, 2]]}, {"request_latency_p99_seconds": "Query unavailable"}))
    backend.devices = mock.AsyncMock()
    backend.workers_series = mock.AsyncMock()
    with (
      mock.patch.object(router.gke, "configuration", return_value=CONFIG),
      mock.patch.object(router.gke, "request", mock.AsyncMock(return_value={})) as request,
      mock.patch.object(router.metrics, "hardware", return_value=(backend, None)),
    ):
      logs = self.client.get("/api/v1/inspect/logs", params={"run_id": "r1", **WINDOW})
      series = self.client.get("/api/v1/inspect/series", params={"run_id": "r1", "kind": "samplers", **WINDOW})
    self.assertEqual(logs.status_code, 200)
    self.assertIn('resource.labels.pod_name="deleted-sampler"', request.await_args.kwargs["json"]["filter"])
    self.assertEqual(series.status_code, 200)
    self.assertTrue(series.json()["available"])
    self.assertEqual(series.json()["samplers_errors"], {"request_latency_p99_seconds": "Query unavailable"})
    self.assertEqual(backend.samplers.await_args.args[0], ["deleted-sampler"])
    backend.devices.assert_not_awaited()
    backend.workers_series.assert_not_awaited()

  def test_invalid_series_kinds_fail_before_reading_sources(self):
    for params, expected in (({"kind": "invalid"}, 422), ({"kind": "samplers"}, 400)):
      response = self.client.get("/api/v1/inspect/series", params=params)
      self.assertEqual(response.status_code, expected)
    router.snapshot.current.assert_not_awaited()
    router.gke.discover.assert_not_awaited()
    self.store.read_samples.assert_not_awaited()

  def test_unavailable_historical_identity_is_not_empty_logs_or_metrics(self):
    self.store.read_samples.side_effect = RuntimeError("store offline")
    for endpoint in ("logs", "series"):
      response = self.client.get(f"/api/v1/inspect/{endpoint}", params={"run_id": "r1", **WINDOW})
      self.assertEqual(response.status_code, 503)
      self.assertIn("Operation records unavailable", response.json()["detail"])

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

  def test_corrupt_telemetry_keeps_valid_records_and_reports_partial_failure(self):
    bad = [
      None,
      [],
      {},
      {"at": "oops"},
      operation(140, float("nan")),
      operation(140, float("inf")),
      operation(160, 150),
      operation(140, 150, metrics={"loss": float("inf")}),
      operation(140, 150, role=[]),
    ]
    self.store.read_samples.side_effect = lambda key: [operation(125, 140), *bad] if key == router.ops.key("r1") else []
    response = self.client.get("/api/v1/inspect/activity", params={**WINDOW, "run_id": "r1"})
    self.assertEqual(response.status_code, 200)
    result = response.json()
    self.assertEqual([s["at"] for s in result["samples"]], [140])
    self.assertIn("invalid", result["partial"].lower())
    self.assertIsNone(result["error"])
    self.assertEqual(result["placements"][0]["intervals"], [[125, 140]])

  def test_pod_logs_preserve_previous_and_unknown_diagnostics(self):
    result = {"pod": "p", "container": None, "previous": True, "text": "untrusted <script> false 0", "future": {"flag": False}}
    with mock.patch.object(router.kubernetes, "pod_logs", return_value=result):
      response = self.client.get("/api/v1/inspect/pods/p/logs?previous=true")
    self.assertEqual(response.json(), result)

  def test_omitted_bounds_anchor_to_snapshot_and_report_the_resolved_window(self):
    self.store.read_samples.side_effect = lambda key: [operation(125, 140)] if key == router.ops.key("r1") else []
    for params in ({"run_id": "r1"}, {"run_id": "r1", "since": WINDOW["since"]}):
      data = self.client.get("/api/v1/inspect/activity", params=params).json()
      self.assertEqual(data["until"], "1970-01-01T00:03:20.000000+00:00")
      self.assertEqual(data["observed_at"], self.state["observed_at"])
      self.assertIn("Missing records do not prove no requests arrived", data["coverage"])
      self.assertEqual(data["placements"][0]["intervals"], [[125, 140]])
      self.assertEqual(len(data["samples"]), 1)
