"""HTTP contracts shared by the inspection API and the restored dashboard views."""

import asyncio
import copy
import json
import unittest
from datetime import datetime
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from server.dashboard import facts, fleet, router, sources, views
from server.dashboard.__main__ import app
from server.store import InMemoryStateStore
from server.telemetry import key

NOW = 2000
BOUNDS = {"since": "1970-01-01T00:16:40Z", "until": "1970-01-01T00:33:20Z"}


class Clock(datetime):
  @classmethod
  def now(cls, tz=None):
    return cls.fromtimestamp(NOW, tz)


class DashboardCompatibilityTest(unittest.TestCase):
  def setUp(self):
    self.store = InMemoryStateStore()
    self.pod = {
      "name": "trainer-0",
      "node": "n1",
      "problem": None,
      "restarts": 1,
      "phase": "Running",
      "ready": True,
      "incidents": [{"at": 1490, "pod": "trainer-0", "reason": "OOMKilled", "message": "trainer exited with code 137"}],
      "events": [{"reason": "OOMKilled", "type": "Warning", "last_seen_at": "1970-01-01T00:24:50Z"}],
      "containers": [
        {
          "name": "trainer",
          "state": "running",
          "restart_count": 1,
          "started_at": "1970-01-01T00:25:00Z",
          "last_termination": {"reason": "OOMKilled", "exit_code": 137, "finished_at": "1970-01-01T00:24:50Z"},
        }
      ],
    }
    self.worker = {
      "id": "process-1",
      "runtime_id": "r1",
      "pod": "trainer-0",
      "role": "trainer",
      "started_at": 900,
      "heartbeat_at": NOW,
      "phase": "ready",
      "turns": [[1080, 1110]],
    }
    self.workload = {
      "name": "trainer",
      "workload": "trainer",
      "role": "trainer",
      "state": "ready",
      "node": "n1",
      "pod": "trainer-0",
      "age_seconds": 1100,
      "detail": None,
    }
    self.run = {
      "run_id": "r1",
      "name": "Run one",
      "model": "Qwen/base",
      "runtime_id": "r1",
      "created_at": 900,
      "completed_at": None,
      "lifecycle": "placed",
      "status": "active",
      "settings": {},
      "session_ids": ["s1"],
      "pods": [self.pod],
      "workloads": [self.workload],
    }
    self.state = {
      "observed_at": NOW,
      "runs": [self.run],
      "workers": [self.worker],
      "pods": [self.pod],
      "workloads": [self.workload],
      "nodes": [{"name": "n1", "devices": [{"id": "GPU-1", "name": "gpu-0"}]}],
      "errors": [],
      "placements": [
        {
          "workload": "trainer",
          "role": "trainer",
          "model_id": "r1",
          "owner": "r1",
          "pod": "trainer-0",
          "node": "n1",
          "gpu_uuids": ["GPU-1"],
          "placed_at": 900,
          "released_at": None,
        }
      ],
    }
    trainer = {
      "at": 1090.2,
      "started_at": 1080,
      "operation": "forward_backward",
      "role": "trainer",
      "status": "succeeded",
      "elapsed_seconds": 10.2,
      "metrics": {"loss:mean": 2.5},
      "run_id": "r1",
      "pod": "trainer-0",
    }
    sample = {"operation": "sample", "role": "sampler", "status": "succeeded", "run_id": "r1", "pod": "sampler-0"}
    self.ops = [
      {**sample, "at": 999, "started_at": 990, "elapsed_seconds": 9},
      {**sample, "at": 1090.1, "started_at": 1058.1, "elapsed_seconds": 32},
      trainer,
      {**sample, "at": 1090.7, "started_at": 1089.7, "elapsed_seconds": 1},
      {**sample, "at": 1200, "started_at": 1195, "elapsed_seconds": 5},
    ]
    asyncio.run(self.store.replace_samples(key("operations", "r1"), self.ops, None))
    asyncio.run(
      self.store.replace_samples(
        key("sdk_errors", "s1"),
        [
          {"at": 1590, "reason": "TimeoutError", "message": "old timeout"},
          {"at": 1600, "reason": "TimeoutError", "message": "new timeout"},
          {"at": 999, "reason": "ValueError", "message": "outside window"},
        ],
        None,
      )
    )
    fleet.cache.clear()
    self.addCleanup(fleet.cache.clear)
    for module in (facts, fleet, router, views):
      self.enterContext(mock.patch.object(module, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=self.state)))
    self.enterContext(mock.patch.object(sources, "discover", mock.AsyncMock()))
    self.enterContext(mock.patch.object(sources, "configuration", return_value={"configured": False, "cluster": "", "project": "", "location": ""}))
    self.enterContext(mock.patch.object(sources, "prometheus_source", return_value=None))
    self.enterContext(mock.patch.object(sources, "datetime", Clock))
    self.enterContext(mock.patch("time.time", return_value=NOW))
    self.client = self.enterContext(TestClient(app))

  def get(self, route, **params):
    response = self.client.get("/api/v1/" + route, params=params)
    self.assertEqual(response.status_code, 200, response.text)
    return response.json()

  def test_dashboard_page_assets_and_root_redirect_are_served(self):
    response = self.client.get("/", follow_redirects=False)
    self.assertEqual((response.status_code, response.headers["location"]), (307, "/dashboard"))
    for path in ("/dashboard", "/dashboard/"):
      with self.subTest(path=path):
        page = self.client.get(path)
        self.assertEqual(page.status_code, 200)
        self.assertTrue(page.headers["content-type"].startswith("text/html"))
        self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertIn('<div id="app">', page.text)
        self.assertIn("/dashboard/assets/app.js", page.text)
    style = self.client.get("/dashboard/assets/style.css")
    self.assertEqual(style.status_code, 200)
    self.assertTrue(style.headers["content-type"].startswith("text/css"))
    self.assertTrue(style.text)
    self.assertEqual(self.client.get("/dashboard/assets/missing.js").status_code, 404)

  def test_inventory_has_worker_and_pod_details_without_expanding_the_inspection_projection(self):
    before = copy.deepcopy(self.state)
    inspected = self.get("inspect/inventory")
    shown = self.get("dashboard/inventory")
    self.assertEqual(inspected["runs"][0]["pods"], [{"name": "trainer-0", "problem": None}])
    self.assertNotIn("workloads", inspected["runs"][0])
    self.assertEqual(shown["runs"][0]["workloads"], [self.workload])
    self.assertEqual(shown["runs"][0]["pods"], [self.pod])
    self.assertEqual(shown["cluster"]["pods"], [self.pod])
    self.assertEqual(shown["placements"], inspected["placements"])
    self.assertEqual(self.state, before)

  def test_a_waiting_workload_is_visible_before_a_process_reports(self):
    self.state["workers"] = []
    self.workload.update(state="waiting_for_placement", node=None, pod=None, detail="No matching device available")
    for route in ("inventory", "activity"):
      with self.subTest(route=route):
        shown = self.get("dashboard/" + route, run_id="r1", **BOUNDS)
        run = shown["runs"][0] if route == "inventory" else shown["run"]
        self.assertEqual(run["workloads"], [self.workload])
        self.assertEqual(run["workloads"][0]["state"], "waiting_for_placement")

  def test_activity_keeps_lanes_and_wall_time_peaks_and_correlates_restart_and_sdk_incidents(self):
    inspected = self.get("inspect/activity", run_id="r1", **BOUNDS)
    shown = self.get("dashboard/activity", run_id="r1", **BOUNDS)
    for field in ("placements", "since", "until", "observed_at"):
      self.assertEqual(shown[field], inspected[field])
    self.assertEqual(shown["run"]["workloads"], [self.workload])
    self.assertEqual([r["at"] for r in shown["samples"]], [1090.1, 1090.2, 1200])
    self.assertEqual(shown["samples"][0]["elapsed_seconds"], 32)
    self.assertEqual(shown["samples"][1]["metrics"], {"loss:mean": 2.5})
    self.assertTrue(all("run_id" not in sample and "pod" not in sample for sample in shown["samples"]))
    self.assertEqual(
      [(e["at"], e["reason"], e["pod"]) for e in shown["incidents"]],
      [
        (1600, "TimeoutError", "client"),
        (1500, "WorkerRestarted", "trainer-0"),
        (1490, "OOMKilled", "trainer-0"),
      ],
    )
    self.assertEqual((shown["incidents"][0]["message"], shown["incidents"][0]["count"]), ("new timeout", 2))
    self.assertEqual(shown["errors"], [])
    self.assertNotIn("samples", inspected)

  def test_activity_preserves_available_context_when_operation_history_is_unavailable(self):
    self.state["errors"] = ["Kubernetes events unavailable"]
    with mock.patch.object(fleet, "operations", mock.AsyncMock(side_effect=RuntimeError("store offline"))):
      shown = self.get("dashboard/activity", run_id="r1", **BOUNDS)
    self.assertEqual(shown["run"]["run_id"], "r1")
    self.assertEqual((shown["placements"], shown["samples"]), ([], []))
    self.assertIn("Kubernetes events unavailable", shown["errors"])
    self.assertIn("Operation records unavailable", shown["errors"])
    self.assertEqual([e["reason"] for e in shown["incidents"]], ["WorkerRestarted", "OOMKilled"])

  def test_history_keeps_the_latest_step_and_only_finite_numeric_metrics(self):
    records = [
      {"step": 2, "metrics": {"reward": 0.5}},
      {"step": 1, "metrics": {"reward": 0.25}},
      {"step": 2, "metrics": {"reward": 0.75, "flag": True, "text": "1", "bad": float("inf")}},
      {"step": "invalid", "metrics": {"reward": 1}},
      None,
    ]
    with mock.patch.object(self.store, "read_samples", mock.AsyncMock(return_value=records)):
      self.assertEqual(
        self.get("dashboard/history", run_id="r1"),
        {
          "available": True,
          "rows": [
            {"step": 1, "metrics": {"reward": 0.25}},
            {"step": 2, "metrics": {"reward": 0.75}},
          ],
        },
      )
    with mock.patch.object(self.store, "read_samples", mock.AsyncMock(side_effect=RuntimeError("store offline"))):
      self.assertEqual(self.get("dashboard/history", run_id="r1"), {"available": False, "error": "Stored history unavailable", "rows": []})

  def test_steps_keep_trainer_history_when_client_metrics_fail(self):
    with mock.patch.object(router, "client_steps", mock.AsyncMock(side_effect=RuntimeError("collector offline"))):
      inspected = self.get("inspect/runs/r1/steps")
      shown = self.get("dashboard/runs/r1/steps")
    self.assertEqual(shown["steps"], inspected["steps"])
    self.assertEqual(shown["steps"][0]["ops"]["forward_backward"]["metrics"]["loss:mean"], {"min": 2.5, "max": 2.5, "mean": 2.5})
    self.assertEqual(shown["client_error"], inspected["error"])
    self.assertNotIn("error", shown)

  def test_telemetry_preserves_metric_identity_and_excludes_nonfinite_and_out_of_window_points(self):
    labels = {"__name__": "openrl_client_metric", "metric": "reward", "session_id": "s1", "openrl_model_id": "r1", "openrl_phase": "train"}
    rows = [
      {"metric": labels, "values": [[999, "1"], [1000, "0.25"], [1500, "NaN"], [1750, "+Inf"], [2000, "0.75"], [2001, "2"]]},
      {"metric": {"__name__": "openrl_server_metric", "metric": "waiting", "k8s_pod_name": "trainer-0"}, "values": [[1100, "2"]]},
      {"metric": {"__name__": "openrl_client_metric"}, "values": [[1100, "1"]]},
    ]
    with (
      mock.patch.object(sources, "prometheus_source", return_value="prometheus"),
      mock.patch.object(sources, "prom", mock.AsyncMock(return_value=rows)) as read,
    ):
      result = self.get("dashboard/telemetry", run_id="r1", metric="reward", **BOUNDS)
    self.assertEqual((result["available"], result["source"], result["limit"], result["truncated"]), (True, "prometheus", 200, False))
    first, second = result["series"]
    self.assertEqual(
      {k: v for k, v in first.items() if k != "id"},
      {
        "name": "reward",
        "source": "client",
        "scope": "model",
        "scope_id": "r1",
        "phase": "train",
        "labels": labels,
        "points": [[1000.0, 0.25], [2000.0, 0.75]],
      },
    )
    self.assertEqual((second["scope"], second["scope_id"], second["source"]), ("pod", "trainer-0", "server"))
    self.assertNotEqual(first["id"], second["id"])
    query, start, end, resolution = read.await_args.args
    self.assertEqual((start, end, resolution), (1000, 2000, 300))
    for selector in ('openrl_model_id="r1"', 'session_id=~"s1"', 'openrl_runtime_id=~"r1"'):
      self.assertIn(selector, query)

  def test_telemetry_reports_its_series_cap_and_source_availability(self):
    rows = [
      {"metric": {"__name__": "openrl_server_metric", "metric": f"metric-{n:03d}", "k8s_node_name": "n1"}, "values": [[1500, "1"]]}
      for n in range(201)
    ]
    with (
      mock.patch.object(sources, "prometheus_source", return_value="prometheus"),
      mock.patch.object(sources, "prom", mock.AsyncMock(return_value=list(reversed(rows)))),
    ):
      result = self.get("dashboard/telemetry", node_id="n1", **BOUNDS)
    self.assertEqual((len(result["series"]), result["truncated"], result["series"][-1]["name"]), (200, True, "metric-199"))
    absent = self.get("dashboard/telemetry", run_id="r1", **BOUNDS)
    self.assertFalse(absent["available"])
    self.assertEqual(absent["series"], [])
    with (
      mock.patch.object(sources, "prometheus_source", return_value="prometheus"),
      mock.patch.object(sources, "prom", mock.AsyncMock(side_effect=RuntimeError("provider offline"))),
    ):
      failed = self.get("dashboard/telemetry", run_id="r1", **BOUNDS)
    self.assertFalse(failed["available"])
    self.assertEqual(failed["error"], "Collector metrics query unavailable")
    for params in ({}, {"run_id": "r1", "node_id": "n1"}):
      self.assertEqual(self.client.get("/api/v1/dashboard/telemetry", params=params).status_code, 400)

  def test_inspection_logs_keep_filters_cursor_and_explicit_source_errors(self):
    params = {**BOUNDS, "run_id": "r1", "pod": "trainer-0", "q": "oom", "severity": "ERROR", "limit": 7, "cursor": "page-2"}
    page = {"records": [{"id": "entry-2", "message": "out of memory"}], "next_cursor": "page-3"}
    with mock.patch.object(sources, "log_page", mock.AsyncMock(return_value=page)) as read:
      shown = self.get("inspect/logs", **params)
    read.assert_awaited_once()
    self.assertEqual(read.await_args.kwargs, {"pod": "trainer-0", "q": "oom", "severity": "ERROR", "limit": 7, "cursor": "page-2"})
    self.assertEqual(shown["next_cursor"], "page-3")
    error = httpx.HTTPStatusError("expired", request=httpx.Request("POST", "https://logging"), response=httpx.Response(400))
    with mock.patch.object(sources, "log_page", mock.AsyncMock(side_effect=error)):
      response = self.client.get("/api/v1/inspect/logs", params=params)
    self.assertEqual(response.status_code, 400)
    self.assertEqual(response.json(), {"detail": "Invalid or expired log cursor; repeat the query without it"})
    for failure, message in (
      (LookupError("GKE telemetry is not configured"), "GKE telemetry is not configured"),
      (RuntimeError("provider unavailable"), "Cloud Logging unavailable"),
    ):
      with self.subTest(failure=type(failure).__name__), mock.patch.object(sources, "log_page", mock.AsyncMock(side_effect=failure)):
        shown = self.get("inspect/logs", run_id="r1", **BOUNDS)
      self.assertFalse(shown["available"])
      self.assertEqual((shown["error"], shown["records"], shown["next_cursor"]), (message, [], None))

  def test_previous_pod_logs_keep_the_selected_container_and_report_source_failure(self):
    result = {"pod": "trainer-0", "container": "trainer", "previous": True, "text": "previous process failed\n"}
    with mock.patch.object(sources, "pod_logs", mock.AsyncMock(return_value=result)) as read:
      self.assertEqual(self.get("dashboard/pods/trainer-0/logs", container="trainer", previous=True, tail=17), result)
    read.assert_awaited_once_with("trainer-0", "trainer", 17, True)
    with mock.patch.object(sources, "pod_logs", mock.AsyncMock(side_effect=RuntimeError("kubelet offline"))):
      response = self.client.get("/api/v1/dashboard/pods/trainer-0/logs")
    self.assertEqual(response.status_code, 503)
    self.assertEqual(response.json(), {"detail": "Pod logs unavailable in the configured namespace"})

  def test_restored_routes_keep_the_existing_window_and_page_validation(self):
    for scope in ("inspect", "dashboard"):
      with self.subTest(scope=scope):
        shown = self.get(f"{scope}/activity")
        self.assertEqual(datetime.fromisoformat(shown["since"]).timestamp(), NOW - 1800)
        self.assertEqual(datetime.fromisoformat(shown["until"]).timestamp(), NOW)
        for since in ("invalid", "0m", "8d"):
          response = self.client.get(f"/api/v1/{scope}/activity", params={"since": since})
          self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get(f"/api/v1/{scope}/activity", params={"run_id": "missing"}).status_code, 404)
    for route, params in (
      ("inspect/logs", {"run_id": "r1", "limit": 1001}),
      ("inspect/logs", {"run_id": "r1", "severity": "WARN"}),
      ("dashboard/pods/trainer-0/logs", {"tail": 0}),
      ("dashboard/pods/trainer-0/logs", {"tail": 1001}),
    ):
      with self.subTest(route=route, params=params):
        self.assertEqual(self.client.get("/api/v1/" + route, params=params).status_code, 422)


class KubernetesCompatibilityTest(unittest.TestCase):
  def test_pod_details_keep_termination_history_and_join_events_by_uid(self):
    termination = {"reason": "OOMKilled", "exitCode": 137, "finishedAt": "1970-01-01T00:24:50Z"}
    status = {
      "name": "trainer",
      "restartCount": 1,
      "state": {"running": {"startedAt": "1970-01-01T00:25:00Z"}},
      "lastState": {"terminated": termination},
    }
    pod = {
      "metadata": {"name": "trainer-0", "uid": "new-pod", "labels": {"openrl.io/worker": "trainer-workload"}},
      "spec": {"nodeName": "n1"},
      "status": {"phase": "Running", "containerStatuses": [status]},
    }
    events = [
      {"involvedObject": {"uid": "old-pod", "name": "trainer-0"}, "reason": "Evicted", "lastTimestamp": "1970-01-01T00:20:00Z"},
      {
        "involvedObject": {"uid": "new-pod", "name": "trainer-0"},
        "reason": "OOMKilled",
        "firstTimestamp": "1970-01-01T00:23:20Z",
        "lastTimestamp": "1970-01-01T00:24:50Z",
        "count": 3,
      },
      {
        "involvedObject": {"uid": "new-pod", "name": "trainer-0"},
        "reason": "FailedScheduling",
        "eventTime": "1970-01-01T00:22:30Z",
        "metadata": {"creationTimestamp": "1970-01-01T00:21:40Z"},
        "series": {"lastObservedTime": "1970-01-01T00:23:10Z", "count": 4},
      },
    ]
    core, custom = mock.Mock(), mock.Mock()
    for method, items in (
      (core.list_namespaced_pod, [pod]),
      (core.list_namespaced_event, events),
      (core.list_node, []),
      (custom.list_namespaced_custom_object, []),
    ):
      method.return_value = mock.Mock(data=json.dumps({"items": items}))
    with mock.patch.object(sources, "k8s_clients", return_value=(core, custom)):
      result = asyncio.run(sources.kubernetes())
      core.list_namespaced_event.side_effect = RuntimeError("events offline")
      partial = asyncio.run(sources.kubernetes())
    found = result["pods"][0]
    self.assertEqual(found["worker"], "trainer-workload")
    self.assertEqual((found["name"], found["node"], found["phase"], found["restarts"], found["problem"]), ("trainer-0", "n1", "Running", 1, None))
    self.assertEqual(
      [(event["reason"], event["first_seen_at"], event["last_seen_at"], event["count"]) for event in found["events"]],
      [
        ("FailedScheduling", "1970-01-01T00:21:40Z", "1970-01-01T00:23:10Z", 4),
        ("OOMKilled", "1970-01-01T00:23:20Z", "1970-01-01T00:24:50Z", 3),
      ],
    )
    self.assertEqual(found["containers"][0]["last_termination"], {"reason": "OOMKilled", "exit_code": 137, "finished_at": "1970-01-01T00:24:50Z"})
    self.assertEqual(found["incidents"], [{"at": 1490, "pod": "trainer-0", "reason": "OOMKilled", "message": "trainer exited with code 137"}])
    self.assertEqual(partial["pods"][0]["events"], [])
    self.assertIn("events read failed: events offline", partial["errors"])

  def test_kubelet_logs_keep_namespace_and_read_bounds(self):
    core = mock.Mock()
    core.read_namespaced_pod_log.return_value = "previous process failed\n"
    with mock.patch.object(sources, "k8s_clients", return_value=(core, mock.Mock())), mock.patch.object(sources, "namespace", return_value="test-ns"):
      found = asyncio.run(sources.pod_logs("trainer-0", "trainer", 17, True))
    self.assertEqual(found, {"pod": "trainer-0", "container": "trainer", "previous": True, "text": "previous process failed\n"})
    core.read_namespaced_pod_log.assert_called_once_with(
      "trainer-0",
      "test-ns",
      container="trainer",
      previous=True,
      tail_lines=17,
      timestamps=True,
      limit_bytes=128 * 1024,
      _request_timeout=10,
    )


if __name__ == "__main__":
  unittest.main()
