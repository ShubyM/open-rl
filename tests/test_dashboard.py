import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from dev.tools import ops
from server import gateway
from server.dashboard import data


class DashboardEndpointsTest(unittest.TestCase):
  def setUp(self) -> None:
    self.client = TestClient(gateway.app)

  def tearDown(self) -> None:
    os.environ.pop("OPEN_RL_DASHBOARD_DEMO", None)

  def test_index_serves_html(self) -> None:
    resp = self.client.get("/dashboard")
    self.assertEqual(resp.status_code, 200)
    self.assertIn("open-rl operations", resp.text)

  def test_static_assets_served(self) -> None:
    for asset in ("style.css", "app.js"):
      resp = self.client.get(f"/dashboard/static/{asset}")
      self.assertEqual(resp.status_code, 200, asset)

  def test_health_reports_all_groups(self) -> None:
    resp = self.client.get("/api/v1/dashboard/health")
    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertFalse(body["demo"])
    groups = {check["group"] for check in body["checks"]}
    self.assertLessEqual({"Gateway", "Storage", "Kubernetes"}, groups)
    statuses = {check["status"] for check in body["checks"]}
    self.assertLessEqual(statuses, {"ok", "warn", "error", "off"})
    stat_ids = {stat["id"] for stat in body["stats"]}
    self.assertLessEqual({"runs.active", "queue.requests", "queue.launch"}, stat_ids)
    for stat in body["stats"]:
      self.assertIn("value_number", stat)
      self.assertIn("unit", stat)
      self.assertIn("context", stat)
      self.assertIn(stat["status"], {"ok", "warn"})
    self.assertIsInstance(body["queues"], list)

  def test_snapshot_reads_kubernetes_once_and_bundles_triage_state(self) -> None:
    k8s = {"available": True, "namespace": "test", "error": None, "pods": [], "nodes": []}
    with mock.patch.object(data, "k8s_snapshot", return_value=k8s) as snapshot:
      resp = self.client.get("/api/v1/dashboard/snapshot")

    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertFalse(body["demo"])
    self.assertEqual(body["schema_version"], 1)
    self.assertIsNotNone(body["generated_at"])
    self.assertEqual({"cluster", "runs", "health", "problems"}, set(body) & {"cluster", "runs", "health", "problems"})
    self.assertEqual(body["cluster"]["kubernetes"]["namespace"], "test")
    self.assertIn("checks", body["health"])
    self.assertIn("problems", body["problems"])
    snapshot.assert_called_once_with()

  def test_cluster_degrades_without_kubernetes(self) -> None:
    resp = self.client.get("/api/v1/dashboard/cluster")
    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertFalse(body["demo"])
    self.assertIn("kubernetes", body)
    self.assertIn("gateway", body)
    service_ids = {s["id"] for s in body["services"]}
    self.assertLessEqual({"redis", "storage"}, service_ids)
    # Edges only exist where connectivity is configured, never invented.
    for edge in body["edges"]:
      self.assertEqual(edge["from"], "gateway")

  def test_runs_lists_filesystem_checkpoints(self) -> None:
    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      adapter_dir = os.path.join(tmp_dir, "peft", "model-abc-123")
      os.makedirs(adapter_dir)
      with open(os.path.join(adapter_dir, "metadata.json"), "w") as f:
        json.dump({"alias": "my-training-run", "base_model": "Qwen/Qwen3-8B"}, f)
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "model-def-456"))

      body = self.client.get("/api/v1/dashboard/runs").json()
      runs = {run["run_id"]: run for run in body["runs"]}
      self.assertIn("model-abc-123", runs)
      self.assertEqual(runs["model-abc-123"]["name"], "my-training-run")
      self.assertEqual(runs["model-abc-123"]["base_model"], "Qwen/Qwen3-8B")
      self.assertIn("model-def-456", runs)
      self.assertIn("checkpoint", runs["model-def-456"]["sources"])

  def test_dashboard_launch_validates_base_model(self) -> None:
    resp = self.client.post("/api/v1/dashboard/runs", json={})
    self.assertEqual(resp.status_code, 400)
    self.assertEqual(resp.json()["error"], "base_model is required")

  def test_dashboard_demo_launch_is_non_mutating(self) -> None:
    os.environ["OPEN_RL_DASHBOARD_DEMO"] = "1"
    resp = self.client.post("/api/v1/dashboard/runs", json={"base_model": "Qwen/Qwen3-8B"})
    self.assertEqual(resp.status_code, 200)
    self.assertTrue(resp.json()["demo"])
    self.assertFalse(resp.json()["launched"])

  def test_ops_make_friendly_positional_arguments(self) -> None:
    cases = [
      (["ops.py", "run", "run-a", "120"], ("GET", "/api/v1/dashboard/runs/run-a?logs=120"), None),
      (["ops.py", "logs", "pod-a", "42"], ("GET", "/api/v1/dashboard/pods/pod-a/logs?tail=42"), None),
      (["ops.py", "launch", "Qwen/Qwen3-0.6B"], ("POST", "/api/v1/dashboard/runs"), {"base_model": "Qwen/Qwen3-0.6B"}),
    ]
    for argv, call, payload in cases:
      with (
        self.subTest(argv=argv),
        mock.patch("sys.argv", argv),
        mock.patch.object(ops, "request", return_value={}) as request,
        mock.patch.object(ops, "emit"),
      ):
        ops.main()
        request.assert_called_once_with(*call, *(() if payload is None else (payload,)))

  def test_run_detail_bundles_run_state(self) -> None:
    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "model-xyz-789"))

      resp = self.client.get("/api/v1/dashboard/runs/model-xyz-789")
      self.assertEqual(resp.status_code, 200)
      detail = resp.json()
      self.assertEqual(detail["run_id"], "model-xyz-789")
      self.assertEqual(detail["queue_depth"], 0)
      self.assertEqual(detail["pods"], [])
      self.assertEqual(detail["gpu_claims"], {})
      self.assertEqual(detail["gpu_devices"], 0)
      self.assertEqual(detail["state"]["phase"], "saved")
      self.assertEqual(detail["diagnostics"], [])
      self.assertNotIn("logs", detail, "logs are only included when requested")

    self.assertEqual(self.client.get("/api/v1/dashboard/runs/no-such-run").status_code, 404)

  def test_run_detail_explains_failed_pods_and_current_gpu_claims(self) -> None:
    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "model-broken"))
      k8s = {
        "available": True,
        "namespace": "open-rl",
        "error": None,
        "nodes": [{"name": "gpu-node", "accelerator": "nvidia-l4", "gpu_capacity": 2}],
        "pods": [
          {
            "name": "trainer-model-broken",
            "labels": {"timeslice.io/job-id": "trainer-model-broken"},
            "phase": "Pending",
            "node": "gpu-node",
            "gpus": 1,
            "problem": "Unschedulable: waiting for a free GPU claim",
            "restarts": 0,
          },
          {
            "name": "sampler-model-broken",
            "labels": {"timeslice.io/job-id": "sampler-model-broken"},
            "phase": "Failed",
            "node": "gpu-node",
            "gpus": 1,
            "problem": "CrashLoopBackOff: CUDA out of memory",
            "restarts": 4,
          },
        ],
      }
      with mock.patch.object(data, "k8s_snapshot", return_value=k8s):
        detail = self.client.get("/api/v1/dashboard/runs/model-broken").json()

    self.assertEqual(detail["state"]["phase"], "failed")
    self.assertEqual(detail["state"]["status"], "error")
    self.assertEqual(detail["state"]["pod_phase_counts"], {"Pending": 1, "Failed": 1})
    self.assertEqual(detail["gpu_claims"], {"nvidia-l4": 1}, "terminal pods must not count as current claims")
    self.assertEqual(detail["gpu_devices"], 1)
    self.assertEqual({item["code"] for item in detail["diagnostics"]}, {"pod.unschedulable", "pod.failed"})
    self.assertTrue(all(item["actions"][0]["command"].startswith("make ops logs") for item in detail["diagnostics"]))

  def test_stop_unknown_run_conflicts(self) -> None:
    resp = self.client.post("/api/v1/dashboard/runs/does-not-exist/stop")
    self.assertEqual(resp.status_code, 409)

  def test_stop_marks_persistent_run_lifecycle(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    store = InMemoryStore()
    asyncio.run(store.set_model_metadata("run-stop", {"status": "ready", "created_at": 100.0}))
    asyncio.run(store.put_request({"model_id": "run-stop", "op": "optim_step"}))
    with mock.patch.object(data, "k8s_core_v1", return_value=(None, "off")):
      result = asyncio.run(data.stop_run(store, None, "run-stop"))

    self.assertTrue(result["stopped"])
    self.assertEqual(result["actions"], ["cleared in-memory queue"])
    self.assertEqual(result["errors"], [])
    metadata = asyncio.run(store.list_model_metadata())["run-stop"]
    self.assertEqual(metadata["status"], "stopped")
    self.assertIn("stopped_at", metadata)

  def test_demo_mode_flags_every_payload(self) -> None:
    os.environ["OPEN_RL_DASHBOARD_DEMO"] = "1"
    for path in ("snapshot", "cluster", "runs", "health", "problems", "pods/any-pod/logs", "runs/demo-run-1?logs=5"):
      body = self.client.get(f"/api/v1/dashboard/{path}").json()
      self.assertTrue(body["demo"], path)
      if path == "snapshot":
        self.assertEqual(body["schema_version"], 1)
        for stat in body["health"]["stats"]:
          self.assertIn("value_number", stat)
          self.assertIn("unit", stat)
          self.assertIn("context", stat)
          self.assertIn(stat["status"], {"ok", "warn"})
      self.assertIn("fictional", body["notice"], path)
    stop = self.client.post("/api/v1/dashboard/runs/demo-run-1/stop").json()
    self.assertTrue(stop["demo"])

  def test_duty_tracker_records_per_job_allocation(self) -> None:
    tracker = data.DutyTracker(max_samples=3, min_interval_s=5.0)
    pools = [{"id": "h100", "nodes": [{"name": "n1", "gpu_capacity": 8}]}]
    pods = [
      {"node": "n1", "phase": "Running", "gpus": 3, "labels": {"timeslice.io/job-id": "trainer-run-a"}},
      {"node": "n1", "phase": "Running", "gpus": 1, "labels": {"timeslice.io/job-id": "sampler-run-a"}},
      {"node": "n1", "phase": "Running", "gpus": 2, "labels": {}, "app": "dcgm-exporter"},
      {"node": "n1", "phase": "Succeeded", "gpus": 2, "labels": {"timeslice.io/job-id": "trainer-run-b"}},
      {"node": "other-node", "phase": "Running", "gpus": 8, "labels": {"timeslice.io/job-id": "trainer-run-c"}},
    ]
    tracker.record(pools, pods, now=100.0)
    tracker.record(pools, pods, now=102.0)

    duty = tracker.duty(pools[0])
    self.assertEqual(duty["capacity"], 8)
    self.assertEqual(duty["current"], 0.75)
    self.assertEqual(len(duty["series"]), 1, "second sample should be throttled")
    self.assertEqual(duty["series"][0][1], {"run-a": 4, "dcgm-exporter": 2}, "trainer+sampler merge per run; unlabeled pods use app")
    self.assertEqual(duty["jobs"], ["run-a", "dcgm-exporter"])

    for i in range(5):
      tracker.record(pools, pods, now=110.0 + i * 10)
    self.assertEqual(len(tracker.duty(pools[0])["series"]), 3, "ring buffer should cap history")

    cpu_pool = {"id": "cpu", "nodes": [{"name": "c1", "gpu_capacity": 0}]}
    self.assertIsNone(tracker.duty(cpu_pool), "pools without GPUs have no duty cycle")

  def test_operational_stats_count_in_memory_queues(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    store = InMemoryStore()
    empty_k8s = {"available": False, "namespace": "default", "error": "off", "pods": [], "nodes": []}
    asyncio.run(store.put_request({"model_id": "run-a", "op": "forward_backward"}))
    asyncio.run(store.put_request({"model_id": "run-a", "op": "optim_step"}))

    stats, queues = asyncio.run(data.operational_stats(store, empty_k8s, worker_manager=None))
    by_id = {stat["id"]: stat for stat in stats}
    self.assertEqual(by_id["runs.active"]["value"], "1")
    self.assertEqual(by_id["queue.requests"]["value"], "2")
    self.assertEqual(queues, [{"model_id": "run-a", "depth": 2}])

  def test_model_pods_match_timeslice_labels(self) -> None:
    pods = [
      {"name": "open-rl-trainer-x", "labels": {"timeslice.io/job-id": "trainer-abc-123"}},
      {"name": "open-rl-sampler-x", "labels": {"timeslice.io/job-id": "sampler-abc-123"}},
      {"name": "other", "labels": {"timeslice.io/job-id": "trainer-zzz"}},
      {"name": "unlabeled", "labels": {}},
    ]
    matched = {p["name"] for p in data.model_pods("abc_123", pods)}
    self.assertEqual(matched, {"open-rl-trainer-x", "open-rl-sampler-x"})
    self.assertEqual(data.model_pods("abc", pods), [], "a run must never match another run's prefix-sharing pods")

  def test_duty_reports_overcommit_honestly(self) -> None:
    tracker = data.DutyTracker()
    pools = [{"id": "shared", "nodes": [{"name": "n1", "gpu_capacity": 1}]}]
    pods = [
      {"node": "n1", "phase": "Running", "gpus": 1, "labels": {"timeslice.io/job-id": "trainer-run-a"}},
      {"node": "n1", "phase": "Running", "gpus": 1, "labels": {"timeslice.io/job-id": "sampler-run-b"}},
    ]
    tracker.record(pools, pods, now=100.0)
    duty = tracker.duty(pools[0])
    self.assertEqual(duty["current"], 2.0, "time-sliced overcommit must not be clamped to 100%")

  def test_operational_stats_report_gpu_overcommit_honestly(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    k8s = {
      "available": True,
      "namespace": "test",
      "error": None,
      "nodes": [{"gpu_capacity": 1}],
      "pods": [
        {"node": "n1", "phase": "Running", "gpus": 1},
        {"node": "n1", "phase": "Running", "gpus": 1},
      ],
    }
    stats, _ = asyncio.run(data.operational_stats(InMemoryStore(), k8s, worker_manager=None))
    gpu = next(stat for stat in stats if stat["id"] == "gpus.claimed")
    self.assertEqual(gpu["value"], "2/1")
    self.assertIn("2.0× allocation overcommit", gpu["detail"])
    self.assertEqual(gpu["value_number"], 2)
    self.assertEqual(gpu["unit"], "devices")
    self.assertEqual(gpu["context"], {"capacity_devices": 1, "allocation_ratio": 2.0, "overcommitted": True})
    self.assertEqual(gpu["status"], "warn")

  def test_problems_have_stable_codes_evidence_and_next_actions(self) -> None:
    checks = [{"id": "storage.shared", "group": "Storage", "label": "Shared filesystem", "status": "warn", "detail": "/tmp missing"}]
    k8s = {
      "namespace": "open-rl",
      "pods": [
        {
          "name": "broken-pod",
          "phase": "Failed",
          "node": "node-a",
          "problem": "CrashLoopBackOff: CUDA out of memory",
          "restarts": 4,
        }
      ],
      "nodes": [{"name": "node-a", "ready": False, "memory_pressure": True, "disk_pressure": False}],
    }
    problems = data.derive_problems(checks, k8s)

    self.assertEqual(problems[0]["severity"], "error", "errors sort before warnings")
    self.assertEqual(len({problem["id"] for problem in problems}), len(problems))
    for problem in problems:
      self.assertTrue(problem["code"])
      self.assertIn("kind", problem["resource"])
      self.assertIsInstance(problem["evidence"], dict)
      self.assertTrue(problem["actions"])
    pod_problem = next(problem for problem in problems if problem["code"] == "pod.failed")
    self.assertEqual(pod_problem["resource"], {"kind": "Pod", "name": "broken-pod", "namespace": "open-rl"})
    self.assertEqual(pod_problem["actions"][0]["command"], "make ops logs broken-pod")

  def test_runs_with_unknown_created_at_sort_last(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    old_tmp_dir = os.environ.get("OPEN_RL_TMP_DIR")
    with tempfile.TemporaryDirectory() as tmp_dir:
      os.environ["OPEN_RL_TMP_DIR"] = tmp_dir
      self.addCleanup(lambda: os.environ.update({"OPEN_RL_TMP_DIR": old_tmp_dir}) if old_tmp_dir else os.environ.pop("OPEN_RL_TMP_DIR", None))
      os.makedirs(os.path.join(tmp_dir, "checkpoints", "run-with-date"))

      store = InMemoryStore()
      asyncio.run(store.put_request({"model_id": "run-no-date", "op": "forward_backward"}))
      snapshot = asyncio.run(data.runs_snapshot(store, None, pods=[]))
      order = [run["run_id"] for run in snapshot["runs"]]
      self.assertEqual(order, ["run-with-date", "run-no-date"], "runs without created_at belong at the end")

  def test_registered_run_survives_queue_drain_and_tracks_readiness(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    store = InMemoryStore()
    asyncio.run(store.set_model_metadata("run-persistent", {"base_model": "Qwen/Qwen3-0.6B", "created_at": 100.0, "status": "queued"}))
    asyncio.run(store.put_request({"request_id": "run-persistent", "model_id": "run-persistent", "op": "create_model"}))
    asyncio.run(store.get_requests())

    starting = asyncio.run(data.runs_snapshot(store, None, pods=[]))["runs"][0]
    self.assertEqual(starting["run_id"], "run-persistent")
    self.assertEqual(starting["queue_depth"], 0)
    self.assertEqual(starting["state"]["phase"], "starting")

    asyncio.run(store.set_future("run-persistent", {"type": "model_created", "model_id": "run-persistent"}))
    ready = asyncio.run(data.runs_snapshot(store, None, pods=[]))["runs"][0]
    self.assertEqual(ready["state"]["phase"], "ready")
    self.assertEqual(ready["lifecycle"]["status"], "ready")
    self.assertIsNotNone(ready["lifecycle"]["ready_at"])


if __name__ == "__main__":
  unittest.main()
