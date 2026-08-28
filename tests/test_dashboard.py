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

  def test_pod_logs_can_read_previous_container_instance(self) -> None:
    with mock.patch.object(data, "k8s_pod_logs", return_value={"demo": False, "pod": "pod-a", "previous": True, "text": "prior"}) as logs:
      resp = self.client.get("/api/v1/dashboard/pods/pod-a/logs?tail=40&previous=true")

    self.assertEqual(resp.status_code, 200)
    self.assertEqual(resp.json()["text"], "prior")
    logs.assert_called_once_with("pod-a", None, 40, True)

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
    k8s = {
      "available": True,
      "namespace": "test",
      "error": None,
      "pods": [],
      "nodes": [],
      "metrics": {
        "installed": True,
        "available": True,
        "error": None,
        "pods_available": True,
        "nodes_available": True,
        "pods": {},
        "nodes": {},
      },
    }
    with mock.patch.object(data, "k8s_snapshot", return_value=k8s) as snapshot:
      resp = self.client.get("/api/v1/dashboard/snapshot")

    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertFalse(body["demo"])
    self.assertEqual(body["schema_version"], 1)
    self.assertIsNotNone(body["generated_at"])
    self.assertEqual({"cluster", "runs", "health", "problems"}, set(body) & {"cluster", "runs", "health", "problems"})
    self.assertEqual(body["cluster"]["kubernetes"]["namespace"], "test")
    self.assertEqual(
      body["cluster"]["kubernetes"]["metrics"],
      {
        "installed": True,
        "available": True,
        "error": None,
        "pods_available": True,
        "nodes_available": True,
        "pods_observed": 0,
        "nodes_observed": 0,
      },
    )
    self.assertIn("checks", body["health"])
    self.assertIn("problems", body["problems"])
    snapshot.assert_called_once_with()

  def test_snapshot_reads_each_queue_observation_once(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    class CountingStore(InMemoryStore):
      def __init__(self) -> None:
        super().__init__()
        self.queue_reads = 0
        self.launch_reads = 0

      async def queue_stats(self) -> list[dict]:
        self.queue_reads += 1
        return await super().queue_stats()

      async def worker_launch_stats(self) -> dict:
        self.launch_reads += 1
        return await super().worker_launch_stats()

    store = CountingStore()
    asyncio.run(store.put_request({"model_id": "run-a", "op": "forward_backward"}))
    k8s = {"available": False, "namespace": "default", "error": "off", "pods": [], "nodes": []}

    snapshot = asyncio.run(data.diagnostic_snapshot(store, None, k8s))

    self.assertEqual(store.queue_reads, 1)
    self.assertEqual(store.launch_reads, 1)
    self.assertEqual(snapshot["runs"]["runs"][0]["queue_depth"], 1)
    self.assertEqual(snapshot["health"]["queues"][0]["depth"], 1)

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

  def test_diagnostics_expose_build_identity_without_endpoint_credentials(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    k8s = {"available": False, "namespace": "default", "error": "off", "pods": [], "nodes": []}
    with mock.patch.dict(
      os.environ,
      {
        "OPEN_RL_BUILD_VERSION": "0123456789abcdef",
        "REDIS_URL": "redis://agent:super-secret@redis.internal:6379/0?token=also-secret",
      },
    ):
      cluster = asyncio.run(data.cluster_snapshot(InMemoryStore(), k8s))

    self.assertEqual(cluster["gateway"]["build"]["revision"], "0123456789abcdef")
    self.assertEqual(cluster["services"][0]["detail"], "redis://redis.internal:6379")
    serialized = json.dumps(cluster)
    self.assertNotIn("super-secret", serialized)
    self.assertNotIn("also-secret", serialized)

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
      (
        ["ops.py", "logs", "pod-a", "42", "--container", "trainer", "--previous"],
        ("GET", "/api/v1/dashboard/pods/pod-a/logs?tail=42&container=trainer&previous=true"),
        None,
      ),
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
    self.assertEqual({item["code"] for item in detail["diagnostics"]}, {"pod.unschedulable", "pod.oom_killed"})
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
    detail = self.client.get("/api/v1/dashboard/runs/demo-run-1").json()
    self.assertEqual(detail["telemetry"]["requests_completed"], 42)
    self.assertEqual(detail["telemetry"]["latest_metrics"]["loss:mean"], 0.7981)
    problems = self.client.get("/api/v1/dashboard/problems").json()["problems"]
    self.assertIn("run.request_failed", {problem["code"] for problem in problems})

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
    self.assertEqual(queues[0]["model_id"], "run-a")
    self.assertEqual(queues[0]["depth"], 2)
    self.assertGreaterEqual(queues[0]["oldest_age_seconds"], 0)

  def test_old_queue_age_becomes_an_actionable_metric_and_problem(self) -> None:
    import asyncio
    import time

    from server.store import InMemoryStore

    store = InMemoryStore()
    asyncio.run(store.put_request({"model_id": "run-stalled", "op": "forward_backward", "enqueued_at": time.time() - 600}))
    k8s = {"available": False, "namespace": "default", "error": "off", "pods": [], "nodes": []}
    with mock.patch.dict("os.environ", {"OPEN_RL_QUEUE_WARN_SECONDS": "300"}):
      stats, _ = asyncio.run(data.operational_stats(store, k8s, worker_manager=None))
      run = asyncio.run(data.runs_snapshot(store, None, pods=[]))["runs"][0]
      detail = asyncio.run(data.run_detail(store, None, "run-stalled", k8s))

    by_id = {stat["id"]: stat for stat in stats}
    self.assertEqual(by_id["queue.request_age"]["status"], "warn")
    self.assertGreaterEqual(by_id["queue.request_age"]["value_number"], 599)
    self.assertEqual(by_id["queue.request_age"]["context"]["model_id"], "run-stalled")
    self.assertGreaterEqual(run["queue_oldest_seconds"], 599)
    stalled = next(item for item in detail["diagnostics"] if item["code"] == "run.queue_stalled")
    self.assertEqual(stalled["actions"][0]["command"], "make ops run run-stalled")
    problems = data.derive_problems([], k8s, stats)
    metric = next(problem for problem in problems if problem["code"] == "metric.queue_request_age")
    self.assertEqual(metric["evidence"]["model_id"], "run-stalled")
    self.assertEqual(metric["actions"][1]["command"], "make ops run run-stalled")

  def test_old_worker_launch_becomes_an_actionable_metric(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    launch = {"depth": 2, "oldest_enqueued_at": 1.0, "oldest_age_seconds": 120.0}
    k8s = {"available": False, "namespace": "default", "error": "off", "pods": [], "nodes": []}
    with mock.patch.dict("os.environ", {"OPEN_RL_LAUNCH_WARN_SECONDS": "60"}):
      stats, _ = asyncio.run(data.operational_stats(InMemoryStore(), k8s, worker_manager=None, queues=[], launch=launch))

    by_id = {stat["id"]: stat for stat in stats}
    self.assertEqual(by_id["queue.launch"]["status"], "warn")
    self.assertEqual(by_id["queue.launch_age"]["value_number"], 120.0)
    problems = data.derive_problems([], k8s, stats)
    metric = next(problem for problem in problems if problem["code"] == "metric.queue_launch_age")
    self.assertEqual(metric["evidence"]["warn_after_seconds"], 60.0)

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
    pod_problem = next(problem for problem in problems if problem["code"] == "pod.oom_killed")
    self.assertEqual(pod_problem["resource"], {"kind": "Pod", "name": "broken-pod", "namespace": "open-rl"})
    self.assertEqual(pod_problem["actions"][0]["command"], "make ops logs broken-pod")

  def test_pod_evidence_preserves_oom_termination_and_previous_logs_action(self) -> None:
    from types import SimpleNamespace as NS

    terminated = NS(
      reason="OOMKilled",
      message="memory cgroup out of memory",
      exit_code=137,
      signal=0,
      started_at=None,
      finished_at=None,
    )
    container_status = NS(
      name="trainer",
      image="open-rl:test",
      image_id="docker-pullable://open-rl@sha256:abc123",
      ready=False,
      restart_count=2,
      state=NS(running=None, waiting=NS(reason="CrashLoopBackOff", message="back-off restarting"), terminated=None),
      last_state=NS(running=None, waiting=None, terminated=terminated),
    )
    condition = NS(type="Ready", status="False", reason="ContainersNotReady", message="trainer is not ready", last_transition_time=None)
    container = NS(name="trainer", image="open-rl:test", resources=NS(requests={}, limits={}))
    pod = NS(
      metadata=NS(name="trainer-run-a", labels={"app": "trainer"}, creation_timestamp=None),
      spec=NS(node_name="gpu-node", containers=[container], resource_claims=[]),
      status=NS(
        phase="Running",
        reason=None,
        message=None,
        init_container_statuses=[],
        container_statuses=[container_status],
        conditions=[condition],
      ),
    )

    observed = data.pod_to_dict(pod)
    observed["events"] = [
      {"reason": "BackOff", "message": "Back-off restarting failed container", "type": "Warning", "count": 2, "last_seen_at": None}
    ]
    diagnostic = data.pod_diagnostic(observed, "open-rl")

    self.assertEqual(observed["problem"], "CrashLoopBackOff: back-off restarting")
    self.assertEqual(observed["containers"][0]["image_id"], "docker-pullable://open-rl@sha256:abc123")
    self.assertEqual(observed["containers"][0]["last_termination"]["exit_code"], 137)
    self.assertEqual(diagnostic["code"], "pod.oom_killed")
    self.assertEqual(diagnostic["evidence"]["containers"][0]["last_termination"]["reason"], "OOMKilled")
    self.assertIn("--previous", diagnostic["actions"][2]["command"])
    self.assertEqual(diagnostic["evidence"]["events"][0]["reason"], "BackOff")

  def test_pod_diagnostics_distinguish_image_pull_and_volume_mount(self) -> None:
    base = {"name": "pod-a", "phase": "Pending", "node": None, "restarts": 0, "conditions": []}
    image = {
      **base,
      "problem": "ImagePullBackOff: image not found",
      "containers": [{"name": "worker", "reason": "ImagePullBackOff", "last_termination": None}],
      "events": [],
    }
    volume = {
      **base,
      "problem": "Pending",
      "containers": [],
      "events": [{"reason": "FailedMount", "message": "persistentvolumeclaim missing", "type": "Warning", "count": 3}],
    }

    self.assertEqual(data.pod_diagnostic(image, "open-rl")["code"], "pod.image_pull")
    volume_diagnostic = data.pod_diagnostic(volume, "open-rl")
    self.assertEqual(volume_diagnostic["code"], "pod.volume_mount")
    self.assertEqual(volume_diagnostic["evidence"]["events"][0]["count"], 3)

  def test_metrics_server_quantities_become_numeric_usage(self) -> None:
    class MetricsApi:
      def list_namespaced_custom_object(self, *_args, **_kwargs):
        return {
          "items": [
            {
              "metadata": {"name": "trainer-a"},
              "timestamp": "2026-08-28T20:00:00Z",
              "window": "30s",
              "containers": [
                {"name": "trainer", "usage": {"cpu": "250m", "memory": "128Mi"}},
                {"name": "sidecar", "usage": {"cpu": "50000000n", "memory": "64Mi"}},
              ],
            }
          ]
        }

      def list_cluster_custom_object(self, *_args, **_kwargs):
        return {"items": [{"metadata": {"name": "node-a"}, "usage": {"cpu": "1500m", "memory": "2Gi"}, "timestamp": "now", "window": "30s"}]}

    with mock.patch.object(data, "k8s_custom_objects", return_value=(MetricsApi(), None)):
      metrics = data.resource_metrics_snapshot("open-rl")

    self.assertTrue(metrics["available"])
    self.assertAlmostEqual(metrics["pods"]["trainer-a"]["cpu_cores"], 0.3)
    self.assertEqual(metrics["pods"]["trainer-a"]["memory_bytes"], 192 * 2**20)
    self.assertEqual(metrics["nodes"]["node-a"]["cpu_cores"], 1.5)
    self.assertEqual(metrics["nodes"]["node-a"]["memory_bytes"], 2 * 2**30)

  def test_metrics_server_preserves_partial_usage_and_explains_missing_scope(self) -> None:
    class Forbidden(Exception):
      status = 403

    class MetricsApi:
      def list_namespaced_custom_object(self, *_args, **_kwargs):
        return {"items": [{"metadata": {"name": "trainer-a"}, "containers": [{"name": "trainer", "usage": {"cpu": "1", "memory": "1Gi"}}]}]}

      def list_cluster_custom_object(self, *_args, **_kwargs):
        raise Forbidden("nodes.metrics.k8s.io is forbidden")

    with mock.patch.object(data, "k8s_custom_objects", return_value=(MetricsApi(), None)):
      metrics = data.resource_metrics_snapshot("open-rl")

    self.assertTrue(metrics["installed"])
    self.assertTrue(metrics["available"])
    self.assertTrue(metrics["pods_available"])
    self.assertFalse(metrics["nodes_available"])
    self.assertEqual(metrics["pods"]["trainer-a"]["memory_bytes"], 2**30)
    self.assertIn("nodes.metrics.k8s.io is forbidden", metrics["error"])

  def test_operational_stats_warn_on_measured_resource_pressure(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    k8s = {
      "available": True,
      "namespace": "open-rl",
      "error": None,
      "pods": [],
      "nodes": [
        {
          "name": "node-a",
          "ready": True,
          "memory_pressure": False,
          "disk_pressure": False,
          "gpu_capacity": 0,
          "cpu_allocatable_cores": 8.0,
          "memory_allocatable_bytes": 16 * 2**30,
          "usage": {"cpu_cores": 7.6, "memory_bytes": 15 * 2**30},
        }
      ],
    }
    stats, _ = asyncio.run(data.operational_stats(InMemoryStore(), k8s, None))
    by_id = {stat["id"]: stat for stat in stats}

    self.assertEqual(by_id["cluster.cpu"]["status"], "warn")
    self.assertAlmostEqual(by_id["cluster.cpu"]["context"]["utilization"], 0.95)
    self.assertEqual(by_id["cluster.memory"]["status"], "warn")
    problems = data.derive_problems([], k8s, stats)
    self.assertLessEqual({"metric.cluster_cpu", "metric.cluster_memory"}, {problem["code"] for problem in problems})

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

  def test_run_telemetry_tracks_worker_metrics_latency_and_failures(self) -> None:
    import asyncio
    import time

    from server.store import InMemoryStore

    store = InMemoryStore()
    asyncio.run(store.set_model_metadata("run-metrics", {"base_model": "Qwen/Qwen3-0.6B", "status": "ready"}))
    asyncio.run(
      store.put_request(
        {
          "request_id": "request-ok",
          "model_id": "run-metrics",
          "op": "forward_backward",
          "enqueued_at": time.time() - 2,
        }
      )
    )
    asyncio.run(store.get_requests())
    asyncio.run(store.mark_request_started("request-ok", "run-metrics", "forward_backward"))
    active_runs = asyncio.run(data.runs_snapshot(store, None, pods=[]))["runs"]
    active = next(run for run in active_runs if run["run_id"] == "run-metrics")["telemetry"]["active_request"]
    self.assertEqual(active["operation"], "forward_backward")
    self.assertGreaterEqual(active["queue_wait_seconds"], 1.9)
    self.assertGreaterEqual(active["age_seconds"], 0)
    asyncio.run(store.set_future("request-ok", {"type": "forward_backward_completed", "metrics": {"loss:mean": 0.75, "ignored": float("nan")}}))
    asyncio.run(store.put_request({"request_id": "request-failed", "model_id": "run-metrics", "op": "optim_step", "enqueued_at": time.time() - 1}))
    asyncio.run(store.get_requests())
    asyncio.run(store.set_future("request-failed", {"type": "RequestFailedResponse", "error_message": "gradient overflow"}))

    telemetry = asyncio.run(store.list_model_metadata())["run-metrics"]["telemetry"]
    self.assertEqual(telemetry["requests_completed"], 2)
    self.assertEqual(telemetry["requests_failed"], 1)
    self.assertEqual(telemetry["failure_rate"], 0.5)
    self.assertEqual(telemetry["operation_counts"], {"forward_backward": 1, "optim_step": 1})
    self.assertEqual(telemetry["latest_metrics"], {"loss:mean": 0.75})
    self.assertEqual(len(telemetry["metric_series"]["loss:mean"]), 1)
    self.assertNotIn("active_request", telemetry)
    self.assertGreaterEqual(telemetry["mean_latency_seconds"], 1.0)
    self.assertEqual(telemetry["last_error"], "gradient overflow")

    k8s = {"available": True, "namespace": "default", "error": None, "pods": [], "nodes": []}
    detail = asyncio.run(data.run_detail(store, None, "run-metrics", k8s))
    diagnostic = next(item for item in detail["diagnostics"] if item["code"] == "run.request_failed")
    self.assertEqual(diagnostic["evidence"]["requests_failed"], 1)
    self.assertEqual(diagnostic["actions"][0]["command"], "make ops run run-metrics 100")
    problem = next(item for item in data.derive_problems([], k8s, runs=[detail]) if item["code"] == "run.request_failed")
    self.assertEqual(problem["resource"], {"kind": "Run", "name": "run-metrics"})

  def test_run_metric_series_is_bounded(self) -> None:
    from server.store import record_request_result

    metadata = {}
    for index in range(25):
      metadata = record_request_result(
        metadata,
        {"operation": "forward_backward", "enqueued_at": float(index)},
        {"type": "forward_backward_completed", "metrics": {"loss:mean": index / 10}},
        completed_at=float(index + 1),
      )

    series = metadata["telemetry"]["metric_series"]["loss:mean"]
    self.assertEqual(len(series), 20)
    self.assertEqual(series[0]["value"], 0.5)
    self.assertEqual(series[-1]["value"], 2.4)

  def test_stalled_active_request_is_a_global_problem(self) -> None:
    telemetry = {
      "active_request": {
        "request_id": "request-stalled",
        "operation": "forward_backward",
        "started_at": 1.0,
        "queue_wait_seconds": 4.0,
        "age_seconds": 700.0,
      }
    }
    k8s = {"available": True, "namespace": "default", "error": None, "pods": [], "nodes": []}
    with mock.patch.dict("os.environ", {"OPEN_RL_OPERATION_WARN_SECONDS": "600"}):
      problems = data.derive_problems([], k8s, runs=[{"run_id": "run-stalled", "telemetry": telemetry}])

    problem = next(item for item in problems if item["code"] == "run.request_stalled")
    self.assertEqual(problem["evidence"]["operation"], "forward_backward")
    self.assertEqual(problem["evidence"]["warn_after_seconds"], 600.0)
    self.assertEqual(problem["actions"][0]["command"], "make ops run run-stalled 100")

  def test_scheduler_snapshot_drives_run_state_and_consistency_diagnostics(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    class CustomObjects:
      def list_namespaced_custom_object(self, _group, _version, _namespace, plural, **_kwargs):
        if plural == "workloads":
          return {
            "items": [
              {
                "metadata": {"name": "trainer-run-scheduled", "uid": "uid-a", "creationTimestamp": "2020-01-01T00:00:00Z", "generation": 2},
                "spec": {
                  "role": "trainer",
                  "modelId": "run-scheduled",
                  "ownerId": "owner-a",
                  "trainingKind": "fft",
                  "accelerator": {"memory": "40Gi", "maxDeviceCount": 1},
                },
                "status": {
                  "phase": "Failed",
                  "reason": "Unsatisfiable: no tier can provide 40Gi",
                  "claimName": "claim-a",
                  "assignmentID": "assignment-a",
                  "podName": "trainer-run-scheduled",
                  "deviceCount": 1,
                  "memoryPerDevice": "40Gi",
                  "observedGeneration": 1,
                },
              }
            ]
          }
        return {
          "items": [
            {
              "metadata": {"name": "claim-a", "creationTimestamp": "2020-01-01T00:00:00Z"},
              "spec": {
                "claimName": "claim-a",
                "seats": [
                  {"workload": "trainer-run-scheduled", "workloadUID": "uid-a", "assignmentID": "assignment-a", "owner": "owner-a"},
                  {"workload": "deleted-workload", "workloadUID": "uid-old", "assignmentID": "assignment-old", "owner": "owner-old"},
                ],
              },
            }
          ]
        }

    with mock.patch.object(data, "k8s_custom_objects", return_value=(CustomObjects(), None)):
      scheduler = data.scheduler_snapshot("open-rl")

    self.assertTrue(scheduler["available"])
    self.assertEqual(scheduler["summary"]["phase_counts"], {"Failed": 1})
    self.assertEqual(scheduler["summary"]["seats"], 2)
    self.assertEqual(scheduler["workloads"][0]["requested_memory"], "40Gi")

    runs = asyncio.run(data.runs_snapshot(InMemoryStore(), None, pods=[], scheduler=scheduler))["runs"]
    self.assertEqual(runs[0]["run_id"], "run-scheduled")
    self.assertEqual(runs[0]["state"]["phase"], "failed")
    self.assertEqual(runs[0]["placement"], {"workloads": 1, "device_count": 1, "phase_counts": {"Failed": 1}})

    k8s = {"available": True, "namespace": "open-rl", "error": None, "pods": [], "nodes": [], "scheduler": scheduler}
    codes = {problem["code"] for problem in data.derive_problems([], k8s)}
    self.assertLessEqual({"scheduler.workload_failed", "scheduler.generation_stale", "scheduler.stale_seat"}, codes)
    self.assertNotIn("scheduler.assignment_mismatch", codes)

    detail = asyncio.run(data.run_detail(InMemoryStore(), None, "run-scheduled", k8s))
    self.assertEqual(detail["workloads"][0]["claim_name"], "claim-a")
    self.assertEqual(detail["claim_ledgers"][0]["seat_count"], 2)
    self.assertLessEqual(
      {"scheduler.workload_failed", "scheduler.generation_stale", "scheduler.stale_seat"}, {item["code"] for item in detail["diagnostics"]}
    )

    stats, _ = asyncio.run(data.operational_stats(InMemoryStore(), k8s, None))
    by_id = {stat["id"]: stat for stat in stats}
    self.assertEqual(by_id["scheduler.workloads"]["context"]["phase_counts"], {"Failed": 1})
    self.assertEqual(by_id["scheduler.seats"]["value_number"], 2)

  def test_missing_scheduler_crd_is_optional_not_an_error(self) -> None:
    import asyncio

    from server.store import InMemoryStore

    class MissingCustomObjects:
      def list_namespaced_custom_object(self, *_args, **_kwargs):
        error = RuntimeError("not found")
        error.status = 404
        raise error

    with mock.patch.object(data, "k8s_custom_objects", return_value=(MissingCustomObjects(), None)):
      scheduler = data.scheduler_snapshot("default")

    self.assertFalse(scheduler["installed"])
    self.assertFalse(scheduler["available"])
    self.assertIsNone(scheduler["error"])
    checks = asyncio.run(
      data.health_checks(
        InMemoryStore(),
        {"available": True, "namespace": "default", "error": None, "pods": [], "nodes": [], "scheduler": scheduler},
      )
    )
    check = next(check for check in checks if check["id"] == "scheduler")
    self.assertEqual(check["status"], "off")


if __name__ == "__main__":
  unittest.main()
