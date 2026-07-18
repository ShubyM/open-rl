import json
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from server import control_plane, gateway
from server.store import InMemoryStore, RedisStore, report_control_event


class ControlPlaneTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.store = InMemoryStore()
    await self.store.set_value(
      "open_rl:model_meta:run-a",
      json.dumps({"base_model": "Qwen/Qwen2.5-0.5B", "created_at": time.time() - 5, "training_kind": "full"}),
    )
    await report_control_event(
      self.store,
      "run-a",
      component="gateway",
      phase="submitted",
      status="queued",
      message="Run submitted",
    )
    await report_control_event(
      self.store,
      "run-a",
      component="sampler",
      phase="initializing_engine",
      status="starting",
      message="Initializing vLLM",
    )
    self.get_store = patch.object(control_plane, "get_store", return_value=self.store)
    self.manager = patch.object(control_plane, "worker_manager", return_value=None)
    self.get_store.start()
    self.manager.start()
    self.addCleanup(self.get_store.stop)
    self.addCleanup(self.manager.stop)

  async def test_runs_expose_live_phase_components_and_queues(self) -> None:
    response = await control_plane.list_runs()

    self.assertEqual(len(response["runs"]), 1)
    run = response["runs"][0]
    self.assertEqual(run["id"], "run-a")
    self.assertEqual(run["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(run["phase"], "initializing_engine")
    self.assertEqual(run["status"], "starting")
    self.assertEqual(run["queue"], {"training": 0, "sampling": 0})
    self.assertEqual({component["id"] for component in run["components"]}, {"gateway", "sampler"})
    self.assertTrue(run["can_stop"])
    self.assertIsNone(run["tracker_url"])
    self.assertNotIn("metrics", run)
    self.assertIn("generated_at", response)

  async def test_run_exposes_only_safe_tracker_url(self) -> None:
    metadata = json.loads(await self.store.get_value("open_rl:model_meta:run-a"))
    metadata["tracker_url"] = "https://wandb.ai/acme/open-rl/runs/abc123"
    await self.store.set_value("open_rl:model_meta:run-a", json.dumps(metadata))

    run = await control_plane.get_run("run-a")

    self.assertEqual(run["tracker_url"], "https://wandb.ai/acme/open-rl/runs/abc123")
    metadata["tracker_url"] = "javascript:alert(1)"
    await self.store.set_value("open_rl:model_meta:run-a", json.dumps(metadata))
    self.assertIsNone((await control_plane.get_run("run-a"))["tracker_url"])

  async def test_history_without_model_metadata_cannot_be_stopped(self) -> None:
    await self.store.delete_values("open_rl:model_meta:run-a")

    run = await control_plane.get_run("run-a")

    self.assertFalse(run["can_stop"])

  async def test_durable_identity_survives_active_model_cleanup(self) -> None:
    await self.store.set_value(
      "open_rl:run_meta:run-a",
      json.dumps(
        {
          "base_model": "Qwen/Qwen2.5-0.5B",
          "created_at": time.time() - 5,
          "name": "durable run",
          "training_kind": "full",
          "tracker_url": "https://wandb.ai/acme/project/runs/run-a",
          "stopped_at": time.time(),
        }
      ),
    )
    await self.store.delete_values("open_rl:model_meta:run-a")

    run = await control_plane.get_run("run-a")

    self.assertEqual(run["name"], "durable run")
    self.assertEqual(run["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(run["training_kind"], "full")
    self.assertEqual(run["status"], "stopped")
    self.assertFalse(run["can_stop"])

  async def test_late_ready_event_cannot_resurrect_explicitly_stopped_history(self) -> None:
    await report_control_event(
      self.store,
      "legacy-run",
      component="gateway",
      phase="submitted",
      status="queued",
      message="submitted",
      details={"base_model": "Qwen/Qwen2.5-0.5B", "training_kind": "full"},
    )
    await report_control_event(
      self.store,
      "legacy-run",
      component="gateway",
      phase="stopped",
      status="stopped",
      message="workers stopped",
    )
    await report_control_event(
      self.store,
      "legacy-run",
      component="sampler",
      phase="ready",
      status="ready",
      message="late readiness",
    )

    run = await control_plane.get_run("legacy-run")

    self.assertEqual(run["status"], "stopped")
    self.assertEqual(run["message"], "workers stopped")
    self.assertEqual(run["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(run["training_kind"], "full")
    self.assertFalse(run["can_stop"])

  async def test_stop_run_preserves_history_and_is_idempotent(self) -> None:
    with patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "false"}):
      first = await control_plane.stop_run("run-a")
      event_count = len(await self.store.get_control_events("run-a", limit=1000))
      second = await control_plane.stop_run("run-a")

    self.assertEqual(first["status"], "accepted")
    self.assertEqual(first["run"]["status"], "stopped")
    self.assertFalse(first["run"]["can_stop"])
    self.assertEqual(first["run"]["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(second["status"], "noop")
    self.assertEqual(len(await self.store.get_control_events("run-a", limit=1000)), event_count)

  async def test_fft_stop_queues_both_worker_sentinels(self) -> None:
    sampling_request = AsyncMock()
    with (
      patch.object(self.store, "put_sampling_request", sampling_request),
      patch.dict(os.environ, {"OPEN_RL_ENABLE_FFT": "true"}),
    ):
      await gateway.request_model_stop("run-a", request_store=self.store, preserve_metadata=True)

    queued = await self.store.get_requests()
    self.assertEqual(queued, [{"request_id": "SHUTDOWN_SENTINEL", "model_id": "run-a", "op": "shutdown_workers"}])
    sampling_request.assert_awaited_once_with({"request_id": "SHUTDOWN_SENTINEL", "model_id": "run-a"})
    metadata = json.loads(await self.store.get_value("open_rl:model_meta:run-a"))
    self.assertIn("stopped_at", metadata)

  async def test_stop_unknown_run_is_404(self) -> None:
    with self.assertRaises(HTTPException) as raised:
      await control_plane.stop_run("missing")
    self.assertEqual(raised.exception.status_code, 404)

  async def test_stop_event_only_live_history_is_rejected(self) -> None:
    await self.store.delete_values("open_rl:model_meta:run-a")

    with self.assertRaises(HTTPException) as raised:
      await control_plane.stop_run("run-a")
    self.assertEqual(raised.exception.status_code, 409)

  async def test_failed_component_controls_run_phase_and_message(self) -> None:
    await report_control_event(
      self.store,
      "run-a",
      component="sampler",
      phase="sample_failed",
      status="failed",
      level="error",
      message="Sampler process crashed",
    )
    await report_control_event(
      self.store,
      "run-a",
      component="gateway",
      phase="heartbeat",
      status="ready",
      message="Gateway is healthy",
    )

    run = await control_plane.get_run("run-a")

    self.assertEqual(run["status"], "failed")
    self.assertEqual(run["phase"], "sample_failed")
    self.assertEqual(run["message"], "Sampler process crashed")

  async def test_worker_metrics_remain_bounded_raw_event_details(self) -> None:
    await report_control_event(
      self.store,
      "run-a",
      component="trainer",
      phase="forward_backward_complete",
      status="ready",
      message="Training step finished",
      details={"metrics": {"loss": 1.25, "nan": float("nan"), "positive_inf": float("inf")}},
    )

    run = await control_plane.get_run("run-a")
    self.assertNotIn("metrics", run)
    json.dumps(run, allow_nan=False)
    events = await control_plane.get_run_events("run-a", limit=200)
    event_metrics = events["events"][-1]["details"]["metrics"]
    self.assertEqual(event_metrics, {"loss": 1.25, "nan": None, "positive_inf": None})
    json.dumps(events, allow_nan=False)

  async def test_events_support_agent_cursor_polling(self) -> None:
    first = await control_plane.get_run_events("run-a", limit=1)
    second = await control_plane.get_run_events("run-a", after=first["next_cursor"], limit=200)

    self.assertEqual(len(first["events"]), 1)
    self.assertEqual(first["events"][0]["phase"], "submitted")
    self.assertEqual(first["events"][0]["run_id"], "run-a")
    self.assertEqual([event["phase"] for event in second["events"]], ["initializing_engine"])
    self.assertTrue(first["events"][0]["timestamp"].endswith("Z"))

  async def test_event_details_and_metric_names_remain_snake_case(self) -> None:
    await report_control_event(
      self.store,
      "run-a",
      component="trainer",
      phase="step_complete",
      status="ready",
      message="Step complete",
      duration_seconds=1.25,
      details={"base_model": "Qwen/Qwen2.5-0.5B", "metrics": {"tokens_per_second": 42.0}},
    )

    response = await control_plane.get_run_events("run-a", limit=200)
    event = response["events"][-1]

    self.assertEqual(event["duration_seconds"], 1.25)
    self.assertEqual(event["details"]["base_model"], "Qwen/Qwen2.5-0.5B")
    self.assertEqual(event["details"]["metrics"], {"tokens_per_second": 42.0})

  async def test_logs_fall_back_to_structured_events(self) -> None:
    response = await control_plane.get_run_logs("run-a", component="sampler", tail=200, previous=False)

    self.assertEqual(response["source"], "events")
    self.assertIn("Initializing vLLM", response["logs"])
    self.assertEqual(response["lines"][0]["stream"], "stdout")

  async def test_kubernetes_byte_logs_are_decoded_into_native_lines(self) -> None:
    class WorkerManager:
      def read_logs(self, *args, **kwargs):
        return {
          "source": "kubernetes",
          "pod_name": "trainer-a",
          "logs": repr(b"2026-07-11T07:20:20.337910369Z first line\n2026-07-11T07:20:21.000000000Z second line\n"),
          "error": None,
        }

    with patch.object(control_plane, "worker_manager", return_value=WorkerManager()):
      response = await control_plane.get_run_logs("run-a", component="trainer", tail=200, previous=False)

    self.assertNotIn("b'", response["logs"])
    self.assertEqual(response["run_id"], "run-a")
    self.assertEqual(response["pod_name"], "trainer-a")
    self.assertEqual(len(response["lines"]), 2)
    self.assertEqual(response["lines"][0]["timestamp"], "2026-07-11T07:20:20.337910369Z")
    self.assertEqual(response["lines"][0]["message"], "first line")

  async def test_event_only_logs_do_not_call_worker_manager(self) -> None:
    class WorkerManager:
      def read_logs(self, *args, **kwargs):
        raise AssertionError("non-worker logs must use control events")

    with patch.object(control_plane, "worker_manager", return_value=WorkerManager()):
      response = await control_plane.get_run_logs("run-a", component="client", tail=200, previous=False)

    self.assertEqual(response["source"], "events")
    self.assertEqual(response["logs"], "")
    self.assertIsNone(response["error"])

  async def test_logs_for_missing_run_are_404(self) -> None:
    with self.assertRaises(HTTPException) as raised:
      await control_plane.get_run_logs("missing", component="trainer", tail=200, previous=False)
    self.assertEqual(raised.exception.status_code, 404)

  async def test_logs_reject_unknown_components(self) -> None:
    with self.assertRaises(HTTPException) as raised:
      await control_plane.get_run_logs("run-a", component="secrets", tail=200, previous=False)
    self.assertEqual(raised.exception.status_code, 400)

  async def test_pod_logs_use_exact_manager_pod_and_native_lines(self) -> None:
    calls = []

    class WorkerManager:
      def read_pod_logs(self, pod_name, tail_lines, previous):
        calls.append((pod_name, tail_lines, previous))
        return {
          "source": "kubernetes",
          "pod_name": pod_name,
          "logs": b"2026-07-11T07:20:20.337910369Z client started\n",
          "error": None,
        }

    with patch.object(control_plane, "worker_manager", return_value=WorkerManager()):
      response = await control_plane.get_pod_logs("open-rl-client-job-abc", tail=500, previous=True)

    self.assertEqual(calls, [("open-rl-client-job-abc", 500, True)])
    self.assertEqual(response["pod_name"], "open-rl-client-job-abc")
    self.assertEqual(response["lines"][0]["timestamp"], "2026-07-11T07:20:20.337910369Z")
    self.assertEqual(response["lines"][0]["message"], "client started")

  async def test_pod_logs_reject_unsafe_names(self) -> None:
    for pod_name in ("../secret", "UPPERCASE", "bad..name", "-bad", "a" * 254):
      with self.subTest(pod_name=pod_name), self.assertRaises(HTTPException) as raised:
        await control_plane.get_pod_logs(pod_name, tail=200, previous=False)
      self.assertEqual(raised.exception.status_code, 400)

  async def test_pod_logs_require_kubernetes_manager(self) -> None:
    with self.assertRaises(HTTPException) as raised:
      await control_plane.get_pod_logs("client-a", tail=200, previous=False)
    self.assertEqual(raised.exception.status_code, 503)

  async def test_missing_run_is_404(self) -> None:
    with self.assertRaises(HTTPException) as raised:
      await control_plane.get_run("missing")
    self.assertEqual(raised.exception.status_code, 404)

  async def test_doctor_is_explicit_about_in_memory_mode(self) -> None:
    response = await control_plane.doctor()

    self.assertEqual(response["status"], "degraded")
    self.assertIn("generated_at", response)
    store_check = next(check for check in response["checks"] if check["name"] == "store")
    self.assertEqual(store_check["status"], "warn")
    cluster_check = next(check for check in response["checks"] if check["name"] == "cluster")
    self.assertEqual(cluster_check["details"]["summary"]["ready_nodes"], 1)

  async def test_cluster_endpoint_preserves_snake_case_and_kubernetes_maps(self) -> None:
    snapshot = {
      "mode": "kubernetes",
      "status": "healthy",
      "summary": {"ready_nodes": 1, "actionable_pending_pods": 0},
      "nodes": [
        {
          "name": "gpu-a",
          "pod_count": 2,
          "labels": {"node_role_name": "gpu"},
          "capacity": {"nvidia.com/gpu": "1", "huge_pages_2Mi": "4Gi"},
          "allocatable": {"vendor_resource_name": "1"},
        }
      ],
      "pods": [
        {
          "pod_name": "trainer-a",
          "model_id": "run-a",
          "annotations": {"open_rl_worker_revision": "abc"},
        }
      ],
      "generated_at": "2026-07-11T00:00:00Z",
    }
    with patch.object(control_plane, "cluster_snapshot", AsyncMock(return_value=snapshot)):
      response = await control_plane.get_cluster()

    self.assertEqual(response["summary"], {"ready_nodes": 1, "actionable_pending_pods": 0})
    self.assertEqual(response["nodes"][0]["pod_count"], 2)
    self.assertEqual(response["nodes"][0]["labels"], {"node_role_name": "gpu"})
    self.assertEqual(response["nodes"][0]["capacity"], {"nvidia.com/gpu": "1", "huge_pages_2Mi": "4Gi"})
    self.assertEqual(response["nodes"][0]["allocatable"], {"vendor_resource_name": "1"})
    self.assertEqual(response["pods"][0]["pod_name"], "trainer-a")
    self.assertEqual(response["pods"][0]["model_id"], "run-a")
    self.assertEqual(response["pods"][0]["annotations"], {"open_rl_worker_revision": "abc"})
    self.assertEqual(response["generated_at"], "2026-07-11T00:00:00Z")

  async def test_problems_have_stable_bounded_agent_contract_and_redact_evidence(self) -> None:
    now = time.time()
    snapshot = {
      "status": "degraded",
      "errors": [],
      "nodes": [
        {
          "name": "gpu-node-a",
          "status": "not_ready",
          "ready": False,
          "conditions": [
            {
              "type": "Ready",
              "status": "False",
              "reason": "KubeletStopped",
              "message": "token=do-not-expose https://user:pass@example.test/debug?key=secret",
            },
            {"type": "DiskPressure", "status": "True", "reason": "DiskFull", "message": "password: do-not-expose"},
          ],
        }
      ],
      "pods": [
        {
          "name": "open-rl-trainer-run-a",
          "pod_name": "open-rl-trainer-run-a",
          "model_id": "run-a",
          "role": "trainer",
          "node": None,
          "status": "pending",
          "reason": "Unschedulable",
          "message": "0/1 nodes have insufficient GPU; api_key=do-not-expose",
          "restarts": 2,
        }
      ],
    }
    runs = [
      {
        "id": "run-a",
        "status": "failed",
        "phase": "launch_failed",
        "message": "Bearer do-not-expose",
        "can_stop": False,
        "updated_at": control_plane.iso_timestamp(now - 600),
        "queue": {"training": 2, "sampling": 0},
        "components": [
          {
            "id": "trainer",
            "role": "trainer",
            "status": "failed",
            "phase": "launch_failed",
            "message": "secret=hunter2",
            "updated_at": control_plane.iso_timestamp(now - 600),
          }
        ],
      }
    ]
    with (
      patch.object(control_plane, "cluster_snapshot", AsyncMock(return_value=snapshot)),
      patch.object(control_plane, "all_runs", AsyncMock(return_value=runs)),
      patch("server.control_plane.time.time", return_value=now),
    ):
      first = await control_plane.get_problems()
      second = await control_plane.get_problems()

    internal_ids = [problem["id"] for problem in control_plane.derive_problems(snapshot, runs, now)]
    self.assertTrue(first["generated_at"].endswith("Z"))
    codes = {problem["code"] for problem in first["problems"]}
    self.assertTrue(
      {
        "node_not_ready",
        "node_disk_pressure",
        "pod_unschedulable",
        "pod_restarting",
        "run_failed",
        "component_failed",
        "training_queue_blocked",
      }.issubset(codes)
    )
    self.assertEqual([problem["id"] for problem in first["problems"]], internal_ids)
    self.assertEqual(
      [problem["id"] for problem in first["problems"]],
      [problem["id"] for problem in second["problems"]],
    )
    for problem in first["problems"]:
      self.assertEqual(
        set(problem),
        {"id", "severity", "code", "summary", "evidence", "resources", "remediation", "actions"},
      )
      self.assertIn(problem["severity"], {"warning", "error"})
      self.assertIsInstance(problem["resources"], dict)
      self.assertEqual([action["name"] for action in problem["actions"]], ["inspect", "logs", "stop"])
      self.assertTrue(all(isinstance(action["allowed"], bool) for action in problem["actions"]))
    serialized = json.dumps(first)
    self.assertNotIn("do-not-expose", serialized)
    self.assertNotIn("hunter2", serialized)

  async def test_problems_ignore_terminal_history_and_surface_stale_run_truth(self) -> None:
    now = time.time()
    snapshot = {
      "status": "healthy",
      "errors": [],
      "nodes": [{"name": "node-a", "status": "ready", "ready": True, "conditions": []}],
      "pods": [
        {"name": "old-failure", "model_id": "done", "role": "trainer", "status": "failed", "restarts": 4},
        {"name": "legacy-trainer", "model_id": "legacy", "role": "trainer", "status": "completed", "restarts": 0},
        {"name": "legacy-sampler", "model_id": "legacy", "role": "sampler", "status": "completed", "restarts": 0},
      ],
    }
    runs = [
      {"id": "done", "status": "completed", "can_stop": False, "components": [], "queue": {}},
      {"id": "stopped", "status": "stopped", "can_stop": False, "components": [], "queue": {}},
      {
        "id": "legacy",
        "status": "ready",
        "phase": "ready",
        "can_stop": False,
        "updated_at": control_plane.iso_timestamp(now - 900),
        "components": [
          {"id": "trainer", "role": "trainer", "status": "completed"},
          {"id": "sampler", "role": "sampler", "status": "completed"},
        ],
        "queue": {},
      },
    ]
    with (
      patch.object(control_plane, "cluster_snapshot", AsyncMock(return_value=snapshot)),
      patch.object(control_plane, "all_runs", AsyncMock(return_value=runs)),
      patch("server.control_plane.time.time", return_value=now),
    ):
      response = await control_plane.get_problems()

    self.assertEqual([problem["code"] for problem in response["problems"]], ["stale_run_state"])
    self.assertEqual(response["problems"][0]["resources"], {"run_id": "legacy"})

  async def test_problems_detect_waiting_stuck_and_blocked_queue_state(self) -> None:
    now = time.time()
    old = control_plane.iso_timestamp(now - 600)
    snapshot = {
      "status": "healthy",
      "errors": [],
      "nodes": [{"name": "node-a", "status": "ready", "ready": True, "conditions": []}],
      "pods": [],
    }
    runs = [
      {
        "id": "waiting",
        "status": "starting",
        "phase": "waiting_for_sampler",
        "can_stop": True,
        "updated_at": old,
        "components": [
          {
            "id": "sampler",
            "role": "sampler",
            "status": "waiting",
            "phase": "waiting_for_gpu",
            "updated_at": old,
          }
        ],
        "queue": {"sampling": 3},
      },
      {
        "id": "stuck",
        "status": "starting",
        "phase": "initializing_engine",
        "can_stop": True,
        "updated_at": old,
        "components": [
          {
            "id": "trainer",
            "role": "trainer",
            "status": "starting",
            "phase": "loading_model",
            "updated_at": old,
          }
        ],
        "queue": {},
      },
    ]
    with (
      patch.object(control_plane, "cluster_snapshot", AsyncMock(return_value=snapshot)),
      patch.object(control_plane, "all_runs", AsyncMock(return_value=runs)),
      patch("server.control_plane.time.time", return_value=now),
    ):
      response = await control_plane.get_problems()

    codes = {problem["code"] for problem in response["problems"]}
    self.assertTrue({"run_waiting", "component_waiting", "sampling_queue_blocked", "run_stuck", "component_stuck"}.issubset(codes))
    waiting_problem = next(problem for problem in response["problems"] if problem["code"] == "run_waiting")
    self.assertTrue(next(action for action in waiting_problem["actions"] if action["name"] == "stop")["allowed"])

  async def test_problems_are_deterministically_limited(self) -> None:
    snapshot = {
      "status": "degraded",
      "errors": [],
      "nodes": [{"name": "node-a", "status": "ready", "ready": True, "conditions": []}],
      "pods": [
        {"name": f"failed-{index:03d}", "status": "failed", "reason": "Error", "restarts": 0}
        for index in reversed(range(control_plane.MAX_PROBLEMS + 25))
      ],
    }
    with (
      patch.object(control_plane, "cluster_snapshot", AsyncMock(return_value=snapshot)),
      patch.object(control_plane, "all_runs", AsyncMock(return_value=[])),
    ):
      response = await control_plane.get_problems()

    self.assertEqual(len(response["problems"]), control_plane.MAX_PROBLEMS)
    self.assertEqual(
      [problem["resources"]["pod_name"] for problem in response["problems"]],
      [f"failed-{index:03d}" for index in range(control_plane.MAX_PROBLEMS)],
    )


class TrackerUrlTest(unittest.TestCase):
  def test_tracker_url_validation(self) -> None:
    self.assertEqual(control_plane.safe_tracker_url("http://wandb.internal:8080/run/1"), "http://wandb.internal:8080/run/1")
    for value in (
      "javascript:alert(1)",
      "data:text/html,hello",
      "/relative/run",
      "https://user:secret@wandb.ai/run",
      "https://wandb.ai:bad/run",
      "https://wandb.ai/run with space",
      "https://wandb.ai\\@evil.example/run",
      "https://wandb.ai/" + "x" * 2048,
    ):
      self.assertIsNone(control_plane.safe_tracker_url(value), value)


class RedisControlEventOrderingTest(unittest.IsolatedAsyncioTestCase):
  async def test_events_are_returned_in_cursor_order_after_concurrent_appends(self) -> None:
    class RedisStub:
      async def lrange(self, key: str, start: int, end: int):
        del key, start, end
        return [
          json.dumps({"cursor": "2", "phase": "second"}),
          json.dumps({"cursor": "1", "phase": "first"}),
          json.dumps({"cursor": "3", "phase": "third"}),
        ]

    store = object.__new__(RedisStore)
    store.redis = RedisStub()

    events = await store.get_control_events("run-a", after="1", limit=10)

    self.assertEqual([event["phase"] for event in events], ["second", "third"])


class ControlRouteTest(unittest.TestCase):
  def test_control_routes_and_static_mount_do_not_replace_tinker_api(self) -> None:
    paths = {route.path for route in gateway.app.routes}
    self.assertIn("/api/v1/create_model", paths)
    self.assertIn("/api/v1/control/runs", paths)
    self.assertIn("/api/v1/control/runs/{run_id}/stop", paths)
    self.assertIn("/api/v1/control/cluster", paths)
    self.assertIn("/api/v1/control/problems", paths)
    self.assertNotIn("/api/v1/control/metrics", paths)
    self.assertIn("/control", paths)

  def test_control_openapi_parameters_are_snake_case(self) -> None:
    schema = gateway.app.openapi()
    parameters = [
      parameter["name"]
      for path, operations in schema["paths"].items()
      if path.startswith("/api/v1/control")
      for operation in operations.values()
      for parameter in operation.get("parameters", [])
    ]

    self.assertIn("run_id", parameters)
    self.assertNotIn("runId", parameters)


class TelemetryTest(unittest.IsolatedAsyncioTestCase):
  async def test_telemetry_is_bounded_and_cannot_impersonate_workers(self) -> None:
    telemetry_store = InMemoryStore()
    await telemetry_store.set_value("open_rl:model_meta:" + "r" * 256, json.dumps({"base_model": "Qwen/Qwen2.5-0.5B"}))
    metrics = {f"metric-{index}": index for index in range(140)}
    metrics.update({"nested": {"value": 1}, "nan": float("nan"), "boolean": True})

    with patch.object(gateway, "store", telemetry_store):
      await gateway.telemetry(
        {
          "run_id": "r" * 300,
          "component": "sampler",
          "phase": "p" * 300,
          "status": "invented",
          "message": "m" * 3000,
          "metrics": metrics,
        }
      )

    events = await telemetry_store.get_control_events("r" * 256)
    self.assertEqual(len(events), 1)
    event = events[0]
    self.assertEqual(event["component"], "client")
    self.assertEqual(event["status"], "running")
    self.assertEqual(len(event["phase"]), 128)
    self.assertEqual(len(event["message"]), 2048)
    self.assertEqual(len(event["details"]["metrics"]), 128)
    self.assertNotIn("nested", event["details"]["metrics"])
    self.assertNotIn("nan", event["details"]["metrics"])
    self.assertNotIn("boolean", event["details"]["metrics"])

  async def test_telemetry_for_unknown_run_is_accepted_but_not_stored(self) -> None:
    telemetry_store = InMemoryStore()

    with patch.object(gateway, "store", telemetry_store):
      response = await gateway.telemetry({"run_id": "phantom", "metrics": {"loss": 1.0}})

    self.assertEqual(response, {"status": "accepted"})
    self.assertEqual(await telemetry_store.list_control_run_ids(), [])
    self.assertEqual(await telemetry_store.get_control_events("phantom"), [])


if __name__ == "__main__":
  unittest.main()
