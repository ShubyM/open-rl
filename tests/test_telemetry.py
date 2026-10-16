import asyncio
import unittest
from datetime import UTC, datetime
from unittest import mock

import httpx
from kubernetes import client

from server.dashboard import metrics
from server.scheduler_worker_manager import SchedulerWorkerManager
from server.telemetry import kubernetes, prometheus
from server.telemetry.prometheus import PromQL

GPU = "GPU-375ff71c-7c41-a89e-3843-c62c7476bc2b"
WINDOW = {"since": "1970-01-01T00:00:00+00:00", "until": "1970-01-01T00:00:10+00:00"}


class KubernetesInventoryTest(unittest.TestCase):
  def test_inventory_and_manager_share_worker_state_including_release_and_failure(self):
    cases = [
      ({"phase": "Pending", "reason": "NoCapacity: needs 40Gi"}, {}, "waiting_for_placement"),
      ({"phase": "Running", "nodeName": "node-a"}, {}, "ready"),
      ({"phase": "Failed", "reason": "PodFailed", "nodeName": "node-a"}, {}, "failed"),
      ({"phase": "Running", "nodeName": "node-a"}, {"deletionTimestamp": "1970-01-01T00:00:09Z"}, "stopped"),
    ]
    for status, metadata, expected in cases:
      with self.subTest(state=expected):
        item = {
          "metadata": {"name": "worker", "uid": "worker-uid", "creationTimestamp": "1970-01-01T00:00:01Z", **metadata},
          "spec": {"role": "trainer", "modelID": "runtime", "ownerID": "owner"},
          "status": status,
        }
        custom = mock.Mock()
        custom.list_namespaced_custom_object.return_value = {"items": [item]}
        with mock.patch.dict("os.environ", {"REDIS_URL": "redis://test"}), mock.patch("time.time", return_value=10):
          manager = SchedulerWorkerManager(custom_api=custom)
          summary = kubernetes.workload_summary(item)
          self.assertEqual(summary["worker"], manager.load_workers()[0])
        self.assertEqual(summary["worker"]["state"], expected)
        self.assertEqual(summary["worker"]["age_seconds"], 9)
        self.assertEqual((summary["phase"], summary["reason"]), (status["phase"], status.get("reason")))

  def test_container_states_preserve_nullable_fields_and_previous_termination(self):
    at = datetime(2026, 1, 1, tzinfo=UTC)
    ended = client.V1ContainerStateTerminated(exit_code=137, reason="OOMKilled", started_at=at, finished_at=at)
    previous = client.V1ContainerState(terminated=ended)
    cases = [
      (None, "unknown", None, None, None, None),
      (client.V1ContainerState(running=client.V1ContainerStateRunning(started_at=at)), "running", None, None, at.isoformat(), None),
      (client.V1ContainerState(waiting=client.V1ContainerStateWaiting(reason="CrashLoopBackOff")), "waiting", "CrashLoopBackOff", None, None, None),
      (previous, "terminated", "OOMKilled", 137, at.isoformat(), at.isoformat()),
    ]
    for state, phase, reason, code, started, finished in cases:
      with self.subTest(phase=phase):
        row = kubernetes.container_summary("worker", "image:v1", state, 2, previous)
        self.assertEqual(
          (row["name"], row["state"], row["reason"], row["exit_code"], row["started_at"], row["finished_at"]),
          ("worker", phase, reason, code, started, finished),
        )
        self.assertEqual(row["restart_count"], 2)
        self.assertEqual(
          row["last_termination"], {"reason": "OOMKilled", "exit_code": 137, "started_at": at.isoformat(), "finished_at": at.isoformat()}
        )

  def test_unstarted_pod_has_the_same_container_shape(self):
    pod = client.V1Pod(
      metadata=client.V1ObjectMeta(name="p", uid="uid"),
      spec=client.V1PodSpec(containers=[client.V1Container(name="worker", image="trainer:v1")]),
      status=client.V1PodStatus(phase="Pending"),
    )
    row = kubernetes.pod_summary(pod)
    self.assertEqual(row["problem"], "Pending")
    self.assertEqual(
      row["containers"],
      [
        {
          "name": "worker",
          "image": "trainer:v1",
          "state": "unknown",
          "reason": None,
          "exit_code": None,
          "started_at": None,
          "finished_at": None,
          "restart_count": 0,
          "last_termination": None,
        }
      ],
    )

  def test_a_probe_traceback_event_keeps_only_its_first_and_last_lines(self):
    probe = 'Readiness probe failed: Traceback (most recent call last):\n  File "<string>", line 2\nURLError: [Errno 111]\n'
    self.assertEqual(kubernetes.event_message(probe), "Readiness probe failed: Traceback (most recent call last): … URLError: [Errno 111]")
    self.assertEqual(kubernetes.event_message("Back-off restarting failed container"), "Back-off restarting failed container")


class PromQLTest(unittest.TestCase):
  def test_one_query_per_metric_scoped_to_the_cluster_and_deduplicated_by_uuid(self) -> None:
    backend = PromQL("gke", "https://example", None, "open-rl-fft-test", "openrl-system")
    queries = []

    async def fake(query, start, end):
      queries.append(query)
      return {GPU: [[1.0, 7.0]]}

    with mock.patch.object(backend, "query_range", side_effect=fake):
      result = asyncio.run(backend.devices([GPU, "GPU-other"], 0, 60))
    self.assertEqual(len(queries), 2)
    self.assertTrue(all(q.startswith("max by (UUID) (DCGM_FI_DEV_") and 'cluster="open-rl-fft-test"' in q for q in queries))
    self.assertIn("GPU\\\\-375ff71c", queries[0])  # regex-escaped inside a PromQL string
    self.assertEqual(result[GPU]["utilization"], [[1.0, 7.0]])
    self.assertEqual(result["GPU-other"], {"utilization": [], "memory_mib": []})

  def test_sampler_series_use_vllm_metrics_summed_over_the_runs_replicas(self) -> None:
    backend = PromQL("gke", "https://example", None, "open-rl-fft-test", "openrl-system")
    queries = []

    async def fake(query, start, end):
      queries.append(query)
      return {"": [[1.0, 2.0]]}

    with mock.patch.object(backend, "query_range", side_effect=fake):
      result = asyncio.run(backend.samplers(["sampler-0", "sampler-1"], 0, 60))
    self.assertEqual(set(result), set(prometheus.SAMPLER_QUERIES))
    match = f'namespace="openrl-system",pod=~{prometheus.any_of(["sampler-0", "sampler-1"])},cluster="open-rl-fft-test"'
    self.assertTrue(all("vllm:" in q and match in q for q in queries), queries)
    self.assertEqual(result["requests_waiting"], [[1.0, 2.0]])

  def test_worker_series_use_gke_container_metrics_by_pod(self) -> None:
    backend = PromQL("gke", "https://example", None, None, "openrl-system")
    queries = []

    async def fake(query, start, end):
      queries.append(query)
      return {"orw-trainer": [[1.0, 3.0]]}

    with mock.patch.object(backend, "query_range", side_effect=fake):
      result = asyncio.run(backend.workers_series(["orw-trainer"], 0, 60))
    self.assertIn("kubernetes_io:container_cpu_core_usage_time", queries[0])
    self.assertIn('namespace_name="openrl-system"', queries[0])
    self.assertIn("kubernetes_io:container_memory_used_bytes", queries[1])
    self.assertEqual(result["orw-trainer"], {"cpu_cores": [[1.0, 3.0]], "memory_bytes": [[1.0, 3.0]]})

  def test_queries_made_at_different_moments_share_sample_times(self) -> None:
    # Each allocation is fetched separately and the window slides with every
    # refresh; unaligned starts gave every response its own timestamps, so the
    # per-sample average across GPUs came out empty until all requests landed.
    backend = PromQL("gke", "https://example", None, None, "openrl-system")
    sent = []

    async def get(self, url, params, headers):
      sent.append(params)
      return httpx.Response(200, json={"status": "success", "data": {"result": []}}, request=httpx.Request("GET", url))

    with mock.patch.object(httpx.AsyncClient, "get", get):
      for start in (1000.4, 1010.4):
        asyncio.run(backend.query_range("q", start, start + 1800))
    self.assertEqual([p["step"] for p in sent], [15, 15])
    self.assertEqual([p["start"] % 15 for p in sent], [0, 0])
    self.assertTrue(all(p["start"] <= s for p, s in zip(sent, (1000.4, 1010.4), strict=True)))


class HardwareSourceTest(unittest.TestCase):
  def test_explicit_prometheus_wins_then_gke_then_nothing(self) -> None:
    gke_on = {"project": "p", "cluster": "c", "location": "l", "mode": "auto", "enabled": True, "configured": True}
    gke_off = {**gke_on, "project": "", "configured": False, "cluster": ""}
    gmp = {"OPEN_RL_PROMETHEUS_URL": "http://gmp:9090/"}
    with mock.patch.dict("os.environ", gmp), mock.patch.object(metrics.gke, "configuration", return_value=gke_on):
      source, _ = metrics.hardware()
      self.assertEqual((source.name, source.url, source.cluster), ("prometheus", "http://gmp:9090", "c"))
    with mock.patch.dict("os.environ", {"OPEN_RL_PROMETHEUS_URL": ""}), mock.patch.object(metrics.gke, "configuration", return_value=gke_on):
      source, _ = metrics.hardware()
      self.assertEqual(source.name, "gke")
      self.assertIn("/projects/p/location/global/prometheus", source.url)
    with mock.patch.dict("os.environ", {"OPEN_RL_PROMETHEUS_URL": ""}), mock.patch.object(metrics.gke, "configuration", return_value=gke_off):
      source, reason = metrics.hardware()
      self.assertIsNone(source)
      self.assertIn("OPEN_RL_PROMETHEUS_URL", reason)


class HardwareSeriesTest(unittest.TestCase):
  def test_node_query_batches_idle_gpus_and_historical_workers(self) -> None:
    state = {
      "observed_at": WINDOW["since"],
      "runs": [],
      "cluster": {"available": True, "scheduler": {}, "nodes": [{"name": "node", "devices": [{"id": "idle", "uuid": "GPU-idle"}]}]},
      "placements": [],
      "processes": [
        {"id": "worker", "pod_uid": "uid", "node": "node", "pod": "old-pod", "runtime_id": None, "gpus": [GPU], "started_at": 0, "heartbeat_at": 10}
      ],
    }
    state["cluster"]["nodes"][0]["devices"].append({"id": "gone", "uuid": GPU})
    backend = mock.Mock(name="source")
    backend.name = "fake"
    backend.devices = mock.AsyncMock(return_value={GPU: {"utilization": [[1, 80]], "memory_mib": []}})
    backend.workers_series = mock.AsyncMock(return_value={"old-pod": {"cpu_cores": [[1, 2]], "memory_bytes": []}})
    with mock.patch.object(metrics, "hardware", return_value=(backend, None)):
      result = asyncio.run(metrics.series(state, node_id="node", **WINDOW))
    self.assertCountEqual(backend.devices.await_args.args[0], ["GPU-idle", GPU])
    self.assertEqual(backend.workers_series.await_args.args[0], ["old-pod"])
    self.assertCountEqual([(d["id"], d["uuid"]) for d in result["devices"]], [("idle", "GPU-idle"), ("gone", GPU)])
    self.assertEqual([(w["id"], w["pod_uid"]) for w in result["workers"]], [("worker", "uid")])
    self.assertTrue(result["available"])

  def test_hardware_failure_preserves_worker_data_and_reports_the_error(self) -> None:
    state = {
      "observed_at": WINDOW["until"],
      "runs": [],
      "cluster": {"available": True, "scheduler": {}, "nodes": [{"name": "node", "devices": [{"id": "d0", "uuid": GPU}]}]},
      "placements": [{"id": "w", "label": "worker", "run_ids": [], "devices": ["d0"], "pod": "p", "node": "node", "created_at": WINDOW["since"]}],
      "processes": [],
    }
    backend = mock.Mock()
    backend.name = "fake"
    backend.devices = mock.AsyncMock(side_effect=httpx.ConnectError("offline"))
    backend.workers_series = mock.AsyncMock(return_value={"p": {"cpu_cores": [[1, 2]], "memory_bytes": []}})
    with mock.patch.object(metrics, "hardware", return_value=(backend, None)):
      result = asyncio.run(metrics.series(state, **WINDOW))
    self.assertTrue(result["available"])
    self.assertEqual(result["reason"], "GPU metrics query unavailable")
    self.assertEqual(result["devices"][0]["reason"], result["reason"])
    self.assertIsNone(result["workers"][0]["reason"])

  def test_fresh_worker_series_include_its_lifetime_before_first_snapshot(self) -> None:
    state = {
      "observed_at": "1970-01-01T00:00:20+00:00",
      "runs": [],
      "cluster": {"available": True, "scheduler": {}, "nodes": [{"devices": [{"id": "d0", "uuid": GPU}, {"id": "d1", "uuid": None}]}]},
      "placements": [
        {"id": "w1", "label": "worker", "run_ids": [], "node": "node", "devices": ["d0", "d1"], "pod": "orw-trainer", "created_at": WINDOW["since"]}
      ],
      "processes": [],
    }

    class Backend:
      name = "fake"

      async def devices(self, uuids, start, end):
        return {u: {"utilization": [[start, 90.0]], "memory_mib": [[start, 1.0]]} for u in uuids}

      async def workers_series(self, pods, start, end):
        return {p: {"cpu_cores": [[start, 4.0]], "memory_bytes": [[start, 8.0]]} for p in pods}

    with mock.patch.object(metrics, "hardware", return_value=(Backend(), None)):
      result = asyncio.run(metrics.series(state, **WINDOW))
    self.assertTrue(result["available"])
    self.assertEqual(result["source"], "fake")
    self.assertEqual(result["devices"][0]["utilization"][0][1], 90.0)
    self.assertEqual(result["devices"][1]["reason"], "GPU UUID unavailable")
    self.assertEqual((result["workers"][0]["pod"], result["workers"][0]["cpu_cores"][0][1]), ("orw-trainer", 4.0))

  def test_a_previous_process_keeps_its_own_gpu_after_the_slot_changes(self) -> None:
    state = {
      "observed_at": "1970-01-01T00:00:20+00:00",
      "runs": [],
      "cluster": {"available": True, "scheduler": {}, "nodes": [{"name": "node", "devices": [{"id": "d0", "uuid": "GPU-new"}]}]},
      "placements": [
        {
          "id": "w",
          "label": "worker",
          "run_ids": [],
          "node": "node",
          "pod": "reused-name",
          "pod_uid": "new",
          "devices": ["d0"],
          "device_uuids": {"d0": "GPU-new"},
          "created_at": "1970-01-01T00:00:15+00:00",
        }
      ],
      # The previous pod of the same name ran on the slot's earlier GPU.
      "processes": [
        {
          "id": "old",
          "node": "node",
          "pod": "reused-name",
          "pod_uid": "old",
          "runtime_id": None,
          "gpus": [GPU],
          "started_at": 2,
          "heartbeat_at": 8,
          "phase": "exited",
        }
      ],
    }
    backend = mock.Mock()
    backend.name = "fake"
    backend.devices = mock.AsyncMock(return_value={GPU: {"utilization": [[3, 50]], "memory_mib": []}})
    backend.workers_series = mock.AsyncMock(return_value={"reused-name": {"cpu_cores": [[1, 1], [3, 2], [9, 3]], "memory_bytes": []}})
    with mock.patch.object(metrics, "hardware", return_value=(backend, None)):
      result = asyncio.run(metrics.series(state, node_id="node", **WINDOW))
    self.assertCountEqual(backend.devices.await_args.args[0], [GPU, "GPU-new"])
    self.assertEqual(next(d for d in result["devices"] if d["uuid"] == GPU)["utilization"], [[3, 50]])
    self.assertEqual(result["workers"][0]["pod_uid"], "old")
    self.assertEqual(result["workers"][0]["cpu_cores"], [[3, 2]])
    state["placements"] = []
    with mock.patch.object(metrics, "hardware", return_value=(backend, None)):
      result = asyncio.run(metrics.series(state, node_id="node", since=WINDOW["since"], until="1970-01-01T00:00:19.999Z"))
    self.assertCountEqual(backend.devices.await_args.args[0], [GPU, "GPU-new"])
    self.assertCountEqual([(d["id"], d["uuid"]) for d in result["devices"]], [(GPU, GPU), ("d0", "GPU-new")])
    self.assertEqual([w["pod_uid"] for w in result["workers"]], ["old"])
