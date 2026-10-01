import asyncio
import unittest
from unittest import mock

from server.dashboard import local, metrics, snapshot, sources

GPU = "GPU-375ff71c-7c41-a89e-3843-c62c7476bc2b"


class LocalInventoryTest(unittest.TestCase):
  def test_a_local_worker_joins_its_run_like_a_scheduled_pod(self) -> None:
    worker = {
      "pod": "trainer-run-a-4242",
      "pid": 4242,
      "job": "trainer-run-a",
      "uid": "trainer-run-a:4242:1",
      "role": "trainer",
      "runtime_id": "run-a",
      "training_kind": "fft",
      "started_at": 1.0,
      "gpu_uuids": [GPU],
      "cpu_cores": 1.5,
      "memory_bytes": 2**30,
    }
    with (
      mock.patch.object(local.sampler, "start"),
      mock.patch.object(local.sampler, "gpus", [{"index": "0", "uuid": GPU, "name": "NVIDIA H100 80GB HBM3"}]),
      mock.patch.object(local.sampler, "gpu_error", None),
      mock.patch.object(local.sampler, "workers", {worker["pod"]: worker}),
      mock.patch.object(local, "host", return_value="b7"),
    ):
      state = local.read()
    (node,) = state["nodes"]
    self.assertEqual((node["name"], node["gpu_capacity"]), ("b7", 1))
    self.assertEqual(node["devices"], [{"id": "local/b7/gpu-0", "name": "gpu-0", "uuid": GPU}])
    runs = snapshot.join_runs([{"model_id": "run-a", "base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "full", "status": "active"}], state)
    self.assertEqual(runs[0]["display_status"], "Running")
    self.assertEqual(runs[0]["pods"][0]["devices"], ["local/b7/gpu-0"])
    (placement,) = snapshot.placements_of(state, runs)
    self.assertEqual((placement["node"], placement["pod"], placement["run_ids"]), ("b7", "trainer-run-a-4242", ["run-a"]))

  def test_series_are_clipped_to_the_window(self) -> None:
    sampler = local.Sampler()
    for t in (10.0, 20.0, 30.0):
      sampler.gpu_series[GPU].append((t, 50.0, 1024.0))
      sampler.worker_series["p"].append((t, 2.0, 100))
    devices = asyncio.run(sampler.devices([GPU, "GPU-missing"], 15, 30))
    self.assertEqual(devices[GPU], {"utilization": [[20.0, 50.0], [30.0, 50.0]], "memory_mib": [[20.0, 1024.0], [30.0, 1024.0]]})
    self.assertEqual(devices["GPU-missing"], {"utilization": [], "memory_mib": []})
    self.assertEqual(asyncio.run(sampler.workers_series(["p"], 0, 15))["p"], {"cpu_cores": [[10.0, 2.0]], "memory_bytes": [[10.0, 100]]})


class PromQLTest(unittest.TestCase):
  def test_one_query_per_metric_scoped_to_the_cluster_and_deduplicated_by_uuid(self) -> None:
    backend = sources.PromQL("gke", "https://example", None, "open-rl-fft-test")
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

  def test_worker_series_use_gke_container_metrics_by_pod(self) -> None:
    backend = sources.PromQL("gke", "https://example", None, None)
    queries = []

    async def fake(query, start, end):
      queries.append(query)
      return {"orw-trainer": [[1.0, 3.0]]}

    with mock.patch.object(backend, "query_range", side_effect=fake), mock.patch.object(sources.cluster, "namespace", return_value="openrl-system"):
      result = asyncio.run(backend.workers_series(["orw-trainer"], 0, 60))
    self.assertIn("kubernetes_io:container_cpu_core_usage_time", queries[0])
    self.assertIn('namespace_name="openrl-system"', queries[0])
    self.assertIn("kubernetes_io:container_memory_used_bytes", queries[1])
    self.assertEqual(result["orw-trainer"], {"cpu_cores": [[1.0, 3.0]], "memory_bytes": [[1.0, 3.0]]})


class SelectionTest(unittest.TestCase):
  def test_explicit_prometheus_wins_then_gke_then_nothing(self) -> None:
    gke_on = {"project": "p", "cluster": "c", "location": "l", "mode": "auto", "enabled": True, "configured": True}
    gke_off = {**gke_on, "project": "", "configured": False, "cluster": ""}
    with mock.patch.object(sources, "mode", return_value="kubernetes"):
      gmp = {"OPEN_RL_PROMETHEUS_URL": "http://gmp:9090/"}
      with mock.patch.dict("os.environ", gmp), mock.patch.object(sources.gke, "configuration", return_value=gke_on):
        backend, _ = sources.hardware()
        self.assertEqual((backend.name, backend.url, backend.cluster), ("prometheus", "http://gmp:9090", "c"))
      with mock.patch.dict("os.environ", {"OPEN_RL_PROMETHEUS_URL": ""}), mock.patch.object(sources.gke, "configuration", return_value=gke_on):
        backend, _ = sources.hardware()
        self.assertEqual(backend.name, "gke")
        self.assertIn("/projects/p/location/global/prometheus", backend.url)
      with mock.patch.dict("os.environ", {"OPEN_RL_PROMETHEUS_URL": ""}), mock.patch.object(sources.gke, "configuration", return_value=gke_off):
        backend, reason = sources.hardware()
        self.assertIsNone(backend)
        self.assertIn("OPEN_RL_PROMETHEUS_URL", reason)


class AllocationMetricsTest(unittest.TestCase):
  def test_gpu_and_worker_series_for_one_placement(self) -> None:
    state = {
      "cluster": {"nodes": [{"devices": [{"id": "d0", "uuid": GPU}, {"id": "d1", "uuid": None}]}]},
      "placements": [{"id": "w1", "devices": ["d0", "d1"], "pod": "orw-trainer"}],
      "history": [],
    }

    class Backend:
      name = "fake"

      async def devices(self, uuids, start, end):
        return {u: {"utilization": [[start, 90.0]], "memory_mib": [[start, 1.0]]} for u in uuids}

      async def workers_series(self, pods, start, end):
        return {p: {"cpu_cores": [[start, 4.0]], "memory_bytes": [[start, 8.0]]} for p in pods}

    with (
      mock.patch.object(metrics.snapshot, "current", mock.AsyncMock(return_value=state)),
      mock.patch.object(metrics.sources, "hardware", return_value=(Backend(), None)),
    ):
      result = asyncio.run(metrics.gpu_history("w1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:10:00+00:00"))
    self.assertTrue(result["available"])
    self.assertEqual(result["source"], "fake")
    self.assertEqual(result["devices"][0]["utilization"][0][1], 90.0)
    self.assertEqual(result["devices"][1]["reason"], "GPU UUID unavailable")
    self.assertEqual((result["worker"]["pod"], result["worker"]["cpu_cores"][0][1]), ("orw-trainer", 4.0))
