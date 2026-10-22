import asyncio
import unittest
from datetime import UTC, datetime
from unittest import mock

import httpx
from kubernetes import client

from server.dashboard import metrics, sources
from server.dashboard.sources import PromQL
from server.scheduler_worker_manager import SchedulerWorkerManager

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
          summary = sources.workload_summary(item)
          self.assertEqual(summary["worker"], manager.load_workers()[0])
        self.assertEqual(summary["worker"]["state"], expected)
        self.assertEqual(summary["worker"]["age_seconds"], 9)
        self.assertEqual((summary["phase"], summary["reason"]), (status["phase"], status.get("reason")))

  def test_a_container_running_again_after_a_crash_is_not_a_problem(self):
    at = datetime(2026, 1, 1, tzinfo=UTC)
    crashed = client.V1ContainerState(terminated=client.V1ContainerStateTerminated(exit_code=1, reason="Error", finished_at=at))
    running = client.V1ContainerState(running=client.V1ContainerStateRunning(started_at=at))
    status = client.V1ContainerStatus(name="dispatcher", image="i", image_id="", ready=True, restart_count=1, state=running, last_state=crashed)
    pod = client.V1Pod(
      metadata=client.V1ObjectMeta(name="p", uid="uid"),
      spec=client.V1PodSpec(containers=[client.V1Container(name="dispatcher")]),
      status=client.V1PodStatus(phase="Running", container_statuses=[status]),
    )
    row = sources.pod_summary(pod)
    self.assertEqual((row["problem"], row["restarts"]), (None, 1))


class PromQLTest(unittest.TestCase):
  def test_identical_requests_share_work_without_sharing_waiter_cancellation(self):
    async def exercise():
      entered, release = asyncio.Event(), asyncio.Event()
      calls = 0

      async def respond(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"answer": 42})

      connection = httpx.AsyncClient(transport=httpx.MockTransport(respond))
      with mock.patch.object(httpx, "AsyncClient", return_value=connection):
        async with sources.lifespan():
          first = asyncio.create_task(sources.get_json("https://example/query", params={"query": "up"}, headers={}))
          await entered.wait()
          second = asyncio.create_task(sources.get_json("https://example/query", params={"query": "up"}, headers={}))
          await asyncio.sleep(0)
          first.cancel()
          with self.assertRaises(asyncio.CancelledError):
            await first
          release.set()
          self.assertEqual(await second, {"answer": 42})
          self.assertEqual(calls, 1)
          await sources.get_json("https://example/query", params={"query": "up"}, headers={})
          self.assertEqual(calls, 2)  # Completed results are not cached.
          entered.clear()
          release.clear()
          abandoned = asyncio.create_task(sources.get_json("https://example/query", params={"query": "up"}, headers={}))
          await entered.wait()
        with self.assertRaises(asyncio.CancelledError):
          await abandoned  # Shutdown cancels and drains the outstanding request.
      self.assertTrue(connection.is_closed)

    asyncio.run(exercise())

  def test_shared_requests_isolate_credentials_and_queries_and_do_not_cache_errors(self):
    async def exercise():
      entered, release = asyncio.Event(), asyncio.Event()
      calls = 0

      async def respond(request):
        nonlocal calls
        calls += 1
        if calls == 6:
          entered.set()
        await release.wait()
        return httpx.Response(503)

      url, params, headers = "https://example/query", {"query": "up", "end": 60}, {"Authorization": "Bearer a", "x-goog-user-project": "p"}
      requests = [
        (url, params, headers),
        (url, params, headers),
        (url, params, {**headers, "Authorization": "Bearer b"}),
        (url, params, {**headers, "x-goog-user-project": "other"}),
        (url, {**params, "query": "down"}, headers),
        (url, {**params, "end": 90}, headers),
        (url + "/other", params, headers),
      ]
      connection = httpx.AsyncClient(transport=httpx.MockTransport(respond))
      with (
        mock.patch.object(httpx, "AsyncClient", return_value=connection),
        mock.patch.object(sources, "MAX_INFLIGHT", 1),
        mock.patch.object(sources, "RETRY_SECONDS", 0),
      ):
        async with sources.lifespan():
          tasks = [asyncio.create_task(sources.get_json(u, params=p, headers=h)) for u, p, h in requests]
          await asyncio.wait_for(entered.wait(), timeout=1)
          release.set()
          errors = await asyncio.gather(*tasks, return_exceptions=True)
          self.assertTrue(all(isinstance(error, httpx.HTTPStatusError) for error in errors))
          # Six distinct requests, each retried once after its 503.
          self.assertEqual(calls, 12)
          with self.assertRaises(httpx.HTTPStatusError):
            await sources.get_json(url, params=params, headers=headers)
          self.assertEqual(calls, 14)
          release.clear()
          with mock.patch.object(sources, "HTTP_SECONDS", 0.01), self.assertRaises(TimeoutError):
            await sources.get_json(url, params=params, headers=headers)

    asyncio.run(exercise())

  def test_partial_failure_keeps_successful_fields_after_one_retry(self):
    calls = []

    async def respond(request):
      query = request.url.params["query"]
      calls.append(query)
      if "FB_USED" in query:
        return httpx.Response(503)
      return httpx.Response(200, json={"status": "success", "data": {"result": [{"metric": {"UUID": GPU}, "values": [[1, "80"]]}]}})

    for denied in (False, True):
      with self.subTest(credentials_denied=denied):
        calls.clear()
        token = mock.Mock(return_value="token", side_effect=RuntimeError("no credentials") if denied else None)
        backend = PromQL("gke", "https://example", token, None, "openrl-system")
        connection = httpx.AsyncClient(transport=httpx.MockTransport(respond))

        async def query(backend=backend, connection=connection):
          async with sources.lifespan():
            self.assertIs(sources.http_client(), connection)
            return await backend.devices([GPU], 0, 60)

        with mock.patch.object(httpx, "AsyncClient", return_value=connection) as factory, mock.patch.object(sources, "RETRY_SECONDS", 0):
          found, errors = asyncio.run(query())
        self.assertEqual(found, {} if denied else {"utilization": {GPU: [[1.0, 80.0]]}})
        expected = dict.fromkeys(sources.GPU_METRICS, "Metrics query unavailable") if denied else {"memory_mib": "Metrics source returned HTTP 503"}
        self.assertEqual(errors, expected)
        # The failed query is retried once; denied credentials make no request at all.
        self.assertEqual(len(calls), 0 if denied else 3)
        token.assert_called_once()
        factory.assert_called_once()
        self.assertTrue(connection.is_closed)
        self.assertIsNone(sources.pool)

  def test_queries_made_at_different_moments_share_sample_times(self) -> None:
    # Each allocation is fetched separately and the window slides with every
    # refresh; unaligned starts gave every response its own timestamps, so the
    # per-sample average across GPUs came out empty until all requests landed.
    backend = PromQL("gke", "https://example", None, None, "openrl-system")
    sent = []

    async def request(self, method, url, params, headers):
      sent.append(params)
      return httpx.Response(200, json={"status": "success", "data": {"result": []}}, request=httpx.Request("GET", url))

    async def query():
      async with sources.lifespan():
        for start in (1000.4, 1010.4):
          await backend.query_range("q", start, start + 1800)

    with mock.patch.object(httpx.AsyncClient, "request", request):
      asyncio.run(query())
    self.assertEqual([p["step"] for p in sent], [15, 15])
    self.assertEqual([p["start"] % 15 for p in sent], [0, 0])
    self.assertTrue(all(p["start"] <= s for p, s in zip(sent, (1000.4, 1010.4), strict=True)))


class HardwareSourceTest(unittest.TestCase):
  def test_explicit_prometheus_wins_then_gke_then_nothing(self) -> None:
    gke_on = {"project": "p", "cluster": "c", "location": "l", "mode": "auto", "enabled": True, "configured": True}
    gke_off = {**gke_on, "project": "", "configured": False, "cluster": ""}
    gmp = {"OPEN_RL_PROMETHEUS_URL": "http://gmp:9090/"}
    with mock.patch.dict("os.environ", gmp), mock.patch.object(metrics.sources, "configuration", return_value=gke_on):
      source, _ = metrics.hardware()
      self.assertEqual((source.name, source.url, source.cluster), ("prometheus", "http://gmp:9090", "c"))
    with mock.patch.dict("os.environ", {"OPEN_RL_PROMETHEUS_URL": ""}), mock.patch.object(metrics.sources, "configuration", return_value=gke_on):
      source, _ = metrics.hardware()
      self.assertEqual(source.name, "gke")
      self.assertIn("/projects/p/location/global/prometheus", source.url)
    with mock.patch.dict("os.environ", {"OPEN_RL_PROMETHEUS_URL": ""}), mock.patch.object(metrics.sources, "configuration", return_value=gke_off):
      source, reason = metrics.hardware()
      self.assertIsNone(source)
      self.assertIn("OPEN_RL_PROMETHEUS_URL", reason)

  def test_node_gpus_come_from_dcgm_by_uuid_in_slot_order(self) -> None:
    backend = mock.Mock(slots={})
    backend.scope.return_value = ',cluster="c"'
    rows = [{"metric": {"Hostname": "n1", "UUID": f"GPU-{slot}", "gpu": str(slot)}} for slot in (10, 2, 0)]
    backend.range_rows = mock.AsyncMock(return_value=[*rows, {"metric": {"Hostname": "n2", "UUID": "GPU-x", "gpu": "0"}}])
    with mock.patch.object(metrics, "hardware", return_value=(backend, None)), mock.patch.object(metrics.sources, "discover", mock.AsyncMock()):
      found = asyncio.run(metrics.node_gpus())
    self.assertEqual(
      [(d["id"], d["uuid"], d["name"]) for d in found["n1"]],
      [("GPU-0", "GPU-0", "gpu-0"), ("GPU-2", "GPU-2", "gpu-2"), ("GPU-10", "GPU-10", "gpu-10")],
    )
    self.assertIn('cluster="c"', backend.range_rows.await_args.args[0])
    with mock.patch.object(metrics, "hardware", return_value=(None, "no source")), mock.patch.object(metrics.sources, "discover", mock.AsyncMock()):
      self.assertIsNone(asyncio.run(metrics.node_gpus()))


def placement(pod: str, gpus: list[str], placed_at: float, released_at: float | None) -> dict:
  return {
    "workload": pod,
    "role": "trainer",
    "model_id": "r",
    "pod": pod,
    "node": "node",
    "gpu_uuids": gpus,
    "placed_at": placed_at,
    "released_at": released_at,
  }


def hardware_state(records: list[dict], devices: list[dict], observed_at: str = WINDOW["until"]) -> dict:
  return {
    "observed_at": observed_at,
    "runs": [],
    "cluster": {"nodes": [{"name": "node", "devices": devices}]},
    "placement_records": records,
    "processes": [],
  }


class HardwareSeriesTest(unittest.TestCase):
  def test_total_failure_is_distinct_from_successful_empty_series(self):
    state = hardware_state([], [{"id": GPU, "uuid": GPU}])
    backend = mock.Mock(slots={})
    backend.name = "fake"
    for errors in ({}, {"memory_mib": "Offline"}, dict.fromkeys(sources.GPU_METRICS, "Offline")):
      with self.subTest(errors=errors):
        backend.devices = mock.AsyncMock(return_value=({}, errors))
        with mock.patch.object(metrics, "hardware", return_value=(backend, None)):
          result = asyncio.run(metrics.series(state, node_id="node", kind="hardware", **WINDOW))
        self.assertEqual(result["error"], "Offline" if len(errors) == 2 else None)
        self.assertEqual(result["devices"][0]["errors"], errors)
        self.assertFalse(result["available"])

  def test_node_query_batches_idle_gpus_and_released_placements(self) -> None:
    state = hardware_state([placement("old-pod", [GPU], 0, 6)], [{"id": "GPU-idle", "uuid": "GPU-idle"}])
    backend = mock.Mock(name="source", slots={})
    backend.name = "fake"
    backend.devices = mock.AsyncMock(return_value=({"utilization": {GPU: [[1, 80]]}}, {}))
    backend.workers_series = mock.AsyncMock(return_value=({"cpu_cores": {"old-pod": [[1, 2], [8, 3]]}}, {}))
    with mock.patch.object(metrics, "hardware", return_value=(backend, None)):
      result = asyncio.run(metrics.series(state, node_id="node", **WINDOW))
    self.assertCountEqual(backend.devices.await_args.args[0], ["GPU-idle", GPU])
    self.assertEqual(backend.workers_series.await_args.args[0], ["old-pod"])
    self.assertCountEqual([(d["id"], d["uuid"]) for d in result["devices"]], [("GPU-idle", "GPU-idle"), (GPU, GPU)])
    # A worker's series stop when its placement was released.
    self.assertEqual([(w["id"], w["cpu_cores"]) for w in result["workers"]], [("old-pod@0", [[1, 2]])])
    self.assertTrue(result["available"])

  def test_a_runs_series_cover_only_the_gpus_its_placements_held(self) -> None:
    state = hardware_state([placement("orw-trainer", [GPU], 0, None)], [{"id": GPU, "uuid": GPU}, {"id": "GPU-other", "uuid": "GPU-other"}])
    state["runs"] = [{"run_id": "r", "runtime_id": "r", "name": "Run", "model": None, "created_at": 0, "completed_at": None}]

    class Backend:
      name = "fake"
      slots = {}

      async def devices(self, uuids, start, end):
        return {"utilization": {u: [[start, 90.0]] for u in uuids}, "memory_mib": {u: [[start, 1.0]] for u in uuids}}, {}

      async def workers_series(self, pods, start, end):
        return {"cpu_cores": {p: [[start, 4.0]] for p in pods}, "memory_bytes": {p: [[start, 8.0]] for p in pods}}, {}

    with mock.patch.object(metrics, "hardware", return_value=(Backend(), None)):
      result = asyncio.run(metrics.series(state, run_id="r", **WINDOW))
    self.assertEqual((result["source"], [d["uuid"] for d in result["devices"]]), ("fake", [GPU]))
    self.assertEqual(result["devices"][0]["utilization"][0][1], 90.0)
    self.assertEqual((result["workers"][0]["pod"], result["workers"][0]["cpu_cores"][0][1]), ("orw-trainer", 4.0))
