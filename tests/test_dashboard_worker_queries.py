"""Worker charts retain runtime process history without changing GPU or vLLM query subjects."""

import copy
import unittest
from unittest import mock

from server.dashboard import facts, fleet, router, sources


class WorkerQueriesTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.run = {"run_id": "r1", "runtime_id": "runtime", "name": "Run", "model": "base", "created_at": 100, "completed_at": None}
    self.state = {
      "observed_at": 1000,
      "runs": [self.run],
      "placements": [],
      "workers": [],
      "nodes": [],
    }
    self.enterContext(mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=self.state)))
    self.enterContext(mock.patch.object(sources, "discover", mock.AsyncMock()))
    self.enterContext(mock.patch.object(sources, "configuration", return_value={"configured": True, "cluster": "cluster"}))
    self.enterContext(mock.patch.object(sources, "namespace", return_value="openrl-system"))
    self.enterContext(mock.patch.object(sources, "prometheus_source", return_value="prometheus"))
    self.prom = self.enterContext(mock.patch.object(sources, "prom", mock.AsyncMock()))

  def worker(self, pod, **fields):
    return {
      "pod": pod,
      "role": "sampler",
      "runtime_id": "runtime",
      "node": "n1",
      "started_at": 200,
      "heartbeat_at": 990,
      "phase": "ready",
      **fields,
    }

  async def query(self, name="worker_cpu_cores", *, run_id="r1", node_id=None, low=100, high=1000, points=()):
    self.prom.return_value = [{"metric": {"pod_name": pod}, "values": [[at, "2"] for at in times]} for pod, times in points]
    return await router.read_query(name, run_id, node_id, facts.iso(low), facts.iso(high), 300)

  async def test_run_queries_include_sampler_and_trainer_processes_for_the_runtime_only(self):
    self.state["workers"] = [
      self.worker("sampler"),
      self.worker("trainer", role="trainer"),
      self.worker("other-sampler", runtime_id="other"),
      self.worker("other-trainer", role="trainer", runtime_id="other"),
    ]
    for name in ("worker_cpu_cores", "worker_memory_bytes"):
      with self.subTest(name=name):
        result = await self.query(name, points=[("sampler", [199, 200, 1000, 1001]), ("trainer", [200, 1000])])
        self.assertIn(f"pod_name=~{sources.any_of(['sampler', 'trainer'])}", result["query"])
        self.assertEqual([row["points"] for row in result["series"]], [[[200.0, 2.0], [1000.0, 2.0]]] * 2)

  async def test_stale_and_exited_processes_stop_at_heartbeat_and_restarts_keep_the_gap(self):
    self.state["workers"] = [
      self.worker("stale", heartbeat_at=400),
      self.worker("exited", started_at=500, heartbeat_at=990, phase="exited"),
      self.worker("reused", heartbeat_at=300, phase="exited"),
      self.worker("reused", started_at=700),
    ]
    result = await self.query(points=[("stale", [200, 400, 401]), ("exited", [499, 500, 990, 1000]), ("reused", [200, 300, 400, 700, 1000])])
    self.assertEqual(
      [row["points"] for row in result["series"]],
      [
        [[200.0, 2.0], [400.0, 2.0]],
        [[500.0, 2.0], [990.0, 2.0]],
        [[200.0, 2.0], [300.0, 2.0], [700.0, 2.0], [1000.0, 2.0]],
      ],
    )

  async def test_ended_run_excludes_processes_that_never_overlapped_its_lifetime(self):
    self.run["completed_at"] = 400
    self.state["workers"] = [
      self.worker("owned", heartbeat_at=350, phase="exited"),
      self.worker("later", started_at=401),
      self.worker("earlier", started_at=0, heartbeat_at=99, phase="exited"),
    ]
    result = await self.query(points=[("owned", [200, 350, 400])])
    self.assertIn('pod_name=~"owned"', result["query"])
    self.assertEqual(result["series"][0]["points"], [[200.0, 2.0], [350.0, 2.0]])

  async def test_node_selection_uses_node_and_clips_to_requested_window(self):
    self.state["workers"] = [self.worker("local", runtime_id="another-runtime"), self.worker("remote", node="n2")]
    before = copy.deepcopy(self.state)
    result = await self.query(run_id=None, node_id="n1", low=800, high=900, points=[("local", [799, 800, 900, 901])])
    self.assertIn('pod_name=~"local"', result["query"])
    self.assertEqual(result["series"][0]["points"], [[800.0, 2.0], [900.0, 2.0]])
    self.assertEqual(self.state, before)

  async def test_placement_history_remains_queryable_without_a_process_record(self):
    self.state["placements"] = [
      {
        "workload": "trainer",
        "model_id": "runtime",
        "role": "trainer",
        "pod": "placed",
        "node": "n1",
        "gpu_uuids": ["GPU-1"],
        "placed_at": 150,
        "released_at": 300,
      }
    ]
    result = await self.query(points=[("placed", [149, 150, 300, 301])])
    self.assertIn('pod_name=~"placed"', result["query"])
    self.assertEqual(result["series"][0]["points"], [[150.0, 2.0], [300.0, 2.0]])

  async def test_process_history_does_not_change_existing_gpu_or_vllm_query_subjects(self):
    self.state["workers"] = [self.worker("sampler")]
    with mock.patch.object(router, "worker_windows", side_effect=AssertionError("Only worker queries read process windows")):
      for name in ("gpu_util", "vllm_requests_running"):
        with self.subTest(name=name):
          result = await self.query(name)
          self.assertEqual(result["error"], "No GPUs or pods recorded for this subject in this window")
    self.prom.assert_not_awaited()


if __name__ == "__main__":
  unittest.main()
