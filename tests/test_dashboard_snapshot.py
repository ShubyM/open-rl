import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from server.dashboard import router, snapshot

STATE = {
  "available": True,
  "pods": [
    {
      "name": "orw-lora-trainer",
      "uid": "p1",
      "phase": "Running",
      "node": "n1",
      "worker": "lora-qwen-0-trainer",
      "owner_uids": ["w1"],
      "restarts": 0,
      "created_at": "t",
      "problem": None,
      "containers": [],
      "events": [],
    },
    {
      "name": "orw-fft-trainer",
      "uid": "p2",
      "phase": "Pending",
      "node": None,
      "worker": "fft-r2-trainer",
      "owner_uids": [],
      "restarts": 0,
      "created_at": "t",
      "problem": "Pending",
      "containers": [],
      "events": [],
    },
  ],
  "scheduler": {
    "workloads": [
      {
        "uid": "w1",
        "name": "lora-qwen-0-trainer",
        "model_id": "Qwen/Qwen3-8B",
        "training_kind": "lora",
        "role": "trainer",
        "node_name": "n1",
        "claim_name": "c1",
        "pod_name": "orw-lora-trainer",
        "device_count": 1,
        "phase": "Running",
      },
      {
        "uid": "w2",
        "name": "fft-r2-trainer",
        "model_id": "r2",
        "training_kind": "fft",
        "role": "trainer",
        "node_name": None,
        "claim_name": None,
        "pod_name": "orw-fft-trainer",
        "device_count": 0,
        "phase": "Pending",
      },
    ]
  },
  "devices": {"claims": {"c1": ["gpu.nvidia.com/n1/gpu-0"]}},
}
METADATA = [
  {"model_id": "r1", "base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "active", "created_at": 2, "total_steps_completed": 3},
  {"model_id": "r3", "base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "active", "created_at": 1},
  {"model_id": "r2", "base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "full", "status": "active", "created_at": 3},
]


class SnapshotJoinTest(unittest.TestCase):
  def test_lora_runs_share_the_runtime_worker_and_fft_runs_own_theirs(self) -> None:
    runs = {r["run_id"]: r for r in snapshot.join_runs(METADATA, STATE)}
    self.assertEqual(runs["r1"]["display_status"], "Running")
    self.assertEqual([p["name"] for p in runs["r1"]["pods"]], ["orw-lora-trainer"])
    self.assertEqual(runs["r1"]["pods"][0]["devices"], ["gpu.nvidia.com/n1/gpu-0"])
    self.assertEqual(sorted(runs["r1"]["runtime_run_ids"]), ["r1", "r3"])
    self.assertEqual(runs["r2"]["display_status"], "Needs attention")
    self.assertEqual([w["uid"] for w in runs["r2"]["workloads"]], ["w2"])

  def test_only_placed_workloads_become_placements_and_carry_their_runs(self) -> None:
    runs = snapshot.join_runs(METADATA, STATE)
    placements = snapshot.placements_of(STATE, runs)
    self.assertEqual(len(placements), 1)
    self.assertEqual(placements[0]["id"], "w1")
    self.assertEqual(sorted(placements[0]["run_ids"]), ["r1", "r3"])
    self.assertEqual(placements[0]["label"], "Qwen/Qwen3-8B")
    self.assertEqual(placements[0]["devices"], ["gpu.nvidia.com/n1/gpu-0"])

  def test_a_finished_lora_run_lets_go_of_the_shared_runtime(self) -> None:
    done = {"model_id": "r0", "base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "completed", "created_at": 0}
    runs = {r["run_id"]: r for r in snapshot.join_runs([*METADATA, done], STATE)}
    self.assertEqual((runs["r0"]["pods"], runs["r0"]["workloads"], runs["r0"]["display_status"]), ([], [], "Completed"))
    placements = snapshot.placements_of(STATE, list(runs.values()))
    self.assertEqual(sorted(placements[0]["run_ids"]), ["r1", "r3"])

  def test_scheduler_outage_is_not_an_unassigned_run(self) -> None:
    state = {**STATE, "scheduler": {"available": False, "workloads": []}}
    run = snapshot.join_runs(METADATA, state)[0]
    self.assertEqual(run["display_status"], "Unknown")
    self.assertIsNotNone(run["delete_blocker"])


class SnapshotRecoveryTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.snapshot = snapshot.Snapshot()
    self.store = SimpleNamespace(list_jobs_metadata=AsyncMock(return_value=copy.deepcopy(METADATA)), get_model_metadata=AsyncMock(return_value=None))
    self.enterContext(patch.object(snapshot.backends, "current", return_value=SimpleNamespace(inventory=lambda: STATE)))
    self.history = self.enterContext(patch.object(snapshot.history, "read", new=AsyncMock(return_value=[{"id": "past"}])))
    self.enterContext(patch.object(router, "snapshot", self.snapshot))
    self.enterContext(patch.object(router, "get_store", return_value=self.store))

  async def test_temporary_source_errors_retain_records_and_report_staleness(self) -> None:
    first = await self.snapshot.current(self.store)
    self.store.list_jobs_metadata.side_effect = TimeoutError()
    self.history.side_effect = TimeoutError()
    self.snapshot.updated = 0
    stale = await self.snapshot.current(self.store)
    self.assertEqual([r["run_id"] for r in first["runs"]], [r["run_id"] for r in stale["runs"]])
    self.assertTrue(all(r["delete_blocker"] for r in stale["runs"]))
    self.assertTrue(stale["store_error"])
    self.assertEqual(stale["history"], first["history"])
    self.assertTrue(stale["history_error"])
    self.store.list_jobs_metadata.side_effect = None
    self.store.list_jobs_metadata.return_value = []
    self.history.side_effect = None
    self.history.return_value = []
    self.snapshot.updated = 0
    recovered = await self.snapshot.current(self.store)
    self.assertEqual(recovered["runs"], [])
    self.assertIsNone(recovered["store_error"])
    self.assertIsNone(recovered["history_error"])

  async def test_new_run_resolves_before_snapshot_cache_expires(self) -> None:
    await self.snapshot.current(self.store)
    self.store.get_model_metadata.return_value = {**METADATA[0], "model_id": "new-run"}
    _, run = await router.run_of("new-run")
    self.assertEqual(run["run_id"], "new-run")
    self.assertCountEqual(run["runtime_run_ids"], ["new-run", "r1", "r3"])
    detail = await router.run_detail("new-run")
    self.assertEqual(len(detail["placements"]), 1)
    self.assertIn("new-run", detail["placements"][0]["run_ids"])

  async def test_store_failure_is_not_a_missing_run(self) -> None:
    self.store.list_jobs_metadata.side_effect = TimeoutError()
    with self.assertRaises(HTTPException) as error:
      await router.run_of("missing")
    self.assertEqual(error.exception.status_code, 503)

  async def test_truly_missing_run_returns_404(self) -> None:
    with self.assertRaises(HTTPException) as error:
      await router.run_of("missing")
    self.assertEqual(error.exception.status_code, 404)
