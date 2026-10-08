import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from server.dashboard import router, snapshot
from server.session_registry import SessionRegistry
from server.store import InMemoryStore
from server.telemetry import ops

# One live FFT worker for run "busy"; nothing for the others.
STATE = {
  "available": True,
  "pods": [
    {
      "name": "orw-busy-trainer",
      "uid": "p1",
      "phase": "Running",
      "node": "n1",
      "worker": "fft-busy-trainer",
      "owner_uids": ["w1"],
      "problem": None,
    }
  ],
  "scheduler": {
    "workloads": [
      {
        "uid": "w1",
        "name": "fft-busy-trainer",
        "model_id": "busy",
        "training_kind": "fft",
        "role": "trainer",
        "node_name": "n1",
        "pod_name": "orw-busy-trainer",
      }
    ]
  },
  "devices": {"claims": {}},
}
RUNS = {
  "done": {"status": "completed"},
  "ended": {"status": "ended"},
  "abandoned": {"status": "active"},
  "busy": {"status": "active"},
  "live-session": {"status": "active", "session_id": "sess-1"},
}


class DeleteRunsTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    for run_id, meta in RUNS.items():
      self.store.kv_store[f"open_rl:model_meta:{run_id}"] = json.dumps({"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "full", **meta})
      self.store.sample_store[ops.key(run_id)] = ["{}"]
    asyncio.run(SessionRegistry(self.store).heartbeat("sess-1"))
    backend = SimpleNamespace(inventory=lambda: STATE)
    self.patches = [
      mock.patch.object(router, "get_store", return_value=self.store),
      mock.patch.object(snapshot.backends, "current", return_value=backend),
    ]
    for patch in self.patches:
      patch.start()
    snapshot.snapshot.invalidate()

  def tearDown(self) -> None:
    for patch in self.patches:
      patch.stop()
    snapshot.snapshot.invalidate()

  def delete(self, *run_ids: str) -> dict:
    return asyncio.run(router.delete_runs(router.DeleteRuns(run_ids=list(run_ids))))

  def remaining(self) -> list[str]:
    return sorted(r["run_id"] for r in asyncio.run(snapshot.snapshot.current(self.store))["runs"])

  def test_finished_and_abandoned_runs_are_forgotten_with_their_ops(self) -> None:
    result = self.delete("done", "ended", "abandoned")
    self.assertEqual(result, {"deleted": ["done", "ended", "abandoned"], "kept": []})
    self.assertEqual(self.remaining(), ["busy", "live-session"])
    self.assertNotIn(ops.key("done"), self.store.sample_store)

  def test_runs_with_workers_or_a_live_session_are_kept_with_the_reason(self) -> None:
    result = self.delete("busy", "live-session", "missing")
    self.assertEqual(result["deleted"], [])
    self.assertEqual(
      {k["run_id"]: k["reason"] for k in result["kept"]},
      {"busy": "The run still has workers", "live-session": "The run's client session is still live", "missing": "Run not found"},
    )
    self.assertEqual(self.remaining(), sorted(RUNS))

  def test_the_snapshot_tells_the_page_which_runs_cannot_go(self) -> None:
    runs = {r["run_id"]: r for r in asyncio.run(snapshot.snapshot.current(self.store))["runs"]}
    self.assertEqual(runs["busy"]["delete_blocker"], "The run still has workers")
    self.assertIsNone(runs["abandoned"]["delete_blocker"])
    self.assertIsNone(runs["done"]["delete_blocker"])

  def test_unfinished_runs_stay_when_the_cluster_cannot_be_read(self) -> None:
    run = {"status": "active", "pods": [], "workloads": []}
    self.assertIsNotNone(snapshot.delete_blocker(run, available=False))
    self.assertIsNone(snapshot.delete_blocker({**run, "status": "completed"}, available=False))

  def test_an_inferred_end_cannot_be_deleted_after_a_client_recovers(self) -> None:
    asyncio.run(self.store.add_to_set("open_rl:run_sessions:ended", "sess-1"))
    self.assertEqual(self.delete("ended")["deleted"], [])

  def test_deleting_a_run_removes_its_session_memberships(self) -> None:
    asyncio.run(self.store.add_to_set("open_rl:run_sessions:abandoned", "expired"))
    self.assertEqual(self.delete("abandoned")["deleted"], ["abandoned"])
    self.assertEqual(asyncio.run(self.store.set_members("open_rl:run_sessions:abandoned")), set())


if __name__ == "__main__":
  unittest.main()
