import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from server import openrl_status
from server.dashboard import router, snapshot
from server.model_metadata import TrainingModelMetadata
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore


def worker(name: str, owner: str, model_id: str, state: str, node: str | None) -> dict:
  return {"workload": name, "role": "trainer", "owner": owner, "model_id": model_id, "state": state, "detail": "", "node": node, "age_seconds": 10}


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
      "created_at": "2026-09-11T09:00:00Z",
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
      "created_at": "2026-09-11T09:00:00Z",
      "problem": "Pending",
      "containers": [],
      "events": [],
    },
  ],
  "scheduler": {
    "installed": True,
    "available": True,
    "error": None,
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
        "worker": worker("lora-qwen-0-trainer", "qwen-qwen3-8b", "Qwen/Qwen3-8B", "ready", "n1"),
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
        "worker": worker("fft-r2-trainer", "r2", "r2", "waiting_for_placement", None),
      },
    ],
  },
  "devices": {"claims": {"c1": ["gpu.nvidia.com/n1/gpu-0"]}},
}
# Stored model records, and the rows metadata_of makes of them.
RECORDS = {
  "r1": {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "active", "created_at": 2},
  "r3": {"base_model": "Qwen/Qwen3-8B", "fine_tuning_type": "lora", "status": "active", "created_at": 1},
  "r2": {"base_model": "Qwen/Qwen3-0.6B", "fine_tuning_type": "full", "status": "active", "created_at": 3},
}
METADATA = [{**record, "model_id": run_id, "steps": 0} for run_id, record in RECORDS.items()]


class SnapshotJoinTest(unittest.TestCase):
  def test_lora_runs_share_the_runtime_worker_and_fft_runs_own_theirs(self) -> None:
    runs = {r["run_id"]: r for r in snapshot.join_runs(METADATA, STATE)}
    for run_id in ("r1", "r3"):
      self.assertEqual((runs[run_id]["runtime_id"], runs[run_id]["shared_runtime"]), ("Qwen/Qwen3-8B", True))
      self.assertEqual([p["name"] for p in runs[run_id]["pods"]], ["orw-lora-trainer"])
    self.assertEqual(runs["r1"]["pods"][0]["devices"], ["gpu.nvidia.com/n1/gpu-0"])
    self.assertEqual((runs["r2"]["runtime_id"], runs["r2"]["shared_runtime"]), ("r2", False))
    self.assertEqual([w["uid"] for w in runs["r2"]["workloads"]], ["w2"])

  def test_a_shared_automodel_run_finds_its_named_runtime(self) -> None:
    run = snapshot.join_runs([{**METADATA[0], "trainer_backend": "automodel"}], STATE)[0]
    self.assertEqual(run["runtime_id"], "automodel-Qwen/Qwen3-8B")
    self.assertEqual(snapshot.join_runs([METADATA[0]], STATE)[0]["runtime_id"], "Qwen/Qwen3-8B")

  def test_runtime_identity_uses_the_model_contract_without_workers(self) -> None:
    state = {**STATE, "scheduler": {"workloads": []}, "pods": []}
    for settings in ({"exclusive": True}, {"trainer_gpus": 2}, {"trainer_backend": "registry.example/trainer:v2"}):
      with self.subTest(settings=settings):
        row = {**METADATA[0], **settings}
        model = TrainingModelMetadata.model_validate(row)
        run = snapshot.join_runs([row], state)[0]
        self.assertEqual((run["runtime_id"], run["shared_runtime"]), (model.runtime("r1"), model.shares_runtime()))

  def test_placements_record_pod_identity_and_physical_devices(self) -> None:
    state = copy.deepcopy(STATE)
    state["devices"]["nodes"] = {"n1": [{"id": "gpu.nvidia.com/n1/gpu-0", "uuid": "GPU-1"}]}
    placement = snapshot.placements_of(state, snapshot.join_runs(METADATA, state))[0]
    self.assertEqual(placement["pod_uid"], "p1")
    self.assertEqual(placement["device_uuids"], {"gpu.nvidia.com/n1/gpu-0": "GPU-1"})

  def test_only_placed_workloads_become_placements_and_carry_their_runs(self) -> None:
    runs = snapshot.join_runs(METADATA, STATE)
    placements = snapshot.placements_of(STATE, runs)
    self.assertEqual(len(placements), 1)
    self.assertEqual(placements[0]["id"], "w1")
    self.assertEqual(sorted(placements[0]["run_ids"]), ["r1", "r3"])
    self.assertEqual(placements[0]["label"], "Qwen/Qwen3-8B")
    self.assertEqual(placements[0]["devices"], ["gpu.nvidia.com/n1/gpu-0"])

  def test_a_finished_lora_run_lets_go_of_the_shared_runtime(self) -> None:
    done = {**METADATA[0], "model_id": "r0", "status": "completed", "created_at": 0}
    runs = {r["run_id"]: r for r in snapshot.join_runs([*METADATA, done], STATE)}
    self.assertEqual((runs["r0"]["pods"], runs["r0"]["workloads"]), ([], []))
    placements = snapshot.placements_of(STATE, list(runs.values()))
    self.assertEqual(sorted(placements[0]["run_ids"]), ["r1", "r3"])


class StatusTest(unittest.IsolatedAsyncioTestCase):
  """Each run reports what the gateway's model status reports for it."""

  async def asyncSetUp(self) -> None:
    self.state = copy.deepcopy(STATE)
    self.store = InMemoryStateStore()
    for run_id, record in RECORDS.items():
      await self.store.set_value(f"open_rl:model_meta:{run_id}", json.dumps({**record, "session_ids": [run_id]}))
    await SessionRegistry(self.store).attach("r1", "qwen-qwen3-8b")

  async def runs(self) -> dict:
    runs = snapshot.join_runs(await snapshot.metadata_of(self.store), self.state)
    ctx = snapshot.cluster_status_context(self.store, self.state)
    for run in runs:
      await snapshot.add_status(ctx, run)
    return {run["run_id"]: run for run in runs}

  async def test_status_matches_the_gateway_as_the_worker_loads_then_announces_ready(self) -> None:
    manager = SimpleNamespace(inspect_owner=lambda _: [self.state["scheduler"]["workloads"][0]["worker"]])
    gateway = openrl_status.StatusContext(self.store, SessionRegistry(self.store), manager, "scheduler")
    for announced, label in ((False, "Starting"), (True, "Ready")):
      if announced:
        await self.store.set_value("open_rl:trainer_ready:Qwen/Qwen3-8B", "1")
      run = (await self.runs())["r1"]
      expected = await openrl_status.model_status(gateway, "r1")
      self.assertEqual({key: run[key] for key in expected}, expected)
      self.assertEqual(run["display_status"], label)

  async def test_a_failing_pod_is_unhealthy_while_its_workload_still_reports_running(self) -> None:
    await self.store.set_value("open_rl:trainer_ready:Qwen/Qwen3-8B", "1")
    self.state["pods"][0]["problem"] = "CrashLoopBackOff"
    run = (await self.runs())["r1"]
    self.assertEqual(run["display_status"], "Unhealthy")
    self.assertIn("CrashLoopBackOff", run["summary"])

  async def test_unreadable_sources_are_unknown_not_unassigned(self) -> None:
    self.state["scheduler"].update(available=False, error="scheduler read failed", workloads=[])
    run = (await self.runs())["r1"]
    self.assertEqual(run["display_status"], "Unknown")
    self.assertIn("scheduler read failed", run["workers_error"])
    self.state = copy.deepcopy(STATE)
    with patch.object(self.store, "get_value", side_effect=TimeoutError):
      run = (await self.runs())["r1"]
    self.assertEqual((run["display_status"], run["session_live"]), ("Unknown", None))


class SnapshotRecoveryTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.snapshot = snapshot.Snapshot()
    self.store = InMemoryStateStore()
    self.metadata = self.enterContext(patch.object(snapshot, "metadata_of", AsyncMock(return_value=copy.deepcopy(METADATA))))
    self.enterContext(patch.object(snapshot.kubernetes, "read", return_value=STATE))
    self.history = self.enterContext(patch.object(snapshot.history, "read", AsyncMock(return_value=[{"id": "past"}])))
    self.enterContext(patch.object(router, "snapshot", self.snapshot))
    self.enterContext(patch.object(router, "get_state_store", return_value=self.store))

  async def test_temporary_source_errors_keep_the_last_records_and_say_so(self) -> None:
    first = await self.snapshot.current(self.store)
    self.metadata.side_effect = self.history.side_effect = TimeoutError()
    self.snapshot.invalidate()
    stale = await self.snapshot.current(self.store)
    self.assertEqual([r["run_id"] for r in stale["runs"]], [r["run_id"] for r in first["runs"]])
    self.assertEqual(stale["history"], first["history"])
    self.assertTrue(stale["store_error"] and stale["history_error"])
    self.metadata.side_effect = self.history.side_effect = None
    self.metadata.return_value = self.history.return_value = []
    self.snapshot.invalidate()
    recovered = await self.snapshot.current(self.store)
    self.assertEqual((recovered["runs"], recovered["store_error"], recovered["history_error"]), ([], None, None))

  async def test_a_run_created_since_the_cached_snapshot_is_found(self) -> None:
    await self.snapshot.current(self.store)
    self.metadata.return_value = [*METADATA, {**METADATA[0], "model_id": "new-run"}]
    await self.store.set_value("open_rl:model_meta:new-run", json.dumps(RECORDS["r1"]))
    _, run = await router.run_of("new-run")
    self.assertEqual(run["run_id"], "new-run")

  async def test_a_store_outage_is_not_a_missing_run(self) -> None:
    self.metadata.side_effect = TimeoutError()
    with patch.object(self.store, "get_value", side_effect=TimeoutError), self.assertRaises(HTTPException) as error:
      await router.run_of("missing")
    self.assertEqual(error.exception.status_code, 503)

  async def test_a_missing_run_is_not_found(self) -> None:
    with self.assertRaises(HTTPException) as error:
      await router.run_of("missing")
    self.assertEqual(error.exception.status_code, 404)
