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

  def test_placements_record_pod_identity(self) -> None:
    placement = snapshot.placements_of(STATE, snapshot.join_runs(METADATA, STATE))[0]
    self.assertEqual(placement["pod_uid"], "p1")

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


RUN = {"run_id": "r1", "runtime_id": "base", "created_at": 1}
NOW = 10_000


def process(*groups, heartbeat_at=NOW, phase="ready"):
  outstanding = [{"op": op, "run_id": run_id, "count": 1, "since": at} for op, run_id, at in groups]
  return {"id": "p", "pod": "trainer-0", "runtime_id": "base", "heartbeat_at": heartbeat_at, "phase": phase, "outstanding": outstanding}


def steps(*ats):
  return [{"at": at, "operation": "optim_step", "status": "succeeded"} for at in ats]


class ProgressTest(unittest.TestCase):
  """Progress comes from what completes and what is still waiting, not from which processes are up."""

  def verdict(self, processes=(), recent=(), queued=None):
    return snapshot.progress(RUN, list(processes), list(recent), queued or {"trainer": 0, "sampler": 0}, NOW)

  def test_completing_requests_are_ok(self):
    result = self.verdict([process(("forward_backward", "r1", NOW - 60))], steps(NOW - 300, NOW - 200, NOW - 100))
    self.assertEqual((result["state"], result["step_seconds"], result["last_completed_at"]), ("OK", 100, NOW - 100))
    self.assertEqual(result["outstanding"][0]["age_seconds"], 60)

  def test_requests_taken_by_a_process_that_stopped_reporting_are_dead(self):
    for gone in (process(("optim_step", "r1", NOW - 60), heartbeat_at=NOW - 120), process(("optim_step", "r1", NOW - 60), phase="exited")):
      result = self.verdict([gone], steps(NOW - 30))
      self.assertEqual(result["state"], "DEAD")
      self.assertIn("trainer-0", result["reason"])

  def test_a_request_on_a_worker_longer_than_three_steps_is_stalled(self):
    recent = steps(*range(NOW - 10_000, NOW - 4_000, 300))  # 300 s steps: patience 900 s
    self.assertEqual(self.verdict([process(("forward_backward", "r1", NOW - 600))], recent)["state"], "OK")
    result = self.verdict([process(("forward_backward", "r1", NOW - 1000))], recent)
    self.assertEqual(result["state"], "STALLED")
    self.assertIn("forward_backward", result["reason"])

  def test_queued_requests_with_nothing_completing_are_stalled(self):
    result = self.verdict(recent=steps(NOW - 3600), queued={"trainer": 0, "sampler": 4})
    self.assertEqual(result["state"], "STALLED")
    self.assertIn("4 request(s) queued", result["reason"])

  def test_nothing_queued_or_outstanding_for_long_is_idle(self):
    self.assertEqual(self.verdict(recent=steps(NOW - 3600))["state"], "IDLE")
    self.assertEqual(self.verdict(recent=steps(NOW - 60))["state"], "OK")
    self.assertEqual(self.verdict(queued={"trainer": None, "sampler": None})["state"], "IDLE")

  def test_only_this_runs_requests_on_a_shared_runtime_count(self):
    other = process(("sample", "r2", NOW - 5000), heartbeat_at=0)
    elsewhere = {**process(("sample", "r1", NOW - 5000)), "runtime_id": "other"}
    result = self.verdict([other, elsewhere], steps(NOW - 30))
    self.assertEqual((result["state"], result["outstanding"]), ("OK", []))


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
    self.processes = self.enterContext(patch.object(snapshot, "processes_of", AsyncMock(return_value=[{"id": "past"}])))
    self.enterContext(patch.object(router, "snapshot", self.snapshot))
    self.enterContext(patch.object(router, "get_state_store", return_value=self.store))

  async def test_temporary_source_errors_keep_the_last_records_and_say_so(self) -> None:
    first = await self.snapshot.current(self.store)
    self.metadata.side_effect = self.processes.side_effect = TimeoutError()
    self.snapshot.invalidate()
    stale = await self.snapshot.current(self.store)
    self.assertEqual([r["run_id"] for r in stale["runs"]], [r["run_id"] for r in first["runs"]])
    self.assertEqual(stale["processes"], first["processes"])
    self.assertTrue(stale["store_error"] and stale["processes_error"])
    self.metadata.side_effect = self.processes.side_effect = None
    self.metadata.return_value = self.processes.return_value = []
    self.snapshot.invalidate()
    recovered = await self.snapshot.current(self.store)
    self.assertEqual((recovered["runs"], recovered["store_error"], recovered["processes_error"]), ([], None, None))

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
