import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from server.dashboard import router, snapshot
from server.model_metadata import TrainingModelMetadata
from server.session_registry import SessionRegistry
from server.store import InMemoryStateStore


def worker(name: str, owner: str, model_id: str, state: str, node: str | None) -> dict:
  return {"workload": name, "role": "trainer", "owner": owner, "model_id": model_id, "state": state, "detail": "", "node": node, "age_seconds": 10}


STATE = {
  "available": True,
  "nodes": [],
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

  def test_the_pool_dispatcher_pod_belongs_to_the_runs_it_samples_for(self) -> None:
    state = copy.deepcopy(STATE)
    pool = TrainingModelMetadata.model_validate(RECORDS["r1"]).pool("r1")
    state["pods"].append(
      {**STATE["pods"][0], "name": "dispatcher", "uid": "p3", "owner_uids": [], "worker": None, "component": "dispatcher", "pool": pool}
    )
    runs = {r["run_id"]: r for r in snapshot.join_runs(METADATA, state)}
    self.assertEqual(runs["r1"]["pool"], pool)
    self.assertEqual({p["name"]: p["role"] for p in runs["r1"]["pods"]}, {"orw-lora-trainer": "trainer", "dispatcher": "dispatcher"})
    self.assertNotIn("dispatcher", [p["name"] for p in runs["r2"]["pods"]])

  def test_a_finished_lora_run_lets_go_of_the_shared_runtime(self) -> None:
    done = {**METADATA[0], "model_id": "r0", "status": "completed", "created_at": 0}
    runs = {r["run_id"]: r for r in snapshot.join_runs([*METADATA, done], STATE)}
    self.assertEqual((runs["r0"]["pods"], runs["r0"]["workloads"]), ([], []))


RUN = {"run_id": "r1", "runtime_id": "base"}
NOW = 10_000


def process(*groups, heartbeat_at=NOW, phase="ready"):
  outstanding = [{"op": op, "run_id": run_id, "count": 1, "since": at} for op, run_id, at in groups]
  return {"id": "p", "pod": "trainer-0", "runtime_id": "base", "heartbeat_at": heartbeat_at, "phase": phase, "outstanding": outstanding}


class OutstandingTest(unittest.TestCase):
  """What request-ledger and progress read: the run's unanswered requests and whether their holder still reports."""

  def test_a_holder_that_stopped_reporting_is_not_alive(self):
    live = snapshot.outstanding(RUN, [process(("forward_backward", "r1", NOW - 60))], NOW)
    self.assertEqual([(o["pod"], o["process_alive"], o["age_seconds"]) for o in live], [("trainer-0", True, 60)])
    for gone in (process(("optim_step", "r1", NOW - 60), heartbeat_at=NOW - 120), process(("optim_step", "r1", NOW - 60), phase="exited")):
      self.assertIs(snapshot.outstanding(RUN, [gone], NOW)[0]["process_alive"], False)

  def test_only_this_runs_requests_on_its_runtime_count(self):
    other = process(("sample", "r2", NOW - 5000), heartbeat_at=0)
    elsewhere = {**process(("sample", "r1", NOW - 5000)), "runtime_id": "other"}
    self.assertEqual(snapshot.outstanding(RUN, [other, elsewhere], NOW), [])


class LifecycleTest(unittest.IsolatedAsyncioTestCase):
  """The label the pages give a run, and whether it counts as finished, before its checks say whether it is healthy."""

  def test_lifecycle_follows_the_workers_and_the_session(self) -> None:
    placed, waiting = STATE["scheduler"]["workloads"]
    stopped = {**placed, "worker": {**placed["worker"], "state": "stopped"}}
    scheduler, unreadable, absent = STATE["scheduler"], {"installed": True, "available": False}, {"installed": False, "available": False}
    cases = [
      ("placed", [placed], True, scheduler),
      ("waiting", [placed, waiting], True, scheduler),
      ("waiting", [], True, scheduler),
      ("ended", [], False, scheduler),
      ("ended", [stopped], True, scheduler),
      ("active", [], False, unreadable),
      ("active", [], True, absent),
      ("ended", [], False, absent),
    ]
    for expected, workloads, live, source in cases:
      with self.subTest(expected=expected, workloads=len(workloads), live=live, scheduler=source):
        self.assertEqual(snapshot.lifecycle({"status": "active", "workloads": workloads}, live, source), expected)
    self.assertEqual(snapshot.lifecycle({"status": "completed", "workloads": [placed]}, True, scheduler), "ended")

  async def test_the_session_decides_for_a_run_without_workers(self) -> None:
    store = InMemoryStateStore()
    await SessionRegistry(store).attach("s1", "owner")
    state = {**STATE, "pods": [], "scheduler": {**STATE["scheduler"], "workloads": []}}
    runs = snapshot.join_runs([{**METADATA[2], "session_ids": ["s1"]}, {**METADATA[2], "model_id": "gone", "session_ids": ["s2"]}], state)
    for run in runs:
      await snapshot.enrich(store, run, [], state["scheduler"], NOW)
    self.assertEqual({r["run_id"]: r["lifecycle"] for r in runs}, {"r2": "waiting", "gone": "ended"})


def held(state: dict) -> list[bool | None]:
  return [o["process_alive"] for run in state["runs"] for o in (run.get("progress") or {}).get("outstanding", [])]


class SnapshotRecoveryTest(unittest.IsolatedAsyncioTestCase):
  async def asyncSetUp(self) -> None:
    self.snapshot = snapshot.Snapshot()
    self.store = InMemoryStateStore()
    self.metadata = self.enterContext(patch.object(snapshot, "metadata_of", AsyncMock(return_value=copy.deepcopy(METADATA))))
    self.cluster = self.enterContext(patch.object(snapshot.sources, "read_cluster", return_value=STATE))
    self.processes = self.enterContext(patch.object(snapshot, "processes_of", AsyncMock(return_value=[{"id": "past"}])))
    self.gpus = self.enterContext(patch.object(snapshot.metrics, "node_gpus", AsyncMock(return_value=None)))
    self.enterContext(patch.object(router, "snapshot", self.snapshot))
    self.enterContext(patch.object(router, "get_state_store", return_value=self.store))

  async def test_temporary_source_errors_keep_the_last_records_and_say_so(self) -> None:
    first = await self.snapshot.current(self.store)
    self.metadata.side_effect = self.processes.side_effect = TimeoutError()
    self.snapshot.invalidate()
    stale = await self.snapshot.current(self.store)
    # A brief outage keeps the last records and says nothing; everything else still refreshes.
    self.assertEqual([r["run_id"] for r in stale["runs"]], [r["run_id"] for r in first["runs"]])
    self.assertEqual(stale["processes"], first["processes"])
    self.assertEqual((stale["store_error"], stale["processes_error"], stale["error"]), (None, None, None))
    self.assertGreater(stale["observed_at"], first["observed_at"])
    with patch.object(snapshot, "GRACE_SECONDS", 0):
      self.snapshot.invalidate()
      lasting = await self.snapshot.current(self.store)
    self.assertTrue(lasting["store_error"] and lasting["processes_error"] and lasting["error"])
    self.assertEqual([r["run_id"] for r in lasting["runs"]], [r["run_id"] for r in first["runs"]])
    self.metadata.side_effect = self.processes.side_effect = None
    self.metadata.return_value = self.processes.return_value = []
    self.snapshot.invalidate()
    recovered = await self.snapshot.current(self.store)
    self.assertEqual((recovered["runs"], recovered["store_error"], recovered["processes_error"]), ([], None, None))

  async def test_a_failed_process_read_cannot_age_live_work_into_dead_work(self) -> None:
    self.processes.return_value = [
      {
        "id": "worker",
        "runtime_id": "Qwen/Qwen3-8B",
        "role": "trainer",
        "heartbeat_at": 1000,
        "started_at": 900,
        "phase": "on_gpu",
        "outstanding": [{"run_id": "r1", "since": 995, "op": "forward_backward", "count": 1}],
      }
    ]
    with patch.object(snapshot.time, "time", return_value=1000):
      first = await self.snapshot.current(self.store)
    self.processes.side_effect = TimeoutError()
    self.snapshot.invalidate()
    with patch.object(snapshot.time, "time", return_value=1100):
      stale = await self.snapshot.current(self.store)
    # The last records are aged as of their read, so live work stays live while the read is failing.
    self.assertGreater(stale["observed_at"], first["observed_at"])
    self.assertIsNone(stale["processes_error"])
    self.assertEqual((held(first), held(stale)), ([True], [True]))
    with patch.object(snapshot.time, "time", return_value=1200), patch.object(snapshot, "GRACE_SECONDS", 0):
      self.snapshot.invalidate()
      lasting = await self.snapshot.current(self.store)
    self.assertTrue(lasting["processes_error"])
    self.assertNotIn(False, held(lasting))

  async def test_cluster_failure_preserves_placements_and_reports_the_error(self) -> None:
    first = await self.snapshot.current(self.store)
    self.cluster.return_value = snapshot.sources.cluster_unavailable("openrl-system", "Kubernetes offline")
    self.snapshot.invalidate()
    stale = await self.snapshot.current(self.store)
    self.assertGreater(stale["observed_at"], first["observed_at"])
    self.assertEqual(stale["placements"], first["placements"])
    self.assertEqual(stale["cluster"]["pods"], first["cluster"]["pods"])
    self.assertIsNone(stale["error"])
    with patch.object(snapshot, "GRACE_SECONDS", 0):
      self.snapshot.invalidate()
      lasting = await self.snapshot.current(self.store)
    self.assertEqual(lasting["placements"], first["placements"])
    self.assertIn("Kubernetes offline", lasting["error"])
    self.cluster.return_value = STATE
    self.snapshot.invalidate()
    self.assertIsNone((await self.snapshot.current(self.store))["cluster"].get("error"))

  async def test_without_kubernetes_runs_and_processes_refresh_then_join_when_it_appears(self) -> None:
    self.cluster.return_value = snapshot.sources.cluster_unavailable("openrl-system", "No cluster configured")
    with patch.object(snapshot.time, "time", return_value=1000):
      first = await self.snapshot.current(self.store)
    self.assertIsNone(first["error"])
    self.metadata.return_value = [{**row, "status": "completed"} for row in METADATA] + [{**METADATA[0], "model_id": "new-run"}]
    self.processes.return_value = [{"id": "new-process"}]
    await self.store.set_value("open_rl:model_meta:new-run", json.dumps(RECORDS["r1"]))
    with patch.object(snapshot.time, "time", return_value=1001):
      updated, run = await router.run_of("new-run")
    self.assertEqual(run["run_id"], "new-run")
    self.assertEqual({r["lifecycle"] for r in updated["runs"] if r["run_id"] != "new-run"}, {"ended"})
    self.assertEqual(updated["processes"], [{"id": "new-process"}])
    self.assertNotEqual(updated["observed_at"], first["observed_at"])
    self.assertFalse(updated["cluster"]["available"])
    self.assertIsNone(updated.get("error"))
    self.metadata.return_value[-1]["steps"] = 3
    self.processes.return_value = []
    with patch.object(snapshot, "GRACE_SECONDS", 0):
      self.snapshot.invalidate()
      later = await self.snapshot.current(self.store)
    self.assertIsNone(later["error"])
    self.assertIn("No cluster configured", later["cluster"]["error"])
    self.assertEqual(later["processes"], [])
    self.assertEqual(next(r["steps"] for r in later["runs"] if r["run_id"] == "new-run"), 3)
    self.cluster.return_value = STATE
    self.snapshot.invalidate()
    recovered = await self.snapshot.current(self.store)
    run = next(r for r in recovered["runs"] if r["run_id"] == "new-run")
    self.assertTrue(recovered["cluster"]["available"])
    self.assertEqual([pod["name"] for pod in run["pods"]], ["orw-lora-trainer"])

  async def test_inventory_placements_are_the_open_placement_records(self) -> None:
    record = {"workload": "lora-qwen-0-trainer", "role": "trainer", "model_id": "Qwen/Qwen3-8B", "owner": "qwen-qwen3-8b", "pod": "orw-lora-trainer"}
    record |= {"node": "n1", "gpu_uuids": ["GPU-1"], "placed_at": 5.0, "released_at": None}
    for placement in ({**record, "placed_at": 1.0, "released_at": 4.0}, record):
      await self.store.append_sample(snapshot.telemetry.placements_key("Qwen/Qwen3-8B"), placement, 10, None)
    # The runtime is shared by r1 and r3, so its one open placement is a lane for each.
    placements = (await self.snapshot.current(self.store))["placements"]
    self.assertEqual(
      [(p["run_ids"], p["allocation_id"], p["devices"], p["start"]) for p in placements],
      [(["r1"], "lora-qwen-0-trainer@5", ["GPU-1"], 5.0), (["r3"], "lora-qwen-0-trainer@5", ["GPU-1"], 5.0)],
    )

  async def test_a_gpu_inventory_failure_keeps_each_nodes_last_gpus(self) -> None:
    gpus = {"n1": [{"id": "GPU-1", "uuid": "GPU-1", "name": "gpu-0"}]}
    self.cluster.return_value = {**copy.deepcopy(STATE), "nodes": [{"name": "n1", "gpu_capacity": 0, "devices": []}]}
    self.gpus.return_value = gpus
    first = await self.snapshot.current(self.store)
    self.assertEqual((first["cluster"]["nodes"][0]["devices"], first["cluster"]["nodes"][0]["gpu_capacity"]), (gpus["n1"], 1))
    self.gpus.side_effect = TimeoutError()
    self.snapshot.invalidate()
    stale = await self.snapshot.current(self.store)
    self.assertEqual((stale["cluster"]["nodes"], stale["error"]), (first["cluster"]["nodes"], None))
    with patch.object(snapshot, "GRACE_SECONDS", 0):
      self.snapshot.invalidate()
      lasting = await self.snapshot.current(self.store)
    self.assertEqual(lasting["cluster"]["nodes"], first["cluster"]["nodes"])
    self.assertIn("GPU inventory unavailable", lasting["error"])

  async def test_cancelled_request_does_not_restart_the_shared_source_read(self) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_metadata(store):
      entered.set()
      await release.wait()
      return copy.deepcopy(METADATA)

    self.metadata.side_effect = slow_metadata
    first = asyncio.create_task(self.snapshot.current(self.store))
    await entered.wait()
    first.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await first
    second = asyncio.create_task(self.snapshot.current(self.store))
    await asyncio.sleep(0)
    release.set()
    result = await second
    self.assertEqual(len(result["runs"]), 3)
    self.metadata.assert_awaited_once()
    self.cluster.assert_called_once()

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
