"""The dashboard's joins over recorded data: lanes and their ops, steps, run lifecycle and pods, and the Cloud Logging read."""

import asyncio
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server.dashboard import fleet, sources
from server.dashboard.__main__ import app
from server.store import InMemoryStateStore
from server.telemetry import key

# A placement record, as the API server keeps it from the scheduler's Workload, and the trainer's own record.
RECORD = {"workload": "trainer", "role": "trainer", "model_id": "r1", "owner": "r1", "pod": "trainer-0", "node": "n1", "gpu_uuids": ["GPU-1"]}
RECORD |= {"placed_at": 90, "released_at": None}
PROCESS = {
  "id": "p1",
  "pod": "trainer-0",
  "role": "trainer",
  "runtime_id": "r1",
  "started_at": 100,
  "heartbeat_at": 200,
  "phase": "ready",
  "turns": [],
}
RUN = {"run_id": "r1", "name": "Run one", "model": "Qwen/base", "runtime_id": "r1", "created_at": 0, "completed_at": None}


def state(records=(RECORD,), workers=(PROCESS,), runs=(RUN,)) -> dict:
  return {"observed_at": 200, "placements": list(records), "workers": list(workers), "runs": list(runs)}


def op(start, end, **fields):
  return {"run_id": "r1", "role": "trainer", "pod": "trainer-0", "operation": "train", "started_at": start, "at": end, **fields}


def activity(snapshot: dict, low: float, high: float, records: dict, run_id=None) -> list[dict]:
  with mock.patch.object(fleet, "operations", mock.AsyncMock(side_effect=lambda name: records.get(name, []))):
    return asyncio.run(fleet.activity(snapshot, low, high, run_id))


class LanesTest(unittest.TestCase):
  def test_a_placement_record_is_one_lane_with_its_process_phase_until_released(self):
    (lane,) = fleet.lanes(state())
    self.assertEqual(
      (lane["id"], lane["label"], lane["run_ids"], lane["phase"], lane["end"], lane["ended"]), ("trainer@90", "Run one", ["r1"], "ready", 200, False)
    )
    (lane,) = fleet.lanes(state(records=[{**RECORD, "released_at": 150}], workers=[]))
    self.assertEqual((lane["end"], lane["ended"], lane["phase"]), (150, True, None))

  def test_a_workload_placed_again_keeps_each_placement_with_its_own_process(self):
    first = {**RECORD, "placed_at": 20, "released_at": 90, "node": "n0", "gpu_uuids": ["GPU-0"]}
    old = {**PROCESS, "id": "p0", "started_at": 20, "heartbeat_at": 89, "phase": "exited", "turns": [[30, 40]]}
    previous, current = sorted(
      activity(state(records=[first, {**RECORD, "placed_at": 95}], workers=[PROCESS, old]), 0, 200, {}), key=lambda p: p["start"]
    )
    self.assertEqual((previous["node"], previous["phase"], previous["intervals"], previous["exact"]), ("n0", "exited", [[30, 40]], True))
    self.assertEqual((current["node"], current["phase"], current["intervals"]), ("n1", "ready", []))

  def test_samples_land_on_the_replica_that_answered_and_ops_are_clipped_to_the_window(self):
    sampler = {**RECORD, "workload": "sampler", "role": "sampler", "pod": "sampler-0"}
    sends = [op(120, 150, role="sampler", pod="sampler-0", operation="sample"), op(160, 170, role="sampler", pod="sampler-1", operation="sample")]
    (lane,) = activity(state(records=[sampler], workers=[]), 130, 200, {"r1": sends})
    self.assertEqual((lane["intervals"], lane["ops"]), ([[130, 150]], {"sample": [[130, 150]]}))
    self.assertNotIn("turns", lane)

  def test_a_shared_runtime_splits_by_run_and_shows_only_each_runs_ops(self):
    runs = [{**RUN, "run_id": name, "runtime_id": "base"} for name in ("r1", "r2")]
    snapshot = state(records=[{**RECORD, "model_id": "base"}], workers=[{**PROCESS, "runtime_id": "base", "turns": [[100, 200]]}], runs=runs)
    lanes = activity(snapshot, 100, 220, {"r1": [op(110, 140)], "r2": [op(160, 170, run_id="r2")]})
    self.assertEqual(
      [(p["id"], p["label"], p["intervals"], p["exact"]) for p in lanes],
      [("trainer@90:r1", "base · r1", [[110, 140]], False), ("trainer@90:r2", "base · r2", [[160, 170]], False)],
    )


class StepsTest(unittest.TestCase):
  def test_steps_group_trainer_ops_by_stamped_step_and_derive_it_for_older_records(self):
    def step_op(at, operation, step=None, **fields):
      return {"at": at, "started_at": at - 1, "operation": operation, "status": "succeeded", "elapsed_seconds": 1.0, "metrics": {}, **fields} | (
        {"step": step} if step is not None else {}
      )

    older = [
      step_op(10, "forward_backward", metrics={"loss": 2.0}),
      step_op(11, "optim_step"),
      step_op(20, "forward_backward", metrics={"loss": 4.0}),
    ]
    stamped = [step_op(30, "forward_backward", step=7, metrics={"loss": 1.0}), step_op(31, "optim_step", step=7, nonfinite=["grad_norm"])]
    found = fleet.steps([*older, {**step_op(15, "sample"), "role": "sampler"}, *stamped])
    self.assertEqual([(s["step"], s["complete"]) for s in found], [(0, True), (1, False), (7, True)])
    self.assertEqual(found[0]["ops"]["forward_backward"]["metrics"]["loss"], {"min": 2.0, "max": 2.0, "mean": 2.0})
    self.assertEqual((found[0]["started_at"], found[0]["ended_at"], found[2]["nonfinite"]), (9, 11, ["grad_norm"]))


class RunJoinTest(unittest.TestCase):
  CLUSTER = {
    "pods": [
      {"name": "orw-r2-trainer", "owner": "r2", "pool": None, "component": None, "node": "n1", "problem": None, "incidents": []},
      {"name": "orw-qwen-trainer", "owner": "qwen-qwen3-8b", "pool": None, "component": None, "node": "n1", "problem": None, "incidents": []},
      {"name": "dispatcher", "owner": None, "pool": "sampler-r2", "component": "dispatcher", "node": "n2", "problem": None, "incidents": []},
    ],
    "workloads": [{"name": "fft-r2-trainer", "model_id": "r2", "training_kind": "fft", "state": "waiting_for_placement", "node": None}],
  }

  def join(self, run_id, live=(), **meta):
    raw = json.dumps({"base_model": "Qwen/Qwen3-8B", "created_at": 1.0, "session_ids": ["s1"], **meta})
    return fleet.join(run_id, raw, {key("steps", run_id): "3"}, set(live), self.CLUSTER)

  def test_pods_join_by_owner_label_and_the_pools_dispatcher(self):
    fft = self.join("r2", fine_tuning_type="full")
    self.assertEqual(([p["name"] for p in fft["pods"]], fft["steps"], fft["lifecycle"]), (["orw-r2-trainer", "dispatcher"], 3, "waiting"))
    lora = self.join("r1", live=["s1"])
    self.assertEqual((lora["runtime_id"], lora["shared_runtime"], [p["name"] for p in lora["pods"]]), ("Qwen/Qwen3-8B", True, ["orw-qwen-trainer"]))
    self.assertEqual(self.join("r0", status="completed")["pods"], [])  # a finished run lets go of the shared runtime

  def test_lifecycle_follows_the_workers_and_the_session(self):
    placed, waiting, stopped = ({"state": state} for state in ("ready", "waiting_for_placement", "stopped"))
    cases = [
      ("placed", [placed], False, []),
      ("waiting", [placed, waiting], True, []),
      ("waiting", [], True, []),
      ("ended", [], False, []),
      ("ended", [stopped], True, []),
      ("active", [], True, None),
      ("ended", [], False, None),
    ]
    for expected, workloads, live, known in cases:
      with self.subTest(expected=expected, workloads=workloads, live=live, known=known):
        self.assertEqual(fleet.lifecycle({"status": "active", "workloads": workloads}, live, known), expected)
    self.assertEqual(fleet.lifecycle({"status": "completed", "workloads": [placed]}, True, []), "ended")


CONFIG = {"project": "proj", "location": "us-central1", "cluster": "c1", "configured": True}
LOG_RUN = {"run_id": "r1", "runtime_id": "Qwen/Qwen3-8B", "pool": "sampler-qwen"}
BOUNDS = ("2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00")


def entry(text, **extra):
  labels = {"k8s-pod/openrl_io/role": "trainer", "compute.googleapis.com/resource_name": "n1"}
  found = {"timestamp": "2026-09-10T10:30:00Z", "insertId": "i1", "logName": "projects/proj/logs/stdout", "severity": "INFO", "textPayload": text}
  return found | {"resource": {"labels": {"pod_name": "orw-a"}}, "labels": labels, **extra}


class LogsTest(unittest.TestCase):
  def setUp(self):
    self.enterContext(mock.patch.object(sources, "configuration", return_value=CONFIG))

  def test_the_filter_takes_the_runs_workers_by_owner_label_its_pool_and_its_client(self):
    text = sources.logs_filter(LOG_RUN, *BOUNDS, "UNKNOWN", "oom")
    for clause in (
      'resource.labels.cluster_name="c1"',
      'labels."k8s-pod/timeslice_io/owner"="qwen-qwen3-8b"',
      'labels."k8s-pod/app_kubernetes_io/instance"="sampler-qwen"',
    ):
      self.assertIn(clause, text)
    self.assertIn('(labels.openrl_run_id="r1" AND labels.openrl_source="client")', text)
    self.assertIn('severity="DEFAULT"', text)
    self.assertIn('textPayload:"oom" OR jsonPayload.message:"oom"', text)
    self.assertNotIn("timeslice_io/owner", sources.logs_filter(LOG_RUN, *BOUNDS, None, "", pod="orw-a"))

  def test_stderr_lines_take_the_level_they_print(self):
    for text, severity in (
      ("\rLoading weights:   0%|          | 0/427 [00:00<?, ?it/s]", "INFO"),
      ("(EngineCore pid=825) 2026-10-14 18:44:50,632 - INFO - autotuner.py:972 - flashinfer.jit", "INFO"),
      ("/app/peft/utils/save_and_load.py:356: UserWarning: Setting", "WARNING"),
      ("Traceback (most recent call last):", "ERROR"),
    ):
      record = sources.entry_record(entry(text, logName="projects/proj/logs/stderr", severity="ERROR"))
      self.assertEqual(record["severity"], severity, text)
    self.assertEqual(sources.entry_record(entry("\rFetching 4 files: 100%", severity="ERROR"))["severity"], "ERROR")  # stdout keeps its level

  def test_the_cursor_is_cloud_loggings_page_token_sent_with_the_same_query(self):
    pages = [mock.Mock(json=lambda: {"nextPageToken": "page-2"}), mock.Mock(json=lambda: {"entries": [entry("older")]})]
    with (
      mock.patch.object(sources, "send", mock.AsyncMock(side_effect=pages)) as send,
      mock.patch.object(sources, "gcp_headers", mock.AsyncMock(return_value={})),
    ):
      first = asyncio.run(sources.log_page(LOG_RUN, *BOUNDS))
      second = asyncio.run(sources.log_page(LOG_RUN, *BOUNDS, cursor=first["next_cursor"]))
    before, after = (call.kwargs["json"] for call in send.await_args_list)
    self.assertEqual((first["next_cursor"], after), ("page-2", {**before, "pageToken": "page-2"}))
    self.assertEqual((second["records"][0]["message"], second["records"][0]["role"], second["records"][0]["node"]), ("older", "trainer", "n1"))


class ApiTest(unittest.TestCase):
  def test_routes_are_read_only_and_a_run_summary_reads_the_store(self):
    store = InMemoryStateStore()
    asyncio.run(
      store.append_sample(key("operations", "r1"), {**op(10, 20), "operation": "optim_step", "status": "succeeded", "metrics": {}}, 10, None)
    )
    run = {**RUN, "pool": "sampler-r1", "lifecycle": "ended", "status": "active", "workloads": [], "pods": [], "settings": {}, "session_ids": []}
    snapshot = {**state(runs=[run]), "nodes": [], "pods": [], "errors": []}
    fleet.cache.clear()
    with (
      mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=snapshot)),
      mock.patch("server.dashboard.facts.get_state_store", return_value=store),
      mock.patch.object(fleet, "get_state_store", return_value=store),
      mock.patch.object(sources, "configuration", return_value={**CONFIG, "configured": False}),
    ):
      client = TestClient(app)
      summary = client.get("/api/v1/inspect/runs/r1/summary").json()
      self.assertEqual((summary["steps"], summary["last_op"]["operation"]), (1, "optim_step"))
      self.assertEqual(client.get("/api/v1/inspect/runs/missing/summary").status_code, 404)
      self.assertEqual(client.get("/api/v1/inspect/activity", params={"since": "not a time"}).status_code, 400)
    self.assertTrue(all(set(methods) == {"get"} for methods in client.get("/openapi.json").json()["paths"].values()))


if __name__ == "__main__":
  unittest.main()
