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


def step_event(step: int, reward: float, at_ns: int) -> dict:
  """An OTLP JSON log export as the collector forwards it, in the shape the cookbook logger emits."""
  values = [{"key": "progress/batch", "value": {"doubleValue": step}}, {"key": "env/all/reward/total", "value": {"doubleValue": reward}}]
  body = {
    "kvlistValue": {"values": [{"key": "metrics", "value": {"kvlistValue": {"values": values}}}, {"key": "step", "value": {"intValue": str(step)}}]}
  }
  record = {
    "timeUnixNano": str(at_ns),
    "eventName": "openrl.training.metrics",
    "body": body,
    "attributes": [{"key": "session.id", "value": {"stringValue": "sess-1"}}],
  }
  return {"resourceLogs": [{"resource": {"attributes": []}, "scopeLogs": [{"logRecords": [record]}]}]}


class ClientStepsTest(unittest.TestCase):
  def test_forwarded_step_events_become_the_runs_client_metrics_by_step_with_no_metrics_backend(self):
    from server import telemetry
    from server.dashboard import router

    store = InMemoryStateStore()
    for step, reward in ((0, 0.2), (1, 0.5)):
      asyncio.run(telemetry.ingest_client_steps(store, step_event(step, reward, (1000 + step) * 10**9)))
    asyncio.run(telemetry.ingest_client_steps(store, {"resourceLogs": [{"scopeLogs": [{"logRecords": [{"eventName": "other"}]}]}]}))
    run = {"run_id": "m", "session_ids": ["sess-1"]}
    with mock.patch.object(router, "get_state_store", return_value=store), mock.patch.object(sources, "prometheus_source", return_value=None):
      found = asyncio.run(router.client_steps(run, 900, 2000, ()))
    self.assertEqual({step: m["env/all/reward/total"] for step, m in found.items()}, {0: 0.2, 1: 0.5})

  def test_a_later_read_asks_prometheus_only_for_new_time_and_keeps_the_earlier_steps(self):
    from server.dashboard import router

    def gauges(*points):
      return [{"metric": {"metric": name}, "values": [[at, value] for at, value in values]} for name, values in points]

    answers = [
      gauges(("progress/batch", [(1000, 0)]), ("reward", [(1000, 0.2)])),
      gauges(("progress/batch", [(2500, 1)]), ("reward", [(2500, 0.5)])),
    ]
    run = {"run_id": "m-incremental", "session_ids": ["s1"]}
    with (
      mock.patch.object(router, "get_state_store", return_value=InMemoryStateStore()),
      mock.patch.object(sources, "prometheus_source", return_value="prom"),
      mock.patch.object(sources, "configuration", return_value={**CONFIG, "cluster": ""}),
      mock.patch.object(sources, "prom", mock.AsyncMock(side_effect=answers)) as prom,
    ):
      asyncio.run(router.client_steps(run, 900, 2000, ()))
      found = asyncio.run(router.client_steps(run, 900, 3000, ()))
    self.assertEqual([call.args[1:3] for call in prom.await_args_list], [(900, 2000), (1700, 3000)])
    self.assertEqual({step: m["reward"] for step, m in found.items()}, {0: 0.2, 1: 0.5})


class CachedTest(unittest.TestCase):
  def test_a_stale_read_is_served_while_its_refresh_runs_and_the_refresh_replaces_it(self):
    reads, release = iter(["first", "second"]), asyncio.Event()

    async def read():
      value = next(reads)
      if value == "second":
        await release.wait()
      return value

    async def scenario():
      fleet.cache.clear()
      first = await fleet.cached("k", 0, read)
      await asyncio.sleep(0.01)  # older than 0 s, so the next call starts a refresh
      during = await asyncio.wait_for(fleet.cached("k", 0, read), 1)
      release.set()
      await asyncio.sleep(0.01)
      return first, during, await fleet.cached("k", 60, read)

    self.assertEqual(asyncio.run(scenario()), ("first", "first", "second"))

  def test_a_failed_refresh_serves_the_last_good_result_and_the_next_call_retries(self):
    reads = iter(["first", ConnectionError("store offline"), "third"])

    async def read():
      if isinstance(value := next(reads), Exception):
        raise value
      return value

    async def scenario():
      fleet.cache.clear()
      first = await fleet.cached("k", 0, read)
      await asyncio.sleep(0.01)
      during = await fleet.cached("k", 0, read)  # starts the refresh that fails
      await asyncio.sleep(0.01)
      after = await fleet.cached("k", 60, read)  # inside the window, after the failure
      await asyncio.sleep(0.01)
      return first, during, after, await fleet.cached("k", 60, read)

    self.assertEqual(asyncio.run(scenario()), ("first", "first", "first", "third"))

  def test_a_failed_first_read_is_not_kept_for_the_window(self):
    reads = iter([ConnectionError("store offline"), "second"])

    async def read():
      if isinstance(value := next(reads), Exception):
        raise value
      return value

    async def scenario():
      fleet.cache.clear()
      with self.assertRaises(ConnectionError):
        await fleet.cached("k", 60, read)
      return await fleet.cached("k", 60, read)

    self.assertEqual(asyncio.run(scenario()), "second")

  def test_the_cache_keeps_only_the_most_recently_used_entries(self):
    async def scenario():
      fleet.cache.clear()
      for n in range(fleet.CACHE_ENTRIES + 8):
        await fleet.cached(f"run-{n}", 60, mock.AsyncMock(return_value=n))
        await fleet.cached("run-0", 60, mock.AsyncMock(return_value="never read"))  # used all along

    asyncio.run(scenario())
    self.assertEqual(len(fleet.cache), fleet.CACHE_ENTRIES)
    self.assertIn("run-0", fleet.cache)
    self.assertNotIn("run-1", fleet.cache)


class RunJoinTest(unittest.TestCase):
  CLUSTER = {
    "pods": [
      {"name": "orw-r2-trainer", "owner": "r2", "node": "n1", "problem": None, "incidents": []},
      {"name": "orw-qwen-trainer", "owner": "qwen-qwen3-8b", "node": "n1", "problem": None, "incidents": []},
    ],
    "workloads": [{"name": "fft-r2-trainer", "model_id": "r2", "training_kind": "fft", "state": "waiting_for_placement", "node": None}],
  }

  def join(self, run_id, live=(), **meta):
    raw = json.dumps({"base_model": "Qwen/Qwen3-8B", "created_at": 1.0, "session_ids": ["s1"], **meta})
    return fleet.join(run_id, raw, {key("steps", run_id): "3"}, set(live), self.CLUSTER)

  def test_pods_join_by_owner_label(self):
    fft = self.join("r2", fine_tuning_type="full")
    self.assertEqual(([p["name"] for p in fft["pods"]], fft["steps"], fft["lifecycle"]), (["orw-r2-trainer"], 3, "waiting"))
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
LOG_RUN = {"run_id": "r1", "runtime_id": "Qwen/Qwen3-8B"}
BOUNDS = ("2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00")


def entry(text, **extra):
  labels = {"k8s-pod/openrl_io/role": "trainer", "compute.googleapis.com/resource_name": "n1"}
  found = {"timestamp": "2026-09-10T10:30:00Z", "insertId": "i1", "logName": "projects/proj/logs/stdout", "severity": "INFO", "textPayload": text}
  return found | {"resource": {"labels": {"pod_name": "orw-a"}}, "labels": labels, **extra}


class LogsTest(unittest.TestCase):
  def setUp(self):
    self.enterContext(mock.patch.object(sources, "configuration", return_value=CONFIG))

  def test_the_filter_takes_the_runs_workers_by_owner_label_and_its_client(self):
    text = sources.logs_filter(LOG_RUN, *BOUNDS, "UNKNOWN", "oom")
    for clause in ('resource.labels.cluster_name="c1"', 'labels."k8s-pod/timeslice_io/owner"="qwen-qwen3-8b"'):
      self.assertIn(clause, text)
    self.assertIn('(labels.openrl_run_id="r1" AND labels.openrl_source="client")', text)
    self.assertIn('severity="DEFAULT"', text)
    self.assertIn('textPayload:"oom" OR jsonPayload.message:"oom"', text)
    one_pod = sources.logs_filter(LOG_RUN, *BOUNDS, None, "", pod="orw-other-run")  # any pod name in the namespace
    self.assertIn('resource.labels.pod_name="orw-other-run"', one_pod)
    self.assertIn('labels."k8s-pod/timeslice_io/owner"="qwen-qwen3-8b"', one_pod)  # still only the run's own pods

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
    run = {**RUN, "lifecycle": "ended", "status": "active", "workloads": [], "pods": [], "settings": {}, "session_ids": []}
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


class TimeSlicedGpuTest(unittest.TestCase):
  """A time-sliced trainer and sampler hold the same GPU at once."""

  def setUp(self):
    from server.dashboard import router

    self.store = InMemoryStateStore()
    asyncio.run(self.store.append_sample(key("operations", "r1"), {**op(100, 150), "operation": "optim_step", "status": "succeeded"}, 10, None))
    sampler = {**RECORD, "workload": "sampler", "role": "sampler", "pod": "sampler-0"}
    records = [RECORD, sampler, {**sampler, "workload": "sampler-1", "pod": "sampler-1", "gpu_uuids": ["GPU-2"]}]
    run = {**RUN, "lifecycle": "placed", "status": "active", "workloads": [], "pods": [], "settings": {}, "session_ids": []}
    snapshot = {**state(records=records, workers=[], runs=[run]), "nodes": [], "pods": [], "errors": []}
    busy = {"GPU-1": 40.0, "GPU-2": 80.0}

    async def query(name, *args):
      series = [{"labels": {"UUID": uuid, "Hostname": "n1", "gpu": uuid[-1]}, "points": [[200, value]]} for uuid, value in busy.items()]
      return {"name": name, "since": "a", "until": "b", "source": "prom", "series": series}

    fleet.cache.clear()
    self.enterContext(mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=snapshot)))
    self.enterContext(mock.patch("server.dashboard.facts.get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(fleet, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(router, "query", query))
    self.client = TestClient(app)

  def test_duty_of_a_gpu_shared_by_two_roles_is_reported_under_both_together(self):
    summary = self.client.get("/api/v1/inspect/runs/r1/summary").json()
    self.assertEqual(summary["gpu_busy"], {"sampler+trainer": 40.0, "sampler": 80.0})

  def test_a_gpu_whose_node_scaled_down_in_the_window_is_still_listed(self):
    from server.dashboard import router

    async def query(name, *args):
      # GPU-2's node is gone, so it has no utilization now, only its duty over the window.
      kept = {"GPU-1", "GPU-2"} if name in ("gpu_busy", "gpu_memory_mib") else {"GPU-1"}
      labels = {"gpu_util_now": lambda u: {"UUID": u, "Hostname": "n1", "gpu": "0", "modelName": "L4"}}.get(name, lambda u: {"UUID": u, "gpu": "0"})
      return {
        "name": name,
        "since": "a",
        "until": "b",
        "source": "prom",
        "series": [{"labels": labels(u), "points": [[200, 50.0]]} for u in sorted(kept)],
      }

    window = {"since": "1970-01-01T00:00:00Z", "until": "1970-01-01T00:05:00Z"}
    with mock.patch.object(router, "query", query):
      gpus = {g["uuid"]: g for g in self.client.get("/api/v1/inspect/accelerators", params=window).json()["gpus"]}
    self.assertEqual(sorted(gpus), ["GPU-1", "GPU-2"])
    self.assertEqual((gpus["GPU-2"]["busy"], gpus["GPU-2"]["node"], "util" in gpus["GPU-2"]), (50.0, "n1", False))

  def test_a_gpu_lists_every_placement_holding_it_at_once(self):
    window = {"since": "1970-01-01T00:00:00Z", "until": "1970-01-01T00:05:00Z"}
    gpus = {g["uuid"]: g for g in self.client.get("/api/v1/inspect/accelerators", params=window).json()["gpus"]}
    self.assertEqual(sorted(h["role"] for h in gpus["GPU-1"]["holders"]), ["sampler", "trainer"])
    self.assertEqual([h["pod"] for h in gpus["GPU-2"]["holders"]], ["sampler-1"])


if __name__ == "__main__":
  unittest.main()
