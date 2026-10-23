"""The dashboard's joins over recorded data: lanes and their ops, steps, run lifecycle and pods, and the Cloud Logging read."""

import asyncio
import json
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server.dashboard import fleet, router, sources
from server.dashboard.__main__ import app
from server.store import InMemoryStateStore
from server.telemetry import key

# A placement record, as the API server keeps it from the scheduler's Workload, and the trainer's own record.
RECORD = {"workload": "trainer", "role": "trainer", "model_id": "r1", "pod": "trainer-0", "node": "n1", "gpu_uuids": ["GPU-1"]}
RECORD |= {"placed_at": 90, "released_at": None}
PROCESS = {"id": "p1", "pod": "trainer-0", "runtime_id": "r1", "started_at": 100, "heartbeat_at": 200, "phase": "ready", "turns": []}
RUN = {"run_id": "r1", "name": "Run one", "model": "Qwen/base", "runtime_id": "r1", "created_at": 0, "completed_at": None}


def state(records=(RECORD,), workers=(PROCESS,), runs=(RUN,)) -> dict:
  return {"observed_at": 200, "placements": list(records), "workers": list(workers), "runs": list(runs)}


def op(start, end, **fields):
  return {"run_id": "r1", "role": "trainer", "pod": "trainer-0", "operation": "train", "started_at": start, "at": end, **fields}


def activity(snapshot: dict, low: float, high: float, records: dict) -> list[dict]:
  with mock.patch.object(fleet, "operations", mock.AsyncMock(side_effect=lambda name: records.get(name, []))):
    return asyncio.run(fleet.activity(snapshot, low, high))


class LanesTest(unittest.TestCase):
  def test_a_workload_placed_again_keeps_each_placement_with_its_own_process(self):
    first = {**RECORD, "placed_at": 20, "released_at": 90, "node": "n0", "gpu_uuids": ["GPU-0"]}
    old = {**PROCESS, "id": "p0", "started_at": 20, "heartbeat_at": 89, "phase": "exited", "turns": [[30, 40]]}
    lanes = activity(state(records=[first, {**RECORD, "placed_at": 95}], workers=[PROCESS, old]), 0, 200, {})
    found = [(p["node"], p["label"], p["phase"], p["end"], p["ended"], p["intervals"]) for p in sorted(lanes, key=lambda p: p["start"])]
    self.assertEqual(found, [("n0", "Run one", "exited", 90, True, [[30, 40]]), ("n1", "Run one", "ready", 200, False, [])])

  def test_samples_land_on_the_replica_that_answered_and_ops_are_clipped_to_the_window(self):
    sampler = {**RECORD, "workload": "sampler", "role": "sampler", "pod": "sampler-0"}
    sends = [op(120, 150, role="sampler", pod="sampler-0", operation="sample"), op(160, 170, role="sampler", pod="sampler-1", operation="sample")]
    (lane,) = activity(state(records=[sampler], workers=[]), 130, 200, {"r1": sends})
    self.assertEqual((lane["intervals"], lane["ops"]), ([[130, 150]], {"sample": [[130, 150]]}))

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
    def step_op(at, operation, **fields):
      return {"at": at, "started_at": at - 1, "operation": operation, "status": "succeeded", "elapsed_seconds": 1.0, "metrics": {}, **fields}

    older = [step_op(10, "forward_backward", metrics={"loss": 2.0}), step_op(11, "optim_step"), step_op(20, "forward_backward")]
    stamped = [step_op(30, "forward_backward", step=7), step_op(31, "optim_step", step=7, nonfinite=["grad_norm"])]
    found = fleet.steps([*older, step_op(15, "sample"), *stamped])
    self.assertEqual([(s["step"], s["complete"]) for s in found], [(0, True), (1, False), (7, True)])
    self.assertEqual(found[0]["ops"]["forward_backward"]["metrics"]["loss"], {"min": 2.0, "max": 2.0, "mean": 2.0})
    self.assertEqual((found[0]["started_at"], found[0]["ended_at"], found[2]["nonfinite"]), (9, 11, ["grad_norm"]))


class KubernetesReadTest(unittest.TestCase):
  def test_a_failed_read_names_its_source_and_a_cluster_without_the_scheduler_has_no_workloads(self):
    from kubernetes.client.exceptions import ApiException

    sources.kubernetes_api.cache_clear()
    self.addCleanup(sources.kubernetes_api.cache_clear)
    core, custom = mock.Mock(), mock.Mock()
    core.list_namespaced_pod.side_effect = ApiException(status=403, reason="Forbidden")
    core.list_node.return_value = mock.Mock(data=json.dumps({"items": [{"metadata": {"name": "n1"}, "status": {}}]}))
    custom.list_namespaced_custom_object.side_effect = ApiException(status=404)  # no scheduler CRD on a local stack
    with (
      mock.patch("kubernetes.config.load_incluster_config"),
      mock.patch("kubernetes.client.CoreV1Api", return_value=core),
      mock.patch("kubernetes.client.CustomObjectsApi", return_value=custom),
    ):
      found = asyncio.run(sources.kubernetes())
    self.assertEqual((found["pods"], [node["name"] for node in found["nodes"]], found["workloads"]), ([], ["n1"], None))
    self.assertEqual([error.split(":")[0] for error in found["errors"]], ["pods read failed"])


def reader(*values):
  """A read that returns, or raises, each value in turn; a value that is an Event is waited on before the next."""
  values = iter(values)

  async def read():
    value = next(values)
    if isinstance(value, asyncio.Event):
      await value.wait()
      value = next(values)
    if isinstance(value, Exception):
      raise value
    return value

  return read


class CachedTest(unittest.TestCase):
  def setUp(self):
    fleet.cache.clear()

  def test_a_stale_or_failed_read_is_refreshed_by_the_next_caller_behind_the_last_good_result(self):
    async def scenario():
      release = asyncio.Event()
      read = reader(ConnectionError("store offline"), "first", release, "second", ConnectionError("store offline"), "fourth")
      with self.assertRaises(ConnectionError):  # a failed first read is not kept
        await fleet.cached("k", 60, read)
      found = [await fleet.cached("k", 60, read)]
      await asyncio.sleep(0.01)  # older than 0 s, so the next call starts a refresh, which waits on release
      found.append(await asyncio.wait_for(fleet.cached("k", 0, read), 1))
      release.set()
      await asyncio.sleep(0.01)
      found.append(await fleet.cached("k", 0, read))  # the refresh landed; this call starts one that fails
      await asyncio.sleep(0.01)
      found.append(await fleet.cached("k", 60, read))  # the failure is retried, behind the last good result
      await asyncio.sleep(0.01)
      return [*found, await fleet.cached("k", 60, read)]

    self.assertEqual(asyncio.run(scenario()), ["first", "first", "second", "second", "fourth"])

  def test_a_result_far_older_than_its_window_is_not_served_after_an_idle_spell(self):
    async def scenario():
      read = reader("first", "second")
      first = await fleet.cached("k", 0, read)
      with mock.patch.object(fleet, "STALE_SECONDS", 0.005):
        await asyncio.sleep(0.02)  # idle far past the stale bound
        return first, await fleet.cached("k", 0, read)

    self.assertEqual(asyncio.run(scenario()), ("first", "second"))

  def test_the_cache_keeps_only_the_most_recently_used_entries(self):
    async def scenario():
      for n in range(fleet.CACHE_ENTRIES + 8):
        await fleet.cached(f"run-{n}", 60, reader(n))
        await fleet.cached("run-0", 60, reader("never read"))  # used all along

    asyncio.run(scenario())
    self.assertEqual(len(fleet.cache), fleet.CACHE_ENTRIES)
    self.assertIn("run-0", fleet.cache)
    self.assertNotIn("run-1", fleet.cache)


class RunJoinTest(unittest.TestCase):
  def test_pods_join_by_owner_label_and_a_finished_run_lets_go_of_a_shared_runtime(self):
    cluster = {
      "pods": [{"name": f"orw-{owner}-trainer", "owner": owner, "node": "n1"} for owner in ("r2", "qwen-qwen3-8b")],
      "workloads": [{"name": "fft-r2-trainer", "model_id": "r2", "training_kind": "fft", "state": "waiting_for_placement", "node": None}],
    }

    def join(run_id, live=(), **meta):
      raw = json.dumps({"base_model": "Qwen/Qwen3-8B", "created_at": 1.0, "session_ids": ["s1"], **meta})
      return fleet.join(run_id, raw, {key("steps", run_id): "3"}, set(live), cluster)

    fft = join("r2", fine_tuning_type="full")
    self.assertEqual(([p["name"] for p in fft["pods"]], fft["steps"], fft["lifecycle"]), (["orw-r2-trainer"], 3, "waiting"))
    lora = join("r1", live=["s1"])
    self.assertEqual(
      (lora["runtime_id"], lora["shared_runtime"], [p["name"] for p in lora["pods"]]), ("Qwen/Qwen3-8B", True, ["orw-qwen-qwen3-8b-trainer"])
    )
    self.assertEqual(join("r0", status="completed")["pods"], [])

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

  def test_the_filter_takes_only_the_runs_own_workers_and_its_client(self):
    owner = 'labels."k8s-pod/timeslice_io/owner"="qwen-qwen3-8b"'
    text = sources.logs_filter(LOG_RUN, *BOUNDS, None, "")
    self.assertIn(f'({owner} OR (labels.openrl_run_id="r1" AND labels.openrl_source="client"))', text)
    self.assertIn('resource.labels.cluster_name="c1"', text)
    one_pod = sources.logs_filter(LOG_RUN, *BOUNDS, None, "", pod="orw-other-run")  # any pod name in the namespace
    self.assertIn(f'{owner} AND resource.labels.pod_name="orw-other-run"', one_pod)  # still only the run's own pods

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


class ApiTest(unittest.TestCase):
  """Routes over a snapshot with one run, a trainer and two samplers time-sliced onto its GPU, and a third sampler."""

  def setUp(self):
    self.store = InMemoryStateStore()
    asyncio.run(self.store.append_sample(key("operations", "r1"), {**op(100, 150), "operation": "optim_step", "status": "succeeded"}, 10, None))
    sampler = {**RECORD, "workload": "sampler", "role": "sampler", "pod": "sampler-0"}
    records = [RECORD, sampler, {**sampler, "workload": "sampler-1", "pod": "sampler-1", "gpu_uuids": ["GPU-2"]}]
    run = {**RUN, "lifecycle": "placed", "status": "active", "workloads": [], "pods": [], "settings": {}, "session_ids": []}
    snapshot = {**state(records=records, workers=[], runs=[run]), "nodes": [], "pods": [], "errors": []}
    fleet.cache.clear()
    self.enterContext(mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=snapshot)))
    self.enterContext(mock.patch("server.dashboard.facts.get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(fleet, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.object(router, "query", self.query))
    self.client = TestClient(app)

  async def query(self, name, *args):
    # GPU-2's node scaled down in the window: it has duty over the window but no utilization now.
    labels = {"GPU-1": {"UUID": "GPU-1", "Hostname": "n1", "gpu": "0"}, "GPU-2": {"UUID": "GPU-2", "gpu": "1"}}
    busy = {"GPU-1": 40.0, "GPU-2": 80.0} if name in ("gpu_busy", "gpu_memory_mib") else {"GPU-1": 50.0}
    series = [{"labels": labels[uuid], "points": [[200, value]]} for uuid, value in busy.items()]
    return {"name": name, "since": "a", "until": "b", "source": "prom", "series": series}

  def test_routes_are_read_only_and_a_run_summary_reads_the_store(self):
    summary = self.client.get("/api/v1/inspect/runs/r1/summary").json()
    self.assertEqual((summary["steps"], summary["last_op"]["operation"]), (1, "optim_step"))
    self.assertEqual(self.client.get("/api/v1/inspect/runs/missing/summary").status_code, 404)
    self.assertEqual(self.client.get("/api/v1/inspect/activity", params={"since": "not a time"}).status_code, 400)
    self.assertTrue(all(set(methods) == {"get"} for methods in self.client.get("/openapi.json").json()["paths"].values()))

  def test_duty_of_a_gpu_shared_by_two_roles_is_reported_under_both_together(self):
    summary = self.client.get("/api/v1/inspect/runs/r1/summary").json()
    self.assertEqual(summary["gpu_busy"], {"sampler+trainer": 40.0, "sampler": 80.0})

  def test_a_gpu_lists_every_placement_holding_it_and_one_whose_node_scaled_down_is_still_listed(self):
    window = {"since": "1970-01-01T00:00:00Z", "until": "1970-01-01T00:05:00Z"}
    gpus = {g["uuid"]: g for g in self.client.get("/api/v1/inspect/accelerators", params=window).json()["gpus"]}
    self.assertEqual(sorted(h["role"] for h in gpus["GPU-1"]["holders"]), ["sampler", "trainer"])
    self.assertEqual([h["pod"] for h in gpus["GPU-2"]["holders"]], ["sampler-1"])
    self.assertEqual((gpus["GPU-2"]["busy"], gpus["GPU-2"]["node"], "util" in gpus["GPU-2"]), (80.0, "n1", False))


if __name__ == "__main__":
  unittest.main()
