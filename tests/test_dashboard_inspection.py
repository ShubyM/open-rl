import unittest

from server.dashboard import inspection


class InspectionIncidentsTest(unittest.TestCase):
  def test_kubelet_oom_and_container_termination_are_one_incident(self):
    run = {
      "pods": [
        {
          "name": "trainer",
          "events": [
            {"reason": "OOMKilled", "last_seen_at": "1970-01-01T00:00:12.5Z"},
            {"reason": "FailedScheduling", "first_seen_at": "1970-01-01T00:00:01Z", "last_seen_at": "1970-01-01T00:00:20Z", "count": 5},
          ],
          "containers": [
            {
              "name": "worker",
              "state": "running",
              "restart_count": 2,
              "started_at": "1970-01-01T00:00:30Z",
              "last_termination": {"reason": "OOMKilled", "finished_at": "1970-01-01T00:00:12Z", "exit_code": 137},
            }
          ],
        }
      ]
    }
    result = inspection.incidents(run, 10, 30)
    self.assertEqual(
      [(item["reason"], item["at"], item["tone"]) for item in result],
      [("WorkerRestarted", 30, "warning"), ("FailedScheduling", 20, "warning"), ("OOMKilled", 12, "error")],
    )
    self.assertEqual((result[0]["restart_count"], result[1]["count"], result[2]["container"]), (2, 5, "worker"))
    self.assertEqual(result[2]["message"], "worker exited with code 137")

  def test_distinct_incidents_are_selected_newest_first_after_time_filtering(self):
    events = [
      {"reason": reason, "last_seen_at": f"1970-01-01T00:00:{second}Z"}
      for reason, second in [
        ("Evicted", 25),
        ("Evicted", 24),
        ("FailedMount", 23),
        ("FailedAttachVolume", 22),
        ("Unschedulable", 21),
        ("Started", 29),
      ]
    ]
    run = {"pods": [{"name": "trainer", "events": [*events, {"reason": "OOMKilled"}, {"reason": "OOMKilled", "last_seen_at": "invalid"}]}]}
    self.assertEqual([event["at"] for event in inspection.incidents(run, 20, 30)], [25, 23, 22, 21])
    self.assertEqual([event["at"] for event in inspection.incidents(run, 20, 24)], [24, 23, 22, 21])
    self.assertEqual(inspection.incidents({"pods": [{"name": "trainer", "problem": "CrashLoopBackOff", "restarts": 5}]}, 0, 30), [])

  def test_backoff_requires_a_timestamped_warning_about_a_restarting_worker(self):
    event = {"reason": "BackOff", "type": "Warning", "last_seen_at": "1970-01-01T00:00:10Z", "message": "Back-off pulling image"}
    run = {
      "pods": [
        {"name": "image-pull", "events": [event]},
        {"name": "restarted", "restarts": 2, "events": [event, {**event, "type": "Normal"}]},
        {"name": "crashing", "problem": "CrashLoopBackOff: failed", "events": [event]},
        {"name": "message", "events": [{**event, "message": "Back-off restarting failed container worker"}]},
      ]
    }
    result = inspection.incidents(run, 0, 20)
    self.assertEqual({event["pod"] for event in result}, {"restarted", "crashing", "message"})
    self.assertTrue(all(event["tone"] == "warning" for event in result))


# A placement record, as the API server keeps it from the scheduler's Workload.
RECORD = {
  "workload": "trainer",
  "role": "trainer",
  "model_id": "r1",
  "owner": "r1",
  "pod": "trainer-0",
  "node": "n1",
  "gpu_uuids": ["GPU-1"],
  "placed_at": 90,
  "released_at": None,
}
# The record the trainer process keeps about itself.
PROCESS = {
  "id": "p1",
  "pod": "trainer-0",
  "pod_uid": "p1",
  "node": "n1",
  "role": "trainer",
  "runtime_id": "r1",
  "gpus": ["GPU-1"],
  "started_at": 100,
  "heartbeat_at": 200,
  "phase": "ready",
  "turns": [],
  "outstanding": [],
}
RUN = {
  "run_id": "r1",
  "name": "Run one",
  "model": "Qwen/base",
  "runtime_id": "r1",
  "pool": "sampler-r1",
  "created_at": 0,
  "completed_at": None,
  "shared_runtime": False,
  "status": "active",
  "lifecycle": "placed",
}


def state(records=None, processes=None, runs=None):
  snapshot = {
    "observed_at": "1970-01-01T00:03:20+00:00",
    "cluster": {
      "available": True,
      "scheduler": {"available": True},
      "pods": [],
      "nodes": [{"name": "n1", "devices": [{"id": "GPU-1", "uuid": "GPU-1", "name": "gpu-0"}]}],
    },
    "placement_records": [RECORD] if records is None else records,
    "processes": [PROCESS] if processes is None else processes,
    "runs": [RUN] if runs is None else runs,
  }
  return {**snapshot, "placements": [lane for lane in inspection.lanes(snapshot) if not lane["ended"]]}


def operation(start, end, **fields):
  return {"run_id": "r1", "role": "trainer", "pod": "trainer-0", "pod_uid": "p1", "operation": "train", "started_at": start, "at": end, **fields}


class InspectionActivityTest(unittest.TestCase):
  def test_a_placement_record_is_one_lane_with_its_process_phase(self):
    (lane,) = inspection.lanes(state())
    self.assertEqual((lane["id"], lane["name"], lane["label"], lane["run_ids"], lane["phase"]), ("trainer@90", "trainer", "Run one", ["r1"], "ready"))
    self.assertEqual((lane["start"], lane["end"], lane["ended"], lane["devices"], lane["device_count"]), (90, 200, False, ["GPU-1"], 1))

  def test_a_released_placement_ends_at_its_release_and_a_new_one_needs_no_process(self):
    (lane,) = inspection.lanes(state(records=[{**RECORD, "released_at": 150}]))
    self.assertEqual((lane["end"], lane["ended"]), (150, True))
    (lane,) = inspection.lanes(state(processes=[]))
    self.assertEqual((lane["start"], lane["end"], lane["ended"], lane["phase"]), (90, 200, False, None))

  def test_a_workload_placed_again_keeps_each_placement_with_its_own_process(self):
    first = {**RECORD, "placed_at": 20, "released_at": 90, "node": "n0", "gpu_uuids": ["GPU-0"]}
    old = {**PROCESS, "id": "p0", "pod_uid": "p0", "node": "n0", "started_at": 20, "heartbeat_at": 89, "phase": "exited", "turns": [[30, 40]]}
    lanes = inspection.activity(state(records=[first, {**RECORD, "placed_at": 95}], processes=[PROCESS, old]), 0, 200, {})["placements"]
    previous, current = sorted(lanes, key=lambda lane: lane["start"])
    self.assertEqual((previous["node"], previous["devices"], previous["phase"], previous["intervals"]), ("n0", ["GPU-0"], "exited", [[30, 40]]))
    self.assertEqual((current["node"], current["phase"], current["intervals"], current["ended"]), ("n1", "ready", [], False))

  def test_sample_ops_land_on_the_replica_that_answered(self):
    sampler = {**RECORD, "workload": "sampler", "role": "sampler", "pod": "sampler-0"}
    sample = {"run_id": "r1", "role": "sampler", "pod": "sampler-0", "operation": "sample", "started_at": 120, "at": 150}
    (lane,) = inspection.activity(state(records=[sampler], processes=[]), 100, 200, {"r1": {"samples": [sample]}})["placements"]
    self.assertEqual((lane["intervals"], list(lane["ops"])), ([[120, 150]], ["sample"]))

  def test_a_shared_runtime_splits_by_run_and_shows_only_each_runs_ops(self):
    shared = {**PROCESS, "runtime_id": "base", "turns": [[100, 200]]}
    runs = [{**RUN, "run_id": identity, "runtime_id": "base", "shared_runtime": True} for identity in ("r1", "r2")]
    operations = {"r1": {"samples": [operation(110, 140)]}, "r2": {"samples": [operation(160, 170, run_id="r2")]}}
    snapshot = state(records=[{**RECORD, "model_id": "base"}], processes=[shared], runs=runs)
    lanes = inspection.activity(snapshot, 100, 220, operations)["placements"]
    self.assertEqual(
      [(p["id"], p["run_ids"], p["label"], p["intervals"], p["exact"]) for p in lanes],
      [("trainer@90:r1", ["r1"], "base · r1", [[110, 140]], False), ("trainer@90:r2", ["r2"], "base · r2", [[160, 170]], False)],
    )
    (lane,) = inspection.activity(snapshot, 100, 220, operations, run_id="r1")["placements"]
    self.assertEqual(lane["allocation_id"], "trainer@90")

  def test_charts_get_every_trainer_op_and_the_slowest_sampler_request_per_bucket(self):
    trainer = [operation(t, t + 5, metrics={"loss": 1.0}) for t in range(0, 1000, 10)]
    sampler = [operation(t / 10, t / 10 + 0.05, role="sampler", operation="sample", elapsed_seconds=(t % 7) + 1.0) for t in range(10000)]
    charted = inspection.chart_samples([*trainer, *sampler, operation(5000, 5001)], 0, 1000)
    self.assertEqual(sum(s["role"] == "trainer" for s in charted), len(trainer))
    slow = [s for s in charted if s["role"] == "sampler"]
    self.assertLessEqual(len(slow), inspection.CHART_POINTS)
    self.assertEqual(max(s["elapsed_seconds"] for s in slow), 7.0)
    self.assertTrue(all(0 <= s["at"] <= 1000 for s in charted))

  def test_a_dedicated_lane_shows_gpu_turns_inside_the_window(self):
    record = {**PROCESS, "turns": [[10, 20], [110, 150]]}
    result = inspection.activity(state(processes=[record]), 120, 190, {})["placements"][0]
    self.assertEqual((result["intervals"], result["ops"], result["exact"]), ([[120, 150]], {}, True))
    self.assertNotIn("turns", result)
    result = inspection.activity(state(processes=[record]), 160, 190, {"r1": {"samples": [operation(170, 180)]}})["placements"][0]
    self.assertEqual((result["intervals"], result["exact"]), ([[170, 180]], False))

  def test_source_errors_and_partial_records_stay_distinct(self):
    operations = {"r1": {"samples": [operation(120, 130)], "partial": "Skipped 1 invalid operation records"}}
    result = inspection.activity(state(), 100, 200, operations)["placements"][0]
    self.assertEqual((result["intervals"], result["error"], result["partial"]), ([[120, 130]], None, operations["r1"]["partial"]))

  def test_seeded_interval_fuzz_preserves_clipping_and_pod_attribution(self):
    import random

    rng = random.Random(20261015)
    for case in range(1000):
      records, expected = [], set()
      for _ in range(rng.randrange(25)):
        low, high = sorted(rng.sample(range(80, 230), 2))
        pod = rng.choice(["trainer-0", "old-pod"])
        records.append(operation(low, high, pod=pod))
        if pod == "trainer-0":
          expected.update(range(max(low, 120), min(high, 190)))
      rng.shuffle(records)
      lane = inspection.activity(state(), 120, 190, {"r1": {"samples": records}})["placements"][0]
      actual = {tick for low, high in lane["intervals"] for tick in range(low, high)}
      self.assertEqual(actual, expected, f"case {case}")
      self.assertEqual(lane["ops"].get("train", []), lane["intervals"])
      self.assertFalse(lane["exact"])
      self.assertTrue(all(120 <= low < high <= 190 for low, high in lane["intervals"]))
      self.assertTrue(all(a[1] < b[0] for a, b in zip(lane["intervals"], lane["intervals"][1:])))


class StepsTest(unittest.TestCase):
  def test_steps_group_trainer_ops_by_stamped_step_and_derive_it_for_older_records(self):
    def op(at, operation, step=None, **fields):
      return {"at": at, "started_at": at - 1, "operation": operation, "status": "succeeded", "elapsed_seconds": 1.0, "metrics": {}, **fields} | (
        {"step": step} if step is not None else {}
      )

    older = [op(10, "forward_backward", metrics={"loss": 2.0}), op(11, "optim_step"), op(20, "forward_backward", metrics={"loss": 4.0})]
    stamped = [op(30, "forward_backward", step=7, metrics={"loss": 1.0}), op(31, "optim_step", step=7, nonfinite=["grad_norm"])]
    sample = {**op(15, "sample"), "role": "sampler"}
    found = inspection.steps([*stamped, sample, *older])
    self.assertEqual([s["step"] for s in found], [0, 1, 7])
    self.assertEqual(found[0]["ops"]["forward_backward"]["metrics"]["loss"], {"min": 2.0, "max": 2.0, "mean": 2.0})
    self.assertEqual((found[0]["started_at"], found[0]["ended_at"]), (9, 11))
    self.assertEqual(found[2]["nonfinite"], ["grad_norm"])
    self.assertNotIn("sample", {name for s in found for name in s["ops"]})
