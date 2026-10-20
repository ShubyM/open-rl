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


PLACEMENT = {
  "id": "w1",
  "name": "trainer",
  "label": "Run one",
  "run_ids": ["r1"],
  "runtime_id": "r1",
  "node": "n1",
  "pod": "trainer-0",
  "pod_uid": "p1",
  "role": "trainer",
  "devices": ["gpu-0"],
  "device_uuids": {"gpu-0": "GPU-1"},
  "device_count": 1,
  "created_at": "1970-01-01T00:01:30+00:00",
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
RUN = {"run_id": "r1", "name": "Run one", "model": "Qwen/base", "runtime_id": "r1", "created_at": 0, "completed_at": None, "shared_runtime": False}


def state(live=None, processes=None, runs=None):
  return {
    "observed_at": "1970-01-01T00:03:20+00:00",
    "cluster": {"available": True, "scheduler": {"available": True}, "nodes": [{"name": "n1", "devices": [{"id": "gpu-0", "uuid": "GPU-1"}]}]},
    "placements": [PLACEMENT] if live is None else live,
    "processes": [PROCESS] if processes is None else processes,
    "runs": [RUN] if runs is None else runs,
  }


def operation(start, end, **fields):
  return {"run_id": "r1", "role": "trainer", "pod": "trainer-0", "pod_uid": "p1", "operation": "train", "started_at": start, "at": end, **fields}


class InspectionActivityTest(unittest.TestCase):
  def test_a_reporting_process_is_one_lane_joined_to_its_placement(self):
    (lane,) = inspection.segments(state())
    self.assertEqual((lane["id"], lane["name"], lane["label"], lane["run_ids"]), ("p1", "trainer", "Run one", ["r1"]))
    self.assertEqual((lane["start"], lane["end"], lane["ended"], lane["devices"]), (100, 200, False, ["gpu-0"]))

  def test_a_process_that_exits_or_stops_reporting_ends_at_its_last_heartbeat(self):
    for record in ({**PROCESS, "heartbeat_at": 150}, {**PROCESS, "phase": "exited"}):
      with self.subTest(phase=record["phase"]):
        (lane,) = inspection.segments(state(live=[], processes=[record]))
        self.assertEqual((lane["end"], lane["ended"]), (record["heartbeat_at"], True))
        self.assertEqual((lane["devices"], lane["device_uuids"], lane["device_count"]), (["gpu-0"], {"gpu-0": "GPU-1"}, 1))

  def test_a_placed_workload_is_a_lane_before_its_process_reports(self):
    (lane,) = inspection.segments(state(processes=[]))
    self.assertEqual((lane["id"], lane["start"], lane["end"], lane["ended"]), ("w1", 90, 200, False))

  def test_a_replaced_pod_keeps_both_processes(self):
    old = {**PROCESS, "id": "p0", "pod_uid": "p0", "started_at": 20, "heartbeat_at": 90}
    current, previous = inspection.segments(state(processes=[PROCESS, old]))
    self.assertEqual((previous["id"], previous["start"], previous["end"], previous["ended"]), ("p0", 20, 90, True))
    self.assertEqual((current["id"], current["ended"]), ("p1", False))

  def test_a_container_restarted_in_place_keeps_both_processes(self):
    restarted = {**PROCESS, "id": "second", "started_at": 160}
    current, previous = inspection.segments(state(processes=[restarted, {**PROCESS, "id": "first", "heartbeat_at": 150}]))
    self.assertEqual((previous["id"], previous["end"], previous["ended"]), ("first", 150, True))
    self.assertEqual((current["id"], current["start"], current["ended"]), ("second", 160, False))

  def test_dispatcher_sample_ops_land_on_the_replica_that_answered(self):
    sampler = {**PLACEMENT, "id": "s1", "name": "sampler", "role": "sampler", "pod": "sampler-0", "pod_uid": "u-s"}
    sample = {"run_id": "r1", "role": "sampler", "pod": "sampler-0", "operation": "sample", "started_at": 120, "at": 150}
    (lane,) = inspection.activity(state(live=[sampler], processes=[]), 100, 200, {"r1": {"samples": [sample]}})["placements"]
    self.assertEqual((lane["intervals"], list(lane["ops"])), ([[120, 150]], ["sample"]))

  def test_a_shared_runtime_splits_by_run_and_shows_only_each_runs_ops(self):
    shared = {**PROCESS, "runtime_id": "base", "turns": [[100, 200]]}
    runs = [{**RUN, "run_id": identity, "runtime_id": "base", "shared_runtime": True} for identity in ("r1", "r2")]
    operations = {"r1": {"samples": [operation(110, 140)]}, "r2": {"samples": [operation(160, 170, run_id="r2")]}}
    lanes = inspection.activity(state(live=[], processes=[shared], runs=runs), 100, 220, operations)["placements"]
    self.assertEqual(
      [(p["id"], p["run_ids"], p["intervals"], p["exact"]) for p in lanes],
      [("p1:r1", ["r1"], [[110, 140]], False), ("p1:r2", ["r2"], [[160, 170]], False)],
    )
    (lane,) = inspection.activity(state(live=[], processes=[shared], runs=runs), 100, 220, operations, run_id="r1")["placements"]
    self.assertEqual(lane["allocation_id"], "p1")

  def test_historical_sampler_and_dispatcher_sources_are_shared_by_every_reader(self):
    dispatcher = {**PROCESS, "id": "dispatcher", "pod": "pool-dispatcher", "pod_uid": "d1", "role": "dispatcher", "pool": "pool", "gpus": []}
    snapshot = state(live=[], processes=[PROCESS, dispatcher], runs=[{**RUN, "pool": "pool"}])
    samples = [operation(start, start + 10, role="sampler", pod="deleted-sampler", operation="sample") for start in (110, 150)]
    samples.append(operation(130, 131, role="sampler", pod="http://127.0.0.1:8081", operation="sample", status="retried"))
    sources = inspection.sources(snapshot, run_id="r1", samples=samples)
    self.assertEqual({p["pod"] for p in sources}, {"trainer-0", "deleted-sampler", "pool-dispatcher"})
    lanes = inspection.segments(snapshot, run_id="r1", samples=samples)
    self.assertEqual({p["pod"] for p in lanes}, {"trainer-0", "deleted-sampler"})
    recorded = inspection.activity(snapshot, 100, 220, {"r1": {"samples": samples}}, run_id="r1")["placements"][1]
    self.assertEqual((recorded["pod"], recorded["ended"], recorded["ops"]), ("deleted-sampler", True, {"sample": [[110, 120], [150, 160]]}))

  def test_current_dispatcher_is_a_source_before_its_process_reports(self):
    pod = {"name": "pool-dispatcher", "uid": "d1", "node": "n1", "role": "dispatcher", "created_at": "1970-01-01T00:01:30Z"}
    snapshot = state(live=[], processes=[], runs=[{**RUN, "pods": [pod]}])
    (source,) = inspection.sources(snapshot, run_id="r1")
    self.assertEqual((source["pod"], source["start"], source["devices"]), ("pool-dispatcher", 90, []))
    self.assertEqual(inspection.segments(snapshot, run_id="r1"), [])

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
        uid = rng.choice(["p1", "old-pod"])
        records.append(operation(low, high, pod_uid=uid))
        if uid == "p1":
          expected.update(range(max(low, 120), min(high, 190)))
      rng.shuffle(records)
      lane = inspection.activity(state(), 120, 190, {"r1": {"samples": records}})["placements"][0]
      actual = {tick for low, high in lane["intervals"] for tick in range(low, high)}
      self.assertEqual(actual, expected, f"case {case}")
      self.assertEqual(lane["ops"].get("train", []), lane["intervals"])
      self.assertFalse(lane["exact"])
      self.assertTrue(all(120 <= low < high <= 190 for low, high in lane["intervals"]))
      self.assertTrue(all(a[1] < b[0] for a, b in zip(lane["intervals"], lane["intervals"][1:])))
