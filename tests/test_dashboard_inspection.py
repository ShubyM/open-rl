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

  def test_latest_three_distinct_incidents_are_selected_after_time_filtering(self):
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
    self.assertEqual([event["at"] for event in inspection.incidents(run, 20, 30)], [25, 23, 22])
    self.assertEqual([event["at"] for event in inspection.incidents(run, 20, 24)], [24, 23, 22])
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
  "runtime_id": "base",
  "node": "n1",
  "pod": "trainer-0",
  "pod_uid": "p1",
  "role": "trainer",
  "devices": ["gpu-0"],
  "device_uuids": {"gpu-0": "GPU-1"},
  "device_count": 1,
}


def state(live=None, past=None, runs=None):
  return {
    "observed_at": "1970-01-01T00:03:20+00:00",
    "cluster": {"available": True, "scheduler": {"available": True}},
    "placements": [PLACEMENT] if live is None else live,
    "history": [{**PLACEMENT, "first_seen": 100, "last_seen": 180}] if past is None else past,
    "runs": [{"run_id": "r1", "model": "Qwen/base"}] if runs is None else runs,
  }


def operation(start, end, **fields):
  return {"run_id": "r1", "role": "trainer", "pod": "trainer-0", "pod_uid": "p1", "operation": "train", "started_at": start, "at": end, **fields}


class InspectionActivityTest(unittest.TestCase):
  def test_claim_teardown_keeps_recorded_devices_and_run_label(self):
    source = state(live=[{**PLACEMENT, "run_ids": [], "label": "base", "devices": [], "device_uuids": {}}])
    result = inspection.segments(source)[0]
    self.assertEqual((result["start"], result["end"], result["ended"]), (100, 200, False))
    self.assertEqual((result["run_ids"], result["label"]), (["r1"], "Run one"))
    self.assertEqual((result["devices"], result["device_uuids"]), (["gpu-0"], {"gpu-0": "GPU-1"}))

  def test_scheduler_outage_does_not_report_allocation_ended(self):
    source = state(live=[])
    self.assertTrue(inspection.segments(source)[0]["ended"])
    source["cluster"]["scheduler"]["available"] = False
    result = inspection.segments(source)[0]
    self.assertFalse(result["ended"])
    self.assertEqual(result["end"], 180)

  def test_replacement_before_recorder_tick_preserves_both_intervals(self):
    source = state(live=[{**PLACEMENT, "pod_uid": "p2", "node": "n2"}])
    old, current = inspection.segments(source)
    self.assertEqual((old["pod_uid"], old["node"], old["start"], old["end"]), ("p1", "n1", 100, 180))
    self.assertEqual((current["pod_uid"], current["node"], current["start"], current["end"]), ("p2", "n2", 200, 200))
    self.assertNotEqual(old["id"], current["id"])

  def test_operations_overlapping_window_edges_are_clipped_and_merged(self):
    operations = {"r1": {"samples": [operation(110, 130), operation(125, 145), operation(170, 210), operation(150, 160, run_id="r2")]}}
    result = inspection.activity(state(), 120, 190, operations, {})["placements"][0]
    self.assertEqual(result["intervals"], [[120, 145], [170, 190]])
    self.assertEqual(result["ops"], {"train": [[120, 145], [170, 190]]})
    self.assertFalse(result["exact"])

  def test_reused_pod_name_does_not_mix_operations_or_turns(self):
    source = state(past=[{**PLACEMENT, "first_seen": 150, "last_seen": 180}])
    operations = {"r1": {"samples": [operation(160, 180, pod_uid="previous-pod"), operation(180, 190)]}}
    turns = {"trainer": {"samples": [{"pod_uid": "previous-pod", "started_at": 150, "at": 200}]}}
    result = inspection.activity(source, 150, 200, operations, turns)["placements"][0]
    self.assertEqual(result["intervals"], [[180, 190]])
    self.assertFalse(result["exact"])

  def test_shared_runtime_attributes_only_tagged_ops_even_with_one_visible_run(self):
    turn = {
      "started_at": 100,
      "at": 200,
      "pod_uid": "p1",
      "ops": [
        {"name": "train", "run_id": "r1", "start": 90, "end": 140},
        {"name": "train", "run_id": "r2", "start": 160, "end": 210},
        {"name": "wake", "start": 140, "end": 160},
      ],
    }
    for ids in (["r1"], ["r1", "r2"]):
      with self.subTest(ids=ids):
        placement = {**PLACEMENT, "run_ids": ids}
        source = state(
          live=[placement],
          past=[{**placement, "first_seen": 100, "last_seen": 180}],
          runs=[{"run_id": identity, "shared_runtime": True} for identity in ids],
        )
        results = inspection.activity(source, 100, 220, {}, {"trainer": {"samples": [turn]}}, run_id="r1")["placements"]
        self.assertEqual(len(results), 1)
        self.assertEqual((results[0]["intervals"], results[0]["ops"]), ([[100, 140]], {"train": [[100, 140]]}))
        self.assertTrue(results[0]["exact"])
        if len(ids) == 1:
          self.assertEqual(results[0]["id"], "w1")

  def test_dedicated_runtime_preserves_unlabelled_gpu_grants(self):
    result = inspection.activity(state(), 120, 190, {}, {"trainer": {"samples": [{"started_at": 100, "at": 200}]}})["placements"][0]
    self.assertEqual((result["intervals"], result["ops"], result["exact"]), ([[120, 190]], {}, True))

  def test_turns_outside_the_query_do_not_suppress_operation_fallback(self):
    operations = {"r1": {"samples": [operation(120, 130)]}}
    turns = {"trainer": {"samples": [{"started_at": 10, "at": 20}]}}
    result = inspection.activity(state(), 100, 200, operations, turns)["placements"][0]
    self.assertEqual((result["intervals"], result["exact"]), ([[120, 130]], False))

  def test_deleted_shared_run_retains_attribution_and_epoch_zero_start(self):
    source = state(
      runs=[],
      live=[{**PLACEMENT, "run_ids": [], "shared_runtime": None}],
      past=[{**PLACEMENT, "shared_runtime": True, "first_seen": 0, "last_seen": 180}],
    )
    turns = {
      "trainer": {
        "samples": [
          {
            "started_at": 100,
            "at": 180,
            "ops": [
              {"name": "train", "run_id": "r1", "start": 120, "end": 130},
              {"name": "train", "run_id": "r2", "start": 140, "end": 150},
            ],
          }
        ]
      }
    }
    result = inspection.activity(source, 100, 200, {}, turns)["placements"][0]
    self.assertEqual((result["intervals"], result["start"], result["shared_runtime"]), ([[120, 130]], 0, True))

  def test_seeded_interval_fuzz_preserves_clipping_and_run_pod_attribution(self):
    import random

    rng = random.Random(20261015)
    for case in range(1000):
      records, expected = [], set()
      for _ in range(rng.randrange(25)):
        low, high = sorted(rng.sample(range(80, 230), 2))
        identity, uid = rng.choice(["r1", "r2"]), rng.choice(["p1", "old-pod"])
        records.append(operation(low, high, run_id=identity, pod_uid=uid))
        if identity == "r1" and uid == "p1":
          expected.update(range(max(low, 120), min(high, 190)))
      rng.shuffle(records)
      lane = inspection.activity(state(), 120, 190, {"r1": {"samples": records}}, {})["placements"][0]
      actual = {tick for low, high in lane["intervals"] for tick in range(low, high)}
      self.assertEqual(actual, expected, f"case {case}")
      self.assertTrue(all(120 <= low < high <= 190 for low, high in lane["intervals"]))
      self.assertTrue(all(a[1] < b[0] for a, b in zip(lane["intervals"], lane["intervals"][1:])))
      self.assertFalse(lane["exact"])
