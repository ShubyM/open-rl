"""Run facts replay recorded incidents: each incident must show up as a fact an agent can read without a verdict."""

import asyncio
import unittest
from unittest import mock

from server import telemetry
from server.dashboard import facts, fleet
from server.store import InMemoryStateStore, request_queue, sampler_queue
from server.telemetry import key

NOW = 100_000.0


def op(at, operation, status="succeeded", role="trainer", **fields):
  return {"at": at, "started_at": at - 10, "operation": operation, "status": status, "role": role, "elapsed_seconds": 10.0, "metrics": {}, **fields}


def healthy_steps(count=5, every=600.0):
  return [op(NOW - (count - i) * every, "optim_step", metrics={"grad_norm:mean": 1.0 + i}) for i in range(count)]


class RunFactsTest(unittest.TestCase):
  """Records go into a real in-memory store, and the facts are read back through the same context the routes use."""

  def setUp(self):
    self.store = InMemoryStateStore()
    for target in (facts, fleet):
      self.enterContext(mock.patch.object(target, "get_state_store", return_value=self.store))
    fleet.cache.clear()

  def facts_of(self, ops=(), workers=(), ledger=None, queued=0, lifecycle="placed", workloads=(), pods=(), samples_queued=0, completed_at=None):
    async def read():
      for record in ops:
        await self.store.append_sample(
          key("operations", "r", "samples") if record["role"] == "sampler" else key("operations", "r"), record, 10_000, None
        )
      for event, value in (ledger or {}).items():
        for _ in range(value):
          await self.store.increment_field(key("ledger", "r"), event, None)
      queues = {request_queue("r"): queued, sampler_queue("r"): samples_queued}  # the in-memory store keeps no queues
      self.store.list_length = mock.AsyncMock(side_effect=queues.get)
      run = {"run_id": "r", "runtime_id": "r", "lifecycle": lifecycle, "created_at": 0.0, "completed_at": completed_at, "workloads": list(workloads)}
      run.update(pods=list(pods), settings={"openrl.fine_tuning_type": "lora", "openrl.trainer_gpus": 1})
      ctx = await facts.context({"observed_at": NOW, "workers": list(workers), "placements": [], "runs": [run]}, run)
      return ctx, facts.summary(ctx), facts.line(ctx)

    return asyncio.run(read())

  def test_requests_held_by_a_trainer_that_stopped_reporting_name_the_dead_holder(self):  # I1
    held = {"op": "forward_backward", "run_id": "r", "count": 3, "since": NOW - 207360}
    dead = {
      "id": "w",
      "pod": "harvey-trainer-0",
      "role": "trainer",
      "runtime_id": "r",
      "started_at": 0,
      "heartbeat_at": NOW - 600,
      "outstanding": [held],
    }
    _, summary, line = self.facts_of(healthy_steps(), workers=[dead])
    (request,) = summary["requests"]["unanswered"]
    self.assertEqual((request["pod"], request["holder_alive"], request["age_s"]), ("harvey-trainer-0", False, 207360))
    self.assertEqual(line["held_by_dead_worker"], 3)

  def test_a_request_lost_between_push_and_pop_shows_in_the_ledger(self):  # I4
    _, lost, _ = self.facts_of(healthy_steps(), ledger={"enqueued": 11, "popped": 10, "answered": 10})
    self.assertEqual(lost["requests"]["ledger"]["lost"], 1)
    self.setUp()
    _, queued, _ = self.facts_of(healthy_steps(), ledger={"enqueued": 11, "popped": 10, "answered": 10}, queued=1)
    self.assertEqual(queued["requests"]["ledger"]["lost"], 0)

  def test_samples_waiting_on_a_sampler_show_so_a_stuck_run_does_not_look_idle(self):
    _, summary, line = self.facts_of(healthy_steps(), samples_queued=64)
    self.assertEqual((summary["requests"]["samples_queued"], line["samples_queued"]), (64, 64))
    self.setUp()
    _, summary, line = self.facts_of(healthy_steps())
    self.assertNotIn("samples_queued", summary.get("requests", {}))
    self.assertEqual(line["samples_queued"], 0)

  def test_a_nan_gradient_names_the_op_step_and_metric(self):  # I30
    _, summary, _ = self.facts_of([*healthy_steps(), op(NOW - 5, "optim_step", nonfinite=["grad_norm:mean"], step=5)])
    first = summary["optimizer"]["nonfinite"]["first"]
    self.assertEqual((first["operation"], first["step"], first["metrics"]), ("optim_step", 5, ["grad_norm:mean"]))

  def test_a_run_that_ended_on_a_failed_create_model_names_the_error_and_where_to_read_it(self):
    _, summary, _ = self.facts_of([op(NOW - 600, "create_model", status="failed", error_type="ValueError", pod="t-0")], lifecycle="ended")
    self.assertEqual((summary["steps"], summary["failures"]["ops"][0]["error_type"]), (0, "ValueError"))
    logs = summary["failures"]["logs"]
    self.assertEqual((logs["read"], logs["args"]["pod"], logs["args"]["q"]), ("logs", "t-0", "ValueError"))

  def test_a_run_abandoned_before_its_first_step_says_what_it_did_last(self):
    _, summary, _ = self.facts_of([op(NOW - 700, "create_model"), op(NOW - 600, "save_weights_for_sampler")], lifecycle="ended")
    self.assertEqual((summary["steps"], summary["last_op"]["operation"]), (0, "save_weights_for_sampler"))
    self.assertNotIn("samplers", summary)

  def test_a_worker_relaunched_in_its_pod_gives_when_it_stopped_and_restarted(self):
    first = {"id": "a", "pod": "t-0", "role": "trainer", "runtime_id": "r", "started_at": NOW - 1000, "heartbeat_at": NOW - 905}
    _, summary, line = self.facts_of(workers=[first, {**first, "id": "b", "started_at": NOW - 880, "heartbeat_at": NOW}])
    restarted = {"pod": "t-0", "stopped_reporting_at": facts.iso(NOW - 905), "restarted_at": facts.iso(NOW - 880)}
    self.assertEqual((summary["relaunches"]["workers"], summary["relaunches"]["logs"]["args"]["pod"], line["relaunches"]), ([restarted], "t-0", 1))

  def test_a_clean_exit_or_a_process_after_the_run_ended_is_not_a_relaunch(self):
    first = {"id": "a", "pod": "t-0", "role": "trainer", "runtime_id": "r", "started_at": NOW - 1000, "heartbeat_at": NOW - 905}
    _, summary, line = self.facts_of(workers=[{**first, "phase": "exited"}, {**first, "id": "b", "started_at": NOW - 880, "heartbeat_at": NOW}])
    self.assertEqual((line["relaunches"], "relaunches" in summary), (0, False))
    self.setUp()
    later = {**first, "id": "b", "started_at": NOW - 300, "heartbeat_at": NOW}  # the next run on the shared runtime, same pod name
    _, summary, line = self.facts_of(workers=[first, later], completed_at=NOW - 500)
    self.assertEqual((line["relaunches"], "relaunches" in summary), (0, False))

  def test_a_slow_sampler_replica_is_named_beside_the_fastest(self):
    sends = [op(NOW - i, "sample", role="sampler", pod=pod, elapsed_seconds=s) for i in range(20) for pod, s in (("fast", 10.0), ("slow", 40.0))]
    _, summary, _ = self.facts_of(sends)
    self.assertEqual(
      (summary["samplers"]["slowest"], summary["samplers"]["fastest"]), ({"pod": "slow", "median_s": 40.0}, {"pod": "fast", "median_s": 10.0})
    )

  def test_sends_a_shutdown_cut_short_count_beside_the_failed_ones(self):
    sends = [
      op(NOW - 3, "sample", role="sampler"),
      op(NOW - 2, "sample", "failed", role="sampler"),
      op(NOW - 1, "sample", "cancelled", role="sampler"),
    ]
    _, summary, _ = self.facts_of(sends)
    self.assertEqual({k: summary["samplers"][k] for k in ("sends", "failed", "cancelled")}, {"sends": 3, "failed": 1, "cancelled": 1})

  def test_a_worker_running_other_settings_than_asked_is_listed_with_both(self):
    wrong = {
      "id": "w",
      "role": "trainer",
      "runtime_id": "r",
      "pod": "t0",
      "gpus": ["GPU-1"],
      "started_at": 0,
      "config": {"env": {"OPEN_RL_FINE_TUNING_TYPE": "full"}},
    }
    _, summary, _ = self.facts_of(workers=[wrong])
    self.assertEqual(summary["config"], [{"pod": "t0", "setting": "openrl.fine_tuning_type", "asked": "lora", "running": "full"}])

  def test_a_trainer_seeing_fewer_gpus_than_asked_is_listed_and_one_that_cannot_tell_is_not(self):
    trainer = {"id": "w", "role": "trainer", "runtime_id": "r", "pod": "t0", "started_at": 0, "config": {"env": {}}}
    for gpus, expected in ((None, None), (["GPU-1"], None), ([], 0)):
      with self.subTest(gpus=gpus):
        self.setUp()
        _, summary, _ = self.facts_of(workers=[{**trainer, "gpus": gpus}])
        drift = [d["running"] for d in summary.get("config") or [] if d["setting"] == "openrl.trainer_gpus"]
        self.assertEqual(drift, [] if expected is None else [expected])

  def test_goodput_splits_trainer_idle_time_into_rollout_and_waiting_on_the_client(self):
    ctx, _, _ = self.facts_of([op(NOW - 100, "optim_step"), op(NOW - 50, "sample", role="sampler")])  # 10 s training, 10 s sampling
    seconds = facts.goodput(ctx)["seconds"]
    self.assertEqual({k: seconds[k] for k in ("wall", "train", "rollout", "client")}, {"wall": 110, "train": 10, "rollout": 10, "client": 90})

  def test_an_oom_killed_container_is_an_incident_and_a_healthy_run_has_no_failure_blocks(self):
    oom = {"at": NOW - 60, "pod": "t-0", "reason": "OOMKilled", "message": "worker exited with code 137"}
    _, summary, _ = self.facts_of(healthy_steps(), pods=[{"name": "t-0", "incidents": [oom]}])
    self.assertEqual(summary["incidents"][0]["reason"], "OOMKilled")
    self.setUp()
    _, healthy, _ = self.facts_of(healthy_steps(), ledger={"enqueued": 5, "popped": 5, "answered": 5})
    self.assertFalse({"failures", "relaunches", "incidents"} & set(healthy))
    self.assertLess(len(repr(healthy)), 3000)


class FleetLineTest(unittest.TestCase):
  """/runs gives every run's line without reading its command history, and agrees with the line built from that history."""

  def setUp(self):
    self.store = InMemoryStateStore()
    for target in (facts, fleet, telemetry):
      self.enterContext(mock.patch.object(target, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch.dict(telemetry.steps_done, clear=True))
    fleet.cache.clear()

  def record(self, operations) -> None:
    async def write():
      for at, operation, status in operations:
        with mock.patch.object(telemetry.time, "time", return_value=at):
          telemetry.record({"model_id": "r", "op": operation}, "trainer", "r", at - 10, status, None, {})
        await asyncio.gather(*list(telemetry.writes))

    asyncio.run(write())

  def lines(self, lifecycle="placed") -> tuple[dict, dict]:
    run = {"run_id": "r", "runtime_id": "r", "lifecycle": lifecycle, "created_at": 0.0, "completed_at": None, "workloads": [], "pods": []}
    run["steps"] = int(asyncio.run(self.store.get_value(key("steps", "r"))) or 0)
    state = {"observed_at": NOW, "workers": [], "placements": [], "runs": [run]}
    self.store.list_length = mock.AsyncMock(return_value=0)
    fleet.cache.clear()
    return asyncio.run(facts.fleet_line(state, run)), facts.line(asyncio.run(facts.context(state, run)))

  def test_the_fleet_line_matches_the_line_from_the_full_history(self):
    self.record([(NOW - 900, "forward_backward", "failed"), *((NOW - 600 + 100 * i, "optim_step", "succeeded") for i in range(4))])
    self.record([(NOW - 50, "forward_backward", "succeeded")])
    brief, full = self.lines()
    self.assertEqual(brief, full)
    self.assertEqual((brief["steps"], brief["failures"], brief["median_step_s"], brief["since_last_op_s"]), (4, 1, 100, 50))

  def test_runs_reads_a_few_records_per_run_whatever_its_history(self):
    from fastapi.testclient import TestClient

    from server.dashboard.__main__ import app

    self.record([(NOW - 5000 + i, "forward_backward", "succeeded") for i in range(3000)] + [(NOW - 1, "optim_step", "succeeded")])
    read, returned = self.store.read_samples, []

    async def counted(list_key, last=None):
      returned.extend(found := await read(list_key, last))
      return found

    run = {"run_id": "r", "name": "r", "runtime_id": "r", "lifecycle": "ended", "created_at": 0.0, "completed_at": None, "steps": 1}
    snapshot = {"observed_at": NOW, "workers": [], "placements": [], "runs": [run], "nodes": [], "pods": [], "errors": []}
    with mock.patch.object(self.store, "read_samples", counted), mock.patch.object(fleet, "snapshot", mock.AsyncMock(return_value=snapshot)):
      (found,) = TestClient(app).get("/api/v1/inspect/runs", params={"ended": "true"}).json()["runs"]
    self.assertEqual((found["steps"], found["since_last_op_s"]), (1, 1))
    self.assertLess(len(returned), 100)


if __name__ == "__main__":
  unittest.main()
