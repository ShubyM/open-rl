"""Run facts replay recorded incidents: each incident must show up as a fact an agent can read without a verdict."""

import unittest

from server.dashboard import facts

NOW = 100_000.0


def op(at, operation, status="succeeded", role="trainer", **fields):
  return {"at": at, "started_at": at - 10, "operation": operation, "status": status, "role": role, "elapsed_seconds": 10.0, "metrics": {}, **fields}


def context(samples=(), outstanding=(), ledger=None, queued=0, finished=False, workers=(), workloads=(), pods=()):
  run = {
    "run_id": "r",
    "progress": {"outstanding": list(outstanding), "queued": {"trainer": queued}},
    "settings": {"openrl.fine_tuning_type": "lora", "openrl.trainer_gpus": 1},
    "workloads": list(workloads),
    "pods": list(pods),
  }
  return {
    "run": run,
    "samples": list(samples),
    "ledger": ledger or {"enqueued": None, "popped": None, "answered": None},
    "workers": list(workers),
    "scheduler": True,
    "finished": finished,
    "now": NOW,
    "since": 0.0,
  }


def healthy_steps(count=5, every=600.0):
  return [op(NOW - (count - i) * every, "optim_step", metrics={"grad_norm:mean": 1.0 + i}) for i in range(count)]


class RunFactsTest(unittest.TestCase):
  def test_requests_held_by_a_trainer_that_stopped_reporting_name_the_dead_holder(self):  # I1
    held = {"op": "forward_backward", "count": 3, "since": NOW - 207360, "age_seconds": 207360, "pod": "harvey-trainer-0", "process_alive": False}
    ctx = context(healthy_steps(), outstanding=[held])
    (request,) = facts.summary(ctx)["requests"]["unanswered"]
    self.assertEqual((request["pod"], request["holder_alive"], request["age_s"]), ("harvey-trainer-0", False, 207360))
    self.assertEqual(facts.line(ctx)["held_by_dead_worker"], 3)

  def test_a_request_lost_between_push_and_pop_shows_in_the_ledger(self):  # I4
    lost = context(healthy_steps(), ledger={"enqueued": 11, "popped": 10, "answered": 10}, queued=0)
    self.assertEqual(facts.summary(lost)["requests"]["ledger"]["lost"], 1)
    queued = context(healthy_steps(), ledger={"enqueued": 11, "popped": 10, "answered": 10}, queued=1)
    self.assertEqual(facts.summary(queued)["requests"]["ledger"]["lost"], 0)

  def test_a_nan_gradient_names_the_op_step_and_metric(self):  # I30
    poisoned = [*healthy_steps(), op(NOW - 5, "optim_step", nonfinite=["grad_norm:mean"], step=5)]
    first = facts.summary(context(poisoned))["optimizer"]["nonfinite"]["first"]
    self.assertEqual((first["operation"], first["step"], first["metrics"]), ("optim_step", 5, ["grad_norm:mean"]))

  def test_a_run_that_ended_on_a_failed_create_model_names_the_error_and_where_to_read_it(self):
    crashed = context([op(NOW - 600, "create_model", status="failed", error_type="ValueError", pod="t-0")], finished=True)
    found = facts.summary(crashed)
    self.assertEqual((found["steps"], found["failures"]["ops"][0]["error_type"]), (0, "ValueError"))
    logs = found["failures"]["logs"]
    self.assertEqual((logs["read"], logs["args"]["pod"], logs["args"]["q"]), ("logs", "t-0", "ValueError"))

  def test_a_run_abandoned_before_its_first_step_says_what_it_did_last(self):
    abandoned = facts.summary(context([op(NOW - 700, "create_model"), op(NOW - 600, "save_weights_for_sampler")], finished=True))
    self.assertEqual((abandoned["steps"], abandoned["last_op"]["operation"]), (0, "save_weights_for_sampler"))
    self.assertNotIn("samplers", abandoned)

  def test_a_worker_relaunched_in_its_pod_gives_when_it_stopped_and_restarted(self):
    first = {"pod": "t-0", "role": "trainer", "started_at": NOW - 1000, "heartbeat_at": NOW - 905}
    workers = [first, {"pod": "t-0", "role": "trainer", "started_at": NOW - 880, "heartbeat_at": NOW}]
    relaunches = facts.summary(context(workers=workers))["relaunches"]
    self.assertEqual(relaunches["workers"], [{"pod": "t-0", "stopped_reporting_at": facts.iso(NOW - 905), "restarted_at": facts.iso(NOW - 880)}])
    self.assertEqual(relaunches["logs"]["args"]["pod"], "t-0")

  def test_samples_waiting_on_a_dispatcher_that_stopped_reporting_name_the_dispatcher(self):
    ctx = context(healthy_steps())
    ctx["pool"] = {"name": "p", "waiting": 32, "oldest": NOW - 900, "dispatcher": "p-dispatcher-0", "alive": False}
    pool = facts.summary(ctx)["pool"]
    self.assertEqual((pool["unanswered"], pool["oldest_s"], pool["dispatcher"], pool["dispatcher_alive"]), (32, 900, "p-dispatcher-0", False))

  def test_a_slow_sampler_replica_is_named_beside_the_fastest(self):
    sends = [op(NOW - i, "sample", role="sampler", pod=pod, elapsed_seconds=s) for i in range(20) for pod, s in (("fast", 10.0), ("slow", 40.0))]
    found = facts.summary(context(sends))["samplers"]
    self.assertEqual((found["slowest"], found["fastest"]), ({"pod": "slow", "median_s": 40.0}, {"pod": "fast", "median_s": 10.0}))

  def test_a_worker_running_other_settings_than_asked_is_listed_with_both(self):
    wrong = {"role": "trainer", "pod": "t0", "gpus": ["GPU-1"], "config": {"env": {"OPEN_RL_FINE_TUNING_TYPE": "full"}}}
    self.assertEqual(
      facts.config(context(workers=[wrong])), [{"pod": "t0", "setting": "openrl.fine_tuning_type", "asked": "lora", "running": "full"}]
    )
    blind = {"role": "trainer", "pod": "t0", "gpus": [], "config": {"env": {}}}
    self.assertEqual(facts.config(context(workers=[blind]))[0]["setting"], "openrl.trainer_gpus")

  def test_goodput_splits_trainer_idle_time_into_rollout_and_waiting_on_the_client(self):
    train = op(NOW - 100, "optim_step")  # 10 s of 110 s training
    send = {**op(NOW - 50, "sample", role="sampler"), "elapsed_seconds": 10.0}  # 10 s with a sample in flight
    result = facts.goodput(context([train, send]))
    self.assertEqual(
      {k: result["seconds"][k] for k in ("wall", "train", "rollout", "client")}, {"wall": 110, "train": 10, "rollout": 10, "client": 90}
    )

  def test_a_healthy_run_has_no_failure_blocks_and_stays_small(self):
    found = facts.summary(context(healthy_steps(), ledger={"enqueued": 5, "popped": 5, "answered": 5}))
    self.assertFalse({"failures", "relaunches", "pool", "incidents"} & set(found))
    self.assertLess(len(repr(found)), 3000)


if __name__ == "__main__":
  unittest.main()
