"""Checks replay recorded incidents: each must fail for the incident and stay ok for a healthy run."""

import unittest

from server.dashboard import checks

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


def verdict(name, ctx):
  return checks.CHECKS[name](ctx)["verdict"]


class ChecksReplayTest(unittest.TestCase):
  def test_requests_held_by_a_trainer_that_stopped_reporting_fail_the_ledger(self):  # I1
    held = {"op": "forward_backward", "count": 3, "age_seconds": 207360, "pod": "harvey-trainer-0", "process_alive": False}
    ctx = context(healthy_steps(), outstanding=[held])
    result = checks.request_ledger(ctx)
    self.assertEqual(result["verdict"], checks.FAIL)
    self.assertIn("harvey-trainer-0", result["reason"])

  def test_a_request_lost_between_push_and_pop_fails_the_ledger_instead_of_reading_idle(self):  # I4
    ctx = context(healthy_steps(), ledger={"enqueued": 11, "popped": 10, "answered": 10}, queued=0)
    result = checks.request_ledger(ctx)
    self.assertEqual((result["verdict"], result["numbers"]["lost"]), (checks.FAIL, 1))
    self.assertEqual(verdict("request-ledger", context(healthy_steps(), ledger={"enqueued": 11, "popped": 10}, queued=1)), checks.OK)

  def test_a_nan_gradient_fails_optimizer_health(self):  # I30
    poisoned = [*healthy_steps(), op(NOW - 5, "optim_step", nonfinite=["grad_norm:mean"])]
    self.assertEqual(verdict("optimizer-health", context(poisoned)), checks.FAIL)

  def test_a_stalled_run_fails_progress_and_a_slow_one_warns(self):
    steps = healthy_steps(every=600.0)
    stalled = context([op(at - 2000, "optim_step") for at in (s["at"] for s in steps)])
    self.assertEqual(verdict("progress", stalled), checks.FAIL)
    slow = context([op(s["at"] - 800, "optim_step") for s in steps])  # last step 1,400 s ago: past 2x, under 3x
    self.assertEqual(verdict("progress", slow), checks.WARN)

  def test_a_healthy_run_is_ok_and_missing_inputs_are_unknown_never_ok(self):
    healthy = context(healthy_steps(), ledger={"enqueued": 5, "popped": 5, "answered": 5})
    for name in ("progress", "request-ledger", "worker-health", "optimizer-health"):
      self.assertEqual(verdict(name, healthy), checks.OK, name)
    empty = context()
    for name in ("progress", "optimizer-health", "effective-config", "worker-health"):
      self.assertEqual(verdict(name, empty), checks.UNKNOWN, name)

  def test_a_worker_running_other_settings_than_asked_fails_effective_config(self):
    wrong = {"role": "trainer", "pod": "t0", "gpus": ["GPU-1"], "config": {"env": {"OPEN_RL_FINE_TUNING_TYPE": "full"}}}
    self.assertEqual(verdict("effective-config", context(workers=[wrong])), checks.FAIL)
    blind = {"role": "trainer", "pod": "t0", "gpus": [], "config": {"env": {}}}
    self.assertEqual(verdict("effective-config", context(workers=[blind])), checks.FAIL)

  def test_answers_stay_small(self):
    ctx = context(healthy_steps(), ledger={"enqueued": 5, "popped": 5, "answered": 5})
    self.assertLess(len(repr(checks.run_checks(ctx))), 5000)


if __name__ == "__main__":
  unittest.main()
