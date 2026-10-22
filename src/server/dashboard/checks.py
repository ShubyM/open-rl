"""Diagnostic checks: each answers one question about a run with a verdict, a reason, a few numbers and evidence.

Thresholds come from past incidents (docs/designs/017-run-diagnostics.md, section 4.5). A check whose inputs are
missing answers unknown, never ok, so absent data cannot read as healthy. Symptoms set a run's verdict; causes
name what may explain a failing symptom. Evidence is a list of reads an agent can run as given.
"""

import statistics
from datetime import UTC, datetime

from server.dashboard import inspection

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"
RANK = {OK: 0, UNKNOWN: 1, WARN: 2, FAIL: 3}
# A failing symptom is what someone notices; causes are where to look first, in this order.
SYMPTOMS = {"progress", "optimizer-health", "client-liveness"}
EXPLAINS = {"sdk-errors": "client-liveness"}  # every other cause explains progress


def read(route: str, head: str | None = None, **args) -> dict:
  """A call against /api/v1/inspect that shows the evidence; `head` is one line of what it returns."""
  return {"read": route, "args": args, **({"head": head[:200]} if head else {})}


def answer(check: str, verdict: str, reason: str, *evidence: dict, **numbers) -> dict:
  kind = {"kind": "symptom"} if check in SYMPTOMS else {"kind": "cause", "explains": EXPLAINS.get(check, "progress")}
  return {"check": check, "verdict": verdict, "reason": reason, **kind, "numbers": numbers, **({"evidence": list(evidence)} if evidence else {})}


def span(ctx: dict) -> dict:
  """since and until covering the run's records, so an evidence read finds them; reads allow at most seven days."""
  times = [s["at"] for s in ctx["samples"]]
  if not times:
    return {}
  until = max(times) + 60
  return {"since": iso(max(min(times) - 60, until - 7 * 86400)), "until": iso(until)}


def holder_logs(run_id: str, held: dict) -> dict:
  """The holder pod's logs from just before it took the request: what it did instead of answering."""
  return read("logs", run_id=run_id, pod=held["pod"], since=iso(held["since"] - 300), until=iso(held["since"] + 3600))


def failure_logs(run_id: str, op: dict) -> dict:
  """The failing pod's log lines naming the error, around the op: where the message and traceback are."""
  return read("logs", run_id=run_id, pod=op.get("pod"), q=op.get("error_type"), since=iso(op["started_at"] - 60), until=iso(op["at"] + 60))


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds")


def trainer_ops(ctx: dict) -> list[dict]:
  return sorted((s for s in ctx["samples"] if s.get("role") != "sampler"), key=lambda s: s["at"])


def optim_steps(ctx: dict) -> list[dict]:
  return [s for s in trainer_ops(ctx) if s["operation"] == "optim_step" and s.get("status") == "succeeded"]


def median_step(steps: list[dict]) -> float | None:
  gaps = [b["at"] - a["at"] for a, b in zip(steps, steps[1:])]
  return statistics.median(gaps) if gaps else None


QUIET_SECONDS = 1200  # a live run completes a sample or a trainer op far more often; the longest sample seen took 568 s


def progress(ctx: dict) -> dict:
  """Is the run completing optimizer steps at its normal pace?"""
  name, run_id, ops, steps = "progress", ctx["run"]["run_id"], trainer_ops(ctx), optim_steps(ctx)
  waiting = [o for o in (ctx["run"].get("progress") or {}).get("outstanding") or [] if o.get("op") != "sample"]
  oldest = max((o.get("age_seconds", 0) for o in waiting), default=0)
  if ctx["finished"]:
    if not steps and waiting:
      reason = f"the run ended without a step, with {waiting[0]['op']} never answered"
      return answer(name, FAIL, reason, holder_logs(run_id, waiting[0]), steps=0, unanswered=len(waiting))
    failed = [o for o in ops if o.get("status") == "failed"]
    if not steps and failed:
      reason = f"the run ended without a step after {failed[0]['operation']} failed ({failed[0].get('error_type') or 'error'})"
      return answer(name, FAIL, reason, failure_logs(run_id, failed[0]), steps=0, failed_ops=len(failed))
    if not steps and not any(s["operation"] == "sample" for s in ctx["samples"]):
      reason = f"the run ended without a step or a sample; the client stopped after {ops[-1]['operation']}" if ops else "the run ended with no ops"
      return answer(name, WARN, reason, read(f"runs/{run_id}/commands"), steps=0)
    return answer(name, OK, "the run finished", steps=len(steps))
  if not ops:
    # Nothing ever completed, but the client is waiting: a run that never started is a failed run, not missing data.
    if oldest > 1800:
      reason = f"no op has completed and {waiting[0]['op']} has waited {oldest / 60:.0f} min"
      return answer(name, FAIL, reason, holder_logs(run_id, waiting[0]), observed_s=round(oldest), threshold_s=1800)
    return answer(name, UNKNOWN, "no trainer ops recorded for this run")
  median = median_step(steps)
  idle = ctx["now"] - (steps[-1]["at"] if steps else ops[0]["started_at"])
  limit = max(1800, 3 * median) if median else 1800
  numbers = {"observed_s": round(idle), "expected_s": median, "threshold_s": limit, "steps": len(steps)}
  evidence = read(f"runs/{run_id}/steps", last=3)
  if not steps:
    verdict = FAIL if idle > 1800 else OK
    return answer(name, verdict, f"no optimizer step {idle / 60:.0f} min after the first op", evidence, **numbers)
  if idle > limit:
    return answer(name, FAIL, f"no optimizer step for {idle / 60:.0f} min, over 3x the {median / 60:.0f} min median", evidence, **numbers)
  quiet = ctx["now"] - max(s["at"] for s in ctx["samples"])
  if quiet > QUIET_SECONDS:
    reason = f"nothing completed for {quiet / 60:.0f} min, neither a trainer op nor a sample"
    return answer(name, FAIL, reason, read(f"runs/{run_id}/commands", limit=5), **{**numbers, "quiet_s": round(quiet), "quiet_threshold_s": QUIET_SECONDS})
  if median and idle > max(600, 2 * median):
    return answer(name, WARN, f"no optimizer step for {idle / 60:.0f} min, over 2x the median", evidence, **numbers)
  return answer(name, OK, f"{len(steps)} steps; the last {idle / 60:.0f} min ago", **numbers)


def request_ledger(ctx: dict) -> dict:
  """Is every request the client waits on accounted for, held by a live worker, and younger than 3 median steps?"""
  name, run_id, progress_ = "request-ledger", ctx["run"]["run_id"], ctx["run"].get("progress") or {}
  outstanding = [o for o in progress_.get("outstanding") or [] if o.get("op") != "sample"]
  dead = [o for o in outstanding if o.get("process_alive") is False]
  numbers = {"outstanding": sum(o.get("count", 0) for o in outstanding)}
  if dead:
    o = dead[0]
    reason = f"{o['count']} {o['op']} held by {o['pod']}, which stopped reporting"
    return answer(name, FAIL, reason, holder_logs(run_id, o), **numbers)
  ledger = ctx["ledger"]
  if ledger.get("enqueued") is not None:
    lost = ledger["enqueued"] - (ledger.get("popped") or 0) - ((progress_.get("queued") or {}).get("trainer") or 0)
    numbers.update(lost=lost, enqueued=ledger["enqueued"], popped=ledger.get("popped"))
    if lost > 0:
      evidence = read("logs", run_id=run_id, q="enqueued", **span(ctx))
      return answer(name, FAIL, f"{lost} requests were enqueued but never popped or queued", evidence, **numbers)
  median = median_step(optim_steps(ctx))
  oldest = max((o.get("age_seconds", 0) for o in outstanding), default=0)
  numbers.update(observed_s=round(oldest), expected_s=median, threshold_s=3 * median if median else None)
  if median and oldest > 3 * median:
    reason = f"a request has waited {oldest / 60:.0f} min, over 3 median steps"
    return answer(name, WARN, reason, read(f"runs/{run_id}/commands", **span(ctx)), **numbers)
  if ledger.get("enqueued") is None and not ctx["finished"]:
    return answer(name, UNKNOWN, "no ledger counters for this run; its workers predate them", **numbers)
  return answer(name, OK, "every request is queued, held by a live worker, or answered", **numbers)


def worker_health(ctx: dict) -> dict:
  """Has any worker of this run crashed, restarted, OOMed or failed an op?"""
  name, run = "worker-health", ctx["run"]
  run_id = run["run_id"]
  events = inspection.incidents(run, ctx["since"], ctx["now"])
  failed = [s for s in trainer_ops(ctx) if s.get("status") == "failed"]
  relaunched = relaunches(ctx)
  restarts = max(len(relaunched), sum(c.get("restart_count") or 0 for p in run.get("pods") or [] for c in p.get("containers") or []))
  held = any(o.get("op") != "sample" for o in (run.get("progress") or {}).get("outstanding") or [])
  numbers = {"failed_ops": len(failed), "restarts": restarts, "incidents": len(events)}
  severe = [e for e in events if e.get("reason") in ("OOMKilled", "CrashLoopBackOff")]
  if severe:
    e = severe[0]
    pod = e.get("pod")
    evidence = read("logs", e.get("message"), run_id=run_id, pod=pod, severity="ERROR", **span(ctx))
    return answer(name, FAIL, f"{e['reason']} on {pod or 'a pod'}: {e.get('message', '')[:120]}", evidence, **numbers)
  if restarts and held:
    if relaunched:
      p = relaunched[0]
      logs = read("logs", run_id=run_id, pod=p["pod"], since=iso(p["started_at"] - 120), until=iso(p["started_at"] + 60))
      return answer(name, FAIL, f"the worker in {p['pod']} restarted at {iso(p['started_at'])} with a request in flight", logs, **numbers)
    errors = read("logs", run_id=run_id, severity="ERROR", **span(ctx))
    return answer(name, FAIL, f"{restarts} restarts while requests were in flight", errors, **numbers)
  if failed or restarts or events:
    kinds = sorted({s.get("error_type") or "failed" for s in failed})
    reason = f"{len(failed)} failed ops ({', '.join(kinds) or 'none'}), {restarts} restarts"
    failures = read(f"runs/{run_id}/commands", role="trainer", **span(ctx))
    logs = failure_logs(run_id, failed[-1]) if failed else read("logs", run_id=run_id, severity="ERROR", **span(ctx))
    return answer(name, WARN, reason, failures, logs, **numbers)
  if not run.get("pods") and not ctx["samples"] and not ctx.get("placements"):
    return answer(name, UNKNOWN, "no pods, placements or ops recorded for this run", **numbers)
  return answer(name, OK, "no failures, restarts or OOMs recorded", **numbers)


def relaunches(ctx: dict) -> list[dict]:
  """Worker processes that replaced an earlier one in the same pod during this run: a crash or a container restart."""
  by_pod: dict[str, list[dict]] = {}
  for p in ctx["workers"]:
    if p.get("pod") and p.get("role") != "dispatcher" and p.get("started_at", 0) >= ctx["since"] - 60:
      by_pod.setdefault(p["pod"], []).append(p)
  return sorted((p for procs in by_pod.values() for p in sorted(procs, key=lambda p: p["started_at"])[1:]), key=lambda p: p["started_at"])


def placement(ctx: dict) -> dict:
  """Does every worker the run needs exist, and is it placed?"""
  name, run = "placement", ctx["run"]
  if not ctx["scheduler"]:
    return answer(name, UNKNOWN, "the OpenRL scheduler is not installed here")
  workloads = run.get("workloads") or []
  age = {w["name"]: (w.get("worker") or {}).get("age_seconds") or 0 for w in workloads}
  pending = [w for w in workloads if not w.get("node_name") and age[w["name"]] > 300]
  numbers = {"workloads": len(workloads), "unplaced": len(pending), "threshold_s": 300}
  if pending:
    w = pending[0]
    why = w.get("placed_message") or w.get("reason") or "waiting for a GPU"
    detail = f"{w['name']} ({w.get('role')}) unplaced for {age[w['name']] / 60:.0f} min: {why}"
    return answer(name, FAIL, detail, **numbers)
  if ctx["finished"]:
    return answer(name, OK, f"the run ended; {len(ctx.get('placements') or [])} placements recorded", **numbers)
  if workloads:
    return answer(name, OK, f"{len(workloads)} workloads, all placed", **numbers)
  return answer(name, UNKNOWN, "no workloads recorded for this run", **numbers)


def optimizer_health(ctx: dict) -> dict:
  """Are gradients finite, and is the grad norm steady rather than spiking or climbing?"""
  name, run_id = "optimizer-health", ctx["run"]["run_id"]
  poisoned = [s for s in trainer_ops(ctx) if s.get("nonfinite")]
  numbers = {"nonfinite_ops": len(poisoned)}
  evidence = read(f"runs/{run_id}/steps", last=10)
  if poisoned:
    s = poisoned[0]
    return answer(name, FAIL, f"{s['operation']} returned non-finite {', '.join(s['nonfinite'])}", evidence, **numbers)
  norms = [s["metrics"]["grad_norm:mean"] for s in optim_steps(ctx) if "grad_norm:mean" in s.get("metrics", {})]
  if not norms:
    return answer(name, UNKNOWN, "no grad norms recorded", **numbers)
  median = statistics.median(norms)
  numbers.update(observed=norms[-1], expected=median, threshold=10 * median)
  if len(norms) >= 2 and norms[-1] == norms[-2] == 0:
    return answer(name, FAIL, "grad norm was exactly 0 for 2 steps", evidence, **numbers)
  if len(norms) >= 5 and norms[-1] > 10 * statistics.median(norms[:-1]):
    return answer(name, WARN, f"grad norm {norms[-1]:.3g} is over 10x its median", evidence, **numbers)
  # Growth often precedes instability (INTELLECT-2): five rising steps ending well above the run's median.
  tail = norms[-5:]
  if len(norms) >= 8 and all(a < b for a, b in zip(tail, tail[1:])) and tail[-1] > 3 * statistics.median(norms[:-5]):
    return answer(name, WARN, f"grad norm rose for 5 steps to {tail[-1]:.3g}", evidence, **numbers)
  return answer(name, OK, f"grad norms finite and steady; last {norms[-1]:.3g}", **numbers)


def effective_config(ctx: dict) -> dict:
  """Is each worker running the settings the run asked for?"""
  name, run = "effective-config", ctx["run"]
  echoed = [w for w in ctx["workers"] if w.get("config")]
  if not echoed:
    return answer(name, UNKNOWN, "no worker echoed its settings; they predate the echo")
  settings = run.get("settings") or {}
  for w in echoed:
    pod = w.get("pod")
    asked, got = settings.get("openrl.fine_tuning_type"), w["config"].get("env", {}).get("OPEN_RL_FINE_TUNING_TYPE")
    if asked and got and asked != got:
      return answer(name, FAIL, f"{pod} runs {got}; the run asked for {asked}", read("logs", run_id=run["run_id"], pod=pod, q="openrl.config"))
    if w.get("role") == "trainer" and settings.get("openrl.trainer_gpus") and not w.get("gpus"):
      reason = f"{pod} sees no GPUs; the run asked for {settings['openrl.trainer_gpus']}"
      return answer(name, FAIL, reason, read("logs", run_id=run["run_id"], pod=pod, q="openrl.config"))
  return answer(name, OK, f"{len(echoed)} workers echo settings consistent with the run", workers=len(echoed))


def sampler_flow(ctx: dict) -> dict:
  """Are samplers answering without many retries or failures, and is no replica far slower than its peers?"""
  name, run_id = "sampler-flow", ctx["run"]["run_id"]
  sends = [s for s in ctx["samples"] if s.get("role") == "sampler"]
  evidence = read(f"runs/{run_id}/commands", role="sampler", **span(ctx))
  pool = ctx.get("pool") or {}
  if pool.get("waiting") and not ctx["finished"] and not pool.get("alive") and ctx["now"] - pool["oldest"] > 120:
    holder = f"dispatcher {pool['dispatcher']}, which stopped reporting" if pool.get("dispatcher") else "a dispatcher, and none is recorded"
    reason = f"{pool['waiting']} samples wait in pool {pool['name']} for {holder}"
    since = pool["oldest"]
    logs = read("logs", run_id=run_id, pod=pool.get("dispatcher"), since=iso(since - 300), until=iso(since + 3600))
    return answer(name, FAIL, reason, logs, waiting=pool["waiting"], observed_s=round(ctx["now"] - since), threshold_s=120)
  if not sends:
    return answer(name, OK if ctx["finished"] else UNKNOWN, "no sample requests recorded")
  retried = sum(s.get("status") == "retried" for s in sends)
  failed = sum(s.get("status") == "failed" for s in sends)
  latency = statistics.median(s.get("elapsed_seconds", 0) for s in sends)
  numbers = {"sends": len(sends), "retried": retried, "failed": failed, "median_s": round(latency, 2), "threshold_share": 0.05}
  if failed / len(sends) > 0.05:
    return answer(name, FAIL, f"{failed} of {len(sends)} sample requests failed", evidence, **numbers)
  if retried / len(sends) > 0.05:
    return answer(name, WARN, f"{retried} of {len(sends)} sends were retried", evidence, **numbers)
  # Each replica against the fastest in the same run, so no stored baseline is needed (NVRx, Minder).
  by_pod = {}
  for s in sends:
    # Answered sends to replica pods only: a fast failure, or a send recorded against the router URL, is not a replica's speed.
    if s.get("status") == "succeeded" and s.get("pod") and "://" not in s["pod"]:
      by_pod.setdefault(s["pod"], []).append(s.get("elapsed_seconds", 0))
  medians = {pod: statistics.median(v) for pod, v in by_pod.items() if len(v) >= 20}
  if len(medians) >= 2:
    fastest, slowest = min(medians, key=medians.get), max(medians, key=medians.get)
    numbers.update(slowest=slowest, slowest_s=round(medians[slowest], 2), fastest_s=round(medians[fastest], 2))
    if medians[slowest] > 2 * medians[fastest]:
      return answer(name, WARN, f"{slowest} answers in {medians[slowest]:.1f} s, over 2x the fastest replica", evidence, **numbers)
  return answer(name, OK, f"{len(sends)} sends, median {latency:.1f} s", **numbers)


def storage(ctx: dict) -> dict:
  """Do checkpoints save, in reasonable time, so a crash loses little?"""
  name, run_id = "storage", ctx["run"]["run_id"]
  saves = [s for s in trainer_ops(ctx) if s["operation"] == "save_state"]
  failed = [s for s in saves if s.get("status") == "failed"]
  slow = [s for s in saves if s.get("elapsed_seconds", 0) > 600]
  numbers = {"saves": len(saves), "failed": len(failed)}
  evidence = read(f"runs/{run_id}/commands", role="trainer", **span(ctx))
  if failed:
    return answer(name, FAIL, f"{len(failed)} of {len(saves)} save_state calls failed", evidence, **numbers)
  last_save = max((s["at"] for s in saves if s.get("status") == "succeeded"), default=0.0)
  since = sum(s["at"] > last_save for s in optim_steps(ctx))
  numbers.update(steps_since_save=since, threshold_steps=5)
  if since > 5:
    return answer(name, WARN, f"{since} steps since the last successful save_state; a crash loses them", evidence, **numbers)
  if slow:
    return answer(name, WARN, f"a save_state took {slow[-1]['elapsed_seconds'] / 60:.0f} min, over the 10 min deadline", evidence, **numbers)
  return answer(name, UNKNOWN, "saves succeed; volume usage is not recorded", **numbers)


CHECKS = {
  "progress": progress,
  "request-ledger": request_ledger,
  "worker-health": worker_health,
  "placement": placement,
  "optimizer-health": optimizer_health,
  "effective-config": effective_config,
  "sampler-flow": sampler_flow,
  "storage": storage,
}


def run_checks(ctx: dict, only: list[str] | None = None) -> list[dict]:
  return [CHECKS[name](ctx) for name in only or CHECKS]


def verdict(results: list[dict]) -> str:
  """A run's verdict is its worst symptom; causes only explain it."""
  symptoms = [r["verdict"] for r in results if r["kind"] == "symptom"]
  return max(symptoms, key=RANK.get, default=UNKNOWN)


def goodput(ctx: dict) -> dict | None:
  """Share of the run's span the trainer spent training, and where the rest went (Google ML goodput)."""
  ops = trainer_ops(ctx)
  if not ops:
    return None
  start = ops[0]["started_at"]
  wall = max((ops[-1]["at"] if ctx["finished"] else ctx["now"]) - start, 1.0)
  spent = {kind: sum(s.get("elapsed_seconds", 0) for s in ops if s["operation"] in names) for kind, names in BUCKETS.items()}
  idle = max(0.0, wall - sum(spent.values()))
  # Trainer idle time with a sample in flight is rollout; with nothing in flight the server waits on the client.
  sends = [s for s in ctx["samples"] if s["operation"] == "sample"]
  rollout = min(idle, max(0.0, covered(ops + sends, start, start + wall) - covered(ops, start, start + wall))) if sends else None
  steps = optim_steps(ctx)
  split = {"rollout": round(rollout), "client": round(idle - rollout)} if rollout is not None else {"other": round(idle)}
  return {
    "goodput": round(spent["train"] / wall, 3),
    "seconds": {"wall": round(wall), **{k: round(v) for k, v in spent.items()}, **split},
    "first_step_after_s": round(steps[0]["at"] - start) if steps else None,
  }


def waiting(ctx: dict) -> dict | None:
  """For a live run, what is in flight now and how long since anything finished: what a stuck run waits on."""
  if ctx["finished"] or not ctx["samples"]:
    return None
  now, trainer = ctx["now"], trainer_ops(ctx)
  held = [o for o in (ctx["run"].get("progress") or {}).get("outstanding") or [] if o.get("op") != "sample"]
  pool = ctx.get("pool") or {}
  return {
    "since_last_op_s": round(now - max(s["at"] for s in ctx["samples"])),
    "trainer_idle_s": round(now - trainer[-1]["at"]) if trainer else None,
    "trainer_requests": sum(o.get("count", 0) for o in held),
    "oldest_trainer_request_s": round(max(o.get("age_seconds", 0) for o in held)) if held else None,
    "samples_queued": pool.get("waiting", 0),
    "oldest_sample_queued_s": round(now - pool["oldest"]) if pool.get("oldest") else None,
  }


def covered(records: list[dict], low: float, high: float) -> float:
  """Seconds of [low, high] during which at least one of the records was in flight."""
  return sum(b - a for a, b in inspection.merge([(max(r["started_at"], low), min(r["at"], high)) for r in records if r["at"] > low and r["started_at"] < high]))


BUCKETS = {"train": {"forward_backward", "optim_step"}, "checkpoint": {"save_state", "save_weights_for_sampler"}, "forward": {"forward"}}
SESSION_LIVE_SECONDS = 120  # the live session key's expiry: a session silent this long is gone


def client_liveness(session: dict) -> dict:
  """Is the client alive, and sending work while it is?"""
  name, record, now = "client-liveness", session["record"], session["now"]
  requests = [s["at"] for ctx in session["models"].values() for s in ctx["samples"]]
  quiet = now - max(requests) if requests else None
  heard = now - record["last_seen_at"]
  unfinished = [m for m, ctx in session["models"].items() if not ctx["finished"] and not session["resumed"].get(m)]
  if record.get("derived") and not session["live"] and unfinished:
    return answer(name, UNKNOWN, "no session record, and no live heartbeat key; its gateway predates session records")
  numbers = {"since_heartbeat_s": round(heard), "observed_s": round(quiet) if quiet is not None else None, "threshold_s": 600}
  evidence = read(f"sessions/{record['session_id']}/summary")
  if session["live"] and unfinished and quiet is not None and quiet > 600:
    return answer(name, FAIL, f"the client heartbeats but sent nothing for {quiet / 60:.0f} min", evidence, **numbers)
  if not session["live"] and unfinished:
    return answer(name, WARN, f"the client stopped {heard / 60:.0f} min ago with {unfinished[0]} unfinished", evidence, **numbers)
  return answer(name, OK, "heartbeating and sending work" if session["live"] else "the session ended", **numbers)


def sdk_errors(session: dict) -> dict:
  """Did the client's SDK report errors in this session?"""
  name, errors = "sdk-errors", session["errors"]
  if not errors:
    return answer(name, OK, "no SDK errors reported", errors=0)
  worst = FAIL if any(e.get("severity") == "CRITICAL" for e in errors) else WARN
  reasons = statistics.mode(e.get("reason") or "error" for e in errors)
  evidence = read(f"sessions/{session['record']['session_id']}/checks", errors[-1].get("message"))
  return answer(name, worst, f"{len(errors)} SDK errors, mostly {reasons}", evidence, errors=len(errors))


def session_checks(session: dict) -> list[dict]:
  """The session's own checks, then each model check rolled up to its worst verdict across the session's models."""
  per_model = {model: run_checks(ctx) for model, ctx in session["models"].items()}
  rolled = {}
  for model, results in per_model.items():
    for result in results:
      held = rolled.get(result["check"])
      if held is None or RANK[result["verdict"]] > RANK[held["verdict"]]:
        rolled[result["check"]] = {**result, "reason": f"{model[:8]}: {result['reason']}", "numbers": {**result["numbers"], "model": model}}
  return [client_liveness(session), sdk_errors(session), *rolled.values()]
