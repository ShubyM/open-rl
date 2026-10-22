"""Diagnostic checks: each answers one question about a run with a verdict, a reason, a few numbers and evidence.

Thresholds come from past incidents (docs/designs/017-run-diagnostics.md, section 4.5). A check whose inputs are
missing answers unknown, never ok, so absent data cannot read as healthy.
"""

import statistics

from server.dashboard import inspection

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"
RANK = {OK: 0, UNKNOWN: 1, WARN: 2, FAIL: 3}


def answer(check: str, verdict: str, reason: str, evidence: str | None = None, **numbers) -> dict:
  return {"check": check, "verdict": verdict, "reason": reason, "numbers": numbers, **({"evidence": evidence} if evidence else {})}


def trainer_ops(ctx: dict) -> list[dict]:
  return sorted((s for s in ctx["samples"] if s.get("role") != "sampler"), key=lambda s: s["at"])


def optim_steps(ctx: dict) -> list[dict]:
  return [s for s in trainer_ops(ctx) if s["operation"] == "optim_step" and s.get("status") == "succeeded"]


def median_step(steps: list[dict]) -> float | None:
  gaps = [b["at"] - a["at"] for a, b in zip(steps, steps[1:])]
  return statistics.median(gaps) if gaps else None


def progress(ctx: dict) -> dict:
  """Is the run completing optimizer steps at its normal pace?"""
  name, run_id, ops, steps = "progress", ctx["run"]["run_id"], trainer_ops(ctx), optim_steps(ctx)
  if ctx["finished"]:
    return answer(name, OK, "the run finished", steps=len(steps))
  if not ops:
    return answer(name, UNKNOWN, "no trainer ops recorded for this run")
  median = median_step(steps)
  idle = ctx["now"] - (steps[-1]["at"] if steps else ops[0]["started_at"])
  numbers = {"steps": len(steps), "median_step_s": median, "since_last_step_s": round(idle)}
  evidence = f"runs/{run_id}/steps?last=3"
  if not steps:
    verdict = FAIL if idle > 1800 else OK
    return answer(name, verdict, f"no optimizer step {idle / 60:.0f} min after the first op", evidence, **numbers)
  if median and idle > max(1800, 3 * median):
    return answer(name, FAIL, f"no optimizer step for {idle / 60:.0f} min, over 3x the {median / 60:.0f} min median", evidence, **numbers)
  if median and idle > max(600, 2 * median):
    return answer(name, WARN, f"no optimizer step for {idle / 60:.0f} min, over 2x the median", evidence, **numbers)
  return answer(name, OK, f"{len(steps)} steps; the last {idle / 60:.0f} min ago", evidence, **numbers)


def request_ledger(ctx: dict) -> dict:
  """Is every request the client waits on accounted for, held by a live worker, and younger than 3 median steps?"""
  name, run_id, progress_ = "request-ledger", ctx["run"]["run_id"], ctx["run"].get("progress") or {}
  outstanding = [o for o in progress_.get("outstanding") or [] if o.get("op") != "sample"]
  dead = [o for o in outstanding if o.get("process_alive") is False]
  numbers = {"outstanding": sum(o.get("count", 0) for o in outstanding)}
  if dead:
    o = dead[0]
    return answer(name, FAIL, f"{o['count']} {o['op']} held by {o['pod']}, which stopped reporting", f"pod {o['pod']}", **numbers)
  ledger = ctx["ledger"]
  if ledger.get("enqueued") is not None:
    lost = ledger["enqueued"] - (ledger.get("popped") or 0) - ((progress_.get("queued") or {}).get("trainer") or 0)
    numbers["lost"] = lost
    if lost > 0:
      return answer(name, FAIL, f"{lost} requests were enqueued but never popped or queued", f"open_rl:ledger:{run_id}:*", **numbers)
  median = median_step(optim_steps(ctx))
  oldest = max((o.get("age_seconds", 0) for o in outstanding), default=0)
  numbers["oldest_s"] = round(oldest)
  if median and oldest > 3 * median:
    return answer(name, WARN, f"a request has waited {oldest / 60:.0f} min, over 3 median steps", f"runs/{run_id}/commands", **numbers)
  if ledger.get("enqueued") is None and not ctx["finished"]:
    return answer(name, UNKNOWN, "no ledger counters for this run; its workers predate them", **numbers)
  return answer(name, OK, "every request is queued, held by a live worker, or answered", **numbers)


def worker_health(ctx: dict) -> dict:
  """Has any worker of this run crashed, restarted, OOMed or failed an op?"""
  name, run = "worker-health", ctx["run"]
  events = inspection.incidents(run, ctx["since"], ctx["now"])
  failed = [s for s in trainer_ops(ctx) if s.get("status") == "failed"]
  restarts = sum(c.get("restart_count") or 0 for p in run.get("pods") or [] for c in p.get("containers") or [])
  held = any(o.get("op") != "sample" for o in (run.get("progress") or {}).get("outstanding") or [])
  numbers = {"failed_ops": len(failed), "restarts": restarts, "incidents": len(events)}
  severe = [e for e in events if e.get("reason") in ("OOMKilled", "CrashLoopBackOff")]
  if severe:
    e = severe[0]
    return answer(name, FAIL, f"{e['reason']} on {e.get('pod', 'a pod')}: {e.get('message', '')[:120]}", f"pod {e.get('pod')}", **numbers)
  if restarts and held:
    return answer(name, FAIL, f"{restarts} restarts while requests were in flight", "pods", **numbers)
  if failed or restarts or events:
    kinds = sorted({s.get("error_type") or "failed" for s in failed})
    return answer(
      name, WARN, f"{len(failed)} failed ops ({', '.join(kinds) or 'none'}), {restarts} restarts", f"runs/{run['run_id']}/commands", **numbers
    )
  if not run.get("pods") and not ctx["samples"]:
    return answer(name, UNKNOWN, "no pods or ops recorded for this run", **numbers)
  return answer(name, OK, "no failures, restarts or OOMs recorded", **numbers)


def placement(ctx: dict) -> dict:
  """Does every worker the run needs exist, and is it placed?"""
  name, run = "placement", ctx["run"]
  if not ctx["scheduler"]:
    return answer(name, UNKNOWN, "the OpenRL scheduler is not installed here")
  workloads = run.get("workloads") or []
  age = {w["name"]: (w.get("worker") or {}).get("age_seconds") or 0 for w in workloads}
  pending = [w for w in workloads if not w.get("node_name") and age[w["name"]] > 300]
  numbers = {"workloads": len(workloads), "unplaced": len(pending)}
  if pending:
    w = pending[0]
    reason = w.get("placed_message") or w.get("reason") or "waiting for a GPU"
    return answer(
      name, FAIL, f"{w['name']} ({w.get('role')}) unplaced for {age[w['name']] / 60:.0f} min: {reason}", f"workload {w['name']}", **numbers
    )
  if ctx["finished"] or workloads:
    return answer(name, OK, f"{len(workloads)} workloads, all placed", **numbers)
  return answer(name, UNKNOWN, "no workloads recorded for this run", **numbers)


def optimizer_health(ctx: dict) -> dict:
  """Are gradients finite and inside their normal range?"""
  name, run_id = "optimizer-health", ctx["run"]["run_id"]
  ops = trainer_ops(ctx)
  poisoned = [s for s in ops if s.get("nonfinite")]
  numbers = {"nonfinite_ops": len(poisoned)}
  if poisoned:
    s = poisoned[0]
    return answer(name, FAIL, f"{s['operation']} returned non-finite {', '.join(s['nonfinite'])}", f"runs/{run_id}/steps", **numbers)
  norms = [s["metrics"]["grad_norm:mean"] for s in optim_steps(ctx) if "grad_norm:mean" in s.get("metrics", {})]
  if not norms:
    return answer(name, UNKNOWN, "no grad norms recorded", **numbers)
  numbers.update(last_grad_norm=norms[-1], median_grad_norm=statistics.median(norms))
  if len(norms) >= 2 and norms[-1] == norms[-2] == 0:
    return answer(name, FAIL, "grad norm was exactly 0 for 2 steps", f"runs/{run_id}/steps", **numbers)
  if len(norms) >= 5 and norms[-1] > 10 * statistics.median(norms[:-1]):
    return answer(name, WARN, f"grad norm {norms[-1]:.3g} is over 10x its median", f"runs/{run_id}/steps", **numbers)
  return answer(name, OK, f"grad norms finite; last {norms[-1]:.3g}", **numbers)


def effective_config(ctx: dict) -> dict:
  """Is each worker running the settings the run asked for?"""
  name, run = "effective-config", ctx["run"]
  echoed = [w for w in ctx["workers"] if w.get("config")]
  if not echoed:
    return answer(name, UNKNOWN, "no worker echoed its settings; they predate the echo")
  settings = run.get("settings") or {}
  for w in echoed:
    env = w["config"].get("env", {})
    asked = settings.get("openrl.fine_tuning_type")
    got = env.get("OPEN_RL_FINE_TUNING_TYPE")
    if asked and got and asked != got:
      return answer(name, FAIL, f"{w.get('pod')} runs {got}; the run asked for {asked}", f"pod {w.get('pod')}")
    if w.get("role") == "trainer" and settings.get("openrl.trainer_gpus") and not w.get("gpus"):
      return answer(name, FAIL, f"{w.get('pod')} sees no GPUs; the run asked for {settings['openrl.trainer_gpus']}", f"pod {w.get('pod')}")
  return answer(name, OK, f"{len(echoed)} workers echo settings consistent with the run", workers=len(echoed))


def sampler_flow(ctx: dict) -> dict:
  """Are samplers answering, without many retries or failures?"""
  name, run_id = "sampler-flow", ctx["run"]["run_id"]
  sends = [s for s in ctx["samples"] if s.get("role") == "sampler"]
  if not sends:
    return answer(name, OK if ctx["finished"] else UNKNOWN, "no sample requests recorded")
  retried = sum(s.get("status") == "retried" for s in sends)
  failed = sum(s.get("status") == "failed" for s in sends)
  latency = statistics.median(s.get("elapsed_seconds", 0) for s in sends)
  numbers = {"sends": len(sends), "retried": retried, "failed": failed, "median_s": round(latency, 2)}
  evidence = f"runs/{run_id}/commands?role=sampler"
  if failed / len(sends) > 0.05:
    return answer(name, FAIL, f"{failed} of {len(sends)} sample requests failed", evidence, **numbers)
  if retried / len(sends) > 0.05:
    return answer(name, WARN, f"{retried} of {len(sends)} sends were retried", evidence, **numbers)
  return answer(name, OK, f"{len(sends)} sends, median {latency:.1f} s", **numbers)


def storage(ctx: dict) -> dict:
  """Do checkpoints save, so a crash loses little?"""
  name, run_id = "storage", ctx["run"]["run_id"]
  saves = [s for s in trainer_ops(ctx) if s["operation"] == "save_state"]
  failed = [s for s in saves if s.get("status") == "failed"]
  numbers = {"saves": len(saves), "failed": len(failed)}
  if failed:
    return answer(name, FAIL, f"{len(failed)} of {len(saves)} save_state calls failed", f"runs/{run_id}/commands", **numbers)
  if not saves and len(optim_steps(ctx)) > 2:
    return answer(name, WARN, "no successful save_state after 2 steps", **numbers)
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


SESSION_LIVE_SECONDS = 120  # the live session key's expiry: a session silent this long is gone


def client_liveness(session: dict) -> dict:
  """Is the client alive, and sending work while it is?"""
  name, record, now = "client-liveness", session["record"], session["now"]
  requests = [s["at"] for ctx in session["models"].values() for s in ctx["samples"]]
  quiet = now - max(requests) if requests else None
  heard = now - record["last_seen_at"]
  unfinished = [m for m, ctx in session["models"].items() if not ctx["finished"] and not session["resumed"].get(m)]
  numbers = {"since_heartbeat_s": round(heard), "since_request_s": round(quiet) if quiet is not None else None, "models": len(session["models"])}
  evidence = f"sessions/{record['session_id']}/summary"
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
  return answer(
    name, worst, f"{len(errors)} SDK errors, mostly {reasons}", f"open_rl:sdk_errors:{session['record']['session_id']}", errors=len(errors)
  )


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
