"""Facts about a run, joined from its records, with no verdicts. A block appears only when it has content, so an absent
block means nothing of its kind was recorded. A log pointer is a read against /api/v1/inspect to run as given."""

import statistics
from datetime import UTC, datetime

from server.dashboard import fleet
from server.store import get_state_store, request_queue, sampler_queue
from server.telemetry import alive, key

BUCKETS = {"train": {"forward_backward", "optim_step"}, "checkpoint": {"save_state", "save_weights_for_sampler"}, "forward": {"forward"}}


async def context(state: dict, run: dict, low: float = 0, high: float = float("inf")) -> dict:
  """What the facts read about one run: its ops in [low, high], its workers, request counters and queue."""
  store, run_id = get_state_store(), run["run_id"]
  ops = [s for s in await fleet.operations(run_id) if low <= s["at"] <= high]
  ledger = {event: await store.get_value(key("ledger", run_id, event)) for event in ("enqueued", "popped", "answered")}
  trainer = [s for s in ops if s.get("role") != "sampler"]
  ctx = {
    "run": run,
    "ops": ops,
    "trainer": trainer,
    "steps": [s for s in trainer if s["operation"] == "optim_step" and s.get("status") == "succeeded"],
  }
  ctx.update(ledger={e: int(v) for e, v in ledger.items() if v is not None}, queued=await store.list_length(request_queue(run_id)))
  ctx.update(now=state["observed_at"], finished=run["lifecycle"] == "ended", lanes=fleet.lanes(state, run_id))
  # Samples wait in the runtime's queue, which co-tenant runs share; a finished run's share of it is gone.
  ctx["samples_queued"] = None if ctx["finished"] else await store.list_length(sampler_queue(run["runtime_id"]))
  ctx["workers"] = [w for w in state["workers"] if w.get("runtime_id") in (run["runtime_id"], run_id)]
  return ctx


def read(route: str, **args) -> dict:
  return {"read": route, "args": {k: v for k, v in args.items() if v is not None}}


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds")


def present(**blocks) -> dict:
  return {k: v for k, v in blocks.items() if v not in (None, [], {})}


def unanswered(ctx: dict) -> list[dict]:
  """Trainer requests the client waits on, with the pod holding each and whether its worker still reports."""
  found = []
  for w in ctx["workers"]:
    for g in (g for g in w.get("outstanding") or [] if g.get("run_id") == ctx["run"]["run_id"] and g["op"] != "sample"):
      found.append({"op": g["op"], "count": g["count"], "age_s": round(ctx["now"] - g["since"]), "pod": w.get("pod") or w["id"]})
      found[-1]["holder_alive"] = alive(w, ctx["now"])
  return found


def failures(ctx: dict) -> dict | None:
  if not (failed := [s for s in ctx["trainer"] if s.get("status") == "failed"]):
    return None
  ops = [{"operation": s["operation"], "error_type": s.get("error_type"), "at": iso(s["at"]), "pod": s.get("pod")} for s in reversed(failed[-5:])]
  last, run_id = failed[-1], ctx["run"]["run_id"]
  logs = read("logs", run_id=run_id, pod=last.get("pod"), q=last.get("error_type"), since=iso(last["started_at"] - 60), until=iso(last["at"] + 60))
  return {"count": len(failed), "ops": ops, "logs": logs}


def relaunches(ctx: dict) -> list[dict]:
  """Worker processes that replaced an earlier one in the same pod during this run: a crash or a container restart."""
  by_pod: dict[str, list[dict]] = {}
  for w in sorted(ctx["workers"], key=lambda w: w["started_at"]):
    if w.get("pod") and w["started_at"] >= (ctx["run"]["created_at"] or 0) - 60:
      by_pod.setdefault(w["pod"], []).append(w)
  found = [
    {"pod": b["pod"], "stopped_at": a.get("heartbeat_at", b["started_at"]), "started_at": b["started_at"]}
    for p in by_pod.values()
    for a, b in zip(p, p[1:])
  ]
  return sorted(found, key=lambda p: p["started_at"])


def saves(ctx: dict) -> dict | None:
  if not (found := [s for s in ctx["trainer"] if s["operation"] == "save_state"]) and not ctx["steps"]:
    return None
  last = max((s["at"] for s in found if s.get("status") == "succeeded"), default=None)
  return present(
    count=len(found),
    failed=sum(s.get("status") == "failed" for s in found),
    last_at=iso(last) if last else None,
    steps_since_last=sum(s["at"] > (last or 0) for s in ctx["steps"]),
    slowest_s=round(max(s.get("elapsed_seconds", 0) for s in found)) if found else None,
  )


def samplers(ctx: dict) -> dict | None:
  if not (sends := [s for s in ctx["ops"] if s.get("role") == "sampler"]):
    return None
  found = {
    "sends": len(sends),
    "cancelled": sum(s.get("status") == "cancelled" for s in sends),
    "failed": sum(s.get("status") == "failed" for s in sends),
  }
  found["median_s"] = round(statistics.median(s.get("elapsed_seconds", 0) for s in sends), 2)
  # Answered sends only: a fast failure is not a replica's speed.
  by_pod: dict[str, list[float]] = {}
  for s in sends:
    if s.get("status") == "succeeded" and s.get("pod"):
      by_pod.setdefault(s["pod"], []).append(s.get("elapsed_seconds", 0))
  medians = {pod: statistics.median(v) for pod, v in by_pod.items() if len(v) >= 20}
  if len(medians) >= 2:
    for end, pick in (("slowest", max), ("fastest", min)):
      found[end] = {"pod": (pod := pick(medians, key=medians.get)), "median_s": round(medians[pod], 2)}
  return found


def config(ctx: dict) -> list[dict] | str | None:
  """Workers whose echoed settings differ from what the run asked for."""
  if not (echoed := [w for w in ctx["workers"] if w.get("config")]):
    return "no echo"
  settings, found = ctx["run"]["settings"], []
  for w in echoed:
    asked, running = settings.get("openrl.fine_tuning_type"), w["config"].get("env", {}).get("OPEN_RL_FINE_TUNING_TYPE")
    if asked and running and asked != running:
      found.append({"pod": w.get("pod"), "setting": "openrl.fine_tuning_type", "asked": asked, "running": running})
    if w.get("role") == "trainer" and settings.get("openrl.trainer_gpus") and not w.get("gpus"):
      found.append({"pod": w.get("pod"), "setting": "openrl.trainer_gpus", "asked": settings["openrl.trainer_gpus"], "running": 0})
  return found or None


def goodput(ctx: dict) -> dict | None:
  """Share of the run's span the trainer spent training, and where the rest went (Google ML goodput)."""
  if not (ops := ctx["trainer"]):
    return None
  start = ops[0]["started_at"]
  wall = max((ops[-1]["at"] if ctx["finished"] else ctx["now"]) - start, 1.0)
  spent = {kind: sum(s.get("elapsed_seconds", 0) for s in ops if s["operation"] in names) for kind, names in BUCKETS.items()}
  idle = max(0.0, wall - sum(spent.values()))
  split = {"other": round(idle)}
  # Trainer idle time with a sample in flight is rollout; with nothing in flight the server waits on the client.
  if sends := [s for s in ctx["ops"] if s["operation"] == "sample"]:
    rollout = min(idle, max(0.0, covered(ops + sends, start, start + wall) - covered(ops, start, start + wall)))
    split = {"rollout": round(rollout), "client": round(idle - rollout)}
  first = round(ctx["steps"][0]["at"] - start) if ctx["steps"] else None
  return {
    "goodput": round(spent["train"] / wall, 3),
    "seconds": {"wall": round(wall), **{k: round(v) for k, v in spent.items()}, **split},
    "first_step_after_s": first,
  }


def covered(records: list[dict], low: float, high: float) -> float:
  """Seconds of [low, high] during which at least one of the records was in flight."""
  return sum(
    b - a for a, b in fleet.merge([(max(r["started_at"], low), min(r["at"], high)) for r in records if r["at"] > low and r["started_at"] < high])
  )


def since_last_op(ctx: dict) -> int | None:
  # The snapshot time lags records written after it, so a fresh op can look a few seconds in the future.
  return max(0, round(ctx["now"] - ctx["ops"][-1]["at"])) if ctx["ops"] else None


def summary(ctx: dict) -> dict:
  steps, ops, run, ledger = ctx["steps"], ctx["trainer"], ctx["run"], dict(ctx["ledger"])
  gaps = [b["at"] - a["at"] for a, b in zip(steps, steps[1:])]
  if "enqueued" in ledger and ctx["queued"] is not None:
    ledger["lost"] = ledger["enqueued"] - ledger.get("popped", 0) - ctx["queued"]
  poisoned = [s for s in ops if s.get("nonfinite")]
  first = present(operation=poisoned[0]["operation"], step=poisoned[0].get("step"), metrics=poisoned[0]["nonfinite"]) if poisoned else None
  norms = [float(f"{s['metrics']['grad_norm:mean']:.4g}") for s in steps if "grad_norm:mean" in s.get("metrics", {})]
  restarts = None
  if relaunched := relaunches(ctx):
    p = relaunched[0]
    logs = read("logs", run_id=run["run_id"], pod=p["pod"], since=iso(p["stopped_at"] - 120), until=iso(p["started_at"] + 60))
    restarts = {
      "workers": [{"pod": p["pod"], "stopped_reporting_at": iso(p["stopped_at"]), "restarted_at": iso(p["started_at"])} for p in relaunched],
      "logs": logs,
    }
  unplaced = [
    {"name": w["name"], "role": w["role"], "age_s": round(w["age_seconds"] or 0), "message": w["message"]} for w in run["workloads"] if not w["node"]
  ]
  incidents = sorted((i for pod in run["pods"] for i in pod["incidents"]), key=lambda i: -i["at"])[:5]
  return present(
    lifecycle=run["lifecycle"],
    steps=len(steps),
    step_seconds={"median": round(statistics.median(gaps), 1), "last": round(gaps[-1], 1)} if gaps else None,
    since_last_step_s=round(ctx["now"] - steps[-1]["at"]) if steps and not ctx["finished"] else None,
    since_last_op_s=since_last_op(ctx) if not ctx["finished"] else None,
    trainer_idle_s=max(0, round(ctx["now"] - ops[-1]["at"])) if ops and not ctx["finished"] else None,
    last_op={"operation": ops[-1]["operation"], "status": ops[-1].get("status"), "at": iso(ops[-1]["at"])} if ops else None,
    last_metrics={k: float(f"{v:.6g}") for s in ops[-50:] for k, v in s.get("metrics", {}).items()},
    goodput=goodput(ctx),
    requests=present(unanswered=unanswered(ctx), ledger=ledger, samples_queued=ctx["samples_queued"] or None),
    failures=failures(ctx),
    relaunches=restarts,
    saves=saves(ctx),
    optimizer=present(nonfinite={"count": len(poisoned), "first": first} if poisoned else None, grad_norm_last=norms[-5:]),
    samplers=samplers(ctx),
    config=config(ctx),
    placement=present(recorded=len(ctx["lanes"]), unplaced=unplaced) or None,
    incidents=[present(at=iso(i["at"]), pod=i["pod"], reason=i["reason"], message=(i.get("message") or "")[:200]) for i in incidents],
  )


def line(ctx: dict) -> dict:
  """A run in one line, for a session or the fleet, enough to see which run to open."""
  steps, held = ctx["steps"], unanswered(ctx)
  return {
    "lifecycle": ctx["run"]["lifecycle"],
    "steps": len(steps),
    "since_last_op_s": since_last_op(ctx),
    "since_last_step_s": round(ctx["now"] - steps[-1]["at"]) if steps else None,
    "median_step_s": round(statistics.median(b["at"] - a["at"] for a, b in zip(steps, steps[1:]))) if len(steps) > 1 else None,
    "trainer_requests_unanswered": sum(o["count"] for o in held),
    "held_by_dead_worker": sum(o["count"] for o in held if not o["holder_alive"]),
    "samples_queued": ctx["samples_queued"],
    "failures": sum(s.get("status") == "failed" for s in ctx["trainer"]),
    "relaunches": len(relaunches(ctx)),
  }
