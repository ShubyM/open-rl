"""Facts about a run, joined from its records. What it did, what failed, and what it waits on, with no verdicts.

A block appears only when it has content, so an absent block means nothing of that kind was recorded. A log pointer
is a read against /api/v1/inspect that an agent runs as given.
"""

import statistics
from datetime import UTC, datetime

from server.dashboard import inspection


def read(route: str, **args) -> dict:
  return {"read": route, "args": {k: v for k, v in args.items() if v is not None}}


def iso(at: float) -> str:
  return datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds")


def present(**blocks) -> dict:
  return {k: v for k, v in blocks.items() if v not in (None, [], {})}


def trainer_ops(ctx: dict) -> list[dict]:
  return sorted((s for s in ctx["samples"] if s.get("role") != "sampler"), key=lambda s: s["at"])


def optim_steps(ctx: dict) -> list[dict]:
  return [s for s in trainer_ops(ctx) if s["operation"] == "optim_step" and s.get("status") == "succeeded"]


def unanswered(ctx: dict) -> list[dict]:
  """Trainer requests the client waits on, with the pod holding each and whether its worker still reports."""
  held = [o for o in (ctx["run"].get("progress") or {}).get("outstanding") or [] if o.get("op") != "sample"]
  return [
    {"op": o["op"], "count": o.get("count", 0), "age_s": round(o.get("age_seconds", 0)), "pod": o.get("pod"), "holder_alive": o.get("process_alive")}
    for o in held
  ]


def requests(ctx: dict) -> dict:
  progress = ctx["run"].get("progress") or {}
  ledger = {k: v for k, v in ctx["ledger"].items() if v is not None}
  queued = (progress.get("queued") or {}).get("trainer")
  if "enqueued" in ledger and queued is not None:
    ledger["lost"] = ledger["enqueued"] - ledger.get("popped", 0) - queued
  return present(unanswered=unanswered(ctx), ledger=ledger)


def failures(ctx: dict) -> dict | None:
  failed = [s for s in trainer_ops(ctx) if s.get("status") == "failed"]
  if not failed:
    return None
  latest = failed[-1]
  ops = [{"operation": s["operation"], "error_type": s.get("error_type"), "at": iso(s["at"]), "pod": s.get("pod")} for s in reversed(failed[-5:])]
  logs = read(
    "logs",
    run_id=ctx["run"]["run_id"],
    pod=latest.get("pod"),
    q=latest.get("error_type"),
    since=iso(latest["started_at"] - 60),
    until=iso(latest["at"] + 60),
  )
  return {"count": len(failed), "ops": ops, "logs": logs}


def relaunches(ctx: dict) -> list[dict]:
  """Worker processes that replaced an earlier one in the same pod during this run: a crash or a container restart."""
  by_pod: dict[str, list[dict]] = {}
  for p in ctx["workers"]:
    if p.get("pod") and p.get("role") != "dispatcher" and p.get("started_at", 0) >= ctx["since"] - 60:
      by_pod.setdefault(p["pod"], []).append(p)
  found = []
  for procs in by_pod.values():
    procs.sort(key=lambda p: p["started_at"])
    found += [
      {"pod": later["pod"], "stopped_at": earlier.get("heartbeat_at", later["started_at"]), "started_at": later["started_at"]}
      for earlier, later in zip(procs, procs[1:])
    ]
  return sorted(found, key=lambda p: p["started_at"])


def relaunch_facts(ctx: dict) -> dict | None:
  found = relaunches(ctx)
  if not found:
    return None
  first = found[0]
  workers = [{"pod": p["pod"], "stopped_reporting_at": iso(p["stopped_at"]), "restarted_at": iso(p["started_at"])} for p in found]
  logs = read("logs", run_id=ctx["run"]["run_id"], pod=first["pod"], since=iso(first["stopped_at"] - 120), until=iso(first["started_at"] + 60))
  return {"workers": workers, "logs": logs}


def pool(ctx: dict) -> dict | None:
  p = ctx.get("pool") or {}
  if not p.get("waiting"):
    return None
  return present(
    name=p["name"],
    unanswered=p["waiting"],
    oldest_s=round(ctx["now"] - p["oldest"]) if p.get("oldest") else None,
    dispatcher=p.get("dispatcher"),
    dispatcher_alive=p.get("alive"),
  )


def saves(ctx: dict) -> dict | None:
  found, steps = [s for s in trainer_ops(ctx) if s["operation"] == "save_state"], optim_steps(ctx)
  if not found and not steps:
    return None
  last = max((s["at"] for s in found if s.get("status") == "succeeded"), default=None)
  return present(
    count=len(found),
    failed=sum(s.get("status") == "failed" for s in found),
    last_at=iso(last) if last else None,
    steps_since_last=sum(s["at"] > (last or 0) for s in steps),
    slowest_s=round(max(s.get("elapsed_seconds", 0) for s in found)) if found else None,
  )


def optimizer(ctx: dict) -> dict:
  poisoned = [s for s in trainer_ops(ctx) if s.get("nonfinite")]
  norms = [s["metrics"]["grad_norm:mean"] for s in optim_steps(ctx) if "grad_norm:mean" in s.get("metrics", {})]
  first = poisoned[0] if poisoned else None
  nonfinite = (
    {"count": len(poisoned), "first": present(operation=first["operation"], step=first.get("step"), metrics=first["nonfinite"])} if first else None
  )
  return present(nonfinite=nonfinite, grad_norm_last=[float(f"{n:.4g}") for n in norms[-5:]])


def samplers(ctx: dict) -> dict | None:
  sends = [s for s in ctx["samples"] if s.get("role") == "sampler"]
  if not sends:
    return None
  found = {
    "sends": len(sends),
    "retried": sum(s.get("status") == "retried" for s in sends),
    "failed": sum(s.get("status") == "failed" for s in sends),
    "median_s": round(statistics.median(s.get("elapsed_seconds", 0) for s in sends), 2),
  }
  # Answered sends to replica pods only: a fast failure, or a send recorded against the router URL, is not a replica's speed.
  by_pod: dict[str, list[float]] = {}
  for s in sends:
    if s.get("status") == "succeeded" and s.get("pod") and "://" not in s["pod"]:
      by_pod.setdefault(s["pod"], []).append(s.get("elapsed_seconds", 0))
  medians = {pod: statistics.median(v) for pod, v in by_pod.items() if len(v) >= 20}
  if len(medians) >= 2:
    for end, pick in (("slowest", max), ("fastest", min)):
      pod = pick(medians, key=medians.get)
      found[end] = {"pod": pod, "median_s": round(medians[pod], 2)}
  return found


def config(ctx: dict) -> list[dict] | str | None:
  """Workers whose echoed settings differ from what the run asked for."""
  echoed = [w for w in ctx["workers"] if w.get("config")]
  if not echoed:
    return "no echo"
  settings, found = ctx["run"].get("settings") or {}, []
  for w in echoed:
    asked, running = settings.get("openrl.fine_tuning_type"), w["config"].get("env", {}).get("OPEN_RL_FINE_TUNING_TYPE")
    if asked and running and asked != running:
      found.append({"pod": w.get("pod"), "setting": "openrl.fine_tuning_type", "asked": asked, "running": running})
    if w.get("role") == "trainer" and settings.get("openrl.trainer_gpus") and not w.get("gpus"):
      found.append({"pod": w.get("pod"), "setting": "openrl.trainer_gpus", "asked": settings["openrl.trainer_gpus"], "running": 0})
  return found or None


def placement(ctx: dict) -> dict | None:
  if not ctx["scheduler"]:
    return None
  unplaced = [
    {
      "name": w["name"],
      "role": w.get("role"),
      "age_s": round((w.get("worker") or {}).get("age_seconds") or 0),
      "message": w.get("placed_message") or w.get("reason"),
    }
    for w in ctx["run"].get("workloads") or []
    if not w.get("node_name")
  ]
  recorded = len(ctx.get("placements") or [])
  return present(recorded=recorded, unplaced=unplaced) if recorded or unplaced else None


def incidents(ctx: dict) -> list[dict]:
  events = inspection.incidents(ctx["run"], ctx["since"], ctx["now"])[:5]
  return [present(at=iso(e["at"]), pod=e["pod"], reason=e["reason"], message=(e.get("message") or "")[:200]) for e in events]


BUCKETS = {"train": {"forward_backward", "optim_step"}, "checkpoint": {"save_state", "save_weights_for_sampler"}, "forward": {"forward"}}


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


def covered(records: list[dict], low: float, high: float) -> float:
  """Seconds of [low, high] during which at least one of the records was in flight."""
  return sum(
    b - a for a, b in inspection.merge([(max(r["started_at"], low), min(r["at"], high)) for r in records if r["at"] > low and r["started_at"] < high])
  )


def waiting(ctx: dict) -> dict | None:
  """For a live run, what is in flight now and how long since anything finished: what a stuck run waits on."""
  if ctx["finished"] or not ctx["samples"]:
    return None
  now, trainer = ctx["now"], trainer_ops(ctx)
  held = unanswered(ctx)
  pool_ = ctx.get("pool") or {}
  return {
    "since_last_op_s": since_last_op(ctx),
    "trainer_idle_s": round(now - trainer[-1]["at"]) if trainer else None,
    "trainer_requests": sum(o["count"] for o in held),
    "oldest_trainer_request_s": max(o["age_s"] for o in held) if held else None,
    "samples_unanswered": pool_.get("waiting", 0),
    "oldest_unanswered_sample_s": round(now - pool_["oldest"]) if pool_.get("oldest") else None,
  }


def since_last_op(ctx: dict) -> int | None:
  # The snapshot time lags records written after it, so a fresh op can look a few seconds in the future.
  return max(0, round(ctx["now"] - max(s["at"] for s in ctx["samples"]))) if ctx["samples"] else None


def summary(ctx: dict) -> dict:
  steps, ops = optim_steps(ctx), trainer_ops(ctx)
  gaps = [b["at"] - a["at"] for a, b in zip(steps, steps[1:])]
  last = ops[-1] if ops else None
  live = not ctx["finished"]
  return present(
    lifecycle=ctx["run"].get("lifecycle"),
    steps=len(steps),
    step_seconds={"median": round(statistics.median(gaps), 1), "last": round(gaps[-1], 1)} if gaps else None,
    since_last_step_s=round(ctx["now"] - steps[-1]["at"]) if live and steps else None,
    last_op={"operation": last["operation"], "status": last.get("status"), "at": iso(last["at"])} if last else None,
    last_metrics={k: float(f"{v:.6g}") for s in ops[-50:] for k, v in s.get("metrics", {}).items()},
    goodput=goodput(ctx),
    waiting=waiting(ctx),
    requests=requests(ctx),
    failures=failures(ctx),
    relaunches=relaunch_facts(ctx),
    pool=pool(ctx),
    saves=saves(ctx),
    optimizer=optimizer(ctx),
    samplers=samplers(ctx),
    config=config(ctx),
    placement=placement(ctx),
    incidents=incidents(ctx),
  )


def line(ctx: dict) -> dict:
  """A run in one line, for a session or the fleet, enough to see which run to open."""
  steps, held = optim_steps(ctx), unanswered(ctx)
  now = ctx["now"]
  return {
    "lifecycle": ctx["run"].get("lifecycle"),
    "steps": len(steps),
    "since_last_op_s": since_last_op(ctx),
    "since_last_step_s": round(now - steps[-1]["at"]) if steps else None,
    "median_step_s": round(statistics.median(b["at"] - a["at"] for a, b in zip(steps, steps[1:]))) if len(steps) > 1 else None,
    "unanswered": sum(o["count"] for o in held),
    "held_by_dead_worker": sum(o["count"] for o in held if o["holder_alive"] is False),
    "failures": sum(s.get("status") == "failed" for s in trainer_ops(ctx)),
    "relaunches": len(relaunches(ctx)),
  }
