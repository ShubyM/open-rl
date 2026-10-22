"""Joins of placement records, worker records, ops and pod events behind the activity view."""

from datetime import datetime


def incidents(run: dict, start: float, end: float) -> list[dict]:
  """Distinct timestamped failures/restarts, newest first; current conditions are not events."""
  errors = ("OOMKilled", "CrashLoopBackOff", "Evicted")
  warnings = ("WorkerRestarted", "FailedScheduling", "Unschedulable", "FailedMount", "FailedAttachVolume")
  events = []

  def add(pod, event, timestamp):
    try:
      at = datetime.fromisoformat(timestamp).timestamp()
    except (TypeError, ValueError):
      return
    if start <= at <= end:
      events.append({**event, "pod": pod["name"], "at": at, "tone": "error" if event["reason"] in errors else "warning"})

  for pod in run.get("pods", []):
    crashing = pod.get("restarts", 0) > 0 or (pod.get("problem") or "").split(":", 1)[0] == "CrashLoopBackOff"
    for event in pod.get("events", []):
      backoff = event.get("reason") == "BackOff" and event.get("type") == "Warning"
      backoff = backoff and (crashing or "back-off restarting failed container" in (event.get("message") or "").lower())
      if event.get("reason") in errors + warnings or backoff:
        add(pod, event, event.get("last_seen_at") or event.get("first_seen_at"))
    for container in pod.get("containers", []):
      name = container["name"]
      for termination in (container if container.get("state") == "terminated" else None, container.get("last_termination")):
        if termination and termination.get("reason") == "OOMKilled":
          code = termination.get("exit_code")
          event = {"container": name, "reason": "OOMKilled", "message": f"{name} exited with code {code if code is not None else 'unknown'}"}
          add(pod, event, termination.get("finished_at"))
      if container.get("state") == "running" and container.get("restart_count", 0) > 0:
        event = {"container": name, "reason": "WorkerRestarted", "restart_count": container["restart_count"]}
        event["message"] = f"{name} is running after a restart"
        add(pod, event, container.get("started_at"))
  events.sort(key=lambda event: event["at"], reverse=True)
  unique = {}
  for event in events:
    if not event.get("container") and any(
      other.get("container") and other["pod"] == event["pod"] and other["reason"] == event["reason"] and abs(other["at"] - event["at"]) < 2
      for other in events
    ):
      continue
    unique.setdefault((event["pod"], event.get("container"), event["reason"]), event)
  return list(unique.values())


def lanes(state: dict, run_id=None, node_id=None) -> list[dict]:
  """One lane per placement record, with its worker process's phase and GPU turns. A lane on a shared runtime
  becomes one lane per run, which shows only that run's ops."""
  observed = datetime.fromisoformat(state["observed_at"]).timestamp()
  runs, found = state["runs"], []
  for record in state["placement_records"]:
    start, end = record["placed_at"], record["released_at"] or observed
    members = [r for r in runs if r["runtime_id"] == record["model_id"] and r["created_at"] <= end and (r["completed_at"] or end) >= start]
    # Pod names repeat across placements of one Workload; the process ran inside this placement's span.
    pod = record["pod"]
    processes = [p for p in state["processes"] if pod and p.get("pod") == pod and p["started_at"] <= end and p["heartbeat_at"] >= start]
    ids = [r["run_id"] for r in members]
    lane = {
      "id": f"{record['workload']}@{record['placed_at']:.0f}",
      "name": record["workload"],
      "label": members[0]["name"] if len(members) == 1 else record["model_id"],
      "run_ids": ids,
      "runtime_id": record["model_id"],
      "shared_runtime": record["model_id"] not in ids if ids else None,
      "role": record["role"],
      "node": record["node"] or "",
      "pod": record["pod"],
      "devices": record["gpu_uuids"],
      "device_count": len(record["gpu_uuids"]),
      "phase": max(processes, key=lambda p: p["started_at"])["phase"] if processes else None,
      "turns": [turn for p in processes for turn in p.get("turns") or []],
      "start": start,
      "end": end,
      "ended": record["released_at"] is not None,
    }
    if lane["shared_runtime"]:
      for member in members:
        label = f"{(member['model'] or lane['label']).split('/')[-1]} · {member['run_id'][:8]}"
        found.append({**lane, "id": f"{lane['id']}:{member['run_id']}", "allocation_id": lane["id"], "run_ids": [member["run_id"]], "label": label})
    else:
      found.append(lane)
  return [p for p in found if (not run_id or run_id in p["run_ids"]) and (not node_id or p["node"] == node_id)]


def merge(intervals: list) -> list:
  result = []
  for start, end in sorted(intervals):
    if result and start <= result[-1][1]:
      result[-1][1] = max(end, result[-1][1])
    else:
      result.append([start, end])
  return result


def matches(sample: dict, placement: dict) -> bool:
  return all(sample[key] == placement[key] for key in ("role", "node", "runtime_id", "pod") if sample.get(key) and placement.get(key))


def activity(state: dict, start: float, end: float, operations: dict, run_id=None, node_id=None) -> dict:
  """GPU turns and recorded ops for each lane in the window. Never utilization."""
  return {"placements": [lane_activity(p, start, end, operations) for p in lanes(state, run_id, node_id) if p["start"] <= end and p["end"] >= start]}


CHART_POINTS = 1000  # sampler requests per chart; a chart cannot show more apart


def chart_samples(samples: list[dict], start: float, end: float) -> list[dict]:
  """A run's op records for its charts: every trainer op, which carries the metrics, and the slowest sampler
  request per time bucket, so a long run ships its peaks instead of tens of thousands of requests."""
  keep = ("at", "started_at", "operation", "role", "status", "elapsed_seconds", "metrics", "nonfinite")
  width = (end - start) / CHART_POINTS or 1
  charted, slowest = [], {}
  for sample in samples:
    if start <= sample["at"] <= end:
      sample = {k: sample[k] for k in keep if k in sample}
      bucket = int((sample["at"] - start) / width)
      if sample.get("role") != "sampler":
        charted.append(sample)
      elif sample.get("elapsed_seconds", 0) >= slowest.get(bucket, {}).get("elapsed_seconds", -1):
        slowest[bucket] = sample
  return sorted([*charted, *slowest.values()], key=lambda sample: sample["at"])


def lane_activity(lane: dict, start: float, end: float, operations: dict) -> dict:
  """A shared lane shows only its run's ops; a dedicated one also its GPU turns and worker-level ops."""
  shared = bool(lane.get("shared_runtime"))
  intervals, named, errors = [], {}, []

  def add(name, low, high):
    span = [max(start, lane["start"], low), min(end, lane["end"], high)]
    if span[1] > span[0]:
      intervals.append(span)
      if name:
        named.setdefault(name, []).append(span)

  turns = [] if shared else [t for t in lane.get("turns") or [] if t[0] < min(end, lane["end"]) and t[1] > max(start, lane["start"])]
  for low, high in turns:
    add(None, low, high)
  for identity in lane["run_ids"] if shared else {*lane["run_ids"], lane.get("runtime_id")} - {None}:
    result = operations.get(identity, {})
    errors.append(result.get("error"))
    for sample in result.get("samples", []):
      if sample.get("role") == lane.get("role") and matches(sample, lane):
        add(sample["operation"], sample.get("started_at") or sample["at"] - sample.get("elapsed_seconds", 0), sample["at"])
  return {
    **{k: v for k, v in lane.items() if k not in ("turns", "outstanding")},
    "intervals": merge(intervals),
    "ops": {name: merge(spans) for name, spans in named.items()},
    "exact": bool(turns),
    "error": next(filter(None, errors), None),
  }


def steps(samples: list[dict]) -> list[dict]:
  """Trainer ops grouped by the step they belong to: the step's window, and per op its count, time and metric spread."""
  found: dict[int, dict] = {}
  completed = 0  # records from before ops carried their step count optim_steps instead
  for s in sorted((s for s in samples if s.get("role") != "sampler"), key=lambda s: s["at"]):
    number = s["step"] if isinstance(s.get("step"), int) else completed
    completed += s["operation"] == "optim_step" and s.get("status") == "succeeded"
    step = found.setdefault(number, {"step": number, "started_at": s["started_at"], "ended_at": s["at"], "ops": {}, "nonfinite": set()})
    step["started_at"], step["ended_at"] = min(step["started_at"], s["started_at"]), max(step["ended_at"], s["at"])
    op = step["ops"].setdefault(s["operation"], {"count": 0, "failed": 0, "seconds": 0.0, "metrics": {}})
    op["count"] += 1
    op["failed"] += s.get("status") != "succeeded"
    op["seconds"] += s.get("elapsed_seconds", 0.0)
    for name, value in s.get("metrics", {}).items():
      spread = op["metrics"].setdefault(name, {"min": value, "max": value, "sum": 0.0})
      spread.update(min=min(spread["min"], value), max=max(spread["max"], value), sum=spread["sum"] + value)
    step["nonfinite"].update(s.get("nonfinite", ()))
  for step in found.values():
    step["nonfinite"] = sorted(step["nonfinite"])
    for op in step["ops"].values():
      op["metrics"] = {name: {"min": m["min"], "max": m["max"], "mean": m["sum"] / op["count"]} for name, m in op["metrics"].items()}
  return [found[k] for k in sorted(found)]
