"""Joins of placements, GPU turns, ops and pod events behind the activity view."""

from datetime import datetime

from server.dashboard.history import reconcile


def incidents(run: dict, start: float, end: float) -> list[dict]:
  """Latest distinct timestamped failures/restarts; current conditions are not events."""
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
  return list(unique.values())[:3]


def segments(state: dict, run_id=None, node_id=None) -> list[dict]:
  live = {p["id"]: p for p in state["placements"]}
  runs = {r["run_id"]: r for r in state["runs"]}
  observed = datetime.fromisoformat(state["observed_at"]).timestamp()
  available = state["cluster"]["available"] and state["cluster"]["scheduler"].get("available", True)
  found = []
  past = {p["id"]: p for p in state["history"]}
  for identity, current in live.items():
    past.update((p["id"], p) for p in reconcile(past.pop(identity, {}), current))
  for entry in past.values():
    started = entry.get("first_seen")
    if started is None:
      started = datetime.fromisoformat(entry["created_at"]).timestamp() if entry.get("created_at") else observed
    placement = {
      **entry,
      "start": started,
      "end": observed if entry["id"] in live else entry.get("last_seen", observed),
      "ended": entry["id"] not in live and available,
    }
    if len(placement["run_ids"]) > 1:
      for member in placement["run_ids"]:
        model = runs.get(member, {}).get("model") or placement["label"]
        found.append(
          {
            **placement,
            "id": f"{entry['id']}:{member}",
            "allocation_id": entry["id"],
            "run_ids": [member],
            "label": f"{model.split('/')[-1]} · {member[:8]}",
          }
        )
    else:
      found.append(placement)
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
  return all(
    not sample.get(key) or not placement.get(key) or sample[key] == placement[key] for key in ("role", "node", "runtime_id", "pod", "pod_uid")
  )


def activity(state: dict, start: float, end: float, operations: dict, turns: dict, run_id=None, node_id=None) -> dict:
  """Actual GPU grants when recorded; completed operations otherwise. Never utilization."""
  shared_runs = {r["run_id"] for r in state["runs"] if r.get("shared_runtime")}
  return {
    "placements": [
      placement_activity(
        p, start, end, operations, turns, p.get("allocation_id") or p.get("shared_runtime") or bool(shared_runs.intersection(p["run_ids"]))
      )
      for p in segments(state, run_id, node_id)
    ]
  }


def placement_activity(placement, start, end, operations, turns, shared):
  held = turns.get(placement["name"], {})
  samples = [
    t
    for t in held.get("samples", [])
    if matches(t, placement) and t["started_at"] < min(end, placement["end"]) and t["at"] > max(start, placement["start"])
  ]
  intervals, named = [], {}

  def add(name, low, high):
    span = [max(start, placement["start"], low), min(end, placement["end"], high)]
    if span[1] > span[0]:
      intervals.append(span)
      if name:
        named.setdefault(name, []).append(span)

  error = None
  if samples:
    for turn in samples:
      if not shared:
        add(None, turn["started_at"], turn["at"])
      for op in turn.get("ops", []):
        if not shared or op.get("run_id") in placement["run_ids"]:
          add(op["name"], max(op["start"], turn["started_at"]), min(op["end"], turn["at"]))
  else:
    for identity in placement["run_ids"]:
      result = operations.get(identity, {})
      error = result.get("error") or error
      for sample in result.get("samples", []):
        if sample.get("run_id", identity) == identity and sample.get("role") == placement.get("role") and matches(sample, placement):
          add(sample.get("operation", "operation"), sample.get("started_at", sample["at"] - sample.get("elapsed_seconds", 0)), sample["at"])
  return {
    **placement,
    "intervals": merge(intervals),
    "ops": {name: merge(spans) for name, spans in named.items()},
    "exact": bool(samples),
    "error": error,
    "warning": held.get("error"),
  }
