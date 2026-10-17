"""Joins of worker records, placements, ops and pod events behind the activity view."""

from datetime import datetime

from server.telemetry.ops import alive


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


def segments(state: dict, run_id=None, node_id=None) -> list[dict]:
  """One lane per worker process from the record it keeps, plus placed workloads whose
  process has not reported yet. A lane on a shared runtime splits into one per run."""
  observed = datetime.fromisoformat(state["observed_at"]).timestamp()
  runs = state["runs"]
  uuids = {d["uuid"]: d["id"] for node in state["cluster"]["nodes"] for d in node["devices"] if d.get("uuid")}
  placed = {p["pod_uid"]: p for p in state["placements"] if p.get("pod_uid")}
  found = []
  for record in state["processes"]:
    if record.get("role") == "dispatcher":
      continue  # a pool's CPU process: it shows in progress, not as a GPU lane
    placement = placed.pop(record.get("pod_uid"), {})
    live = alive(record, observed)
    end = observed if live else record["heartbeat_at"]
    members = [
      r for r in runs if r["runtime_id"] == record["runtime_id"] and r["created_at"] <= end and (r["completed_at"] or end) >= record["started_at"]
    ]
    ids, gpus = [r["run_id"] for r in members], record.get("gpus") or []
    # A GPU no current device slot reports keeps its UUID as its identity, so its series stay queryable.
    found.append(
      {
        **record,
        "name": placement.get("name") or record.get("pod") or record["id"],
        "label": members[0]["name"] if len(members) == 1 else record["runtime_id"] or record["id"],
        "run_ids": ids,
        "shared_runtime": record["runtime_id"] not in ids if ids else None,
        "devices": placement.get("devices") or [uuids.get(uuid, uuid) for uuid in gpus],
        "device_uuids": {uuids.get(uuid, uuid): uuid for uuid in gpus},
        "device_count": placement.get("device_count") or len(gpus),
        "start": record["started_at"],
        "end": end,
        "ended": not live,
      }
    )
  for placement in [*placed.values(), *(p for p in state["placements"] if not p.get("pod_uid"))]:
    started = datetime.fromisoformat(placement["created_at"]).timestamp() if placement.get("created_at") else observed
    found.append({**placement, "start": started, "end": observed, "ended": False})
  lanes = []
  for lane in found:
    if lane.get("shared_runtime") and lane["run_ids"]:
      for member in lane["run_ids"]:
        model = next((r["model"] for r in runs if r["run_id"] == member), None) or lane["label"]
        lanes.append(
          {
            **lane,
            "id": f"{lane['id']}:{member}",
            "allocation_id": lane["id"],
            "run_ids": [member],
            "label": f"{model.split('/')[-1]} · {member[:8]}",
          }
        )
    else:
      lanes.append(lane)
  return [p for p in lanes if (not run_id or run_id in p["run_ids"]) and (not node_id or p["node"] == node_id)]


def merge(intervals: list) -> list:
  result = []
  for start, end in sorted(intervals):
    if result and start <= result[-1][1]:
      result[-1][1] = max(end, result[-1][1])
    else:
      result.append([start, end])
  return result


def matches(sample: dict, placement: dict) -> bool:
  # A plain loop: activity calls this once per record per lane.
  for key in ("role", "node", "runtime_id", "pod", "pod_uid"):
    if (mine := sample.get(key)) and (theirs := placement.get(key)) and mine != theirs:
      return False
  return True


def activity(state: dict, start: float, end: float, operations: dict, run_id=None, node_id=None) -> dict:
  """GPU turns and recorded ops for each lane in the window. Never utilization."""
  lanes = segments(state, run_id, node_id)
  if run_id and not node_id:
    lanes += recorded_lanes(run_id, lanes, operations.get(run_id, {}).get("samples", []))
  return {"placements": [lane_activity(p, start, end, operations) for p in lanes if p["start"] <= end and p["end"] >= start]}


def recorded_lanes(run_id: str, lanes: list[dict], samples: list[dict]) -> list[dict]:
  """A vllm serve replica keeps no process record, so once its pod is gone its lane
  comes from the ops the run recorded there, which name the pod."""
  shown, pods = {p.get("pod") for p in lanes}, {}
  for sample in samples:
    if sample.get("role") == "sampler" and sample.get("pod") and sample["pod"] not in shown:
      pods.setdefault(sample["pod"], []).append(sample)
  return [
    {
      "id": f"recorded:{pod}",
      "name": pod,
      "label": pod,
      "pod": pod,
      "role": ops[0].get("role"),
      "node": "",  # not recorded for a replica
      "run_ids": [run_id],
      "runtime_id": None,
      "shared_runtime": True,
      "devices": [],
      "device_count": 0,
      "start": min(o.get("started_at") or o["at"] for o in ops),
      "end": max(o["at"] for o in ops),
      "ended": True,
    }
    for pod, ops in pods.items()
  ]


def lane_activity(lane: dict, start: float, end: float, operations: dict) -> dict:
  """A shared lane shows only its run's ops; a dedicated one also its GPU turns and worker-level ops."""
  shared = bool(lane.get("shared_runtime"))
  intervals, named, errors, partial = [], {}, [], []

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
    partial.append(result.get("partial"))
    for sample in result.get("samples", []):
      if sample.get("role") == lane.get("role") and matches(sample, lane):
        add(sample["operation"], sample.get("started_at") or sample["at"] - sample.get("elapsed_seconds", 0), sample["at"])
  return {
    **{k: v for k, v in lane.items() if k not in ("turns", "outstanding")},
    "intervals": merge(intervals),
    "ops": {name: merge(spans) for name, spans in named.items()},
    "exact": bool(turns),
    "error": next(filter(None, errors), None),
    "partial": next(filter(None, partial), None),
  }
