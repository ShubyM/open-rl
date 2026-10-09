"""A session's recorded operations as a trace for ui.perfetto.dev.

The Tinker SDK's export_session_trace() asks the API server for this. It is the
Chrome JSON trace format, which Perfetto opens directly: one process per run and
role, and as many lanes as that role ran operations at once, since slices on
one track must nest.
"""

from typing import Any


def lanes(samples: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
  """Give each operation the first lane free at its start."""
  ends: list[float] = []
  placed = []
  for sample in sorted(samples, key=lambda s: s["started_at"]):
    lane = next((i for i, end in enumerate(ends) if end <= sample["started_at"]), len(ends))
    if lane == len(ends):
      ends.append(sample["at"])
    else:
      ends[lane] = sample["at"]
    placed.append((lane, sample))
  return placed


def chrome_trace(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
  events: list[dict[str, Any]] = []
  pid = 0
  for run_id, samples in sorted(runs.items()):
    by_role: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
      if sample.get("started_at") is not None and sample.get("at") is not None:
        by_role.setdefault(sample.get("role") or "process", []).append(sample)
    for role, ops in sorted(by_role.items()):
      pid += 1
      events.append({"ph": "M", "name": "process_name", "pid": pid, "args": {"name": f"{role} · {run_id[:8]}"}})
      for lane, op in lanes(ops):
        events.append(
          {
            "ph": "X",
            "name": op.get("operation") or "operation",
            "cat": role,
            "pid": pid,
            "tid": lane,
            "ts": op["started_at"] * 1e6,
            "dur": max(op["at"] - op["started_at"], 0) * 1e6,
            "args": {k: op[k] for k in ("request_id", "status", "error_type", "run_id", "queue_seconds") if op.get(k) is not None},
          }
        )
  return {"traceEvents": events, "displayTimeUnit": "ms"}
