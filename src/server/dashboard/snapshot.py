"""The one view of the cluster the pages and agents read: runs joined to their
workers and their progress, current placements on GPUs, nodes with their
devices, and the records worker processes keep about themselves."""

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime

from server import openrl_status
from server.model_metadata import decode_model_metadata
from server.session_registry import SessionRegistry
from server.store import StateStore, request_queue, sampling_queue
from server.telemetry import kubernetes, ops
from server.worker_manager import FAILED, READY, STARTING, STOPPED, WAITING_FOR_PLACEMENT

CACHE_SECONDS = 5
HISTORY_WINDOW_SECONDS = 24 * 3600
METADATA_PREFIX = "open_rl:model_meta:"
# A request on a worker longer than this, or three median steps, is stalled.
STALL_SECONDS = 600


class InventoryWorkers:
  """The worker manager's inspect_owner, answered from the inventory already read."""

  def __init__(self, scheduler: dict) -> None:
    self.scheduler = scheduler

  def inspect_owner(self, owner: str) -> list[dict]:
    if not self.scheduler["available"]:
      raise RuntimeError(self.scheduler["error"] or "worker inventory unavailable")
    return [w["worker"] for w in self.scheduler["workloads"] if w["worker"]["owner"] == owner]


def cluster_status_context(store: StateStore, cluster: dict) -> openrl_status.StatusContext:
  """The gateway's status semantics over the dashboard's own inventory."""
  if cluster["scheduler"]["installed"] is False:
    return openrl_status.StatusContext(store, SessionRegistry(store), None, "none")
  return openrl_status.StatusContext(store, SessionRegistry(store), InventoryWorkers(cluster["scheduler"]), "scheduler")


def display_status(run: dict, launcher: str) -> str:
  """One label for the run list. Ready means the model is loaded, not that an op is running."""
  if run["status"] != "active":
    return run["status"].capitalize()
  if run["workers_error"]:
    return "Unknown"
  if any(pod["problem"] for pod in run["pods"]):
    return "Unhealthy"
  if launcher == "single_process":
    return "Active"
  states = {worker["state"] for worker in run["workers"]}
  for state, label in ((FAILED, "Unhealthy"), (WAITING_FOR_PLACEMENT, "Queued"), (STARTING, "Starting")):
    if state in states:
      return label
  if states == {READY}:
    return "Ready"
  if states == {STOPPED}:
    return "Stopped"
  if states:
    return "Unknown"
  return "Unassigned" if run["session_live"] else "Ended"


async def add_status(ctx: openrl_status.StatusContext, run: dict) -> None:
  """Worker and session state as the gateway reports it, for runs that may still have workers."""
  if run["status"] == "active":
    try:
      run.update(await asyncio.wait_for(openrl_status.model_status(ctx, run["run_id"]), timeout=5) or {})
    except Exception as exc:
      run.update(workers_error=f"status unavailable: {type(exc).__name__}", summary="status unavailable")
  run["summary"] = "; ".join([run["summary"], *(pod["problem"] for pod in run["pods"] if pod["problem"])])
  run["display_status"] = display_status(run, ctx.worker_launcher)


async def metadata_of(store: StateStore) -> list[dict]:
  steps = await store.scan_values(ops.steps_key(""))
  rows = []
  for key, value in (await store.scan_values(METADATA_PREFIX)).items():
    run_id = key.removeprefix(METADATA_PREFIX)
    rows.append({**json.loads(value), "model_id": run_id, "steps": int(steps.get(ops.steps_key(run_id), 0))})
  return rows


def join_runs(metadata: list[dict], state: dict) -> list[dict]:
  """LoRA workers serve a base-model runtime shared by every unfinished LoRA
  run on it; an FFT worker's modelID is the run itself."""
  workloads = state["scheduler"]["workloads"]
  claims = state["devices"]["claims"]
  runs = []
  for row in metadata:
    run_id = row["model_id"]
    meta = decode_model_metadata(json.dumps(row))
    kind = "lora" if meta.fine_tuning_type == "lora" else "fft"
    runtime = meta.runtime(run_id)
    shared = runtime != run_id
    matched = [w for w in workloads if w.get("model_id") == runtime and w.get("training_kind") == kind and not (meta.status != "active" and shared)]
    pods = []
    for pod in state["pods"]:
      owned = [w for w in matched if w["uid"] in pod["owner_uids"] or (w.get("pod_name") == pod["name"] and pod.get("worker") == w["name"])]
      if owned:
        pod = pod.copy()
        pod.update(role=owned[0].get("role"), shared_runtime=shared, runtime_id=runtime)
        pod["devices"] = sorted({device for w in owned for device in claims.get(w.get("claim_name"), [])})
        pods.append(pod)
    runs.append(
      {
        "run_id": run_id,
        "name": row.get("name") or row.get("run_name") or f"{(meta.base_model or 'Run').split('/')[-1]} · {run_id[:8]}",
        "model": meta.base_model,
        "display_name": row.get("name") or row.get("run_name"),
        "recipe_name": row.get("recipe_name"),
        "git_rev": row.get("git_rev"),
        "fine_tuning_type": meta.fine_tuning_type,
        "runtime_id": runtime,
        "shared_runtime": shared,
        "status": meta.status,
        "created_at": meta.created_at,
        "completed_at": meta.completed_at,
        "steps": row["steps"],
        "pods": pods,
        "workloads": matched,
        "nodes": sorted({p["node"] for p in pods if p.get("node")}),
        # Until add_status: what the record alone says.
        "session_live": None,
        "settings": openrl_status.settings_of(meta),
        "workers": [],
        "workers_error": None,
        "summary": f"model {meta.status}",
      }
    )
  return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def progress(run: dict, processes: list[dict], recent: list, queued: dict, now: float) -> dict:
  """Whether an active run is moving, judged by what completes rather than by which processes are up.

  DEAD: a process that popped the run's requests stopped reporting, so nothing will answer them.
  STALLED: a request has been on a worker, or requests have sat queued, for longer than
  three median steps (at least STALL_SECONDS). IDLE: nothing is queued or running and nothing
  completed for that long, so the client is not sending work. OK otherwise."""
  outstanding = [
    {
      **group,
      "pod": process.get("pod") or process["id"],
      "phase": process.get("phase"),
      "process_alive": ops.alive(process, now),
      "age_seconds": now - group["since"],
    }
    for process in processes
    if process.get("runtime_id") == run["runtime_id"]
    for group in process.get("outstanding", [])
    if group.get("run_id") == run["run_id"]
  ]
  done = [sample for sample in recent if isinstance(sample, dict) and isinstance(sample.get("at"), int | float)]
  last = max((sample["at"] for sample in done), default=None)
  steps = sorted(sample["at"] for sample in done if sample.get("operation") == "optim_step" and sample.get("status") == "succeeded")
  step_seconds = statistics.median(b - a for a, b in zip(steps, steps[1:])) if len(steps) > 2 else None
  patience = max(STALL_SECONDS, 3 * (step_seconds or 0))
  quiet = now - (last if last is not None else run["created_at"] or now)
  waiting = sum(count for count in queued.values() if count)
  lost = [o for o in outstanding if not o["process_alive"]]
  oldest = max(outstanding, key=lambda o: o["age_seconds"], default=None)
  held = oldest and f"{oldest['op']} has been on {oldest['pod']} ({oldest['phase']}) for {oldest['age_seconds'] / 60:.0f} min"
  if lost:
    state, reason = "DEAD", f"{sum(o['count'] for o in lost)} {lost[0]['op']} request(s) were taken by {lost[0]['pod']}, which stopped reporting"
  elif oldest and oldest["age_seconds"] > patience:
    state, reason = "STALLED", held
  elif waiting and quiet > patience:
    state, reason = "STALLED", f"{waiting} request(s) queued and nothing completed for {quiet / 60:.0f} min"
  elif not outstanding and not waiting and quiet > patience:
    state, reason = "IDLE", f"nothing queued or running, and nothing completed for {quiet / 60:.0f} min: the client is not sending work"
  else:
    state, reason = "OK", held or (f"last op completed {quiet / 60:.0f} min ago" if last is not None else "no ops yet")
  return {"state": state, "reason": reason, "last_completed_at": last, "step_seconds": step_seconds, "outstanding": outstanding, "queued": queued}


async def add_progress(store: StateStore, run: dict, processes: list[dict], now: float) -> None:
  reads = (
    store.read_samples(ops.key(run["run_id"]), last=50),
    store.list_length(request_queue(run["run_id"])),
    store.list_length(sampling_queue(run["runtime_id"])),
  )
  try:
    recent, trainer, sampler = await asyncio.wait_for(asyncio.gather(*reads), timeout=5)
  except Exception as exc:
    run["progress"] = {"state": "UNKNOWN", "reason": f"progress unavailable: {type(exc).__name__}"}
    return
  run["progress"] = progress(run, processes, recent, {"trainer": trainer, "sampler": sampler}, now)
  if run["progress"]["state"] in ("DEAD", "STALLED") and run["display_status"] in ("Ready", "Active"):
    run["display_status"] = "Stalled"


async def processes_of(store: StateStore, since: float) -> list[dict]:
  records = [json.loads(raw) for raw in (await store.scan_values(ops.worker_key(""))).values()]
  return [record for record in records if record.get("heartbeat_at", 0) >= since]


def placements_of(state: dict, runs: list[dict]) -> list[dict]:
  """Every placed workload as one allocation: who, where, on which devices."""
  claims = state["devices"]["claims"]
  pods = {pod["name"]: pod for pod in state["pods"]}
  placements = []
  for workload in state["scheduler"]["workloads"]:
    if not workload.get("node_name"):
      continue
    associated = [r for r in runs if workload in r["workloads"]]
    placements.append(
      {
        "id": workload.get("uid") or workload["name"],
        "name": workload["name"],
        "label": associated[0]["name"] if len(associated) == 1 else workload.get("model_id") or workload["name"],
        "run_ids": [r["run_id"] for r in associated],
        "runtime_id": workload.get("model_id"),
        "shared_runtime": associated[0]["shared_runtime"] if associated else None,
        "node": workload["node_name"],
        "pod": workload.get("pod_name"),
        "pod_uid": pods.get(workload.get("pod_name"), {}).get("uid"),
        "created_at": pods.get(workload.get("pod_name"), {}).get("created_at") or workload.get("created_at"),
        "role": workload.get("role"),
        "owner_id": workload.get("owner_id"),
        "phase": workload.get("phase"),
        "device_count": workload.get("device_count", 0),
        "devices": claims.get(workload.get("claim_name"), []),
      }
    )
  return placements


class Snapshot:
  """Reads the cluster at most once per CACHE_SECONDS however many pages poll."""

  def __init__(self) -> None:
    self.lock = asyncio.Lock()
    self.latest: dict | None = None
    self.updated = 0.0
    self.metadata: list[dict] = []
    # The gateway mount swaps in its own worker manager's context.
    self.status_context: Callable[[StateStore, dict], openrl_status.StatusContext] = cluster_status_context

  async def current(self, store: StateStore) -> dict:
    async with self.lock:
      if self.latest is not None and time.monotonic() - self.updated < CACHE_SECONDS:
        return self.latest
      cluster = await asyncio.to_thread(kubernetes.read)
      store_error = None
      try:
        self.metadata = await asyncio.wait_for(metadata_of(store), timeout=5)
      except Exception:
        store_error = "Run store unavailable; showing last known run records" if self.metadata else "Run store unavailable"
      runs = join_runs(self.metadata, cluster)
      ctx = self.status_context(store, cluster)
      await asyncio.gather(*(add_status(ctx, run) for run in runs))
      now = time.time()
      processes_error = None
      try:
        processes = await asyncio.wait_for(processes_of(store, now - HISTORY_WINDOW_SECONDS), timeout=5)
      except Exception:
        processes = self.latest["processes"] if self.latest else []
        processes_error = "Worker process records unavailable; showing last known" if processes else "Worker process records unavailable"
      await asyncio.gather(*(add_progress(store, run, processes, now) for run in runs if run["status"] == "active"))
      self.latest = {
        "observed_at": datetime.fromtimestamp(now, UTC).isoformat(timespec="microseconds"),
        "cluster": cluster,
        "runs": runs,
        "store_error": store_error,
        "placements": placements_of(cluster, runs),
        "processes": processes,
        "processes_error": processes_error,
      }
      self.updated = time.monotonic()
      return self.latest

  def invalidate(self) -> None:
    """Read again on the next call, still falling back to this snapshot's records."""
    self.updated = float("-inf")


snapshot = Snapshot()
