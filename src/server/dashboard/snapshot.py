"""The one view of the cluster the pages and agents read: runs joined to their
workers, current placements on GPUs, nodes with their devices, and the
placement history that outlives them."""

import asyncio
import copy
import time

from server.dashboard import history
from server.telemetry import backends
from server.worker_manager import owner_id

CACHE_SECONDS = 5
HISTORY_WINDOW_SECONDS = 24 * 3600
TERMINAL_STATUSES = {"completed", "failed", "ended"}


def iso(ts: float) -> str:
  return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + f".{int((ts % 1) * 1e6):06d}+00:00"


def observed_status(metadata: dict, workloads: list[dict], pods: list[dict], available: bool) -> str:
  """Recorded lifecycle first; otherwise what Kubernetes shows right now."""
  recorded = str(metadata.get("status", "")).lower()
  if recorded in TERMINAL_STATUSES:
    return recorded.capitalize()
  if not available:
    return "Unknown"
  if any(pod.get("problem") for pod in pods):
    return "Needs attention"
  if any(pod.get("phase") == "Running" for pod in pods):
    return "Running"
  if pods or any(w.get("node_name") for w in workloads):
    return "Starting"
  if workloads:
    return "Queued"
  return "Unassigned"


async def mark_ended(store, runs: list[dict]) -> None:
  """A run with no workers whose owner has no live client session has ended,
  even when its API server never recorded it finished."""
  for run in runs:
    if run["display_status"] != "Unassigned":
      continue
    sessions = await store.set_members(f"open_rl:owner:{owner_id(run['runtime_id'])}")
    if not [s for s in sessions if await store.get_value(f"open_rl:session:{s}") is not None]:
      run["display_status"] = "Ended"


def delete_blocker(run: dict, available: bool) -> str | None:
  """Why the run's record must stay, or None. Finished runs can always go; an
  unfinished one only once the cluster shows it has no workers."""
  if str(run["status"]).lower() in TERMINAL_STATUSES:
    return None
  if not available:
    return "Cluster state is unavailable, so the run may still have workers"
  if run["pods"] or run["workloads"]:
    return "The run still has workers"
  return None


def shared_runtime(row: dict) -> str:
  """The runtime a shared LoRA run's workers serve, named as the API server names it."""
  base = row.get("base_model") or ""
  backend = row.get("trainer_backend") or "pytorch"
  return base if backend == "pytorch" else f"{backend}-{base}"


def join_runs(metadata: list[dict], state: dict) -> list[dict]:
  """LoRA workers serve a base-model runtime shared by every unfinished LoRA
  run on it; an FFT worker's modelID is the run itself."""
  workloads = state["scheduler"]["workloads"]
  claims = state["devices"]["claims"]
  runs = []
  for row in metadata:
    run_id = row["model_id"]
    lora = row.get("fine_tuning_type", "lora") == "lora"
    finished = str(row.get("status", "")).lower() in TERMINAL_STATUSES
    kind = "lora" if lora else "fft"
    # Workloads keyed by the run itself are its own: FFT, and LoRA runs that are
    # exclusive or train on several GPUs. Otherwise a LoRA run shares the
    # runtime the API server names after its base model and trainer.
    own = [w for w in workloads if w.get("model_id") == run_id and w.get("training_kind") == kind]
    shared = lora and not own
    runtime = shared_runtime(row) if shared else run_id
    matched = own or ([] if finished else [w for w in workloads if w.get("model_id") == runtime and w.get("training_kind") == kind])
    pods = []
    for pod in state["pods"]:
      owned = [w for w in matched if w["uid"] in pod["owner_uids"] or (w.get("pod_name") == pod["name"] and pod.get("worker") == w["name"])]
      if owned:
        pod = copy.deepcopy(pod)
        pod.update(role=owned[0].get("role"), shared_runtime=shared, runtime_id=runtime)
        pod["devices"] = sorted({device for w in owned for device in claims.get(w.get("claim_name"), [])})
        pods.append(pod)
    runs.append(
      {
        "run_id": run_id,
        "name": row.get("name") or row.get("run_name") or f"{(row.get('base_model') or 'Run').split('/')[-1]} · {run_id[:8]}",
        "model": row.get("base_model"),
        "display_name": row.get("name") or row.get("run_name"),
        "recipe_name": row.get("recipe_name"),
        "fine_tuning_type": row.get("fine_tuning_type"),
        "runtime_id": runtime,
        "shared_runtime": shared,
        "status": row.get("status", "unknown"),
        "display_status": observed_status(row, matched, pods, state["available"] and state["scheduler"].get("available", True)),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "completed_at": row.get("completed_at"),
        "steps": row.get("total_steps_completed"),
        "max_steps": row.get("max_steps"),
        "pods": pods,
        "workloads": matched,
        "nodes": sorted({p["node"] for p in pods if p.get("node")}),
      }
    )
  for run in runs:
    run["runtime_run_ids"] = [r["run_id"] for r in runs if r["runtime_id"] == run["runtime_id"]]
    run["delete_blocker"] = delete_blocker(run, state["available"] and state["scheduler"].get("available", True))
  return sorted(runs, key=lambda r: str(r.get("created_at") or ""), reverse=True)


def placements_of(state: dict, runs: list[dict]) -> list[dict]:
  """Every placed workload as one allocation: who, where, on which devices."""
  claims = state["devices"]["claims"]
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
        "node": workload["node_name"],
        "pod": workload.get("pod_name"),
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

  async def current(self, store) -> dict:
    async with self.lock:
      if self.latest is not None and time.monotonic() - self.updated < CACHE_SECONDS:
        return self.latest
      state = await asyncio.to_thread(lambda: backends.current().inventory())
      store_error = None
      try:
        self.metadata = await asyncio.wait_for(store.list_jobs_metadata(), timeout=5)
      except Exception:
        store_error = "Run store unavailable; showing last known run records" if self.metadata else "Run store unavailable"
      runs = join_runs(self.metadata, state)
      if not store_error:
        try:
          await asyncio.wait_for(mark_ended(store, runs), timeout=5)
        except Exception:
          pass  # the statuses from the cluster alone still stand
      if store_error:
        for run in runs:
          run["delete_blocker"] = "Run store unavailable"
      placements = placements_of(state, runs)
      now = time.time()
      history_error = None
      try:
        past = await history.read(store, now - HISTORY_WINDOW_SECONDS)
      except Exception:
        past = self.latest["history"] if self.latest else []
        history_error = "Placement history unavailable; showing last known placements" if past else "Placement history unavailable"
      self.latest = {
        "schema_version": 2,
        "observed_at": iso(now),
        "cluster": state,
        "runs": runs,
        "store_error": store_error,
        "placements": placements,
        "history": past,
        "history_error": history_error,
      }
      self.updated = time.monotonic()
      return self.latest

  async def placements(self, store) -> list[dict]:
    return (await self.current(store))["placements"]

  def invalidate(self) -> None:
    self.latest = None


snapshot = Snapshot()
