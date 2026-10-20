"""The one view of the cluster the pages and agents read: runs joined to their
workers and their progress, current placements on GPUs, nodes with their
devices, and the records worker processes keep about themselves."""

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime

from server import openrl_status, telemetry
from server.dashboard import sources
from server.model_metadata import decode_model_metadata
from server.sampling_store import SamplingStore
from server.session_registry import SessionRegistry
from server.store import StateStore, request_queue
from server.worker_manager import FAILED, READY, STARTING, STOPPED, WAITING_FOR_PLACEMENT

CACHE_SECONDS = 5
# A source that fails briefly, say over a dropped tunnel, is ridden out on its last good
# section; its error is reported only once it has failed for longer than this.
GRACE_SECONDS = 30
# The longest window the dashboard shows, matching the logs API's seven-day limit.
HISTORY_WINDOW_SECONDS = 7 * 86400
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
  # A pod that is only pending is starting, not sick; the worker states below say which.
  if any(pod["problem"] and pod["problem"] != "Pending" for pod in run["pods"]):
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
  meta = run.pop("_metadata")
  if run["status"] == "active":
    try:
      run.update(await asyncio.wait_for(openrl_status.model_status(ctx, run["run_id"], meta), timeout=5) or {})
    except Exception as exc:
      run.update(workers_error=f"status unavailable: {type(exc).__name__}", summary="status unavailable")
  run["summary"] = "; ".join([run["summary"], *(pod["problem"] for pod in run["pods"] if pod["problem"])])
  run["display_status"] = display_status(run, ctx.worker_launcher)


async def metadata_of(store: StateStore) -> list[dict]:
  steps = await store.scan_values(telemetry.steps_key(""))
  rows = []
  for key, value in (await store.scan_values(METADATA_PREFIX)).items():
    run_id = key.removeprefix(METADATA_PREFIX)
    rows.append({**json.loads(value), "model_id": run_id, "steps": int(steps.get(telemetry.steps_key(run_id), 0))})
  return rows


def join_runs(metadata: list[dict], state: dict) -> list[dict]:
  """LoRA workers serve a base-model runtime shared by every unfinished LoRA
  run on it; an FFT worker's modelID is the run itself."""
  workloads = state["scheduler"]["workloads"]
  claims = state["devices"]["claims"]
  runs = []
  for row in metadata:
    run_id = row["model_id"]
    meta = decode_model_metadata(row)
    kind = "lora" if meta.fine_tuning_type == "lora" else "fft"
    runtime = meta.runtime(run_id)
    shared, pool = runtime != run_id, meta.pool(run_id)
    matched = [w for w in workloads if w.get("model_id") == runtime and w.get("training_kind") == kind and not (meta.status != "active" and shared)]
    pods = []
    for pod in state["pods"]:
      owned = [w for w in matched if w["uid"] in pod["owner_uids"] or (w.get("pod_name") == pod["name"] and pod.get("worker") == w["name"])]
      # The pool's CPU pod (dispatcher, Envoy, EPP) is a Deployment, not a Workload; its label names the pool.
      dispatcher = pod.get("component") == "dispatcher" and pod.get("pool") == pool
      if owned or dispatcher:
        pod = pod.copy()
        pod.update(role=owned[0].get("role") if owned else "dispatcher", shared_runtime=shared, runtime_id=runtime)
        pod["devices"] = sorted({device for w in owned for device in claims.get(w.get("claim_name"), [])})
        pods.append(pod)
    runs.append(
      {
        "_metadata": meta,
        "run_id": run_id,
        "name": row.get("name") or row.get("run_name") or f"{(meta.base_model or 'Run').split('/')[-1]} · {run_id[:8]}",
        "model": meta.base_model,
        "display_name": row.get("name") or row.get("run_name"),
        "recipe_name": row.get("recipe_name"),
        "git_rev": row.get("git_rev"),
        "fine_tuning_type": meta.fine_tuning_type,
        "runtime_id": runtime,
        "pool": pool,
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


def progress(run: dict, processes: list[dict], recent: list, queued: dict, now: float, backlog: tuple = (0, None)) -> dict:
  """Whether an active run is moving, judged by what completes rather than by which processes are up.

  DEAD: requests are held by a process that stopped reporting: a trainer that popped them, or the
  dispatcher of the pool whose stream they wait in. STALLED: a request has been on a trainer, waited
  in the sampling pool, or sat in the trainer queue for longer than three median steps (at least
  STALL_SECONDS). IDLE: nothing is queued or running and nothing completed for that long, so the
  client is not sending work. OK otherwise."""
  outstanding = [
    {
      **group,
      "pod": process.get("pod") or process["id"],
      "phase": process.get("phase"),
      "process_alive": telemetry.alive(process, now),
      "age_seconds": now - group["since"],
      "where": f"on {process.get('pod') or process['id']} ({process.get('phase')})",
    }
    for process in processes
    if process.get("runtime_id") == run["runtime_id"] and process.get("role") != "dispatcher"
    for group in process.get("outstanding", [])
    if group.get("run_id") == run["run_id"]
  ]
  # The pool's stream holds every sample not yet answered; its dispatcher is what answers them.
  dispatchers = [p for p in processes if p.get("role") == "dispatcher" and p.get("pool") == run["pool"]]
  dispatcher = max(dispatchers, key=lambda p: p.get("heartbeat_at", 0), default=None)
  count, since = backlog
  if count:
    pod, phase = (dispatcher.get("pod") or dispatcher["id"], dispatcher.get("phase")) if dispatcher else ("no dispatcher", None)
    error = f"; last error: {dispatcher['last_error']}" if dispatcher and dispatcher.get("last_error") else ""
    outstanding.append(
      {
        "op": "sample",
        "pool": run["pool"],
        "count": count,
        "since": since,
        "pod": pod,
        "phase": phase,
        "process_alive": telemetry.alive(dispatcher, now) if dispatcher else None,
        "age_seconds": now - since,
        "where": f"waiting in pool {run['pool']} for dispatcher {pod} ({phase}{error})",
      }
    )
  done = [sample for sample in recent if isinstance(sample, dict) and isinstance(sample.get("at"), int | float)]
  last = max((sample["at"] for sample in done), default=None)
  steps = sorted(sample["at"] for sample in done if sample.get("operation") == "optim_step" and sample.get("status") == "succeeded")
  step_seconds = statistics.median(b - a for a, b in zip(steps, steps[1:])) if len(steps) > 2 else None
  patience = max(STALL_SECONDS, 3 * (step_seconds or 0))
  quiet = now - (last if last is not None else run["created_at"] or now)
  waiting = sum(count for count in queued.values() if count)
  lost = [o for o in outstanding if o["process_alive"] is False]
  oldest = max(outstanding, key=lambda o: o["age_seconds"], default=None)
  held = oldest and f"{oldest['op']} has been {oldest['where']} for {oldest['age_seconds'] / 60:.0f} min"
  if lost:
    holder = f"waiting in pool {lost[0]['pool']} for dispatcher" if lost[0]["op"] == "sample" else "held by"
    state, reason = "DEAD", f"{lost[0]['count']} {lost[0]['op']} request(s) {holder} {lost[0]['pod']}, which stopped reporting"
  elif oldest and oldest["age_seconds"] > patience:
    state, reason = "STALLED", held
  elif waiting and quiet > patience:
    state, reason = "STALLED", f"{waiting} request(s) queued and nothing completed for {quiet / 60:.0f} min"
  elif not outstanding and not waiting and quiet > patience:
    state, reason = "IDLE", f"nothing queued or running, and nothing completed for {quiet / 60:.0f} min: the client is not sending work"
  else:
    state, reason = "OK", held or (f"last op completed {quiet / 60:.0f} min ago" if last is not None else "no ops yet")
  result = {"state": state, "reason": reason, "last_completed_at": last, "step_seconds": step_seconds, "outstanding": outstanding, "queued": queued}
  if dispatcher:
    result["dispatcher"] = {key: value for key, value in dispatcher.items() if key not in ("turns", "outstanding")}
  return result


async def pool_backlog(store: StateStore, pool: str) -> tuple[int, float | None]:
  redis = getattr(store, "redis", None)  # pools need Redis; the in-memory store has none
  return await SamplingStore(redis).backlog(pool) if redis is not None else (0, None)


async def add_progress(store: StateStore, run: dict, processes: list[dict], now: float) -> None:
  run_id = run["run_id"]
  reads = (store.read_samples(telemetry.ops_key(run_id), last=50), store.list_length(request_queue(run_id)), pool_backlog(store, run["pool"]))
  try:
    recent, trainer, backlog = await asyncio.wait_for(asyncio.gather(*reads), timeout=5)
  except Exception as exc:
    run["progress"] = {"state": "UNKNOWN", "reason": f"progress unavailable: {type(exc).__name__}"}
    return
  run["progress"] = progress(run, processes, recent, {"trainer": trainer}, now, backlog)


async def processes_of(store: StateStore, since: float) -> list[dict]:
  records = [json.loads(raw) for raw in (await store.scan_values(telemetry.worker_key(""))).values()]
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
    self.refresh: asyncio.Task | None = None
    self.latest: dict | None = None
    self.updated = 0.0
    self.metadata: list[dict] | None = None
    self.processes_at = 0.0
    self.failing: dict[str, float] = {}
    # The gateway mount swaps in its own worker manager's context.
    self.status_context: Callable[[StateStore, dict], openrl_status.StatusContext] = cluster_status_context

  async def current(self, store: StateStore) -> dict:
    if self.latest is not None and time.monotonic() - self.updated < CACHE_SECONDS:
      return self.latest
    if self.refresh is None or self.refresh.done():
      self.refresh = asyncio.create_task(self.read(store))
      self.refresh.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
    # A client timeout cannot cancel the shared read and start overlapping Kubernetes threads.
    return await asyncio.shield(self.refresh)

  async def read(self, store: StateStore) -> dict:
    now = time.time()
    cluster, metadata, processes = await asyncio.gather(
      asyncio.to_thread(sources.read_cluster),
      asyncio.wait_for(metadata_of(store), timeout=5),
      asyncio.wait_for(processes_of(store, now - HISTORY_WINDOW_SECONDS), timeout=5),
      return_exceptions=True,
    )
    if isinstance(cluster, Exception):
      cluster = sources.cluster_unavailable(sources.namespace(), "Cluster inventory unavailable")
    # Each failed source falls back to its own last good section; the rest refreshes.
    cluster = self.merge(cluster, self.latest["cluster"] if self.latest else None)
    if isinstance(metadata, Exception):
      metadata, store_error = self.metadata or [], self.lasting("store", "Run store unavailable", self.metadata is None)
    else:
      self.metadata, store_error = metadata, self.lasting("store", None)
    if isinstance(processes, Exception):
      previous = self.latest["processes"] if self.latest else None
      # The last records, aged as of when they were read: a failed read must not make live work look dead.
      shift = now - self.processes_at
      processes = [{**record, "heartbeat_at": record["heartbeat_at"] + shift} if "heartbeat_at" in record else record for record in previous or []]
      processes_error = self.lasting("processes", "Worker process records unavailable", previous is None)
    else:
      self.processes_at, processes_error = now, self.lasting("processes", None)
    errors = [store_error, processes_error]
    # Kubernetes is optional locally. Section diagnostics remain visible, but
    # must not make the browser reject fresh run data when no cluster was read.
    if cluster["available"]:
      errors.extend([cluster.get("error"), cluster.get("nodes_error"), cluster["scheduler"].get("error"), cluster["devices"].get("error")])
    runs = join_runs(metadata, cluster)
    ctx = self.status_context(store, cluster)

    async def enrich(run):
      tasks = [add_status(ctx, run)]
      if run["status"] == "active":
        if processes_error:
          run["progress"] = {"state": "UNKNOWN", "reason": processes_error}
        else:
          tasks.append(add_progress(store, run, processes, now))
      await asyncio.gather(*tasks)
      if run.get("progress", {}).get("state") in ("DEAD", "STALLED") and run["display_status"] in ("Ready", "Active"):
        run["display_status"] = "Stalled"

    await asyncio.gather(*(enrich(run) for run in runs))
    self.latest = {
      "observed_at": datetime.fromtimestamp(now, UTC).isoformat(timespec="microseconds"),
      "cluster": cluster,
      "runs": runs,
      "store_error": store_error,
      "placements": placements_of(cluster, runs),
      "processes": processes,
      "processes_error": processes_error,
      "error": "; ".join(dict.fromkeys(error for error in errors if error)) or None,
    }
    self.updated = time.monotonic()
    return self.latest

  def lasting(self, source: str, error: str | None, at_once: bool = False) -> str | None:
    """The error to report for a source: none while it works or has failed only briefly."""
    if error is None:
      self.failing.pop(source, None)
      return None
    since = self.failing.setdefault(source, time.monotonic())
    return error if at_once or time.monotonic() - since > GRACE_SECONDS else None

  def merge(self, cluster: dict, previous: dict | None) -> dict:
    """The fresh cluster, with each section that failed replaced by its last good one."""
    if previous is None:
      return cluster
    if cluster.get("error"):
      return {**previous, "error": self.lasting("cluster", cluster["error"])}
    self.lasting("cluster", None)
    merged = dict(cluster)
    if cluster.get("nodes_error") and previous.get("nodes"):
      merged["nodes"] = previous["nodes"]
    merged["nodes_error"] = self.lasting("nodes", cluster.get("nodes_error"))
    if cluster["scheduler"].get("error") and previous["scheduler"].get("workloads"):
      merged["scheduler"] = previous["scheduler"]
    merged["scheduler"] = {**merged["scheduler"], "error": self.lasting("scheduler", cluster["scheduler"].get("error"))}
    if cluster["devices"].get("error") and any(node.get("devices") for node in previous.get("nodes", [])):
      known = {node["name"]: node.get("devices", []) for node in previous["nodes"]}
      merged["nodes"] = [{**node, "devices": node.get("devices") or known.get(node["name"], [])} for node in merged["nodes"]]
      merged["devices"] = {**previous["devices"], "error": cluster["devices"]["error"]}
    merged["devices"] = {**merged["devices"], "error": self.lasting("devices", cluster["devices"].get("error"))}
    return merged

  def invalidate(self) -> None:
    """Read again on the next call, still falling back to this snapshot's records."""
    self.updated = float("-inf")


snapshot = Snapshot()
