"""The one view of the cluster the pages and checks read: runs joined to their
workers, the requests each still waits on, where each run's workers were placed,
nodes with their GPUs, and the records worker processes keep about themselves."""

import asyncio
import json
import time
from contextlib import suppress
from datetime import UTC, datetime

from server import openrl_status, telemetry
from server.dashboard import inspection, metrics, sources
from server.model_metadata import decode_model_metadata
from server.session_registry import SessionRegistry
from server.store import StateStore, request_queue
from server.worker_manager import STOPPED, WAITING_FOR_PLACEMENT

CACHE_SECONDS = 5
# A source that fails briefly, say over a dropped tunnel, is ridden out on its last good
# section; its error is reported only once it has failed for longer than this.
GRACE_SECONDS = 30
# The longest window the dashboard shows, matching the logs API's seven-day limit.
HISTORY_WINDOW_SECONDS = 7 * 86400
METADATA_PREFIX = "open_rl:model_meta:"


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
        pods.append(pod)
    runs.append(
      {
        "run_id": run_id,
        "name": row.get("name") or row.get("run_name") or f"{(meta.base_model or 'Run').split('/')[-1]} · {run_id[:8]}",
        "model": meta.base_model,
        "display_name": row.get("name") or row.get("run_name"),
        "recipe_name": row.get("recipe_name"),
        "session_ids": meta.session_ids,
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
        "settings": openrl_status.settings_of(meta),
        "last_completed_at": None,
      }
    )
  return sorted(runs, key=lambda r: r["created_at"], reverse=True)


def lifecycle(run: dict, live: bool | None, scheduler: dict) -> str:
  """ended: marked finished, or nothing serves it and no session is live. waiting: a worker is not placed yet.
  placed: every worker is placed. active: the scheduler cannot be read, so placement is unknown."""
  states = {(w.get("worker") or {}).get("state") for w in run["workloads"]}
  if run["status"] != "active" or (states and states <= {STOPPED}):
    return "ended"
  if scheduler["available"]:
    if states:
      return "waiting" if WAITING_FOR_PLACEMENT in states else "placed"
    return "ended" if live is False else "waiting"
  return "ended" if live is False and scheduler["installed"] is False else "active"


def outstanding(run: dict, processes: list[dict], now: float) -> list[dict]:
  """The run's requests that trainers took and have not answered, with whether the holder still reports."""
  return [
    {**group, "pod": process.get("pod") or process["id"], "process_alive": telemetry.alive(process, now), "age_seconds": now - group["since"]}
    for process in processes
    if process.get("runtime_id") == run["runtime_id"] and process.get("role") != "dispatcher"
    for group in process.get("outstanding", [])
    if group.get("run_id") == run["run_id"]
  ]


async def enrich(store: StateStore, run: dict, processes: list[dict] | None, scheduler: dict, now: float) -> None:
  """The run's lifecycle; for an active run, also what it waits on and when work last completed. A failed read leaves them unknown."""
  registry, run_id, live = SessionRegistry(store), run["run_id"], None
  with suppress(Exception):
    live = any(await asyncio.wait_for(asyncio.gather(*(registry.live(session) for session in run["session_ids"])), timeout=5))
  run["lifecycle"] = lifecycle(run, live, scheduler)
  if run["status"] != "active":
    return
  with suppress(Exception):
    reads = (store.read_samples(telemetry.ops_key(run_id), last=50), store.list_length(request_queue(run_id)))
    recent, queued = await asyncio.wait_for(asyncio.gather(*reads), timeout=5)
    run["last_completed_at"] = max((s["at"] for s in recent if isinstance(s, dict) and isinstance(s.get("at"), int | float)), default=None)
    if processes is not None:
      run["progress"] = {"outstanding": outstanding(run, processes, now), "queued": {"trainer": queued}}


async def processes_of(store: StateStore, since: float) -> list[dict]:
  records = [json.loads(raw) for raw in (await store.scan_values(telemetry.worker_key(""))).values()]
  return [record for record in records if record.get("heartbeat_at", 0) >= since]


async def placement_records(store: StateStore, runtimes: set[str]) -> list[dict]:
  lists = await asyncio.gather(*(store.read_samples(telemetry.placements_key(runtime)) for runtime in runtimes))
  return [record for records in lists for record in records]


class Snapshot:
  """Reads the cluster at most once per CACHE_SECONDS however many pages poll."""

  def __init__(self) -> None:
    self.refresh: asyncio.Task | None = None
    self.latest: dict | None = None
    self.updated = 0.0
    self.metadata: list[dict] | None = None
    self.processes_at = 0.0
    self.failing: dict[str, float] = {}

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
    cluster, metadata, processes, gpus = await asyncio.gather(
      asyncio.to_thread(sources.read_cluster),
      asyncio.wait_for(metadata_of(store), timeout=5),
      asyncio.wait_for(processes_of(store, now - HISTORY_WINDOW_SECONDS), timeout=5),
      asyncio.wait_for(metrics.node_gpus(), timeout=10),
      return_exceptions=True,
    )
    if isinstance(cluster, Exception):
      cluster = sources.cluster_unavailable(sources.namespace(), "Cluster inventory unavailable")
    # Each failed source falls back to its own last good section; the rest refreshes.
    cluster = self.merge(cluster, self.latest["cluster"] if self.latest else None)
    known = {node["name"]: node["devices"] for node in self.latest["cluster"]["nodes"]} if self.latest else {}
    if isinstance(gpus, Exception):
      gpus, gpus_error = known, self.lasting("gpus", "GPU inventory unavailable", not known)
    else:
      gpus, gpus_error = gpus or {}, self.lasting("gpus", None)
    nodes = [{**node, "devices": gpus.get(node["name"], [])} for node in cluster["nodes"]]
    cluster["nodes"] = [{**node, "gpu_capacity": max(node["gpu_capacity"], len(node["devices"]))} for node in nodes]
    cluster["devices"] = {"available": bool(gpus), "error": gpus_error}
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
    runs = join_runs(metadata, cluster)
    try:
      records, records_error = await asyncio.wait_for(placement_records(store, {r["runtime_id"] for r in runs}), timeout=5), None
    except Exception:
      records, records_error = self.latest["placement_records"] if self.latest else [], "Placement records unavailable"
    records_error = self.lasting("placements", records_error, not self.latest)
    errors = [store_error, processes_error, records_error]
    # Kubernetes is optional locally. Section diagnostics remain visible, but
    # must not make the browser reject fresh run data when no cluster was read.
    if cluster["available"]:
      errors.extend([cluster.get("error"), cluster.get("nodes_error"), cluster["scheduler"].get("error"), gpus_error])
    await asyncio.gather(*(enrich(store, run, None if processes_error else processes, cluster["scheduler"], now) for run in runs))
    state = {
      "observed_at": datetime.fromtimestamp(now, UTC).isoformat(timespec="microseconds"),
      "cluster": cluster,
      "runs": runs,
      "store_error": store_error,
      "placement_records": records,
      "processes": processes,
      "processes_error": processes_error,
      "error": "; ".join(dict.fromkeys(error for error in errors if error)) or None,
    }
    self.latest = {**state, "placements": [lane for lane in inspection.lanes(state) if not lane["ended"]]}
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
    return merged

  def invalidate(self) -> None:
    """Read again on the next call, still falling back to this snapshot's records."""
    self.updated = float("-inf")


snapshot = Snapshot()
