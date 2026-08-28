# Real data sources for the operational dashboard: gateway process state, Redis, the shared
# filesystem, and (when reachable) the Kubernetes API. Every accessor degrades to an explicit
# "unavailable" result instead of raising so the dashboard can always render something truthful.

import asyncio
import concurrent.futures
import functools
import json
import os
import shutil
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from server.store import InMemoryStore, RedisStore, RequestStore
from server.worker_launch_processor import FFTWorkerManager

START_TIME = time.time()
K8S_REQUEST_TIMEOUT = 4
NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
SCHEDULER_GROUP = "openrl.io"
SCHEDULER_VERSION = "v1alpha1"


def demo_mode_enabled() -> bool:
  return os.getenv("OPEN_RL_DASHBOARD_DEMO", "").lower() in {"1", "true", "yes"}


def tmp_dir() -> str:
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def iso_timestamp(ts: float | str | None) -> str | None:
  if ts is None:
    return None
  if isinstance(ts, str):
    return ts
  return datetime.fromtimestamp(ts, tz=UTC).isoformat()


# *** Kubernetes ***


def k8s_namespace() -> str:
  if ns := os.getenv("OPEN_RL_WORKER_NAMESPACE"):
    return ns
  try:
    with open(NAMESPACE_FILE) as f:
      return f.read().strip()
  except OSError:
    return "default"


@functools.cache
def k8s_core_v1() -> tuple[Any, str | None]:
  """Return (CoreV1Api, None) or (None, reason). The client library and cluster credentials
  are both optional; the first outcome is cached for the lifetime of the process."""
  try:
    from kubernetes import client, config
  except ImportError:
    return None, "kubernetes python client not installed"
  try:
    config.load_incluster_config()
  except Exception:
    try:
      config.load_kube_config()
    except Exception as exc:
      return None, f"no cluster credentials: {exc}"
  return client.CoreV1Api(), None


@functools.cache
def k8s_custom_objects() -> tuple[Any, str | None]:
  _, err = k8s_core_v1()
  if err:
    return None, err
  from kubernetes import client

  return client.CustomObjectsApi(), None


def object_age_seconds(timestamp: str | None) -> int | None:
  if not timestamp:
    return None
  try:
    created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return max(0, int((datetime.now(tz=UTC) - created).total_seconds()))
  except (TypeError, ValueError):
    return None


def empty_scheduler_snapshot(*, installed: bool | None, error: str | None = None) -> dict:
  return {
    "installed": installed,
    "available": False,
    "error": error,
    "workloads": [],
    "ledgers": [],
    "summary": {"workloads": 0, "phase_counts": {}, "ledgers": 0, "seats": 0, "shared_ledgers": 0},
  }


def scheduler_snapshot(namespace: str) -> dict:
  """Read the optional scheduler CRDs as unstructured objects. A missing CRD is a supported
  configuration, while RBAC or API failures remain visible diagnostic facts."""
  api, err = k8s_custom_objects()
  if api is None:
    return empty_scheduler_snapshot(installed=None, error=err)

  def list_objects(plural: str) -> list[dict]:
    return api.list_namespaced_custom_object(
      SCHEDULER_GROUP,
      SCHEDULER_VERSION,
      namespace,
      plural,
      _request_timeout=K8S_REQUEST_TIMEOUT,
    ).get("items", [])

  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    workloads_future = executor.submit(list_objects, "workloads")
    ledgers_future = executor.submit(list_objects, "claimledgers")
    try:
      workload_items = workloads_future.result()
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        return empty_scheduler_snapshot(installed=False)
      return empty_scheduler_snapshot(installed=True, error=f"workload list failed: {exc}")
    try:
      ledger_items = ledgers_future.result()
    except Exception as exc:
      return empty_scheduler_snapshot(installed=True, error=f"claim ledger list failed: {exc}")

  workloads = []
  for item in workload_items:
    metadata = item.get("metadata") or {}
    spec = item.get("spec") or {}
    accelerator = spec.get("accelerator") or {}
    status = item.get("status") or {}
    conditions = status.get("conditions") or []
    placed = next((condition for condition in conditions if condition.get("type") == "Placed"), None)
    workloads.append(
      {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "created_at": metadata.get("creationTimestamp"),
        "age_seconds": object_age_seconds(metadata.get("creationTimestamp")),
        "deleting": bool(metadata.get("deletionTimestamp")),
        "generation": metadata.get("generation"),
        "role": spec.get("role"),
        "model_id": spec.get("modelId"),
        "owner_id": spec.get("ownerId"),
        "training_kind": spec.get("trainingKind"),
        "requested_memory": accelerator.get("memory"),
        "max_devices": accelerator.get("maxDeviceCount", 1),
        "phase": status.get("phase") or "Pending",
        "reason": status.get("reason"),
        "claim_name": status.get("claimName"),
        "assignment_id": status.get("assignmentID"),
        "pod_name": status.get("podName"),
        "node_name": status.get("nodeName"),
        "device_count": status.get("deviceCount", 0),
        "memory_per_device": status.get("memoryPerDevice"),
        "observed_generation": status.get("observedGeneration"),
        "generation_current": status.get("observedGeneration") == metadata.get("generation"),
        "placed": None if placed is None else placed.get("status") == "True",
        "placed_reason": placed.get("reason") if placed else None,
        "placed_message": placed.get("message") if placed else None,
        "placed_transition_at": placed.get("lastTransitionTime") if placed else None,
      }
    )

  ledgers = []
  for item in ledger_items:
    metadata = item.get("metadata") or {}
    spec = item.get("spec") or {}
    seats = spec.get("seats") or []
    ledgers.append(
      {
        "name": metadata.get("name"),
        "created_at": metadata.get("creationTimestamp"),
        "age_seconds": object_age_seconds(metadata.get("creationTimestamp")),
        "claim_name": spec.get("claimName"),
        "seat_count": len(seats),
        "owners": sorted({owner for seat in seats if (owner := seat.get("owner") or seat.get("workload"))}),
        "seats": [
          {
            "workload": seat.get("workload"),
            "workload_uid": seat.get("workloadUID"),
            "assignment_id": seat.get("assignmentID"),
            "owner": seat.get("owner"),
            "host_request": seat.get("hostRequest"),
          }
          for seat in seats
        ],
      }
    )

  phase_counts: dict[str, int] = {}
  for workload in workloads:
    phase_counts[workload["phase"]] = phase_counts.get(workload["phase"], 0) + 1
  return {
    "installed": True,
    "available": True,
    "error": None,
    "workloads": workloads,
    "ledgers": ledgers,
    "summary": {
      "workloads": len(workloads),
      "phase_counts": phase_counts,
      "ledgers": len(ledgers),
      "seats": sum(ledger["seat_count"] for ledger in ledgers),
      "shared_ledgers": sum(ledger["seat_count"] > 1 for ledger in ledgers),
    },
  }


def pod_problem(pod: Any) -> str | None:
  phase = pod.status.phase or "Unknown"
  if phase == "Failed":
    return f"Failed: {pod.status.reason or 'see logs'}"
  for cs in pod.status.container_statuses or []:
    waiting = cs.state.waiting if cs.state else None
    if waiting and waiting.reason not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting.reason}: {waiting.message or ''}".strip(": ")
  if phase == "Pending":
    for cond in pod.status.conditions or []:
      if cond.type == "PodScheduled" and cond.status != "True":
        return f"Unschedulable: {cond.message or cond.reason or 'no node available'}"
    return "Pending"
  return None


def pod_gpu_count(pod: Any) -> int:
  """GPUs a pod claims: nvidia.com/gpu requests/limits, or DRA resource claims (1 device each)."""
  gpus = 0
  for container in pod.spec.containers or []:
    resources = container.resources
    for source in (resources.requests if resources else None, resources.limits if resources else None):
      if source and "nvidia.com/gpu" in source:
        gpus += int(float(source["nvidia.com/gpu"]))
        break
  if gpus == 0:
    gpus = len(pod.spec.resource_claims or [])
  return gpus


def pod_to_dict(pod: Any) -> dict:
  statuses = pod.status.container_statuses or []
  containers = []
  for cs in statuses:
    state = "unknown"
    if cs.state:
      state = "running" if cs.state.running else "waiting" if cs.state.waiting else "terminated" if cs.state.terminated else "unknown"
    containers.append({"name": cs.name, "image": cs.image, "ready": bool(cs.ready), "state": state})
  if not containers:
    containers = [{"name": c.name, "image": c.image, "ready": False, "state": "unknown"} for c in pod.spec.containers or []]
  ready_count = sum(1 for c in statuses if c.ready)
  return {
    "name": pod.metadata.name,
    "phase": pod.status.phase or "Unknown",
    "node": pod.spec.node_name,
    "app": (pod.metadata.labels or {}).get("app"),
    "labels": pod.metadata.labels or {},
    "ready": f"{ready_count}/{len(pod.spec.containers or [])}",
    "restarts": sum(cs.restart_count or 0 for cs in statuses),
    "created_at": pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else None,
    "problem": pod_problem(pod),
    "containers": containers,
    "gpus": pod_gpu_count(pod),
  }


def node_to_dict(node: Any) -> dict:
  labels = node.metadata.labels or {}
  capacity = node.status.capacity or {}
  allocatable = node.status.allocatable or {}
  conditions = node.status.conditions or []
  return {
    "name": node.metadata.name,
    "ready": any(c.type == "Ready" and c.status == "True" for c in conditions),
    "memory_pressure": any(c.type == "MemoryPressure" and c.status == "True" for c in conditions),
    "disk_pressure": any(c.type == "DiskPressure" and c.status == "True" for c in conditions),
    "unschedulable": bool(node.spec.unschedulable),
    "instance_type": labels.get("node.kubernetes.io/instance-type") or labels.get("beta.kubernetes.io/instance-type"),
    "accelerator": labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"),
    "gpu_capacity": int(capacity.get("nvidia.com/gpu", 0)),
    "gpu_allocatable": int(allocatable.get("nvidia.com/gpu", 0)),
  }


def k8s_snapshot() -> dict:
  """List pods in our namespace and (when RBAC allows) cluster nodes. Blocking; call in a thread."""
  api, err = k8s_core_v1()
  namespace = k8s_namespace()
  if api is None:
    return {
      "available": False,
      "namespace": namespace,
      "error": err,
      "pods": [],
      "nodes": [],
      "scheduler": empty_scheduler_snapshot(installed=None, error=err),
    }
  try:
    pods = [pod_to_dict(p) for p in api.list_namespaced_pod(namespace, _request_timeout=K8S_REQUEST_TIMEOUT).items]
  except Exception as exc:
    error = f"pod list failed: {exc}"
    return {
      "available": False,
      "namespace": namespace,
      "error": error,
      "pods": [],
      "nodes": [],
      "scheduler": empty_scheduler_snapshot(installed=None, error=error),
    }
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    nodes_future = executor.submit(api.list_node, _request_timeout=K8S_REQUEST_TIMEOUT)
    scheduler_future = executor.submit(scheduler_snapshot, namespace)
    try:
      nodes = [node_to_dict(n) for n in nodes_future.result().items]
    except Exception:
      # Namespaced service accounts often cannot list nodes; pods alone are still useful.
      nodes = []
    try:
      scheduler = scheduler_future.result()
    except Exception as exc:
      scheduler = empty_scheduler_snapshot(installed=True, error=f"scheduler snapshot failed: {exc}")
  return {
    "available": True,
    "namespace": namespace,
    "error": None,
    "pods": pods,
    "nodes": nodes,
    "scheduler": scheduler,
  }


def k8s_pod_logs(pod: str, container: str | None, tail: int) -> dict:
  api, err = k8s_core_v1()
  if api is None:
    raise RuntimeError(err or "kubernetes unavailable")
  text = api.read_namespaced_pod_log(
    pod,
    k8s_namespace(),
    container=container,
    tail_lines=tail,
    _request_timeout=K8S_REQUEST_TIMEOUT + 4,
  )
  return {"demo": False, "pod": pod, "container": container, "text": text}


# *** GPU duty cycle ***
#
# Allocation duty per pool, broken down by job: GPUs claimed by non-terminal pods / pool GPU
# capacity, attributed to runs via the timeslice job-id labels the k8s worker manager stamps,
# sampled into an in-memory ring buffer whenever the cluster is polled. This is truthful
# scheduler state, not device utilization — DCGM owns that. History lives for the gateway's
# lifetime. A series sample is [unix_ts, {job: claimed_gpus}].

TERMINAL_POD_PHASES = {"Succeeded", "Failed"}


def pool_gpu_capacity(pool: dict) -> int:
  return sum(node["gpu_capacity"] for node in pool["nodes"])


def pod_job(pod: dict) -> str:
  """The run a pod's GPUs belong to: the model id from its timeslice job-id label, else its
  app label, else 'other'. Trainer and sampler pods of one run share one job."""
  job_id = (pod.get("labels") or {}).get("timeslice.io/job-id", "")
  for role in ("trainer-", "sampler-"):
    if job_id.startswith(role):
      return job_id.removeprefix(role)
  return pod.get("app") or "other"


class DutyTracker:
  """Ring buffer of per-job allocation-duty samples per GPU pool."""

  def __init__(self, max_samples: int = 120, min_interval_s: float = 5.0):
    self.max_samples = max_samples
    self.min_interval_s = min_interval_s
    self.series: dict[str, list[list]] = {}
    self.last_sample_at = 0.0

  def record(self, pools: list[dict], pods: list[dict], now: float | None = None) -> None:
    """Append one duty sample per GPU pool, throttled so overlapping polls don't stack points."""
    now = now if now is not None else time.time()
    if now - self.last_sample_at < self.min_interval_s:
      return
    self.last_sample_at = now
    claims_by_node: dict[str, dict[str, int]] = {}
    for pod in pods:
      gpus = pod.get("gpus", 0)
      if not gpus or not pod["node"] or pod["phase"] in TERMINAL_POD_PHASES:
        continue
      node_claims = claims_by_node.setdefault(pod["node"], {})
      job = pod_job(pod)
      node_claims[job] = node_claims.get(job, 0) + gpus
    for pool in pools:
      if not pool_gpu_capacity(pool):
        continue
      claims: dict[str, int] = {}
      for node in pool["nodes"]:
        for job, gpus in claims_by_node.get(node["name"], {}).items():
          claims[job] = claims.get(job, 0) + gpus
      series = self.series.setdefault(pool["id"], [])
      series.append([int(now), claims])
      del series[: -self.max_samples]

  def duty(self, pool: dict) -> dict | None:
    capacity = pool_gpu_capacity(pool)
    if not capacity:
      return None
    series = self.series.get(pool["id"], [])
    jobs: list[str] = []
    for _, claims in series:
      for job in claims:
        if job not in jobs:
          jobs.append(job)
    # Deliberately unclamped: time-sliced pools can be overcommitted, and current > 1 is the
    # honest way to report that.
    current = sum(series[-1][1].values()) / capacity if series else 0.0
    return {"capacity": capacity, "current": round(current, 4), "jobs": jobs, "series": series}


duty_tracker = DutyTracker()


# *** Cluster assembly ***


def gateway_summary() -> dict:
  from server import gateway

  return {
    "title": "open-rl gateway",
    "mode": "single-process" if gateway.is_single_process_mode() else "distributed",
    "fft_enabled": gateway.is_fft_enabled(),
    "redis_configured": bool(os.getenv("REDIS_URL")),
    "vllm_url": gateway.VLLM_URL if gateway.get_sampler_backend() == "vllm" else None,
    "sampler_backend": gateway.get_sampler_backend(),
  }


async def ping_redis(store: RequestStore) -> bool | None:
  """True/False for a Redis-backed store, None when the store is in-memory."""
  if not isinstance(store, RedisStore):
    return None
  try:
    return bool(await store.redis.ping())
  except Exception:
    return False


async def cluster_snapshot(store: RequestStore, k8s: dict) -> dict:
  gateway_card = gateway_summary()
  redis_ok = await ping_redis(store)

  shared = tmp_dir()
  services = [
    {
      "id": "redis",
      "label": "Redis",
      "configured": gateway_card["redis_configured"],
      "ok": redis_ok,
      "detail": os.getenv("REDIS_URL") or "not set — in-memory store",
    },
    {
      "id": "storage",
      "label": "Shared storage",
      "configured": True,
      "ok": os.path.isdir(shared) and os.access(shared, os.W_OK),
      "detail": shared,
    },
  ]
  if gateway_card["vllm_url"]:
    services.append({"id": "vllm", "label": "vLLM worker", "configured": True, "ok": None, "detail": gateway_card["vllm_url"]})

  # Edges only where we know the gateway actually connects: its configured Redis and vLLM URLs.
  edges = []
  if gateway_card["redis_configured"]:
    edges.append({"from": "gateway", "to": "redis", "reason": "REDIS_URL configured"})
  if gateway_card["vllm_url"]:
    edges.append({"from": "gateway", "to": "vllm", "reason": "VLLM_URL configured"})

  pools: dict[str, dict] = {}
  pods_by_node: dict[str, list[str]] = {}
  for pod in k8s["pods"]:
    if pod["node"]:
      pods_by_node.setdefault(pod["node"], []).append(pod["name"])
  for node in k8s["nodes"]:
    pool_id = node["accelerator"] or ("gpu" if node["gpu_capacity"] else "cpu")
    pool = pools.setdefault(pool_id, {"id": pool_id, "label": pool_id, "nodes": []})
    pool["nodes"].append(
      {**{k: node[k] for k in ("name", "ready", "instance_type", "gpu_capacity", "gpu_allocatable")}, "pods": pods_by_node.get(node["name"], [])}
    )
  duty_tracker.record(list(pools.values()), k8s["pods"])
  for pool in pools.values():
    pool["duty"] = duty_tracker.duty(pool)

  known_nodes = {n["name"] for n in k8s["nodes"]}
  unplaced = [p["name"] for p in k8s["pods"] if not p["node"] or p["node"] not in known_nodes]
  if unplaced:
    pools["unscheduled"] = {
      "id": "unscheduled",
      "label": "not scheduled",
      "duty": None,
      "nodes": [{"name": "—", "ready": None, "instance_type": None, "gpu_capacity": 0, "gpu_allocatable": 0, "pods": unplaced}],
    }

  return {
    "demo": False,
    "kubernetes": {"available": k8s["available"], "namespace": k8s["namespace"], "error": k8s["error"]},
    "gateway": gateway_card,
    "scheduler": k8s.get("scheduler") or empty_scheduler_snapshot(installed=None),
    "services": services,
    "edges": edges,
    "pools": sorted(pools.values(), key=lambda p: (p["id"] == "cpu", p["id"] == "unscheduled", p["id"])),
    "pods": k8s["pods"],
  }


async def diagnostic_snapshot(store: RequestStore, worker_manager: FFTWorkerManager | None, k8s: dict) -> dict:
  """Build one coherent dashboard payload from one Kubernetes observation."""
  try:
    queues, launch = await asyncio.gather(store.queue_stats(), store.worker_launch_stats())
  except Exception:
    queues, launch = [], {"depth": 0, "oldest_enqueued_at": None, "oldest_age_seconds": None}
  cluster, runs, checks, operational = await asyncio.gather(
    cluster_snapshot(store, k8s),
    runs_snapshot(store, worker_manager, k8s["pods"], k8s.get("scheduler"), queues),
    health_checks(store, k8s),
    operational_stats(store, k8s, worker_manager, queues, launch),
  )
  stats, queues = operational
  return {
    "demo": False,
    "generated_at": iso_timestamp(time.time()),
    "cluster": cluster,
    "runs": runs,
    "health": {"demo": False, "checks": checks, "stats": stats, "queues": queues},
    "problems": {"demo": False, "problems": derive_problems(checks, k8s, stats, runs["runs"])},
  }


# *** Runs ***


def filesystem_runs() -> dict[str, dict]:
  found: dict[str, dict] = {}
  peft_dir = os.path.join(tmp_dir(), "peft")
  if os.path.isdir(peft_dir):
    for entry in os.scandir(peft_dir):
      if not entry.is_dir():
        continue
      info = found.setdefault(entry.name, {"sources": set()})
      info["sources"].add("adapter")
      info.setdefault("created_at", iso_timestamp(entry.stat().st_ctime))
      metadata_path = os.path.join(entry.path, "metadata.json")
      if os.path.exists(metadata_path):
        try:
          with open(metadata_path) as f:
            meta = json.load(f)
          info.setdefault("name", meta.get("alias"))
          info.setdefault("base_model", meta.get("base_model"))
          info.setdefault("wandb_url", meta.get("wandb_url"))
        except Exception:
          pass
  ckpt_dir = os.path.join(tmp_dir(), "checkpoints")
  if os.path.isdir(ckpt_dir):
    for entry in os.scandir(ckpt_dir):
      if entry.is_dir():
        info = found.setdefault(entry.name, {"sources": set()})
        info["sources"].add("checkpoint")
        info.setdefault("created_at", iso_timestamp(entry.stat().st_ctime))
  return found


async def redis_runs(store: RequestStore, queues: list[dict] | None = None) -> dict[str, dict]:
  found: dict[str, dict] = {}
  try:
    for model_id, meta in (await store.list_model_metadata()).items():
      info = found.setdefault(model_id, {"sources": set()})
      info["sources"].add("registered")
      for field in (
        "base_model",
        "created_at",
        "wandb_url",
        "status",
        "error",
        "operation",
        "state_path",
        "ready_at",
        "failed_at",
        "stopped_at",
        "telemetry",
      ):
        if meta.get(field) is not None:
          info[field] = iso_timestamp(meta[field]) if field in {"created_at", "ready_at", "failed_at", "stopped_at"} else meta[field]
  except Exception:
    pass
  try:
    for queue in queues if queues is not None else await store.queue_stats():
      model_id = queue["model_id"]
      if model_id != "default":
        info = found.setdefault(model_id, {"sources": set()})
        info["sources"].add("queue")
        info["queue_depth"] = queue["depth"]
        info["queue_oldest_at"] = iso_timestamp(queue["oldest_enqueued_at"])
        info["queue_oldest_seconds"] = queue["oldest_age_seconds"]
  except Exception:
    pass
  return found


def worker_processes(worker_manager: FFTWorkerManager | None) -> dict[str, bool]:
  """model_id -> process alive, for gateway-launched local FFT workers."""
  if worker_manager is None:
    return {}
  return {model_id: proc.poll() is None for model_id, proc in worker_manager.processes.items()}


def sanitize_job_id(model_id: str) -> str:
  return "".join(c if c.isalnum() else "-" for c in model_id.lower()).strip("-")


def model_pods(model_id: str, pods: list[dict]) -> list[dict]:
  """Pods launched for this model, matched exactly on the timeslice job-id labels the k8s
  worker manager stamps. Exact match only: prefix matching would cross-match runs that share
  an id prefix, and stop_run deletes what this returns. (Branches whose sanitize_job_id
  hash-truncates long ids should swap that helper in here.)"""
  wanted = {f"{role}-{sanitize_job_id(model_id)}" for role in ("trainer", "sampler")}
  return [p for p in pods if (p.get("labels") or {}).get("timeslice.io/job-id") in wanted]


def model_workloads(model_id: str, scheduler: dict | None) -> list[dict]:
  return [workload for workload in (scheduler or {}).get("workloads", []) if workload.get("model_id") == model_id]


def summarize_run(
  pods: list[dict],
  queue_depth: int,
  worker_alive: bool | None,
  sources: set[str] | list[str],
  model_status: str | None = None,
  model_error: str | None = None,
  workloads: list[dict] | None = None,
) -> dict:
  """Derive a compact lifecycle verdict from observable state without pretending an artifact
  is a currently running process. The verdict is shared by the list and detail endpoints."""
  phases: dict[str, int] = {}
  for pod in pods:
    phase = pod.get("phase") or "Unknown"
    phases[phase] = phases.get(phase, 0) + 1

  failed = next(
    (
      pod
      for pod in pods
      if pod.get("phase") == "Failed" or any(marker in (pod.get("problem") or "") for marker in ("BackOff", "OOM", "Error", "Failed"))
    ),
    None,
  )
  waiting = next((pod for pod in pods if pod.get("problem") or pod.get("phase") == "Pending"), None)
  workloads = workloads or []
  workload_phases: dict[str, int] = {}
  for workload in workloads:
    workload_phases[workload["phase"]] = workload_phases.get(workload["phase"], 0) + 1
  failed_workload = next((workload for workload in workloads if workload["phase"] == "Failed"), None)
  waiting_workload = next((workload for workload in workloads if workload["phase"] in {"Pending", "Placing"}), None)
  running = phases.get("Running", 0)
  if failed or failed_workload or model_status == "failed":
    if failed:
      reason = failed.get("problem") or f"pod {failed['name']} failed"
    elif failed_workload:
      reason = failed_workload.get("reason") or f"workload {failed_workload['name']} failed"
    else:
      reason = model_error or "model creation failed"
    phase, status = "failed", "error"
  elif waiting and running:
    phase, status, reason = "degraded", "warn", waiting.get("problem") or f"pod {waiting['name']} is pending"
  elif waiting:
    phase, status, reason = "waiting", "warn", waiting.get("problem") or f"pod {waiting['name']} is pending"
  elif waiting_workload:
    phase, status, reason = (
      "waiting",
      "warn",
      waiting_workload.get("reason") or f"workload {waiting_workload['name']} is {waiting_workload['phase'].lower()}",
    )
  elif worker_alive or running or workload_phases.get("Running"):
    count = running or 1
    if running:
      reason = f"{count} running {'pod' if count == 1 else 'pods'}"
    elif workload_phases.get("Running"):
      count = workload_phases["Running"]
      reason = f"{count} scheduler {'workload is' if count == 1 else 'workloads are'} running"
    else:
      reason = "local worker process is alive"
    phase, status = "running", "ok"
  elif queue_depth:
    phase, status, reason = "queued", "ok", f"{queue_depth} queued {'request' if queue_depth == 1 else 'requests'}"
  elif model_status == "queued":
    phase, status, reason = "starting", "ok", "model creation is in progress"
  elif model_status == "ready":
    phase, status, reason = "ready", "ok", "model is ready in the gateway"
  elif model_status == "stopped":
    phase, status, reason = "stopped", "off", "run was stopped by an operator"
  elif phases.get("Succeeded"):
    phase, status, reason = "succeeded", "ok", f"{phases['Succeeded']} pod{'s' if phases['Succeeded'] != 1 else ''} succeeded"
  elif set(sources) & {"adapter", "checkpoint"}:
    phase, status, reason = "saved", "off", "saved artifacts are present; no active worker is visible"
  else:
    phase, status, reason = "inactive", "off", "no active worker, queued work, or pod is visible"
  return {
    "phase": phase,
    "status": status,
    "reason": reason,
    "pod_phase_counts": phases,
    "workload_phase_counts": workload_phases,
  }


def current_gpu_claims(pods: list[dict], nodes: list[dict]) -> dict[str, int]:
  """Current GPU claims for one run grouped by the same pool identifiers as Cluster."""
  pools_by_node = {node["name"]: node["accelerator"] or ("gpu" if node["gpu_capacity"] else "cpu") for node in nodes}
  claims: dict[str, int] = {}
  for pod in pods:
    if not pod.get("node") or pod.get("phase") in TERMINAL_POD_PHASES or not pod.get("gpus"):
      continue
    pool = pools_by_node.get(pod["node"], "unknown")
    claims[pool] = claims.get(pool, 0) + pod["gpus"]
  return claims


def diagnostic_entry(
  code: str,
  severity: str,
  source: str,
  message: str,
  *,
  resource: dict,
  evidence: dict | None = None,
  actions: list[dict] | None = None,
) -> dict:
  resource_id = f"{resource['kind'].lower()}/{resource['name']}"
  return {
    "id": f"{code}:{resource_id}",
    "code": code,
    "severity": severity,
    "source": source,
    "resource": resource,
    "message": message,
    "evidence": evidence or {},
    "actions": actions or [],
  }


def pod_diagnostic(pod: dict, namespace: str | None = None) -> dict | None:
  source = f"pod/{pod['name']}"
  resource = {"kind": "Pod", "name": pod["name"], "namespace": namespace or k8s_namespace()}
  action = {
    "label": "Read pod logs",
    "method": "GET",
    "path": f"/api/v1/dashboard/pods/{pod['name']}/logs?tail=500",
    "command": f"make ops logs {pod['name']}",
  }
  problem = pod.get("problem")
  if problem:
    severity = "error" if pod.get("phase") == "Failed" or any(marker in problem for marker in ("BackOff", "OOM", "Error")) else "warn"
    code = "pod.unschedulable" if "Unschedulable" in problem else "pod.failed" if severity == "error" else "pod.waiting"
    return diagnostic_entry(
      code,
      severity,
      source,
      problem,
      resource=resource,
      evidence={"phase": pod.get("phase"), "node": pod.get("node"), "restarts": pod.get("restarts", 0)},
      actions=[action],
    )
  if pod.get("restarts", 0) >= 3:
    return diagnostic_entry(
      "pod.restarts",
      "warn",
      source,
      f"{pod['restarts']} container restarts",
      resource=resource,
      evidence={"phase": pod.get("phase"), "node": pod.get("node"), "restarts": pod["restarts"]},
      actions=[action],
    )
  return None


def run_diagnostics(
  run_id: str,
  state: dict,
  pods: list[dict],
  queue_depth: int,
  k8s: dict,
  queue_age_seconds: float | None = None,
  telemetry: dict | None = None,
) -> list[dict]:
  diagnostics = []
  if not k8s["available"]:
    diagnostics.append(
      diagnostic_entry(
        "kubernetes.unavailable",
        "warn",
        "kubernetes",
        k8s.get("error") or "Kubernetes is unavailable; pod state cannot be inspected",
        resource={"kind": "Cluster", "name": k8s.get("namespace") or "unknown"},
        evidence={"available": False, "namespace": k8s.get("namespace")},
        actions=[{"label": "Check dashboard health", "method": "GET", "path": "/api/v1/dashboard/health", "command": "make ops health"}],
      )
    )
  for pod in pods:
    if diagnostic := pod_diagnostic(pod, k8s.get("namespace")):
      diagnostics.append(diagnostic)
  if state["phase"] == "queued" and queue_depth:
    warn_after = float(os.getenv("OPEN_RL_QUEUE_WARN_SECONDS", "300"))
    stalled = queue_age_seconds is not None and queue_age_seconds >= warn_after
    diagnostics.append(
      diagnostic_entry(
        "run.queue_stalled" if stalled else "run.queued",
        "warn" if stalled else "info",
        f"run/{run_id}",
        f"oldest request has waited {format_duration(queue_age_seconds)}" if stalled else state["reason"],
        resource={"kind": "Run", "name": run_id},
        evidence={"queue_depth": queue_depth, "oldest_age_seconds": queue_age_seconds, "warn_after_seconds": warn_after},
        actions=[
          {"label": "Refresh run inspection", "method": "GET", "path": f"/api/v1/dashboard/runs/{run_id}", "command": f"make ops run {run_id}"}
        ],
      )
    )
  diagnostics.extend(run_telemetry_diagnostics(run_id, telemetry or {}))
  return diagnostics


def run_telemetry_diagnostics(run_id: str, telemetry: dict) -> list[dict]:
  diagnostics = []
  completed = int(telemetry.get("requests_completed") or 0)
  failures = int(telemetry.get("requests_failed") or 0)
  last_failed = telemetry.get("last_outcome") == "error"
  elevated_failures = completed >= 5 and failures / completed >= 0.2
  action = {
    "label": "Inspect run and pod logs",
    "method": "GET",
    "path": f"/api/v1/dashboard/runs/{run_id}?logs=100",
    "command": f"make ops run {run_id} 100",
  }
  if last_failed or elevated_failures:
    diagnostics.append(
      diagnostic_entry(
        "run.request_failed" if last_failed else "run.request_errors",
        "error" if last_failed else "warn",
        f"run/{run_id}",
        telemetry.get("last_error") or f"{failures} of {completed} completed requests failed",
        resource={"kind": "Run", "name": run_id},
        evidence={
          "requests_completed": completed,
          "requests_failed": failures,
          "failure_rate": failures / completed if completed else 0,
          "last_operation": telemetry.get("last_operation"),
          "last_error_at": iso_timestamp(telemetry.get("last_error_at")),
        },
        actions=[action],
      )
    )
  active = telemetry.get("active_request") or {}
  active_age = active.get("age_seconds") or 0
  warn_after = float(os.getenv("OPEN_RL_OPERATION_WARN_SECONDS", "600"))
  if active and active_age >= warn_after:
    diagnostics.append(
      diagnostic_entry(
        "run.request_stalled",
        "warn",
        f"run/{run_id}",
        f"{active.get('operation') or 'request'} has been executing for {format_duration(active_age)}",
        resource={"kind": "Run", "name": run_id},
        evidence={**active, "warn_after_seconds": warn_after},
        actions=[action],
      )
    )
  return diagnostics


def telemetry_snapshot(raw: dict | None) -> dict:
  telemetry = {**(raw or {})}
  if active := telemetry.get("active_request"):
    active = {**active}
    try:
      active["age_seconds"] = max(0.0, time.time() - float(active["started_at"]))
    except (KeyError, TypeError, ValueError):
      active["age_seconds"] = None
    telemetry["active_request"] = active
  return telemetry


def scheduler_run_diagnostics(workloads: list[dict], ledgers: list[dict], k8s: dict) -> list[dict]:
  if not workloads:
    return []
  workload_names = {workload["name"] for workload in workloads}
  ledger_names = {ledger["name"] for ledger in ledgers}
  scheduler_only = {**k8s, "pods": [], "nodes": []}
  return [
    problem
    for problem in derive_problems([], scheduler_only)
    if (problem["resource"]["kind"] == "Workload" and problem["resource"]["name"] in workload_names)
    or (problem["resource"]["kind"] == "ClaimLedger" and problem["resource"]["name"] in ledger_names)
  ]


async def runs_snapshot(
  store: RequestStore,
  worker_manager: FFTWorkerManager | None,
  pods: list[dict],
  scheduler: dict | None = None,
  queues: list[dict] | None = None,
) -> dict:
  found = await redis_runs(store, queues)
  for workload in (scheduler or {}).get("workloads", []):
    model_id = workload.get("model_id")
    if not model_id:
      continue
    info = found.setdefault(model_id, {"sources": set()})
    info["sources"].add("scheduler")
    if info.get("created_at") is None:
      info["created_at"] = workload.get("created_at")
  for model_id, info in filesystem_runs().items():
    merged = found.setdefault(model_id, {"sources": set()})
    merged["sources"] |= info.pop("sources")
    for key, value in info.items():
      if merged.get(key) is None:
        merged[key] = value
  workers = worker_processes(worker_manager)
  for model_id, alive in workers.items():
    info = found.setdefault(model_id, {"sources": set()})
    info["sources"].add("worker")
    info["worker_alive"] = alive

  runs = []
  for model_id, info in found.items():
    run_pods = model_pods(model_id, pods)
    run_workloads = model_workloads(model_id, scheduler)
    queue_depth = int(info.get("queue_depth") or 0)
    worker_alive = info.get("worker_alive")
    stoppable = bool(info.get("worker_alive") or "queue" in info["sources"] or run_pods)
    runs.append(
      {
        "run_id": model_id,
        "name": info.get("name") or f"run-{model_id[:8]}",
        "base_model": info.get("base_model"),
        "created_at": info.get("created_at"),
        "wandb_url": info.get("wandb_url"),
        "stoppable": stoppable,
        "sources": sorted(info["sources"]),
        "pods": [p["name"] for p in run_pods],
        "workloads": [workload["name"] for workload in run_workloads],
        "placement": {
          "workloads": len(run_workloads),
          "device_count": sum(workload.get("device_count") or 0 for workload in run_workloads),
          "phase_counts": {
            phase: sum(workload["phase"] == phase for workload in run_workloads)
            for phase in sorted({workload["phase"] for workload in run_workloads})
          },
        },
        "queue_depth": queue_depth,
        "queue_oldest_at": info.get("queue_oldest_at"),
        "queue_oldest_seconds": info.get("queue_oldest_seconds"),
        "telemetry": telemetry_snapshot(info.get("telemetry")),
        "worker_alive": worker_alive,
        "model_status": info.get("status"),
        "lifecycle": {
          "operation": info.get("operation"),
          "status": info.get("status"),
          "error": info.get("error"),
          "ready_at": info.get("ready_at"),
          "failed_at": info.get("failed_at"),
          "stopped_at": info.get("stopped_at"),
        },
        "state": summarize_run(run_pods, queue_depth, worker_alive, info["sources"], info.get("status"), info.get("error"), run_workloads),
      }
    )
  runs.sort(key=lambda r: (r["created_at"] is not None, r["created_at"] or "", r["run_id"]), reverse=True)
  return {"demo": False, "runs": runs}


async def run_detail(
  store: RequestStore,
  worker_manager: FFTWorkerManager | None,
  run_id: str,
  k8s: dict,
  log_tail: int = 0,
) -> dict | None:
  """Everything about one run in a single payload: its record, full pod state, queue depth,
  current GPU claims per pool, and (when log_tail > 0) a log tail per pod."""
  scheduler = k8s.get("scheduler")
  snapshot = await runs_snapshot(store, worker_manager, k8s["pods"], scheduler)
  run = next((r for r in snapshot["runs"] if r["run_id"] == run_id), None)
  if run is None:
    return None

  pods = model_pods(run_id, k8s["pods"])
  workloads = model_workloads(run_id, scheduler)
  claim_names = {workload["claim_name"] for workload in workloads if workload.get("claim_name")}
  ledgers = [ledger for ledger in (scheduler or {}).get("ledgers", []) if ledger.get("claim_name") in claim_names]
  queue_depth = run["queue_depth"]
  diagnostics = run_diagnostics(run_id, run["state"], pods, queue_depth, k8s, run.get("queue_oldest_seconds"), run.get("telemetry"))
  diagnostics.extend(scheduler_run_diagnostics(workloads, ledgers, k8s))
  gpu_claims = current_gpu_claims(pods, k8s["nodes"])
  detail = {
    **run,
    "demo": False,
    "pods": pods,
    "workloads": workloads,
    "claim_ledgers": ledgers,
    "queue_depth": queue_depth,
    "gpu_claims": gpu_claims,
    "gpu_devices": sum(gpu_claims.values()),
    "scheduled_devices": sum(workload.get("device_count") or 0 for workload in workloads),
    "diagnostics": diagnostics,
  }
  if log_tail:

    async def read_log(pod: dict) -> tuple[str, str]:
      try:
        text = (await asyncio.to_thread(k8s_pod_logs, pod["name"], None, log_tail))["text"]
      except Exception as exc:
        text = f"(logs unavailable: {exc})"
      return pod["name"], text

    detail["logs"] = dict(await asyncio.gather(*(read_log(pod) for pod in pods)))
  return detail


async def stop_run(store: RequestStore, worker_manager: FFTWorkerManager | None, model_id: str) -> dict:
  """Stop everything we can truthfully stop for a run: the gateway-launched worker process,
  queued work in Redis, and any pods labeled for the model. Reports each action taken."""
  actions = []
  errors = []
  changed = False

  proc = worker_manager.processes.get(model_id) if worker_manager is not None else None
  if proc is not None and proc.poll() is None:
    proc.terminate()
    actions.append("terminated local worker process")
    changed = True

  if isinstance(store, RedisStore):
    try:
      removed = await store.redis.delete(
        f"open_rl:queue:{model_id}",
        f"open_rl:sampler_queue:{model_id}",
        f"open_rl:sampler_ready:{model_id}",
      )
      await store.redis.lrem(store.active_list, 0, model_id)
      await store.redis.srem(store.active_set, model_id)
      if removed:
        actions.append(f"cleared {removed} queue key(s) in Redis")
        changed = True
    except Exception as exc:
      errors.append(f"redis cleanup failed: {exc}")
  elif isinstance(store, InMemoryStore):
    async with store.active_tenants_cv:
      if model_id in store.queues:
        del store.queues[model_id]
        store.queue_oldest_at.pop(model_id, None)
        actions.append("cleared in-memory queue")
        changed = True
      if model_id in store.active_tenants:
        store.active_tenants.remove(model_id)

  api, _ = k8s_core_v1()
  if api is not None:
    try:
      k8s = k8s_snapshot()
      for pod in model_pods(model_id, k8s["pods"]):
        api.delete_namespaced_pod(pod["name"], k8s["namespace"], _request_timeout=K8S_REQUEST_TIMEOUT)
        actions.append(f"deleted pod {pod['name']}")
        changed = True
    except Exception as exc:
      errors.append(f"pod deletion failed: {exc}")

  if changed:
    try:
      await store.set_model_metadata(model_id, {"status": "stopped", "stopped_at": time.time()})
    except Exception as exc:
      errors.append(f"lifecycle update failed: {exc}")
  return {"run_id": model_id, "stopped": changed, "actions": actions, "errors": errors}


# *** Operational load stats ***


def format_bytes(n: float) -> str:
  for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
    if n < 1024 or unit == "TiB":
      return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
    n /= 1024
  return f"{n:.1f} TiB"


def format_duration(seconds: float | None) -> str:
  seconds = max(0, int(seconds or 0))
  if seconds < 60:
    return f"{seconds}s"
  if seconds < 3600:
    return f"{seconds // 60}m {seconds % 60}s"
  return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def gateway_rss_bytes() -> int | None:
  try:
    with open("/proc/self/status") as f:
      for line in f:
        if line.startswith("VmRSS:"):
          return int(line.split()[1]) * 1024
  except (OSError, ValueError, IndexError):
    pass
  return None


def stat_entry(
  stat_id: str,
  label: str,
  value: str,
  detail: str = "",
  *,
  value_number: int | float,
  unit: str,
  context: dict | None = None,
  status: str = "ok",
) -> dict:
  return {
    "id": stat_id,
    "label": label,
    "value": value,
    "value_number": value_number,
    "unit": unit,
    "detail": detail,
    "context": context or {},
    "status": status,
  }


async def operational_stats(
  store: RequestStore,
  k8s: dict,
  worker_manager: FFTWorkerManager | None,
  queues: list[dict] | None = None,
  launch: dict | None = None,
) -> tuple[list[dict], list[dict]]:
  """Load numbers for the Health screen: queue depths, active runs, Redis and gateway memory,
  disk, and pod totals. Everything is measured, never estimated."""
  queues_provided = queues is not None
  launch_provided = launch is not None
  queues = queues or []
  launch = launch or {"depth": 0, "oldest_enqueued_at": None, "oldest_age_seconds": None}
  redis_stats: list[dict] = []
  try:
    if not queues_provided:
      queues = await store.queue_stats()
    if not launch_provided:
      launch = await store.worker_launch_stats()
  except Exception:
    pass
  if isinstance(store, RedisStore):
    try:
      memory = await store.redis.info("memory")
      used = memory.get("used_memory", 0)
      maxmemory = memory.get("maxmemory", 0)
      value = f"{used / maxmemory:.0%} of {format_bytes(maxmemory)}" if maxmemory else format_bytes(used)
      limit = f"limit {format_bytes(maxmemory)}" if maxmemory else "no maxmemory limit"
      peak = memory.get("used_memory_peak", 0)
      utilization = used / maxmemory if maxmemory else None
      redis_stats.append(
        stat_entry(
          "redis.memory",
          "Redis memory",
          value,
          f"peak {format_bytes(peak)} · {limit}",
          value_number=used,
          unit="bytes",
          context={"peak_bytes": peak, "limit_bytes": maxmemory or None, "utilization": utilization},
          status="warn" if utilization is not None and utilization >= 0.8 else "ok",
        )
      )
      clients = await store.redis.info("clients")
      connected = clients.get("connected_clients", 0)
      redis_stats.append(stat_entry("redis.clients", "Redis clients", str(connected), "connected", value_number=connected, unit="clients"))
    except Exception:
      pass
  queues.sort(key=lambda q: -q["depth"])

  workers = worker_processes(worker_manager)
  active = {model_id for model_id, alive in workers.items() if alive} | {q["model_id"] for q in queues if q["model_id"] != "default"}
  queue_warn_seconds = float(os.getenv("OPEN_RL_QUEUE_WARN_SECONDS", "300"))
  launch_warn_seconds = float(os.getenv("OPEN_RL_LAUNCH_WARN_SECONDS", "60"))
  oldest_queue = max(queues, key=lambda queue: queue["oldest_age_seconds"] or 0, default=None)
  oldest_queue_seconds = (oldest_queue or {}).get("oldest_age_seconds") or 0
  launch_depth = launch["depth"]
  launch_age = launch["oldest_age_seconds"] or 0

  stats = [
    stat_entry("runs.active", "Active runs", str(len(active)), "live worker or queued work", value_number=len(active), unit="runs"),
    stat_entry(
      "queue.requests",
      "Queued requests",
      str(sum(q["depth"] for q in queues)),
      f"across {len(queues)} queue{'' if len(queues) == 1 else 's'}",
      value_number=sum(q["depth"] for q in queues),
      unit="requests",
      context={
        "queue_count": len(queues),
        "oldest_model_id": oldest_queue["model_id"] if oldest_queue else None,
        "oldest_age_seconds": oldest_queue_seconds,
      },
      status="warn" if oldest_queue_seconds >= queue_warn_seconds else "ok",
    ),
    stat_entry(
      "queue.request_age",
      "Oldest request wait",
      format_duration(oldest_queue_seconds),
      oldest_queue["model_id"] if oldest_queue else "no queued requests",
      value_number=oldest_queue_seconds,
      unit="seconds",
      context={"model_id": oldest_queue["model_id"] if oldest_queue else None, "warn_after_seconds": queue_warn_seconds},
      status="warn" if oldest_queue_seconds >= queue_warn_seconds else "ok",
    ),
    stat_entry(
      "queue.launch",
      "Launches pending",
      str(launch_depth),
      f"oldest waiting {format_duration(launch_age)}" if launch_depth else "worker launch queue",
      value_number=launch_depth,
      unit="runs",
      context={"oldest_age_seconds": launch_age},
      status="warn" if launch_age >= launch_warn_seconds else "ok",
    ),
    stat_entry(
      "queue.launch_age",
      "Oldest launch wait",
      format_duration(launch_age),
      "worker launch queue" if launch_depth else "no pending launches",
      value_number=launch_age,
      unit="seconds",
      context={"warn_after_seconds": launch_warn_seconds},
      status="warn" if launch_age >= launch_warn_seconds else "ok",
    ),
    *redis_stats,
  ]
  rss = gateway_rss_bytes()
  if rss is not None:
    stats.append(stat_entry("gateway.rss", "Gateway memory", format_bytes(rss), "resident set size", value_number=rss, unit="bytes"))
  shared = tmp_dir()
  if os.path.isdir(shared):
    usage = shutil.disk_usage(shared)
    free_ratio = usage.free / usage.total if usage.total else 0
    stats.append(
      stat_entry(
        "storage.disk",
        "Disk free",
        format_bytes(usage.free),
        f"of {format_bytes(usage.total)} at {shared}",
        value_number=usage.free,
        unit="bytes",
        context={"total_bytes": usage.total, "free_ratio": free_ratio, "path": shared},
        status="warn" if usage.free < 20 * 2**30 else "ok",
      )
    )
  if k8s["available"]:
    phases: dict[str, int] = {}
    for pod in k8s["pods"]:
      phases[pod["phase"]] = phases.get(pod["phase"], 0) + 1
    running = phases.pop("Running", 0)
    others = " · ".join(f"{count} {phase.lower()}" for phase, count in sorted(phases.items())) or "no other phases"
    stats.append(
      stat_entry(
        "pods.running",
        "Pods running",
        str(running),
        others,
        value_number=running,
        unit="pods",
        context={"phase_counts": {"Running": running, **phases}},
        status="warn" if phases.get("Failed", 0) else "ok",
      )
    )
    total_gpus = sum(node["gpu_capacity"] for node in k8s["nodes"])
    if total_gpus:
      claimed = sum(pod.get("gpus", 0) for pod in k8s["pods"] if pod["node"] and pod["phase"] not in TERMINAL_POD_PHASES)
      ratio = claimed / total_gpus
      detail = "across all pools" if ratio <= 1 else f"{ratio:.1f}× allocation overcommit across all pools"
      stats.append(
        stat_entry(
          "gpus.claimed",
          "GPUs claimed",
          f"{claimed}/{total_gpus}",
          detail,
          value_number=claimed,
          unit="devices",
          context={"capacity_devices": total_gpus, "allocation_ratio": ratio, "overcommitted": ratio > 1},
          status="warn" if ratio > 1 else "ok",
        )
      )
    scheduler = k8s.get("scheduler") or {}
    if scheduler.get("available"):
      summary = scheduler["summary"]
      failed = summary["phase_counts"].get("Failed", 0)
      waiting = summary["phase_counts"].get("Pending", 0) + summary["phase_counts"].get("Placing", 0)
      stats.extend(
        [
          stat_entry(
            "scheduler.workloads",
            "Scheduler workloads",
            str(summary["workloads"]),
            f"{waiting} waiting · {failed} failed",
            value_number=summary["workloads"],
            unit="workloads",
            context={"phase_counts": summary["phase_counts"]},
            status="warn" if failed else "ok",
          ),
          stat_entry(
            "scheduler.seats",
            "Claim ledger seats",
            str(summary["seats"]),
            f"across {summary['ledgers']} ledgers · {summary['shared_ledgers']} shared",
            value_number=summary["seats"],
            unit="seats",
            context={"ledgers": summary["ledgers"], "shared_ledgers": summary["shared_ledgers"]},
          ),
        ]
      )
  return stats, queues


# *** Health ***


def check_entry(check_id: str, group: str, label: str, status: str, detail: str) -> dict:
  return {"id": check_id, "group": group, "label": label, "status": status, "detail": detail}


async def health_checks(store: RequestStore, k8s: dict) -> list[dict]:
  from server import gateway

  checks = []

  uptime = int(time.time() - START_TIME)
  mode = "single-process" if gateway.is_single_process_mode() else "distributed"
  fft = "FFT enabled" if gateway.is_fft_enabled() else "LoRA mode"
  checks.append(check_entry("gateway", "Gateway", "Gateway process", "ok", f"{mode}, {fft}, up {uptime // 3600}h {uptime % 3600 // 60}m"))

  if isinstance(store, RedisStore):
    started = time.perf_counter()
    ok = await ping_redis(store)
    latency_ms = (time.perf_counter() - started) * 1000
    if ok:
      checks.append(check_entry("storage.redis", "Storage", "Redis", "ok", f"PING {latency_ms:.1f} ms — {os.getenv('REDIS_URL')}"))
    else:
      checks.append(check_entry("storage.redis", "Storage", "Redis", "error", f"PING failed — {os.getenv('REDIS_URL')}"))
  else:
    checks.append(check_entry("storage.redis", "Storage", "Redis", "off", "REDIS_URL not set — using in-memory store"))

  shared = tmp_dir()
  if os.path.isdir(shared) and os.access(shared, os.W_OK):
    free_gib = shutil.disk_usage(shared).free / 2**30
    status = "warn" if free_gib < 20 else "ok"
    checks.append(check_entry("storage.shared", "Storage", "Shared filesystem", status, f"{shared} writable, {free_gib:.0f} GiB free"))
  else:
    checks.append(check_entry("storage.shared", "Storage", "Shared filesystem", "warn", f"{shared} missing or not writable"))

  if k8s["available"]:
    checks.append(check_entry("kubernetes", "Kubernetes", "API server", "ok", f"{len(k8s['pods'])} pods visible in namespace {k8s['namespace']}"))
  else:
    status = "off" if "not installed" in (k8s["error"] or "") or "credentials" in (k8s["error"] or "") else "error"
    checks.append(check_entry("kubernetes", "Kubernetes", "API server", status, k8s["error"] or "unavailable"))

  scheduler = k8s.get("scheduler") or empty_scheduler_snapshot(installed=None)
  if scheduler["installed"] is False:
    checks.append(check_entry("scheduler", "Scheduler", "Placement API", "off", "Workload CRD is not installed"))
  elif scheduler["available"]:
    summary = scheduler["summary"]
    checks.append(
      check_entry(
        "scheduler",
        "Scheduler",
        "Placement API",
        "ok",
        f"{summary['workloads']} workloads, {summary['ledgers']} claim ledgers, {summary['seats']} seats",
      )
    )
  elif scheduler["installed"] is True:
    checks.append(check_entry("scheduler", "Scheduler", "Placement API", "error", scheduler["error"] or "scheduler API unavailable"))

  if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
    checks.append(check_entry("visibility.trace", "Visibility", "Trace export", "ok", "GCP Cloud Trace exporter configured"))
  else:
    checks.append(check_entry("visibility.trace", "Visibility", "Trace export", "off", "ENABLE_GCP_TRACE=0 — tracing not configured"))

  if gateway.get_sampler_backend() == "vllm" and gateway.is_single_process_mode():
    healthz = f"{gateway.VLLM_URL.rstrip('/')}/healthz"
    try:
      async with httpx.AsyncClient(timeout=2.0) as client:
        (await client.get(healthz)).raise_for_status()
      checks.append(check_entry("visibility.sampler", "Visibility", "vLLM worker", "ok", f"reachable at {gateway.VLLM_URL}"))
    except Exception:
      checks.append(check_entry("visibility.sampler", "Visibility", "vLLM worker", "error", f"unreachable at {gateway.VLLM_URL}"))

  return checks


def derive_problems(checks: list[dict], k8s: dict, stats: list[dict] | None = None, runs: list[dict] | None = None) -> list[dict]:
  problems = []
  for check in checks:
    if check["status"] in {"warn", "error"}:
      actions = [{"label": "Refresh health checks", "method": "GET", "path": "/api/v1/dashboard/health", "command": "make ops health"}]
      if check["id"] == "kubernetes":
        actions.append({"label": "Verify pod visibility", "command": f"kubectl auth can-i list pods -n {k8s['namespace']}"})
      problems.append(
        diagnostic_entry(
          f"check.{check['id']}",
          check["status"],
          check["label"],
          check["detail"],
          resource={"kind": "Check", "name": check["id"]},
          evidence={"group": check["group"], "status": check["status"]},
          actions=actions,
        )
      )
  alerting_stats = {"queue.request_age", "queue.launch_age", "redis.memory", "storage.disk"}
  for stat in stats or []:
    if stat["status"] != "warn" or stat["id"] not in alerting_stats:
      continue
    actions = [{"label": "Inspect load metrics", "method": "GET", "path": "/api/v1/dashboard/health", "command": "make ops health"}]
    if (model_id := stat["context"].get("model_id")) and model_id != "default":
      actions.append(
        {
          "label": "Inspect waiting run",
          "method": "GET",
          "path": f"/api/v1/dashboard/runs/{model_id}",
          "command": f"make ops run {model_id}",
        }
      )
    problems.append(
      diagnostic_entry(
        f"metric.{stat['id'].replace('.', '_')}",
        "warn",
        stat["label"],
        f"{stat['value']} — {stat['detail']}",
        resource={"kind": "Metric", "name": stat["id"]},
        evidence={"value_number": stat["value_number"], "unit": stat["unit"], **stat["context"]},
        actions=actions,
      )
    )
  for run in runs or []:
    problems.extend(run_telemetry_diagnostics(run["run_id"], run.get("telemetry") or {}))
  for pod in k8s["pods"]:
    if diagnostic := pod_diagnostic(pod, k8s.get("namespace")):
      problems.append(diagnostic)
  for node in k8s["nodes"]:
    conditions = [
      (not node.get("ready"), "node.not_ready", "Node not ready"),
      (node.get("memory_pressure"), "node.memory_pressure", "Node under memory pressure"),
      (node.get("disk_pressure"), "node.disk_pressure", "Node under disk pressure"),
    ]
    for active, code, message in conditions:
      if not active:
        continue
      problems.append(
        diagnostic_entry(
          code,
          "warn",
          f"node/{node['name']}",
          message,
          resource={"kind": "Node", "name": node["name"]},
          evidence={
            "ready": node.get("ready"),
            "memory_pressure": bool(node.get("memory_pressure")),
            "disk_pressure": bool(node.get("disk_pressure")),
          },
          actions=[
            {
              "label": "Describe node",
              "method": "GET",
              "path": "/api/v1/dashboard/cluster",
              "command": f"kubectl describe node {node['name']}",
            }
          ],
        )
      )
  scheduler = k8s.get("scheduler") or {}
  if scheduler.get("available"):
    namespace = k8s["namespace"]
    workloads_by_name = {workload["name"]: workload for workload in scheduler["workloads"]}
    ledgers_by_claim = {ledger["claim_name"]: ledger for ledger in scheduler["ledgers"]}
    for workload in scheduler["workloads"]:
      source = f"workload/{workload['name']}"
      resource = {"kind": "Workload", "name": workload["name"], "namespace": namespace}
      actions = [
        {
          "label": "Inspect workload",
          "method": "GET",
          "path": "/api/v1/dashboard/snapshot",
          "command": f"kubectl get workload {workload['name']} -n {namespace} -o yaml",
        }
      ]
      evidence = {
        "phase": workload["phase"],
        "reason": workload["reason"],
        "age_seconds": workload["age_seconds"],
        "claim_name": workload["claim_name"],
        "assignment_id": workload["assignment_id"],
        "pod_name": workload["pod_name"],
        "node_name": workload["node_name"],
      }
      if workload["phase"] == "Failed":
        problems.append(
          diagnostic_entry(
            "scheduler.workload_failed",
            "error",
            source,
            workload["reason"] or "scheduler marked the workload failed",
            resource=resource,
            evidence=evidence,
            actions=actions,
          )
        )
      elif workload["phase"] in {"Pending", "Placing"} and (workload["age_seconds"] or 0) >= (300 if workload["phase"] == "Pending" else 120):
        problems.append(
          diagnostic_entry(
            f"scheduler.workload_{workload['phase'].lower()}_slow",
            "warn",
            source,
            workload["reason"] or f"workload has remained {workload['phase'].lower()} for {workload['age_seconds']} seconds",
            resource=resource,
            evidence=evidence,
            actions=actions,
          )
        )
      if workload["observed_generation"] is not None and not workload["generation_current"]:
        problems.append(
          diagnostic_entry(
            "scheduler.generation_stale",
            "warn",
            source,
            f"controller observed generation {workload['observed_generation']}, current generation is {workload['generation']}",
            resource=resource,
            evidence=evidence,
            actions=actions,
          )
        )
      if not workload["deleting"] and workload["claim_name"] and workload["assignment_id"] and (workload["age_seconds"] or 0) >= 60:
        ledger = ledgers_by_claim.get(workload["claim_name"])
        seat = next((seat for seat in (ledger or {}).get("seats", []) if seat["workload"] == workload["name"]), None)
        if seat is None or seat["assignment_id"] != workload["assignment_id"]:
          problems.append(
            diagnostic_entry(
              "scheduler.assignment_mismatch",
              "error",
              source,
              "workload status assignment does not match its ClaimLedger seat",
              resource=resource,
              evidence={**evidence, "ledger": ledger["name"] if ledger else None, "seat_assignment_id": seat["assignment_id"] if seat else None},
              actions=actions,
            )
          )
    for ledger in scheduler["ledgers"]:
      for seat in ledger["seats"]:
        if seat["workload"] in workloads_by_name or (ledger["age_seconds"] or 0) < 60:
          continue
        problems.append(
          diagnostic_entry(
            "scheduler.stale_seat",
            "warn",
            f"claimledger/{ledger['name']}",
            f"seat references missing workload {seat['workload']}",
            resource={"kind": "ClaimLedger", "name": ledger["name"], "namespace": namespace},
            evidence={"claim_name": ledger["claim_name"], "workload": seat["workload"], "assignment_id": seat["assignment_id"]},
            actions=[
              {
                "label": "Inspect claim ledger",
                "method": "GET",
                "path": "/api/v1/dashboard/snapshot",
                "command": f"kubectl get claimledger {ledger['name']} -n {namespace} -o yaml",
              }
            ],
          )
        )
  priority = {"error": 0, "warn": 1, "info": 2}
  problems.sort(key=lambda problem: (priority.get(problem["severity"], 3), problem["id"]))
  return problems
