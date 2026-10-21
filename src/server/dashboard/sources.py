"""Everything the dashboard reads outside the run store: Kubernetes, Prometheus and Cloud Logging.

Kubernetes reads block and run in a thread; a missing client, credentials or scheduler CRD come back as explicit
errors, never exceptions. Prometheus is any compatible query API; on GKE that is Managed Service for Prometheus,
with DCGM GPU metrics and GKE's own container metrics. Cloud Logging is the run-log source on GKE: identity from
the metadata server, Application Default Credentials, and stateless paging through a signed cursor that keeps
the query and pod lifetimes fixed while the live inventory changes.
"""

import asyncio
import base64
import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import zlib
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import google.auth
import google.oauth2.credentials
import httpx
from google.auth.transport.requests import Request

from server.telemetry import CLIENT, RUN_LABEL, SOURCE_LABEL, iso, logs_key

# *** http: one connection pool, owned by the server lifespan ***

pool: httpx.AsyncClient | None = None
fallback: httpx.AsyncClient | None = None  # outside the lifespan, for a script or a test
inflight: dict[tuple, asyncio.Task] = {}
MAX_INFLIGHT, HTTP_SECONDS = 128, 10
RETRY_SECONDS = 1  # a quota refusal or server error is usually momentary; one retry spares the page an error


@asynccontextmanager
async def lifespan(app=None):
  global pool
  async with httpx.AsyncClient(timeout=HTTP_SECONDS) as connection:
    pool = connection
    try:
      yield
    finally:
      pending = list(inflight.values())
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
      inflight.clear()
      pool = None


def http_client() -> httpx.AsyncClient:
  global fallback
  if pool is not None:
    return pool
  if fallback is None or fallback.is_closed:
    fallback = httpx.AsyncClient(timeout=HTTP_SECONDS)
  return fallback


async def send(method: str, url: str, **kwargs) -> httpx.Response:
  """One request, retried once after a transient refusal."""
  response = await http_client().request(method, url, **kwargs)
  if response.status_code == 429 or response.status_code >= 500:
    await asyncio.sleep(RETRY_SECONDS)
    response = await http_client().request(method, url, **kwargs)
  response.raise_for_status()
  return response


async def fetch_json(url: str, params: dict, headers: dict) -> dict:
  async with asyncio.timeout(HTTP_SECONDS + RETRY_SECONDS):  # a shared request can outlive its callers
    return (await send("GET", url, params=params, headers=headers)).json()


async def get_json(url: str, *, params: dict, headers: dict) -> dict:
  """Share only overlapping identical requests, including their actual credentials."""
  key = (url, tuple(sorted(params.items())), tuple(sorted(headers.items())))
  if (task := inflight.get(key)) is None:
    if len(inflight) >= MAX_INFLIGHT:
      return await fetch_json(url, params, headers)
    task = inflight[key] = asyncio.create_task(fetch_json(url, params, headers))
    task.add_done_callback(lambda t: (inflight.pop(key, None), t.cancelled() or t.exception()))  # observed even with no waiter left
  return await asyncio.shield(task)


# *** Kubernetes ***

K8S_SECONDS = 6
SCHEDULER_GROUP, SCHEDULER_VERSION = "openrl.io", "v1alpha1"
MAX_POD_EVENTS = 10
k8s_cache: tuple[Any, Any] | None = None


def namespace() -> str:
  return os.getenv("OPEN_RL_WORKER_NAMESPACE") or os.getenv("POD_NAMESPACE") or "openrl-system"


def k8s_clients() -> tuple[Any, Any, str | None]:
  """(CoreV1Api, CustomObjectsApi, None) or (None, None, reason). Only success is cached, so credentials
  mounted after startup are picked up."""
  global k8s_cache
  if k8s_cache is None:
    try:
      from kubernetes import client, config
    except ImportError:
      return None, None, "kubernetes python client not installed"
    try:
      config.load_incluster_config()
    except Exception:
      try:
        config.load_kube_config()
      except Exception as exc:
        return None, None, f"no cluster credentials: {exc}"
    k8s_cache = client.CoreV1Api(), client.CustomObjectsApi()
  return *k8s_cache, None


def stamp(value: Any) -> str | None:
  return value.isoformat() if value is not None and hasattr(value, "isoformat") else value


def termination(state: Any) -> dict | None:
  if state is None:
    return None
  return {"reason": state.reason, "exit_code": state.exit_code, "finished_at": stamp(state.finished_at), "started_at": stamp(state.started_at)}


def container_summary(name: str, image: str | None, state=None, restarts: int = 0, last_state=None) -> dict:
  states = state.to_dict() if state else {}
  phase, values = next(((key, states[key]) for key in ("running", "waiting", "terminated") if states.get(key) is not None), ("unknown", {}))
  return {
    "name": name,
    "image": image,
    "state": phase,
    **{field: stamp(values.get(field)) for field in ("reason", "exit_code", "started_at", "finished_at")},
    "restart_count": restarts or 0,
    "last_termination": termination(last_state.terminated if last_state else None),
  }


def pod_problem(pod: Any) -> str | None:
  """The one line that explains a pod that is not simply running. A container running again after a restart
  is not one; its restart count and incident say so."""
  phase = pod.status.phase or "Unknown"
  if phase == "Failed":
    detail = ": ".join(part for part in (pod.status.reason, pod.status.message) if part)
    return f"Failed: {detail or 'see logs'}"
  for cs in pod.status.container_statuses or []:
    waiting = cs.state.waiting if cs.state else None
    if waiting and waiting.reason not in (None, "ContainerCreating", "PodInitializing"):
      return f"{waiting.reason}: {waiting.message or ''}".strip(": ")
    ended = cs.state.terminated if cs.state else None
    if ended and (ended.exit_code or ended.reason not in (None, "Completed")):
      return f"{ended.reason or 'Terminated'}: exit code {ended.exit_code}{f' — {ended.message}' if ended.message else ''}"
  if phase == "Pending":
    for cond in pod.status.conditions or []:
      if cond.type == "PodScheduled" and cond.status != "True":
        return f"Unschedulable: {cond.message or cond.reason or 'no node available'}"
    return "Pending"
  return None


def pod_summary(pod: Any) -> dict:
  statuses, labels = pod.status.container_statuses or [], pod.metadata.labels or {}
  containers = [container_summary(cs.name, cs.image, cs.state, cs.restart_count, cs.last_state) for cs in statuses]
  return {
    "name": pod.metadata.name,
    "uid": str(pod.metadata.uid),
    "phase": pod.status.phase or "Unknown",
    "node": pod.spec.node_name,
    "worker": labels.get("openrl.io/worker"),
    # Sampling pool pods: their role in the pool and the pool's name.
    "component": labels.get("app.kubernetes.io/component"),
    "pool": labels.get("app.kubernetes.io/instance"),
    "owner_uids": [str(owner.uid) for owner in pod.metadata.owner_references or []],
    "restarts": sum(cs.restart_count or 0 for cs in statuses),
    "created_at": stamp(pod.metadata.creation_timestamp),
    "problem": pod_problem(pod),
    "containers": containers or [container_summary(c.name, c.image) for c in pod.spec.containers or []],
    "events": [],
  }


def event_message(message: str | None) -> str | None:
  """A failed exec probe reports a whole traceback; its first and last lines say what happened."""
  lines = (message or "").strip().splitlines()
  return f"{lines[0]} … {lines[-1].strip()}" if len(lines) > 2 else message


def event_summary(event: Any) -> dict:
  series = event.series
  return {
    "reason": event.reason,
    "message": event_message(event.message),
    "type": event.type,
    "count": event.count or (series.count if series else None) or 1,
    "first_seen_at": stamp(event.first_timestamp or event.metadata.creation_timestamp),
    "last_seen_at": stamp((series.last_observed_time if series else None) or event.event_time or event.last_timestamp),
    "pod_uid": event.involved_object.uid,
  }


def node_summary(node: Any) -> dict:
  labels = node.metadata.labels or {}
  return {
    "name": node.metadata.name,
    "ready": any(c.type == "Ready" and c.status == "True" for c in node.status.conditions or []),
    "accelerator": labels.get("cloud.google.com/gke-accelerator") or labels.get("nvidia.com/gpu.product"),
    "gpu_capacity": int((node.status.capacity or {}).get("nvidia.com/gpu", 0)),
    # The scheduler places workers only on nodes labelled openrl.io/enabled=true.
    "openrl_enabled": labels.get("openrl.io/enabled") == "true",
    "openrl_labels": {key: value for key, value in labels.items() if key.startswith("openrl.io/")},
    "devices": [],
  }


def workload_summary(item: dict) -> dict:
  from server.scheduler_worker_manager import workload_worker  # needs the kubernetes client, which only the cluster extra installs

  metadata, spec, status = item.get("metadata") or {}, item.get("spec") or {}, item.get("status") or {}
  placed = next((c for c in status.get("conditions") or [] if c.get("type") == "Placed"), None) or {}
  return {
    "name": metadata.get("name"),
    "uid": metadata.get("uid"),
    "created_at": metadata.get("creationTimestamp"),
    "role": spec.get("role"),
    "model_id": spec.get("modelID"),
    "owner_id": spec.get("ownerID"),
    "training_kind": spec.get("trainingKind"),
    "exclusive": spec.get("exclusive", False),
    "requested_memory": (spec.get("accelerator") or {}).get("memory"),
    "phase": status.get("phase") or "Pending",
    "reason": status.get("reason"),
    "claim_name": status.get("claimName"),
    "pod_name": status.get("podName"),
    "node_name": status.get("nodeName"),
    "device_count": status.get("deviceCount", 0),
    "placed_reason": placed.get("reason"),
    "placed_message": placed.get("message"),
    "worker": workload_worker(item, time.time()),
  }


def ledger_summary(item: dict) -> dict:
  spec = item.get("spec") or {}
  seats = [
    {"workload": s.get("workload"), "workload_uid": s.get("workloadUID"), "owner": s.get("ownerID"), "exclusive": s.get("exclusive", False)}
    for s in spec.get("seats") or []
  ]
  return {"name": (item.get("metadata") or {}).get("name"), "claim_name": spec.get("claimName"), "seats": seats}


def scheduler_state(custom: Any, ns: str) -> dict:
  def listing(plural: str) -> list[dict]:
    return custom.list_namespaced_custom_object(SCHEDULER_GROUP, SCHEDULER_VERSION, ns, plural, _request_timeout=K8S_SECONDS).get("items", [])

  try:
    workloads = [workload_summary(item) for item in listing("workloads")]
    ledgers = [ledger_summary(item) for item in listing("claimledgers")]
  except Exception as exc:
    if getattr(exc, "status", None) == 404:
      return {"installed": False, "available": False, "error": None, "workloads": [], "ledgers": []}
    return {"installed": True, "available": False, "error": f"scheduler read failed: {exc}", "workloads": [], "ledgers": []}
  return {"installed": True, "available": True, "error": None, "workloads": workloads, "ledgers": ledgers}


def complete_slices(slices: list[dict], driver: str) -> tuple[list[dict], bool]:
  """Only the latest complete generation of each DRA pool describes its devices."""
  pools: dict[tuple, list[dict]] = {}
  for item in slices:
    spec = item.get("spec", {})
    if spec.get("driver") == driver and spec.get("nodeName"):
      pools.setdefault((spec["nodeName"], spec["pool"]["name"]), []).append(spec)
  complete, incomplete = [], False
  for specs in pools.values():
    generation = max(spec["pool"].get("generation", -1) for spec in specs)
    current = [spec for spec in specs if spec["pool"].get("generation", -1) == generation]
    if generation < 0 or any(spec["pool"].get("resourceSliceCount") != len(current) for spec in current):
      incomplete = True
    else:
      complete.extend(current)
  return complete, incomplete


def device_uuid(attributes: dict) -> str | None:
  return next((v.get("string") for k, v in attributes.items() if k.rsplit("/", 1)[-1].lower() == "uuid" and isinstance(v, dict)), None)


def device_inventory(custom: Any, ns: str) -> dict:
  """GPUs per node from ResourceSlices, and the devices each ResourceClaim holds."""
  driver = os.getenv("OPEN_RL_GPU_DRA_DRIVER", "gpu.nvidia.com")
  for version in ("v1", "v1beta1"):
    try:
      claims = custom.list_namespaced_custom_object("resource.k8s.io", version, ns, "resourceclaims", _request_timeout=K8S_SECONDS)["items"]
      slices = custom.list_cluster_custom_object("resource.k8s.io", version, "resourceslices", _request_timeout=K8S_SECONDS)["items"]
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        continue
      return {"available": False, "error": f"DRA discovery failed (status {getattr(exc, 'status', 'unavailable')})", "nodes": {}, "claims": {}}
    slices, incomplete = complete_slices(slices, driver)
    nodes: dict[str, list[dict]] = {}
    for spec in slices:
      for device in spec.get("devices", []):
        attributes = device.get("attributes") or device.get("basic", {}).get("attributes", {})
        identity = "/".join((spec["driver"], spec["pool"]["name"], device["name"]))
        nodes.setdefault(spec["nodeName"], []).append({"id": identity, "name": device["name"], "uuid": device_uuid(attributes)})
    held = {
      item["metadata"]["name"]: [
        "/".join((d["driver"], d["pool"], d["device"])) for d in item.get("status", {}).get("allocation", {}).get("devices", {}).get("results", [])
      ]
      for item in claims
    }
    return {"available": True, "error": "Some DRA pools have incomplete device inventory" if incomplete else None, "nodes": nodes, "claims": held}
  return {"available": False, "error": "DRA API unavailable", "nodes": {}, "claims": {}}


def cluster_unavailable(ns: str, error: str) -> dict:
  return {
    "available": False,
    "namespace": ns,
    "error": error,
    "pods": [],
    "nodes": [],
    "nodes_error": None,
    "events_error": None,
    "scheduler": {"installed": None, "available": False, "error": error, "workloads": [], "ledgers": []},
    "devices": {"available": False, "error": error, "nodes": {}, "claims": {}},
  }


def read_cluster() -> dict:
  """Pods, nodes, events, scheduler objects and DRA devices in one pass."""
  core, custom, error = k8s_clients()
  ns = namespace()
  if core is None:
    return cluster_unavailable(ns, error)

  def attempt(call, *args, **kwargs):
    try:
      return call(*args, **kwargs), None
    except Exception as exc:
      return None, str(exc)

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as threads:
    pods_f = threads.submit(attempt, core.list_namespaced_pod, ns, _request_timeout=K8S_SECONDS)
    nodes_f = threads.submit(attempt, core.list_node, _request_timeout=K8S_SECONDS)
    events_f = threads.submit(attempt, core.list_namespaced_event, ns, field_selector="involvedObject.kind=Pod", _request_timeout=K8S_SECONDS)
    scheduler_f, devices_f = threads.submit(scheduler_state, custom, ns), threads.submit(device_inventory, custom, ns)
    (node_list, nodes_error), (event_list, events_error) = nodes_f.result(), events_f.result()
    scheduler, devices, (pod_list, pods_error) = scheduler_f.result(), devices_f.result(), pods_f.result()
  if pods_error:
    return cluster_unavailable(ns, f"pod list failed: {pods_error}")
  pods, device_nodes = [pod_summary(p) for p in pod_list.items], devices.pop("nodes")
  nodes = [node_summary(n) for n in node_list.items] if node_list else []
  for node in nodes:
    node["devices"] = device_nodes.get(node["name"], [])
    node["gpu_capacity"] = max(node["gpu_capacity"], len(node["devices"]))
  events_by_pod: dict[str, list[dict]] = {}
  for event in event_list.items if event_list else []:
    summary = event_summary(event)
    events_by_pod.setdefault(summary.pop("pod_uid"), []).append(summary)
  for pod in pods:
    pod["events"] = sorted(events_by_pod.get(pod["uid"], []), key=lambda e: e["last_seen_at"] or "")[-MAX_POD_EVENTS:]
  return {
    "available": True,
    "namespace": ns,
    "error": None,
    "pods": pods,
    "nodes": nodes,
    "nodes_error": f"node list failed: {nodes_error}" if nodes_error else None,
    "events_error": f"event list failed: {events_error}" if events_error else None,
    "scheduler": scheduler,
    "devices": devices,
  }


def pod_logs(pod: str, container: str | None, tail: int, previous: bool = False) -> dict:
  core, _, error = k8s_clients()
  if core is None:
    raise RuntimeError(error or "kubernetes unavailable")
  text = core.read_namespaced_pod_log(
    pod,
    namespace(),
    container=container,
    previous=previous,
    tail_lines=tail,
    timestamps=True,
    limit_bytes=128 * 1024,
    _request_timeout=K8S_SECONDS + 4,
  )
  return {"pod": pod, "container": container, "previous": previous, "text": text}


# *** Prometheus: hardware series keep the metric-to-label shape until the dashboard assembles them ***

GPU_METRICS = {"utilization": "DCGM_FI_DEV_GPU_UTIL", "memory_mib": "DCGM_FI_DEV_FB_USED"}
# From vLLM's and llm-d's published Grafana dashboards; M is the replica matcher.
SAMPLER_QUERIES = {
  "generation_tokens_per_second": "sum(rate(vllm:generation_tokens_total{M}[2m]))",
  "prompt_tokens_per_second": "sum(rate(vllm:prompt_tokens_total{M}[2m]))",
  "requests_running": "sum(vllm:num_requests_running{M})",
  "requests_waiting": "sum(vllm:num_requests_waiting{M})",
  "kv_cache_usage": "max(vllm:kv_cache_usage_perc{M})",
  "preemptions_per_second": "sum(rate(vllm:num_preemptions_total{M}[2m]))",
  "time_to_first_token_p99_seconds": "histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{M}[2m])))",
  "request_latency_p99_seconds": "histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{M}[2m])))",
}
MAX_POINTS, MIN_STEP_SECONDS = 600, 15


def any_of(values: list[str]) -> str:
  """A PromQL regex matcher for exactly these label values."""
  return json.dumps("|".join(re.escape(v) for v in values))


class PromQL:
  def __init__(self, name, url, token: Callable[[], str] | None, cluster_name, namespace, extra_headers: Callable[[], dict] | None = None):
    self.name, self.url, self.token, self.extra_headers, self.cluster, self.namespace = name, url, token, extra_headers, cluster_name, namespace
    self.headers_task: asyncio.Task[dict] | None = None
    self.slots: dict[str, str] = {}  # GPU UUID -> the slot DCGM reported it in, which outlives the node

  async def load_headers(self) -> dict:
    headers = {}
    if self.token is not None:
      headers["Authorization"] = f"Bearer {await asyncio.wait_for(asyncio.to_thread(self.token), timeout=TOKEN_SECONDS)}"
    if self.extra_headers is not None:
      headers.update(self.extra_headers())
    return headers

  async def range_rows(self, query: str, start: float, end: float, *, limit: int | None = None) -> list[dict]:
    """Prometheus matrix rows, labels kept for discovered application metrics."""
    step = max(MIN_STEP_SECONDS, int((end - start) / MAX_POINTS))
    # Samples land on multiples of step, so queries made at different moments share timestamps and add up sample by sample.
    start -= start % step
    if self.headers_task is None:  # share credential success or failure across this request's queries
      self.headers_task = asyncio.create_task(self.load_headers())
    params = {"query": query, "start": start, "end": end, "step": step, **({"limit": limit} if limit is not None else {})}
    payload = await get_json(f"{self.url}/api/v1/query_range", params=params, headers=await self.headers_task)
    if payload.get("status") != "success":
      raise ValueError(payload.get("error") or "query failed")
    return payload["data"]["result"]

  async def query_range(self, query: str, start: float, end: float, by: str | None = None) -> dict[str, list[list[float]]]:
    """{series label value: points} for a query aggregated `by (<one label>)`, or keyed by the label `by` names."""
    found = {}
    for r in await self.range_rows(query, start, end):
      labels = r["metric"]
      if "UUID" in labels and "gpu" in labels:
        self.slots[labels["UUID"]] = labels["gpu"]
      key = labels.get(by, "") if by else next(iter(labels.values()), "")
      found[key] = [[float(t), float(v)] for t, v in r.get("values", []) if v not in ("NaN", "+Inf", "-Inf")]
    return found

  async def query_many(self, queries: dict[str, str], start: float, end: float, by: str | None = None) -> tuple[dict, dict[str, str]]:
    results = await asyncio.gather(*(self.query_range(q, start, end, by) for q in queries.values()), return_exceptions=True)
    found, errors = {}, {}
    for name, result in zip(queries, results, strict=True):
      if isinstance(result, TimeoutError | httpx.TimeoutException):
        errors[name] = "Metrics query timed out"
      elif isinstance(result, httpx.HTTPStatusError):
        errors[name] = f"Metrics source returned HTTP {result.response.status_code}"
      elif isinstance(result, Exception):
        errors[name] = "Metrics query unavailable"
      else:
        found[name] = result
    return found, errors

  def scope(self, cluster_label: str) -> str:
    return f",{cluster_label}={json.dumps(self.cluster)}" if self.cluster else ""

  async def devices(self, uuids: list[str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    # max by UUID: a GPU scraped by two exporters must not double-count.
    queries = {name: f"max by (UUID, gpu) ({metric}{{UUID=~{any_of(uuids)}{self.scope('cluster')}}})" for name, metric in GPU_METRICS.items()}
    return await self.query_many(queries, start, end, by="UUID")

  async def workers_series(self, pods: list[str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    # GKE system metrics, free on every GKE cluster; absent on other Prometheus servers.
    match = f'monitored_resource="k8s_container",namespace_name={json.dumps(self.namespace)},pod_name=~{any_of(pods)}{self.scope("cluster_name")}'
    queries = {
      "cpu_cores": f"sum by (pod_name) (rate(kubernetes_io:container_cpu_core_usage_time{{{match}}}[2m]))",
      "memory_bytes": f'sum by (pod_name) (kubernetes_io:container_memory_used_bytes{{{match},memory_type="non-evictable"}})',
    }
    return await self.query_many(queries, start, end)

  async def samplers(self, pods: list[str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    """Summed over a run's vLLM replicas."""
    match = f"{{namespace={json.dumps(self.namespace)},pod=~{any_of(pods)}{self.scope('cluster')}}}"
    found, errors = await self.query_many({name: query.replace("{M}", match) for name, query in SAMPLER_QUERIES.items()}, start, end)
    return {name: found.get(name, {}).get("", []) for name in SAMPLER_QUERIES}, errors


# *** GKE identity and Cloud Logging ***

MAX_MESSAGE, MAX_SOURCES, MAX_CURSOR, MAX_PLAN_BYTES = 16 * 1024, 64, 16 * 1024, 128 * 1024
CURSOR_SECONDS, TOKEN_SECONDS = 900, 10
# Share this secret between dashboard replicas to keep cursors valid across them.
cursor_secret = os.getenv("OPEN_RL_LOG_CURSOR_SECRET", "").encode() or None
CURSOR_KEY_RECORD = "open_rl:dashboard:log_cursor_key"
REFUSED = "Cloud Logging refused this gateway's credentials (missing logging read scope or IAM role)"
UNAVAILABLE = "Cloud Logging unavailable; check credentials, IAM and telemetry configuration"
METADATA = {"project": "project/project-id", "cluster": "instance/attributes/cluster-name", "location": "instance/attributes/cluster-location"}
detected: dict | None = None
discovered_at = 0.0
discovery_lock = asyncio.Lock()
credentials = None
credentials_lock = threading.Lock()
refused_until = 0.0
# GKE files every stderr line as ERROR, though most are progress bars, INFO logs and warnings; such a line takes the level it prints.
RELABELED = ("DEBUG", "INFO", "WARNING")
PROCESS_PREFIX = re.compile(r"^\([\w ]+ pid=\d+\)\s*")  # vLLM's "(EngineCore pid=826) "
PROGRESS = re.compile(r"\r|\d+%(\||\s+Completed)")
LEVEL = re.compile(r"\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b")
WARNING_TEXT = re.compile(r"Warning\b|warnings\.warn\(")


async def cursor_key() -> bytes:
  """One signing key for every dashboard on the run store, so a log cursor survives a restart and works on both
  the standalone and the gateway-mounted dashboard. Without the store it is random to this process."""
  global cursor_secret
  if cursor_secret is None:
    try:
      from server.store import get_state_store

      store = get_state_store()
      if not await store.get_value(CURSOR_KEY_RECORD):
        await store.set_value(CURSOR_KEY_RECORD, secrets.token_hex(32))
      cursor_secret = bytes.fromhex(await store.get_value(CURSOR_KEY_RECORD))  # read back, so two dashboards settle on one key
    except Exception:
      cursor_secret = secrets.token_bytes(32)
  return cursor_secret


def log_views() -> list[str]:
  """The log views in OPEN_RL_LOG_VIEWS, such as a long-retention bucket's _AllLogs view, or the project's default buckets."""
  return [v.strip() for v in os.getenv("OPEN_RL_LOG_VIEWS", "").split(",") if v.strip()] or [f"projects/{configuration()['project']}"]


def configuration() -> dict:
  mode, found = os.getenv("OPEN_RL_GKE_TELEMETRY", "auto").strip().lower(), detected or {}
  fields = {key: os.getenv(f"OPEN_RL_GKE_{key.upper()}") or found.get(key, "") for key in ("project", "cluster", "location")}
  complete = all(fields.values())
  enabled = mode in {"1", "true"} or (mode == "auto" and complete)
  return {**fields, "mode": mode, "enabled": enabled, "configured": enabled and complete}


async def discover() -> dict:
  """Resolve GKE identity once; retry unavailable metadata after a minute."""
  global detected, discovered_at
  config = configuration()
  if config["mode"] not in {"auto", "1", "true"} or config["configured"] or not os.getenv("KUBERNETES_SERVICE_HOST"):
    return config
  async with discovery_lock:
    if time.monotonic() - discovered_at < 60:
      return configuration()
    async with httpx.AsyncClient(timeout=1, trust_env=False, follow_redirects=False) as client:

      async def read(key, path):
        response = await client.get(f"http://metadata.google.internal/computeMetadata/v1/{path}", headers={"Metadata-Flavor": "Google"})
        response.raise_for_status()
        value = response.text.strip()
        if response.headers.get("Metadata-Flavor") != "Google" or not value or len(value) > 256:
          raise ValueError("Invalid metadata response")
        return key, value

      values = await asyncio.gather(*(read(key, path) for key, path in METADATA.items()), return_exceptions=True)
    if all(isinstance(value, tuple) for value in values):  # never from partial metadata: generic GCE also exposes a project
      detected = dict(values)
    discovered_at = time.monotonic()
    return configuration()


def access_token() -> str:
  global credentials
  with credentials_lock:
    if credentials is None:
      credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform.read-only"])
    if not credentials.valid:
      credentials.refresh(Request())
    return credentials.token


def quota_headers() -> dict[str, str]:
  """Bill API reads to this project on a person's gcloud login, which otherwise draws on gcloud's shared quota
  and fails with 429. Service-account credentials, as in the cluster, bill their own project already."""
  if isinstance(credentials, google.oauth2.credentials.Credentials) and not credentials.quota_project_id and configuration()["project"]:
    return {"x-goog-user-project": configuration()["project"]}
  return {}


async def gcp_request(method: str, url: str, **kwargs) -> dict:
  # A hung token refresh holds the credentials lock; bound it rather than a thread per call.
  token = await asyncio.wait_for(asyncio.to_thread(access_token), timeout=TOKEN_SECONDS)
  return (await send(method, url, headers={"Authorization": f"Bearer {token}", **quota_headers()}, **kwargs)).json()


def timestamp(value) -> str:
  """ISO-8601 in UTC with microseconds, the form Cloud Logging filters compare."""
  if isinstance(value, int | float):
    return datetime.fromtimestamp(value, UTC).isoformat(timespec="microseconds")
  parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
  return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(UTC).isoformat(timespec="microseconds")


def log_window(since=None, until=None) -> tuple[str, str]:
  end = datetime.fromisoformat(timestamp(until)) if until else datetime.now(UTC)
  start = datetime.fromisoformat(timestamp(since)) if since else end - timedelta(minutes=30)
  if start >= end or end - start > timedelta(days=7):
    raise ValueError("Time range must be positive and no longer than seven days")
  return start.isoformat(timespec="microseconds"), end.isoformat(timespec="microseconds")


def pod_sources(run: dict | None, lanes: list[dict], observed_at: str) -> list[dict]:
  """The pods a run's logs can come from: live ones until now, and each worker process of the run for its lifetime.
  Names are bound to lifetimes, since Cloud Logging labels carry pod names and not UIDs."""
  sources: dict[tuple, dict] = {}
  for lane in lanes:
    if lane.get("pod"):
      sources[lane["pod"], lane.get("pod_uid") or lane["start"]] = {
        "pod": lane["pod"],
        "node": lane.get("node"),
        "role": lane.get("role") or "unknown",
        "created_at": timestamp(lane["start"]),
        "until": timestamp(lane["end"] + 60),
      }
  for pod in (run or {}).get("pods", []):
    if pod.get("created_at"):
      sources[pod["name"], pod.get("uid") or pod["created_at"]] = {
        "pod": pod["name"],
        "node": pod.get("node"),
        "role": pod.get("role", "unknown"),
        "created_at": timestamp(pod["created_at"]),
        "until": timestamp(observed_at),
        "shared_runtime": pod.get("shared_runtime", False),
      }
  return sorted(sources.values(), key=lambda source: source["created_at"], reverse=True)


def page_cursor(plan: dict) -> str:
  raw = json.dumps({**plan, "expires": time.time() + CURSOR_SECONDS}, separators=(",", ":")).encode()
  if len(raw) > MAX_PLAN_BYTES:
    raise ValueError("Log query is too large to page; narrow the pod filter")
  payload = zlib.compress(raw)
  token = base64.urlsafe_b64encode(hmac.digest(cursor_secret, payload, "sha256") + payload).decode()
  if len(token) > MAX_CURSOR:
    raise ValueError("Log query is too large to page; narrow the pod filter")
  return token


def page_plan(cursor: str, selection: list, since, until) -> dict:
  try:
    if len(cursor) > MAX_CURSOR:
      raise ValueError
    signed = base64.urlsafe_b64decode(cursor)
    signature, payload = signed[:32], signed[32:]
    if not hmac.compare_digest(signature, hmac.digest(cursor_secret, payload, "sha256")):
      raise ValueError
    decoder = zlib.decompressobj()
    raw = decoder.decompress(payload, MAX_PLAN_BYTES + 1)
    if len(raw) > MAX_PLAN_BYTES or not decoder.eof:
      raise ValueError
    plan = json.loads(raw)
    if plan["expires"] < time.time() or plan["selection"] != selection:
      raise ValueError
    if any(value is not None and timestamp(value) != bound for value, bound in zip((since, until), plan["window"], strict=True)):
      raise ValueError
    return plan
  except (ValueError, KeyError, TypeError, zlib.error):
    raise ValueError("Invalid or expired log cursor; repeat the query without it") from None


def scope_filter() -> list[str]:
  config = configuration()
  labels = (("project_id", config["project"]), ("location", config["location"]), ("cluster_name", config["cluster"]), ("namespace_name", namespace()))
  return ['resource.type="k8s_container"', *[f"resource.labels.{label}={json.dumps(value)}" for label, value in labels]]


def source_filter(sources: list[dict], client_of: str | None = None) -> str:
  clauses = [
    f"(resource.labels.pod_name={json.dumps(s['pod'])} AND timestamp>={json.dumps(s['created_at'])} AND timestamp<={json.dumps(s['until'])})"
    for s in sources
  ]
  if client_of:
    clauses.append(f"(labels.{RUN_LABEL}={json.dumps(client_of)} AND labels.{SOURCE_LABEL}={json.dumps(CLIENT)})")
  return "(" + " OR ".join(clauses) + ")"


def logs_filter(sources: list[dict], start: str, end: str, severity: str | None, q: str, client_of: str | None = None) -> str:
  clauses = scope_filter() + [source_filter(sources, client_of), f"timestamp>={json.dumps(start)}", f"timestamp<={json.dumps(end)}"]
  if severity in RELABELED:
    stderr = json.dumps(f"projects/{configuration()['project']}/logs/stderr")
    clauses.append(f'(severity={json.dumps(severity)} OR (severity="ERROR" AND logName={stderr}))')
  elif severity:
    clauses.append(f"severity={json.dumps('DEFAULT' if severity == 'UNKNOWN' else severity)}")
  if q:
    clauses.append(f"(textPayload:{json.dumps(q)} OR jsonPayload.message:{json.dumps(q)})")
  return " AND ".join(clauses)


def stderr_severity(text: str) -> str:
  body = PROCESS_PREFIX.sub("", text)
  if not body.strip() or PROGRESS.search(body):
    return "INFO"
  if level := LEVEL.search(body[:80]):
    return "WARNING" if level[1] == "WARN" else level[1]
  return "WARNING" if WARNING_TEXT.search(body) else "ERROR"


def entry_record(entry: dict, sources: list[dict], run_id: str) -> dict | None:
  """One log entry as a dashboard record, or None when it belongs to no source lifetime."""
  labels, at = entry.get("resource", {}).get("labels", {}), timestamp(entry["timestamp"])
  if entry.get("labels", {}).get(SOURCE_LABEL) == CLIENT and entry["labels"].get(RUN_LABEL) == run_id:
    match = {"pod": CLIENT, "role": CLIENT}
  else:
    match = next((s for s in sources if s["pod"] == labels.get("pod_name") and s["created_at"] <= at <= s["until"]), None)
  if match is None:
    return None
  fields = entry.get("jsonPayload") or {}
  text = entry.get("textPayload")
  if text is None:
    text = json.dumps(fields or entry.get("protoPayload", {}), ensure_ascii=False)
  severity = entry.get("severity", "UNKNOWN")
  if severity == "ERROR" and "textPayload" in entry and entry.get("logName", "").endswith("/logs/stderr"):
    severity = stderr_severity(text)
  return {
    "id": hashlib.sha256(json.dumps([entry.get("logName"), entry.get("insertId"), entry.get("timestamp"), text]).encode()).hexdigest(),
    "run_id": run_id,
    "timestamp": entry.get("timestamp"),
    "pod": match["pod"],
    "container": labels.get("container_name"),
    "node": match.get("node"),
    "role": match.get("role", "unknown"),
    "severity": severity,
    "rank": fields.get("rank"),
    "request_id": fields.get("request_id"),
    "message": text[:MAX_MESSAGE],
    "message_truncated": len(text) > MAX_MESSAGE,
  }


async def run_logs(
  run_id: str, sources: list[dict], *, since=None, until=None, cursor=None, limit=200, q="", pod=None, node=None, severity=None, shared_runtime=False
) -> dict:
  global refused_until
  await cursor_key()
  selection = [run_id, limit, q, pod, node, severity]
  plan = page_plan(cursor, selection, since, until) if cursor else None
  result = {
    "run_id": run_id,
    "source": "gke",
    "records": [],
    "sources": sources,
    "order": "newest_first",
    "next_cursor": None,
    "available": False,
    "shared_runtime": plan["shared_runtime"] if plan else shared_runtime,
  }
  if not configuration()["configured"]:
    return {**result, "error": "GKE telemetry is not configured"}
  if time.monotonic() < refused_until:
    return {**result, "error": REFUSED}
  if plan is None:
    start, end = log_window(since, until)
    selected = [s for s in sources if (pod is None or s["pod"] == pod) and (node is None or s.get("node") == node)]
    client_of = run_id if pod in (None, CLIENT) and node is None else None  # the client's output is one more source, unless a filter names another
    if not selected and not client_of:
      return {**result, "available": True, "note": "No pod lifetimes match this run and filter"}
    if len(selected) > MAX_SOURCES:
      raise ValueError(f"Select a pod to narrow this query to at most {MAX_SOURCES} pod lifetimes")
    query = {
      "resourceNames": log_views(),
      "filter": logs_filter(selected, start, end, severity, q, client_of),
      "orderBy": "timestamp desc",
      "pageSize": limit,
    }
    plan = {"selection": selection, "shared_runtime": shared_runtime, "window": [start, end], "sources": selected, "request": query}
  result.update(sources=plan["sources"], since=plan["window"][0], until=plan["window"][1])
  try:
    payload = await gcp_request("POST", "https://logging.googleapis.com/v2/entries:list", json=plan["request"])
    records = [record for entry in payload.get("entries", []) if (record := entry_record(entry, plan["sources"], run_id))]
    if severity in (*RELABELED, "ERROR"):
      records = [record for record in records if record["severity"] == severity]
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code in (401, 403):
      refused_until = time.monotonic() + 600  # a scope or IAM refusal does not clear itself; stop asking for a while
      return {**result, "error": REFUSED}
    if exc.response.status_code == 400 and cursor:
      raise ValueError("Invalid or expired log cursor; repeat the query without it") from None
    return {**result, "error": UNAVAILABLE}
  except Exception:
    return {**result, "error": UNAVAILABLE}
  token = payload.get("nextPageToken")
  next_cursor = page_cursor({**plan, "request": {**plan["request"], "pageToken": token}}) if token else None
  return {**result, "records": records, "available": True, "next_cursor": next_cursor}


async def client_tail(store, run_id: str, limit: int) -> list[dict]:
  """The newest client lines the gateway kept in the run store, newest first, as log records."""
  records = []
  for line in reversed(await store.read_samples(logs_key(run_id), last=limit)):
    identity = hashlib.sha256(json.dumps([run_id, line.get("at"), line.get("message")]).encode()).hexdigest()
    at, severity, message = iso(line.get("at", 0.0)), line.get("severity", "INFO"), line.get("message", "")
    records.append({"id": identity, "run_id": run_id, "timestamp": at, "pod": CLIENT, "role": CLIENT, "severity": severity, "message": message})
  return records
