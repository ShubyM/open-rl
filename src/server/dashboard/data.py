"""Read-only joins between logical runs, scheduler runtimes, pods and DRA devices."""

import asyncio
import copy
import os
import time
from collections import deque

from server.dashboard import kubernetes as kube


def complete_device_slices(slices: list[dict], driver: str) -> tuple[list[dict], bool]:
  """Only the latest complete generation of each DRA pool describes its devices."""
  pools = {}
  for item in slices:
    spec = item.get("spec", {})
    if spec.get("driver") == driver and spec.get("nodeName"):
      pools.setdefault((spec["nodeName"], spec["pool"]["name"]), []).append(spec)
  complete = []
  incomplete = False
  for specs in pools.values():
    generation = max(spec["pool"].get("generation", -1) for spec in specs)
    current = [spec for spec in specs if spec["pool"].get("generation", -1) == generation]
    if generation < 0 or any(spec["pool"].get("resourceSliceCount") != len(current) for spec in current):
      incomplete = True
    else:
      complete.extend(current)
  return complete, incomplete


def device_inventory() -> dict:
  api, error = kube.k8s_custom_objects()
  if api is None:
    return {"available": False, "error": error, "claims": {}, "nodes": {}}
  for version in ("v1", "v1beta1"):
    try:
      claims = api.list_namespaced_custom_object("resource.k8s.io", version, kube.k8s_namespace(), "resourceclaims", _request_timeout=4)["items"]
      slices = api.list_cluster_custom_object("resource.k8s.io", version, "resourceslices", _request_timeout=4)["items"]
      slices, incomplete = complete_device_slices(slices, os.getenv("OPEN_RL_GPU_DRA_DRIVER", "gpu.nvidia.com"))
      nodes = {}
      for spec in slices:
        node = spec["nodeName"]
        for device in spec.get("devices", []):
          identity = "/".join((spec["driver"], spec["pool"]["name"], device["name"]))
          nodes.setdefault(node, {})[identity] = {
            "id": identity,
            "name": device["name"],
            "attributes": device.get("attributes") or device.get("basic", {}).get("attributes", {}),
          }
      return {
        "available": True,
        "error": "Some DRA pools have incomplete device inventory" if incomplete else None,
        "nodes": {node: list(devices.values()) for node, devices in nodes.items()},
        "claims": {
          item["metadata"]["name"]: [
            "/".join((device["driver"], device["pool"], device["device"]))
            for device in item.get("status", {}).get("allocation", {}).get("devices", {}).get("results", [])
          ]
          for item in claims
        },
      }
    except Exception as exc:
      if getattr(exc, "status", None) == 404:
        continue
      return {"available": False, "error": f"DRA discovery failed (status {getattr(exc, 'status', 'unavailable')})", "claims": {}, "nodes": {}}
  return {"available": False, "error": "DRA API unavailable", "claims": {}, "nodes": {}}


def observed_run_status(metadata: dict, workloads: list[dict], pods: list[dict], available: bool) -> str:
  """Distinguish recorded lifecycle from current Kubernetes execution evidence."""
  recorded = str(metadata.get("status", "")).lower()
  if recorded in {"completed", "failed"}:
    return recorded.capitalize()
  if not available:
    return "Unknown"
  if any(pod.get("problem") for pod in pods):
    return "Needs attention"
  if any(pod.get("phase") == "Running" for pod in pods):
    return "Running"
  if pods or any(workload.get("node_name") for workload in workloads):
    return "Starting"
  if workloads:
    return "Queued"
  return "Unassigned"


def join_runs(metadata: list[dict], cluster: dict, inventory: dict) -> list[dict]:
  """LoRA modelID is the base-model runtime; FFT modelID is the logical run ID."""
  workloads = cluster.get("scheduler", {}).get("workloads", [])
  runs = []
  for row in metadata:
    run_id = row["model_id"]
    lora = row.get("fine_tuning_type", "lora") == "lora"
    runtime = row.get("base_model") if lora else run_id
    matched = [w for w in workloads if w.get("model_id") == runtime and w.get("training_kind") == ("lora" if lora else "fft")]
    pods = []
    for pod in cluster.get("pods", []):
      owners = {owner["uid"] for owner in pod.get("owner_references", [])}
      owned = [
        w
        for w in matched
        if w.get("uid") in owners or (w.get("pod_name") == pod["name"] and pod.get("labels", {}).get("openrl.io/worker") == w["name"])
      ]
      if owned:
        pod = copy.deepcopy(pod)
        pod["role"] = owned[0].get("role")
        pod["shared_runtime"] = lora
        pod["runtime_id"] = runtime
        pod["devices"] = sorted({device for w in owned for device in inventory["claims"].get(w.get("claim_name"), [])})
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
        "shared_runtime": lora,
        "status": row.get("status", "unknown"),
        "display_status": observed_run_status(row, matched, pods, bool(cluster.get("available"))),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "completed_at": row.get("completed_at"),
        "steps": row.get("total_steps_completed"),
        "max_steps": row.get("max_steps"),
        "pods": pods,
        "workloads": matched,
        "nodes": sorted({p["node"] for p in pods if p.get("node")}),
        "metrics": {"gpu_utilization": None, "gpu_memory": None, "mfu": None, "reason": "GPU telemetry is not reported by this gateway"},
      }
    )
  for run in runs:
    run["runtime_run_ids"] = [r["run_id"] for r in runs if r["runtime_id"] == run["runtime_id"]]
  return sorted(runs, key=lambda r: str(r.get("created_at") or ""), reverse=True)


class Observations:
  """Bounded in-process observations; no inference about unobserved historical duty."""

  def __init__(self):
    self.samples = deque(maxlen=361)
    self.lock = asyncio.Lock()
    self.latest = None
    self.updated = 0.0

  async def snapshot(self, store):
    async with self.lock:
      age = time.monotonic() - self.updated
      if self.latest is not None and age < 5:
        cluster = self.latest["cluster"]
        return {
          **self.latest,
          "cluster": {**cluster, "observation": {**cluster["observation"], "source": "cache", "age_seconds": round(age, 6)}},
        }
      cluster, inventory = await asyncio.gather(asyncio.to_thread(kube.k8s_snapshot), asyncio.to_thread(device_inventory))
      store_error = None
      try:
        metadata = await asyncio.wait_for(store.list_jobs_metadata(), timeout=5)
      except Exception:
        metadata, store_error = [], "Run store unavailable"
      runs = join_runs(metadata, cluster, inventory)
      placements = []
      for workload in cluster.get("scheduler", {}).get("workloads", []):
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
            "devices": inventory["claims"].get(workload.get("claim_name"), []),
          }
        )
      now = time.time()
      self.samples.append({"at": now, "available": cluster["available"], "placements": placements})
      # Keep the observation covering the window's left edge, including between polls.
      while len(self.samples) > 1 and self.samples[1]["at"] <= now - 1800:
        self.samples.popleft()
      for node in cluster["nodes"]:
        node["devices"] = inventory["nodes"].get(node["name"], [])
        node["gpu_capacity"] = max(node["gpu_capacity"], len(node["devices"]))
      self.latest = {
        "schema_version": 1,
        "observed_at": kube.iso_timestamp(now),
        "build": kube.build_summary(),
        "cluster": cluster,
        "device_inventory": inventory,
        "runs": runs,
        "store_error": store_error,
        "placements": placements,
        "history": list(self.samples),
        "coverage": {
          "history_complete": False,
          "history_start": self.samples[0]["at"],
          "scope": cluster["namespace"],
          "identity": "shared_operator",
          "gpu_telemetry": "queried_on_demand",
        },
      }
      self.updated = time.monotonic()
      return self.latest


observations = Observations()
