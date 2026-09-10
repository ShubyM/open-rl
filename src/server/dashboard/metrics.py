"""Optional DCGM/Prometheus GPU history, joined by UUID rather than guessed index."""

import json
import math
import os

import httpx

from server.dashboard import gke
from server.dashboard.data import observations
from server.store import get_store


def device_uuids(state: dict) -> dict:
  return {
    device["id"]: next((value.get("string") for key, value in device.get("attributes", {}).items() if key.rsplit("/", 1)[-1].lower() == "uuid"), None)
    for node in state["cluster"]["nodes"]
    for device in node["devices"]
  }


async def gpu_history(placement_id: str, since=None, until=None) -> dict:
  start, end = gke.time_range(since, until)
  state = await observations.snapshot(get_store())
  placement = next((p for p in state["placements"] if p["id"] == placement_id), None)
  if placement is None:
    return {"available": False, "reason": "Allocation no longer present", "devices": []}
  url = os.getenv("OPEN_RL_PROMETHEUS_URL", "").rstrip("/")
  if not url:
    await gke.discover()
  if not url and gke.configuration()["enabled"]:
    return await gke_gpu_history(state, placement, since, until)
  if not url:
    return {"available": False, "reason": "GPU metrics source is not configured", "devices": []}
  uuids = device_uuids(state)
  devices = []
  async with httpx.AsyncClient(timeout=8) as client:
    for identity in placement["devices"]:
      uuid = uuids.get(identity)
      if not uuid:
        devices.append({"id": identity, "utilization": [], "memory_mib": [], "reason": "GPU UUID unavailable"})
        continue
      series = {"id": identity, "uuid": uuid}
      for name, metric in (("utilization", "DCGM_FI_DEV_GPU_UTIL"), ("memory_mib", "DCGM_FI_DEV_FB_USED")):
        try:
          response = await client.get(
            url + "/api/v1/query_range",
            params={
              "query": metric + "{UUID=" + json.dumps(uuid) + "}",
              "start": start,
              "end": end,
              "step": 15,
            },
          )
          response.raise_for_status()
          payload = response.json()
          if payload.get("status") != "success":
            raise ValueError("Query failed")
          matches = [r for r in payload["data"]["result"] if r["metric"].get("UUID") == uuid]
          # Duplicate scrape targets must not inflate utilization or memory.
          values = {}
          for result in matches:
            for at, value in result.get("values", []):
              number = float(value)
              if math.isfinite(number):
                values[float(at)] = max(number, values.get(float(at), number))
          series[name] = sorted(values.items())
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
          series[name] = []
          series["reason"] = "GPU metrics query unavailable"
      devices.append(series)
  available = any(d.get("utilization") for d in devices)
  return {
    "available": available,
    "reason": None if available else "No GPU samples matched this allocation",
    "devices": devices,
    "scope": "Physical GPUs; utilization can include other workloads sharing the device",
    "mfu": None,
  }


async def gke_gpu_history(state: dict, placement: dict, since=None, until=None) -> dict:
  pods = {p["name"]: p for run in state["runs"] for p in run["pods"] if p["name"] == placement.get("pod")}
  sources = gke.pod_sources({"pods": list(pods.values())}, [], state["observed_at"])
  history = await gke.resource_metrics(sources, since, until)
  uuids = device_uuids(state)
  devices = []
  for identity in placement["devices"]:
    uuid = uuids.get(identity)
    device = {"id": identity, "utilization": [], "memory_mib": []}
    if uuid:
      for name, target, divisor in (("gpu_utilization", "utilization", 1), ("gpu_memory", "memory_mib", 1024 * 1024)):
        points = {}
        for series in history["series"]:
          if series["name"] == name and series["device"] == uuid:
            for at, value in series["points"]:
              points[at] = max(points.get(at, value), value)
        device[target] = [[at, value / divisor] for at, value in sorted(points.items())]
    devices.append(device)
  available = any(d["utilization"] for d in devices)
  return {
    "source": "gke",
    "available": available,
    "devices": devices,
    "coverage": history["coverage"],
    "mfu": None,
    "reason": history.get("error") or (None if available else "GKE accelerator IDs did not match observed GPU UUIDs"),
  }
