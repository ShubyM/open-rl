"""GPU and worker hardware series from any Prometheus-compatible query API.

On GKE that is Managed Service for Prometheus, which serves DCGM GPU metrics
and GKE's own container CPU/memory in one place; through the in-cluster
gmp-frontend, or Cloud Monitoring's PromQL endpoint directly.

Answers the two questions every hardware source answers (local.Sampler too),
keyed the way the inventory names things, over [start, end]:
  devices(uuids, start, end)        -> {uuid: {"utilization": [[t, %]], "memory_mib": [[t, MiB]]}}
  workers_series(pods, start, end)  -> {pod: {"cpu_cores": [[t, cores]], "memory_bytes": [[t, B]]}}
"""

import asyncio
import json
import re
from collections.abc import Callable

import httpx

GPU_METRICS = {"utilization": "DCGM_FI_DEV_GPU_UTIL", "memory_mib": "DCGM_FI_DEV_FB_USED"}
MAX_POINTS = 600
MIN_STEP_SECONDS = 15


def any_of(values: list[str]) -> str:
  """A PromQL regex matcher for exactly these label values."""
  return json.dumps("|".join(re.escape(v) for v in values))


class PromQL:
  def __init__(self, name: str, url: str, token: Callable[[], str] | None, cluster_name: str | None, namespace: str) -> None:
    self.name = name
    self.url = url
    self.token = token
    self.cluster = cluster_name
    self.namespace = namespace

  async def query_range(self, query: str, start: float, end: float) -> dict[str, list[list[float]]]:
    """{series label value: points} for a query aggregated `by (<one label>)`."""
    headers = {}
    if self.token is not None:
      headers["Authorization"] = f"Bearer {await asyncio.wait_for(asyncio.to_thread(self.token), timeout=10)}"
    step = max(MIN_STEP_SECONDS, int((end - start) / MAX_POINTS))
    async with httpx.AsyncClient(timeout=10) as client:
      params = {"query": query, "start": start, "end": end, "step": step}
      response = await client.get(f"{self.url}/api/v1/query_range", params=params, headers=headers)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
      raise ValueError(payload.get("error") or "query failed")
    found = {}
    for result in payload["data"]["result"]:
      label = next(iter(result["metric"].values()), "")
      found[label] = [[float(t), float(v)] for t, v in result.get("values", []) if v not in ("NaN", "+Inf", "-Inf")]
    return found

  def scope(self, cluster_label: str) -> str:
    return f",{cluster_label}={json.dumps(self.cluster)}" if self.cluster else ""

  async def devices(self, uuids: list[str], start: float, end: float) -> dict[str, dict]:
    # max by UUID: a GPU scraped by two exporters must not double-count.
    queries = {name: f"max by (UUID) ({metric}{{UUID=~{any_of(uuids)}{self.scope('cluster')}}})" for name, metric in GPU_METRICS.items()}
    results = await asyncio.gather(*(self.query_range(q, start, end) for q in queries.values()))
    return {uuid: {name: series.get(uuid, []) for name, series in zip(queries, results, strict=True)} for uuid in uuids}

  async def workers_series(self, pods: list[str], start: float, end: float) -> dict[str, dict]:
    # GKE system metrics, free on every GKE cluster; absent on other Prometheus servers.
    namespace = json.dumps(self.namespace)
    match = f'monitored_resource="k8s_container",namespace_name={namespace},pod_name=~{any_of(pods)}{self.scope("cluster_name")}'
    cpu, memory = await asyncio.gather(
      self.query_range(f"sum by (pod_name) (rate(kubernetes_io:container_cpu_core_usage_time{{{match}}}[2m]))", start, end),
      self.query_range(f'sum by (pod_name) (kubernetes_io:container_memory_used_bytes{{{match},memory_type="non-evictable"}})', start, end),
    )
    return {pod: {"cpu_cores": cpu.get(pod, []), "memory_bytes": memory.get(pod, [])} for pod in pods}
