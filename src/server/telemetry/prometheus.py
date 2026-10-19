"""GPU and worker hardware series from any Prometheus-compatible query API.

On GKE that is Managed Service for Prometheus, which serves DCGM GPU metrics
and GKE's own container CPU/memory in one place; through the in-cluster
gmp-frontend, or Cloud Monitoring's PromQL endpoint directly.

Answers two questions, keyed the way the inventory names things, over [start, end]:
  devices(uuids, start, end)        -> {uuid: {"utilization": [[t, %]], "memory_mib": [[t, MiB]]}}
  workers_series(pods, start, end)  -> {pod: {"cpu_cores": [[t, cores]], "memory_bytes": [[t, B]]}}
  samplers(pods, start, end)        -> {name: [[t, value]]}, summed over a run's vLLM replicas
"""

import asyncio
import json
import re
from collections.abc import Callable

import httpx

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
MAX_POINTS = 600
MIN_STEP_SECONDS = 15


def any_of(values: list[str]) -> str:
  """A PromQL regex matcher for exactly these label values."""
  return json.dumps("|".join(re.escape(v) for v in values))


class PromQL:
  def __init__(
    self,
    name: str,
    url: str,
    token: Callable[[], str] | None,
    cluster_name: str | None,
    namespace: str,
    extra_headers: Callable[[], dict] | None = None,
  ) -> None:
    self.name = name
    self.url = url
    self.token = token
    self.extra_headers = extra_headers
    self.cluster = cluster_name
    self.namespace = namespace

  async def query_range(self, query: str, start: float, end: float) -> dict[str, list[list[float]]]:
    """{series label value: points} for a query aggregated `by (<one label>)`."""
    headers = {}
    if self.token is not None:
      headers["Authorization"] = f"Bearer {await asyncio.wait_for(asyncio.to_thread(self.token), timeout=10)}"
    if self.extra_headers is not None:
      headers.update(self.extra_headers())
    step = max(MIN_STEP_SECONDS, int((end - start) / MAX_POINTS))
    # Samples land on multiples of step, not offsets from start, so queries made
    # at different moments (or for different allocations) share timestamps and
    # can be averaged or summed sample by sample.
    start -= start % step
    async with httpx.AsyncClient(timeout=10) as client:
      params = {"query": query, "start": start, "end": end, "step": step}
      response = await client.get(f"{self.url}/api/v1/query_range", params=params, headers=headers)
      if response.status_code == 429 or response.status_code >= 500:
        await asyncio.sleep(1)
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

  async def samplers(self, pods: list[str], start: float, end: float) -> dict[str, list[list[float]]]:
    match = f"{{namespace={json.dumps(self.namespace)},pod=~{any_of(pods)}{self.scope('cluster')}}}"
    results = await asyncio.gather(*(self.query_range(query.replace("{M}", match), start, end) for query in SAMPLER_QUERIES.values()))
    return {name: found.get("", []) for name, found in zip(SAMPLER_QUERIES, results, strict=True)}
