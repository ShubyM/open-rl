"""GPU and worker hardware series from any Prometheus-compatible query API.

On GKE that is Managed Service for Prometheus, which serves DCGM GPU metrics
and GKE's own container CPU/memory in one place; through the in-cluster
gmp-frontend, or Cloud Monitoring's PromQL endpoint directly.

Hardware queries keep the metric-to-label shape until dashboard assembly:
  devices(uuids, start, end)        -> {metric: {uuid: [[t, value]]}}
  workers_series(pods, start, end)  -> {metric: {pod: [[t, value]]}}
  samplers(pods, start, end)        -> {name: [[t, value]]}, summed over a run's vLLM replicas
"""

import asyncio
import json
import re
from collections.abc import Callable

import httpx

from server.telemetry import http

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
    self._headers_task: asyncio.Task[dict] | None = None

  async def _load_headers(self) -> dict:
    headers = {}
    if self.token is not None:
      headers["Authorization"] = f"Bearer {await asyncio.wait_for(asyncio.to_thread(self.token), timeout=10)}"
    if self.extra_headers is not None:
      headers.update(self.extra_headers())
    return headers

  async def query_range(self, query: str, start: float, end: float) -> dict[str, list[list[float]]]:
    """{series label value: points} for a query aggregated `by (<one label>)`."""
    step = max(MIN_STEP_SECONDS, int((end - start) / MAX_POINTS))
    # Samples land on multiples of step, not offsets from start, so queries made
    # at different moments (or for different allocations) share timestamps and
    # can be averaged or summed sample by sample.
    start -= start % step
    # Share credential success or failure across this inspection request's queries.
    if self._headers_task is None:
      self._headers_task = asyncio.create_task(self._load_headers())
    headers = await self._headers_task
    params = {"query": query, "start": start, "end": end, "step": step}
    payload = await http.get_json(f"{self.url}/api/v1/query_range", params=params, headers=headers)
    if payload.get("status") != "success":
      raise ValueError(payload.get("error") or "query failed")
    found = {}
    for result in payload["data"]["result"]:
      label = next(iter(result["metric"].values()), "")
      found[label] = [[float(t), float(v)] for t, v in result.get("values", []) if v not in ("NaN", "+Inf", "-Inf")]
    return found

  async def query_many(self, queries: dict[str, str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    results = await asyncio.gather(*(self.query_range(q, start, end) for q in queries.values()), return_exceptions=True)
    found, errors = {}, {}
    for name, result in zip(queries, results, strict=True):
      if isinstance(result, Exception):
        errors[name] = (
          "Metrics query timed out"
          if isinstance(result, TimeoutError | httpx.TimeoutException)
          else f"Metrics source returned HTTP {result.response.status_code}"
          if isinstance(result, httpx.HTTPStatusError)
          else "Metrics query unavailable"
        )
      else:
        found[name] = result
    return found, errors

  def scope(self, cluster_label: str) -> str:
    return f",{cluster_label}={json.dumps(self.cluster)}" if self.cluster else ""

  async def devices(self, uuids: list[str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    # max by UUID: a GPU scraped by two exporters must not double-count.
    queries = {name: f"max by (UUID) ({metric}{{UUID=~{any_of(uuids)}{self.scope('cluster')}}})" for name, metric in GPU_METRICS.items()}
    return await self.query_many(queries, start, end)

  async def workers_series(self, pods: list[str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    # GKE system metrics, free on every GKE cluster; absent on other Prometheus servers.
    namespace = json.dumps(self.namespace)
    match = f'monitored_resource="k8s_container",namespace_name={namespace},pod_name=~{any_of(pods)}{self.scope("cluster_name")}'
    queries = {
      "cpu_cores": f"sum by (pod_name) (rate(kubernetes_io:container_cpu_core_usage_time{{{match}}}[2m]))",
      "memory_bytes": f'sum by (pod_name) (kubernetes_io:container_memory_used_bytes{{{match},memory_type="non-evictable"}})',
    }
    return await self.query_many(queries, start, end)

  async def samplers(self, pods: list[str], start: float, end: float) -> tuple[dict, dict[str, str]]:
    match = f"{{namespace={json.dumps(self.namespace)},pod=~{any_of(pods)}{self.scope('cluster')}}}"
    found, errors = await self.query_many({name: query.replace("{M}", match) for name, query in SAMPLER_QUERIES.items()}, start, end)
    return {name: found.get(name, {}).get("", []) for name in SAMPLER_QUERIES}, errors
