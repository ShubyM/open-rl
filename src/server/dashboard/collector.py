"""Discovered OpenRL gauges, read from the same Prometheus as hardware metrics."""

import hashlib
import json
import math
import re

from server.dashboard.metrics import hardware
from server.telemetry.prometheus import any_of

SERIES_LIMIT = 200
IDENTITIES = (
  ("model", "openrl_model_id"),
  ("session", "session_id"),
  ("runtime", "openrl_runtime_id"),
  ("pod", "k8s_pod_name"),
  ("node", "k8s_node_name"),
)


async def series(start: float, end: float, *, run_id=None, session_ids=(), runtime_ids=(), node_id=None, metric="") -> dict:
  backend, _ = hardware()
  result = {"available": False, "series": [], "limit": SERIES_LIMIT, "truncated": False}
  if backend is None:
    return {**result, "reason": "Collector metrics need a Prometheus backend. Configure OPEN_RL_PROMETHEUS_URL or GKE monitoring."}

  # A session or runtime may serve several models. Include its unassigned values,
  # clearly labelled with their wider scope; never claim another model's values.
  selectors = []
  session_ids = [value for value in session_ids if value]
  runtime_ids = [value for value in runtime_ids if value]
  if run_id:
    selectors.append(f"openrl_model_id={json.dumps(run_id)}")
    if session_ids:
      selectors.append(f'openrl_model_id="",session_id=~{any_of(session_ids)}')
    if runtime_ids:
      selectors.append(f'openrl_model_id="",session_id="",openrl_runtime_id=~{any_of(runtime_ids)}')
  if node_id:
    selectors.append(f"k8s_node_name={json.dumps(node_id)}")
  name = json.dumps("(?i).*" + re.escape(metric.strip()) + ".*")
  namespace = json.dumps("|" + re.escape(backend.namespace))
  common = f'__name__=~"openrl_(client|server)_metric",metric=~{name},namespace=~{namespace}{backend.scope("cluster")}'
  query = " or ".join("{" + common + "," + selector + "}" for selector in selectors)
  try:
    # Prometheus bounds the response at the source. Cloud Monitoring does not
    # expose this parameter; its response is still bounded below for the UI.
    rows = await backend.range_rows(query, start, end, limit=SERIES_LIMIT + 1 if backend.name == "prometheus" else None)
  except Exception:
    return {**result, "source": backend.name, "error": "Collector metrics query unavailable"}

  # Stable order and identity keep polling from moving charts around. All labels
  # matter: two workers reporting the same metric are distinct time series.
  rows.sort(key=lambda row: (row["metric"].get("metric", ""), json.dumps(row["metric"], sort_keys=True)))
  charts = []
  for row in rows:
    labels = row["metric"]
    points = [[float(at), float(value)] for at, value in row.get("values", []) if math.isfinite(float(value)) and start <= float(at) <= end]
    if not points or not labels.get("metric"):
      continue
    if len(charts) == SERIES_LIMIT:
      result["truncated"] = True
      break
    scope, identity = next(
      ((scope, labels[field]) for scope, field in IDENTITIES if labels.get(field)), ("service", labels.get("service_name", "unknown"))
    )
    charts.append(
      {
        "id": hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()[:20],
        "name": labels["metric"],
        "source": "client" if labels["__name__"] == "openrl_client_metric" else "server",
        "scope": scope,
        "scope_id": identity,
        "phase": labels.get("openrl_phase"),
        "labels": labels,
        "points": points,
      }
    )
  return {**result, "available": True, "source": backend.name, "series": charts}
