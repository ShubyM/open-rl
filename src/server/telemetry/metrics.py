"""Opt-in server gauges, exported by the standard OpenTelemetry SDK."""

import logging
import math
import os
import threading
import uuid
from numbers import Real

from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger(__name__)
SERIES_LIMIT = int(os.getenv("OPEN_RL_SERVER_METRIC_SERIES_LIMIT", "2048"))
_provider: MeterProvider | None = None
_gauge = None
_series: set[tuple] = set()
_lock = threading.Lock()
_warned = False


def numeric_values(values: dict) -> dict[str, float]:
  """Native numbers only, including nonfinite values so operation diagnostics can name them."""
  result = {}
  for name, value in values.items():
    if not isinstance(name, str) or not name or len(name) > 200 or not isinstance(value, Real) or isinstance(value, bool):
      continue
    try:
      result[name] = float(value)
    except OverflowError:
      continue
  return result


def configure(service_name: str) -> None:
  """Configure once when an OTLP endpoint is explicit; ordinary local runs need no collector."""
  global _provider, _gauge
  if os.getenv("OTEL_METRICS_EXPORTER", "").strip().lower() == "none":
    return
  if _gauge is not None or not (os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")):
    return
  resource = Resource.create(
    {
      "service.name": os.getenv("OTEL_SERVICE_NAME", service_name),
      "service.instance.id": uuid.uuid4().hex,
      **{
        key: os.environ[env]
        for key, env in (("k8s.pod.name", "POD_NAME"), ("k8s.pod.uid", "POD_UID"), ("k8s.node.name", "NODE_NAME"))
        if os.getenv(env)
      },
    }
  )
  if not isinstance(metrics.get_meter_provider(), MeterProvider):
    _provider = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    metrics.set_meter_provider(_provider)
  _gauge = metrics.get_meter("openrl.server").create_gauge("openrl.server.metric", description="Latest reported server metric")


def record_metrics(
  values: dict, *, model_id: str | None = None, runtime_id: str | None = None, session_id: str | None = None, phase: str | None = None
) -> None:
  """Record latest values without network I/O. Omitted identities mean process scope, never an inferred run.

  Names and scope attributes must stay stable: step/request IDs are not labels.
  At most SERIES_LIMIT distinct name/scope combinations are accepted per process.
  """
  global _warned
  gauge = _gauge
  if gauge is None:
    return
  attributes = {
    key: value
    for key, value in (("openrl.model.id", model_id), ("openrl.runtime.id", runtime_id), ("session.id", session_id), ("openrl.phase", phase))
    if value is not None
  }
  if any(not isinstance(value, str) or not value or len(value) > 200 for value in attributes.values()):
    return
  dropped = 0
  with _lock:
    for name, value in numeric_values(values).items():
      if not math.isfinite(value):
        continue
      labels = {**attributes, "metric": name}
      identity = tuple(labels.items())
      if identity not in _series and len(_series) >= SERIES_LIMIT:
        dropped += 1
        continue
      _series.add(identity)
      gauge.set(value, labels)
    if dropped and not _warned:
      _warned = True
      logger.warning(
        "Server metric series limit reached: retained %s, dropped %s new series; limit=%s per process", len(_series), dropped, SERIES_LIMIT
      )


def shutdown() -> None:
  """Drain our SDK provider; leave externally configured providers to their owner."""
  global _provider, _gauge
  if _provider is not None:
    _gauge = None
    provider, _provider = _provider, None
    provider.shutdown()
