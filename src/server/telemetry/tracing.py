"""OpenTelemetry setup."""

import importlib.metadata
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from server.telemetry import metrics


def configure_tracing(service_name: str) -> TracerProvider:
  """Install the global TracerProvider, exporting to each exporter named in OTEL_TRACES_EXPORTER."""
  provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

  exporters = os.environ.get("OTEL_TRACES_EXPORTER", "").split(",")
  for name in exporters:
    name = name.strip()
    if not name or name == "none":
      continue
    eps = importlib.metadata.entry_points(group="opentelemetry_traces_exporter", name=name)
    if not eps:
      raise ValueError(f"Unknown OTEL_TRACES_EXPORTER: {name}")
    (entry,) = eps
    provider.add_span_processor(BatchSpanProcessor(entry.load()()))

  trace.set_tracer_provider(provider)
  metrics.configure(service_name)
  return provider
