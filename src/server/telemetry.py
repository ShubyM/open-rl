"""Process-owned tracing setup. Importing this module has no global side effects."""

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_provider: TracerProvider | None = None


def initialize_tracing(service_name: str) -> None:
  global _provider
  if _provider is not None or isinstance(trace.get_tracer_provider(), TracerProvider):
    return
  provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
  if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
  trace.set_tracer_provider(provider)
  _provider = provider


def shutdown_tracing() -> None:
  if _provider is not None:
    _provider.shutdown()
