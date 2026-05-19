"""Shared OpenTelemetry setup for autoresearch clients."""

import os
from collections.abc import Callable

from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


def configure_telemetry(service_name: str) -> tuple[trace.Tracer, Callable[[], None]]:
  provider = TracerProvider()
  trace.set_tracer_provider(provider)

  if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
    provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
  elif os.getenv("ENABLE_CONSOLE_TRACE", "0") == "1":
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    print("OpenTelemetry: Configured ConsoleSpanExporter")

  HTTPXClientInstrumentor().instrument()
  print("OpenTelemetry: Attached HTTPXClientInstrumentor for context propagation")

  return trace.get_tracer(service_name), provider.shutdown
