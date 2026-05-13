"""Shared OpenTelemetry setup."""

from __future__ import annotations

import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider: TracerProvider | None = None


def setup_tracing(service_name: str) -> TracerProvider:
  global provider
  if provider is not None:
    return provider

  provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
  trace.set_tracer_provider(provider)

  if os.getenv("ENABLE_GCP_TRACE", "0") != "1":
    print("OpenTelemetry: No exporter configured")
    return provider

  provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
  print(f"OpenTelemetry: Configured GCP CloudTraceSpanExporter for {service_name}")
  return provider


def get_tracer(name: str):
  setup_tracing(name)
  return trace.get_tracer(name)


def instrument_fastapi(app: Any, excluded_urls: str | None = None) -> None:
  FastAPIInstrumentor.instrument_app(app, excluded_urls=excluded_urls)
