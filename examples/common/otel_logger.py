"""Cookbook metric logging through standard OpenTelemetry HTTP exporters."""

import math
import time
from numbers import Real
from typing import Any

from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from tinker_cookbook.utils.ml_log import Logger


class OTelLogger(Logger):
  """Append to ``ml_log.setup_logging(...).loggers`` to export metric observations.

  Session identifies the client; model is optional. Each call produces a log
  event and updates gauges. OTEL_EXPORTER_OTLP_* configures HTTP/protobuf export;
  OTEL_METRIC_EXPORT_INTERVAL controls gauge export. Existing OTel globals and
  cookbook file logging are untouched. Call the cookbook logger's close() to
  drain the SDK's best-effort buffers.
  """

  def __init__(self, session_id: str, model_id: str | None = None):
    if not session_id:
      raise ValueError("session_id is required")
    self._attributes = {"session.id": session_id}
    if model_id is not None:
      self._attributes["openrl.model.id"] = model_id
    self._closed = False
    resource = Resource.create()
    self._logs = LoggerProvider(resource=resource)
    self._logs.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    self._logger = self._logs.get_logger("openrl.cookbook")
    self._metrics = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    meter = self._metrics.get_meter("openrl.cookbook")
    self._values = meter.create_gauge("openrl.client.metric", description="Latest reported client metric")
    self._step = meter.create_gauge("openrl.client.step", description="Latest reported client step")

  def log_hparams(self, config: Any) -> None:
    """Configuration remains with the cookbook's existing loggers."""

  def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
    if self._closed:
      return
    # Copy into native finite numbers before handing the event to a batch thread.
    values = {}
    for name, value in metrics.items():
      if not isinstance(name, str) or not name or not isinstance(value, Real) or isinstance(value, bool):
        continue
      try:
        value = float(value)
      except OverflowError:
        continue
      if math.isfinite(value):
        values[name] = value

    attributes = self._attributes.copy()
    if isinstance(phase := metrics.get("phase"), str):
      attributes["openrl.phase"] = phase
    body = {"metrics": values}
    if step is not None:
      body["step"] = step
      self._step.set(step, attributes)
    self._logger.emit(
      timestamp=time.time_ns(), body=body, attributes=attributes, event_name="openrl.training.metrics", severity_number=SeverityNumber.INFO
    )
    for name, value in values.items():
      self._values.set(value, {**attributes, "metric": name})

  def sync(self) -> None:
    if not self._closed:
      try:
        self._metrics.force_flush()
      finally:
        self._logs.force_flush()

  def close(self) -> None:
    if not self._closed:
      self._closed = True
      try:
        self._metrics.shutdown()
      finally:
        self._logs.shutdown()
