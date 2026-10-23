"""Cookbook metric logging through standard OpenTelemetry HTTP exporters."""

import logging
import math
import threading
import time
import uuid
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

MAX_EVENT_METRICS = 500
MAX_METRIC_NAME = 200


class OTelLogger(Logger):
  """Append to ``ml_log.setup_logging(...).loggers`` to export metric observations.

  Session identifies the client; model is optional. Each call produces a log
  event and updates gauges. OTEL_EXPORTER_OTLP_* configures HTTP/protobuf export;
  OTEL_METRIC_EXPORT_INTERVAL controls gauge export. Existing OTel globals and
  cookbook file logging are untouched. Call the cookbook logger's close() to
  drain the SDK's best-effort buffers. max_series caps distinct metric/phase and
  step/phase gauge series; log events retain up to 500 numeric values per call.
  """

  def __init__(self, session_id: str, model_id: str | None = None, *, max_series: int = 500):
    if not session_id:
      raise ValueError("session_id is required")
    if not isinstance(max_series, int) or isinstance(max_series, bool) or max_series < 1:
      raise ValueError("max_series must be a positive integer")
    self._attributes = {"session.id": session_id}
    if model_id is not None:
      self._attributes["openrl.model.id"] = model_id
    self._closed = False
    self._max_series = max_series
    self._series: set[tuple[str | None, str | None]] = set()
    self._warned: set[str] = set()
    self._limit_lock = threading.Lock()
    resource = Resource.create({"service.instance.id": uuid.uuid4().hex})
    self._logs = LoggerProvider(resource=resource)
    self._logs.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    self._logger = self._logs.get_logger("openrl.cookbook")
    self._metrics = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    meter = self._metrics.get_meter("openrl.cookbook")
    self._values = meter.create_gauge("openrl.client.metric", description="Latest reported client metric")
    self._step = meter.create_gauge("openrl.client.step", description="Latest reported client step")

  def log_hparams(self, config: Any) -> None:
    """Configuration remains with the cookbook's existing loggers."""

  def admit(self, name: str | None, phase: str | None) -> bool:
    # None names the step gauge, independently of any user metric called step.
    key = (name, phase)
    with self._limit_lock:
      if key in self._series:
        return True
      if len(self._series) >= self._max_series:
        return False
      self._series.add(key)
      return True

  def warn_once(self, limit: str, message: str, *args) -> None:
    with self._limit_lock:
      if limit in self._warned:
        return
      self._warned.add(limit)
    logging.getLogger(__name__).warning(message, *args)

  def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
    if self._closed:
      return
    # Copy into native finite numbers before handing the event to a batch thread.
    values, dropped = {}, 0
    for name, value in metrics.items():
      if not isinstance(value, Real) or isinstance(value, bool):
        continue
      try:
        value = float(value)
      except OverflowError:
        continue
      if not math.isfinite(value):
        continue
      if not isinstance(name, str) or not name or len(name) > MAX_METRIC_NAME or len(values) >= MAX_EVENT_METRICS:
        dropped += 1
      else:
        values[name] = value

    attributes = self._attributes.copy()
    phase = metrics.get("phase") if isinstance(metrics.get("phase"), str) else None
    if phase is not None:
      attributes["openrl.phase"] = phase
    body = {"metrics": values}
    skipped = 0
    if step is not None:
      body["step"] = step
      if self.admit(None, phase):
        self._step.set(step, attributes)
      else:
        skipped += 1
    for name, value in values.items():
      if self.admit(name, phase):
        self._values.set(value, {**attributes, "metric": name})
      else:
        skipped += 1
    if dropped:
      body["dropped_metrics"] = dropped
      self.warn_once(
        "event",
        "OTel event limits reached (%d values, %d-character names); dropped %d values in this call.",
        MAX_EVENT_METRICS,
        MAX_METRIC_NAME,
        dropped,
      )
    if skipped:
      body["dropped_gauge_updates"] = skipped
      self.warn_once(
        "series",
        "OTel gauge series limit (%d) reached; skipped %d updates in this call. Observations remain in log events.",
        self._max_series,
        skipped,
      )
    self._logger.emit(
      timestamp=time.time_ns(), body=body, attributes=attributes, event_name="openrl.training.metrics", severity_number=SeverityNumber.INFO
    )

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
