"""Exercise cookbook logging against a local OTLP HTTP/protobuf receiver."""

import os
import threading
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from common.otel_logger import OTelLogger
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from tinker_cookbook.utils import ml_log


def value_of(value):
  kind = value.WhichOneof("value")
  return attributes(value.kvlist_value.values) if kind == "kvlist_value" else getattr(value, kind)


def attributes(items):
  return {item.key: value_of(item.value) for item in items}


class OTelLoggerTest(unittest.TestCase):
  def setUp(self):
    self.requests = []
    requests = self.requests

    class Receiver(BaseHTTPRequestHandler):
      def do_POST(self):
        requests.append((self.path, self.headers, self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.send_header("Content-Length", "0")
        self.end_headers()

      def log_message(self, *_):
        pass

    server = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def stop():
      server.shutdown()
      thread.join()
      server.server_close()

    self.addCleanup(stop)
    self.endpoint = f"http://127.0.0.1:{server.server_port}"
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    env.update(
      OTEL_EXPORTER_OTLP_ENDPOINT=self.endpoint,
      OTEL_EXPORTER_OTLP_HEADERS="x-test=shared%20token",
      OTEL_METRIC_EXPORT_INTERVAL="3600000",
      OTEL_BLRP_SCHEDULE_DELAY="3600000",
      NO_PROXY="127.0.0.1",
    )
    self.enterContext(patch.dict(os.environ, env, clear=True))

  def logs(self):
    return [
      record
      for path, _, body in self.requests
      if path.endswith("/logs")
      for resource in ExportLogsServiceRequest.FromString(body).resource_logs
      for scope in resource.scope_logs
      for record in scope.log_records
    ]

  def gauges(self):
    return [
      (metric.name, attributes(point.attributes), getattr(point, point.WhichOneof("value")))
      for path, _, body in self.requests
      if path.endswith("/metrics")
      for resource in ExportMetricsServiceRequest.FromString(body).resource_metrics
      for scope in resource.scope_metrics
      for metric in scope.metrics
      for point in metric.gauge.data_points
    ]

  def test_cookbook_files_and_distinct_observations_survive_sync_and_shutdown(self):
    with TemporaryDirectory() as directory, closing(ml_log.setup_logging(directory, do_configure_logging_module=False)) as logger:
      store = logger.store
      logger.loggers.append(OTelLogger("session-a", "model-a"))
      self.assertIs(logger.store, store)
      metric = "env/all/reward/total"
      original = {metric: 1.5, "phase": "train"}
      logger.log_metrics(original, step=7)
      original.update({metric: 999, "phase": "mutated"})
      logger.log_metrics({metric: 2.5, "phase": "train"}, step=7)
      logger.log_metrics({metric: 4.5, "phase": "eval"}, step=7)
      logger.sync()
      self.assertEqual([value_of(record.body) for record in self.logs()], [{"metrics": {metric: value}, "step": 7} for value in (1.5, 2.5, 4.5)])
      logger.log_metrics({metric: 5.5, "phase": "train"}, step=8)
      self.assertEqual(
        store.read_metrics(),
        [
          {metric: value, "phase": phase, "step": step}
          for value, phase, step in ((1.5, "train", 7), (2.5, "train", 7), (4.5, "eval", 7), (5.5, "train", 8))
        ],
      )

    records = self.logs()
    self.assertEqual(len(records), 4)
    self.assertEqual(value_of(records[-1].body), {"metrics": {metric: 5.5}, "step": 8})
    for record, phase in zip(records, ("train", "train", "eval", "train"), strict=True):
      self.assertEqual(record.event_name, "openrl.training.metrics")
      self.assertEqual(attributes(record.attributes), {"session.id": "session-a", "openrl.model.id": "model-a", "openrl.phase": phase})
    latest = {}
    for name, attrs, value in self.gauges():
      phase = attrs["openrl.phase"]
      self.assertEqual(
        attrs,
        {
          "session.id": "session-a",
          "openrl.model.id": "model-a",
          "openrl.phase": phase,
          **({"metric": metric} if name == "openrl.client.metric" else {}),
        },
      )
      latest[name, phase] = value
    self.assertEqual(
      latest,
      {
        ("openrl.client.metric", "train"): 5.5,
        ("openrl.client.metric", "eval"): 4.5,
        ("openrl.client.step", "train"): 8,
        ("openrl.client.step", "eval"): 7,
      },
    )
    self.assertEqual({path for path, _, _ in self.requests}, {"/v1/logs", "/v1/metrics"})
    self.assertTrue(all(headers["x-test"] == "shared token" for _, headers, _ in self.requests))

  def test_optional_identity_and_step_filter_invalid_values_and_honor_signal_configuration(self):
    with (
      patch.dict(
        os.environ,
        {
          "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": self.endpoint + "/custom/logs",
          "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": self.endpoint + "/custom/metrics",
          "OTEL_EXPORTER_OTLP_LOGS_HEADERS": "x-test=logs%20token",
          "OTEL_EXPORTER_OTLP_METRICS_HEADERS": "x-test=metrics%20token",
        },
      ),
      closing(OTelLogger("session-b")) as logger,
    ):
      logger.log_hparams({"secret": "never export configuration"})
      logger.log_metrics(
        {
          "loss/train": np.float32(2),
          "nan": float("nan"),
          "infinity": float("inf"),
          "negative_infinity": -float("inf"),
          "overflow": 10**1000,
          "flag": True,
          "text": "2",
          "nested": {"x": 1},
          "none": None,
          "": 3,
          123: 4,
        }
      )
    (record,) = self.logs()
    self.assertEqual(value_of(record.body), {"metrics": {"loss/train": 2}})
    self.assertEqual(attributes(record.attributes), {"session.id": "session-b"})
    self.assertEqual(self.gauges(), [("openrl.client.metric", {"session.id": "session-b", "metric": "loss/train"}, 2)])
    self.assertEqual(
      {(path, headers["x-test"]) for path, headers, _ in self.requests},
      {
        ("/custom/logs", "logs token"),
        ("/custom/metrics", "metrics token"),
      },
    )
