"""Exercise cookbook logging against a local OTLP HTTP/protobuf receiver."""

import os
import threading
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
  import numpy as np
  from common.otel_logger import OTelLogger
  from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceRequest
  from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
  from tinker_cookbook.utils import ml_log
except ImportError:  # the server environment; make test-client-telemetry runs this in the examples one
  OTelLogger = None


def value_of(value):
  kind = value.WhichOneof("value")
  return attributes(value.kvlist_value.values) if kind == "kvlist_value" else getattr(value, kind)


def attributes(items):
  return {item.key: value_of(item.value) for item in items}


@unittest.skipIf(OTelLogger is None, "needs the examples environment: make test-client-telemetry")
class OTelLoggerTest(unittest.TestCase):
  def setUp(self):
    self.requests = requests = []

    class Receiver(BaseHTTPRequestHandler):
      def do_POST(self):
        requests.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

      def log_message(self, *_):
        pass

    server = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    self.addCleanup(lambda: (server.shutdown(), thread.join(), server.server_close()))
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    env.update(
      OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{server.server_port}",
      OTEL_METRIC_EXPORT_INTERVAL="3600000",
      OTEL_BLRP_SCHEDULE_DELAY="3600000",
      NO_PROXY="127.0.0.1",
    )
    self.enterContext(patch.dict(os.environ, env, clear=True))

  def events(self):
    """Each exported log event as (body, attributes)."""
    return [
      (value_of(record.body), attributes(record.attributes))
      for path, body in self.requests
      if path.endswith("/logs")
      for resource in ExportLogsServiceRequest.FromString(body).resource_logs
      for scope in resource.scope_logs
      for record in scope.log_records
    ]

  def gauges(self, *labels):
    """The latest exported value of each gauge series, by its name and the given labels."""
    found = {}
    for path, body in self.requests:
      for resource in ExportMetricsServiceRequest.FromString(body).resource_metrics if path.endswith("/metrics") else []:
        for metric, point in ((m, p) for scope in resource.scope_metrics for m in scope.metrics for p in m.gauge.data_points):
          values = attributes(point.attributes)
          found[(metric.name, *(values.get(label) for label in labels))] = getattr(point, point.WhichOneof("value"))
    return found

  def test_every_call_is_one_event_beside_the_cookbook_files_and_gauges_keep_the_latest(self):
    metric = "env/all/reward/total"
    with TemporaryDirectory() as directory, closing(ml_log.setup_logging(directory, do_configure_logging_module=False)) as logger:
      logger.loggers.append(OTelLogger("session-a", "model-a"))
      original = {metric: 1.5, "phase": "train"}
      logger.log_metrics(original, step=7)
      original.update({metric: 999, "phase": "mutated"})  # the event keeps what was logged
      logger.log_metrics({metric: 4.5, "phase": "eval"}, step=7)
      logger.log_metrics({metric: 5.5, "phase": "train"}, step=8)
      self.assertEqual([row[metric] for row in logger.store.read_metrics()], [1.5, 4.5, 5.5])

    expected = [
      ({"metrics": {metric: value}, "step": step}, phase) for value, step, phase in ((1.5, 7, "train"), (4.5, 7, "eval"), (5.5, 8, "train"))
    ]
    self.assertEqual(
      self.events(),
      [(body, {"session.id": "session-a", "openrl.model.id": "model-a", "openrl.phase": phase}) for body, phase in expected],
    )
    # The server reads the gauges by session and model when a run has no forwarded events.
    self.assertEqual(
      self.gauges("metric", "openrl.phase", "session.id", "openrl.model.id"),
      {
        ("openrl.client.metric", metric, "train", "session-a", "model-a"): 5.5,
        ("openrl.client.metric", metric, "eval", "session-a", "model-a"): 4.5,
        ("openrl.client.step", None, "train", "session-a", "model-a"): 8,
        ("openrl.client.step", None, "eval", "session-a", "model-a"): 7,
      },
    )

  def test_only_finite_numbers_are_exported_within_limits_that_are_reported_once(self):
    invalid = {"nan": float("nan"), "infinity": float("inf"), "overflow": 10**1000, "flag": True, "text": "2", "nested": {"x": 1}, "none": None}
    with self.assertLogs("common.otel_logger", level="WARNING") as warnings, closing(OTelLogger("session-a", max_series=3)) as logger:
      logger.log_hparams({"secret": "never export configuration"})
      logger.log_metrics({"phase": "train", "loss": np.float32(1), "reward": 2, **invalid}, step=7)
      logger.log_metrics({"phase": "eval", "loss": 3}, step=7)
      logger.log_metrics({"phase": "train", **{f"m{i}": i for i in range(501)}, "x" * 201: 1, "": 2, 123: 3}, step=8)
      logger.log_metrics({"phase": "train", "loss": 6, "x" * 201: 1}, step=9)

    events = self.events()
    self.assertEqual(events[0], ({"metrics": {"loss": 1, "reward": 2}, "step": 7}, {"session.id": "session-a", "openrl.phase": "train"}))
    bodies = [body for body, _ in events]
    self.assertEqual(len(bodies), 4)
    self.assertEqual(bodies[1], {"metrics": {"loss": 3}, "step": 7, "dropped_gauge_updates": 2})
    self.assertEqual(bodies[2], {"metrics": {f"m{i}": i for i in range(500)}, "step": 8, "dropped_metrics": 4, "dropped_gauge_updates": 500})
    self.assertEqual(bodies[3], {"metrics": {"loss": 6}, "step": 9, "dropped_metrics": 1})
    self.assertEqual(
      self.gauges("metric", "openrl.phase"),
      {("openrl.client.step", None, "train"): 9, ("openrl.client.metric", "loss", "train"): 6, ("openrl.client.metric", "reward", "train"): 2},
    )
    self.assertEqual(len(warnings.output), 2)
