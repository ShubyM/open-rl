"""Server gauges use the SDK; standalone observations never become completed work."""

import asyncio
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import numpy as np
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import ExportMetricsServiceRequest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from server import telemetry
from server.dashboard.snapshot import progress
from server.store import InMemoryStateStore


class ServerMetricsTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.reader = InMemoryMetricReader()
    self.provider = MeterProvider(metric_readers=[self.reader])
    self.addCleanup(self.provider.shutdown)
    gauge = self.provider.get_meter("openrl.server").create_gauge("openrl.server.metric")
    self.enterContext(patch.object(telemetry, "gauge", gauge))
    self.enterContext(patch.object(telemetry, "series", set()))
    self.enterContext(patch.object(telemetry, "warned", set()))
    self.store = InMemoryStateStore()
    self.enterContext(patch.object(telemetry, "get_state_store", return_value=self.store))

  def points(self):
    data = self.reader.get_metrics_data()
    return [
      point for resource in data.resource_metrics for scope in resource.scope_metrics for metric in scope.metrics for point in metric.data.data_points
    ]

  async def test_operation_and_standalone_metrics_keep_honest_scopes_and_do_not_fake_progress(self):
    async def step():
      await asyncio.to_thread(telemetry.metrics, {"internal": 1})
      return {"metrics": {"loss": 2}}

    await telemetry.observe_operation({"model_id": "run", "op": "optim_step"}, "trainer", "run", step)
    await asyncio.gather(*list(telemetry.writes))
    before = await self.store.read_samples(telemetry.ops_key("run"))
    run = {"run_id": "run", "runtime_id": "run", "pool": "pool", "created_at": 1}
    state = progress(run, [], before, {}, before[0]["at"] + 1000)
    for value in range(60):
      telemetry.record_metrics({"temperature": value}, model_id="run", session_id="session", phase="background")
    telemetry.record_metrics({"queue": 4}, runtime_id="shared")
    telemetry.record_metrics({"process_load": 0.5})
    async with telemetry.serving("trainer", "shared"):
      with telemetry.record_op("weight_sync"):
        await asyncio.to_thread(telemetry.metrics, {"weight_bytes": 10})
    await asyncio.gather(*list(telemetry.writes))
    self.assertEqual(await self.store.read_samples(telemetry.ops_key("run")), before)
    self.assertEqual(progress(run, [], before, {}, before[0]["at"] + 1000), state)
    self.assertEqual(await self.store.get_value(telemetry.steps_key("run")), "1")
    points = {point.attributes["metric"]: point for point in self.points()}
    self.assertEqual(
      points["internal"].attributes, {"openrl.model.id": "run", "openrl.runtime.id": "run", "openrl.phase": "optim_step", "metric": "internal"}
    )
    self.assertEqual(points["temperature"].value, 59)
    self.assertEqual(points["queue"].attributes, {"openrl.runtime.id": "shared", "metric": "queue"})
    self.assertEqual(points["weight_bytes"].attributes, {"openrl.runtime.id": "shared", "openrl.phase": "weight_sync", "metric": "weight_bytes"})
    self.assertEqual(points["process_load"].attributes, {"metric": "process_load"})

  async def test_invalid_values_and_cardinality_are_bounded_without_losing_updates(self):
    with patch.object(telemetry, "SERIES_LIMIT", 2), self.assertLogs(telemetry.logger) as captured:
      telemetry.record_metrics(
        {"loss": np.float32(2), "nan": float("nan"), "inf": float("inf"), "bool": True, "huge": 10**1000, "": 1, 3: 4}, model_id="a"
      )
      telemetry.record_metrics({"loss": 3}, model_id="b")
      telemetry.record_metrics({"loss": 4, "new": 5}, model_id="c")
      telemetry.record_metrics({"loss": 6}, model_id="a")
      telemetry.record_metrics({"loss": 7}, model_id="d")
    self.assertEqual(len(telemetry.series), 2)
    self.assertEqual({point.attributes["openrl.model.id"]: point.value for point in self.points()}, {"a": 6, "b": 3})
    self.assertEqual(len(captured.output), 1)
    self.assertIn("retained 2, dropped 2", captured.output[0])


class ServerMetricsExportTest(unittest.TestCase):
  def test_explicit_endpoint_exports_on_shutdown_with_resource_identity_and_no_global_tracer_change(self):
    received = []

    class Receiver(BaseHTTPRequestHandler):
      def do_POST(self):
        received.append((self.path, self.headers, ExportMetricsServiceRequest.FromString(self.rfile.read(int(self.headers["Content-Length"])))))
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

      def log_message(self, *_):
        pass

    server = HTTPServer(("127.0.0.1", 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {key: value for key, value in os.environ.items() if not key.startswith("OTEL_")}
    env.update(
      POD_NAME="pod",
      POD_UID="uid",
      NODE_NAME="node",
      NO_PROXY="127.0.0.1",
      OTEL_SERVICE_NAME="test-server",
      OTEL_METRIC_EXPORT_INTERVAL="3600000",
      OTEL_EXPORTER_OTLP_TIMEOUT="1",
      OTEL_EXPORTER_OTLP_HEADERS="x-test=token",
    )
    script = f"""
import os
from opentelemetry import trace
from server import telemetry
tracer = trace.get_tracer_provider()
telemetry.configure_metrics('trainer')
assert telemetry.gauge is None
os.environ['OTEL_EXPORTER_OTLP_METRICS_ENDPOINT'] = 'http://127.0.0.1:{server.server_port}/custom/metrics'
os.environ['OTEL_METRICS_EXPORTER'] = 'none'
telemetry.configure_metrics('trainer')
assert telemetry.gauge is None
del os.environ['OTEL_METRICS_EXPORTER']
telemetry.configure_metrics('trainer')
provider = telemetry.provider
telemetry.configure_metrics('trainer')
assert telemetry.provider is provider
telemetry.record_metrics({{'loss': 1}}, model_id='run')
telemetry.shutdown()
telemetry.shutdown()
assert trace.get_tracer_provider() is tracer
"""
    try:
      subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True, timeout=20)
    finally:
      server.shutdown()
      thread.join()
      server.server_close()
    ((path, headers, request),) = received
    self.assertEqual((path, headers["x-test"]), ("/custom/metrics", "token"))
    (resource,) = request.resource_metrics
    attrs = {item.key: item.value.string_value for item in resource.resource.attributes}
    self.assertEqual(
      {name: attrs[name] for name in ("service.name", "k8s.pod.name", "k8s.pod.uid", "k8s.node.name")},
      {"service.name": "test-server", "k8s.pod.name": "pod", "k8s.pod.uid": "uid", "k8s.node.name": "node"},
    )
    self.assertTrue(attrs["service.instance.id"])
    (gauge,) = resource.scope_metrics[0].metrics
    self.assertEqual((gauge.name, gauge.gauge.data_points[0].as_double), ("openrl.server.metric", 1))
