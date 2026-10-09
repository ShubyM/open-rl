"""export_session_trace: a session's recorded operations, as a trace Perfetto opens."""

import asyncio
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import api_server
from server.store import InMemoryStateStore, InMemoryStore
from server.telemetry import ops
from server.telemetry.trace_export import chrome_trace, lanes


def op(start: float, end: float, operation: str = "sample", role: str = "sampler", **extra) -> dict:
  return {"started_at": start, "at": end, "operation": operation, "role": role, "status": "succeeded", **extra}


class LaneTest(unittest.TestCase):
  def test_overlapping_operations_get_their_own_lane_and_lanes_are_reused(self) -> None:
    placed = lanes([op(0, 10), op(2, 5), op(6, 8), op(11, 12)])
    self.assertEqual([lane for lane, _ in placed], [0, 1, 1, 0])

  def test_each_run_and_role_is_its_own_process(self) -> None:
    trace = chrome_trace({"run-a": [op(1, 2, "forward_backward", "trainer"), op(1, 3)], "run-b": [op(4, 5)]})
    names = [e["args"]["name"] for e in trace["traceEvents"] if e["ph"] == "M"]
    self.assertEqual(names, ["sampler · run-a", "trainer · run-a", "sampler · run-b"])
    [slice_] = [e for e in trace["traceEvents"] if e.get("name") == "forward_backward"]
    self.assertEqual((slice_["ts"], slice_["dur"]), (1e6, 1e6))


class TraceExportApiTest(unittest.TestCase):
  def setUp(self) -> None:
    self.store, self.state = InMemoryStore(), InMemoryStateStore()
    self.enterContext(patch.object(api_server, "store", self.store))
    self.enterContext(patch.object(api_server, "state", self.state))
    self.enterContext(patch.object(api_server, "worker_manager", None))
    self.client = TestClient(api_server.app)

  def test_a_session_exports_its_runs_operations(self) -> None:
    asyncio.run(api_server.bind_session("sess-1", "run-a"))
    asyncio.run(self.store.append_sample(ops.key("run-a"), op(1, 2, "optim_step", "trainer", request_id="r1"), 10))
    asyncio.run(self.store.append_sample(ops.key("run-a"), op(1, 4, request_id="sampled:set:x#1"), 10))
    exported = self.client.get("/api/v1/sessions/sess-1/trace_export").json()
    self.assertEqual(exported["status"], "ready")
    trace = self.client.get(exported["url"].removeprefix("http://testserver")).json()
    slices = {e["name"]: e for e in trace["traceEvents"] if e["ph"] == "X"}
    self.assertEqual(set(slices), {"optim_step", "sample"})
    self.assertEqual(slices["sample"]["args"]["request_id"], "sampled:set:x#1")

  def test_an_unknown_session_fails_and_an_old_url_expires(self) -> None:
    self.assertEqual(self.client.get("/api/v1/sessions/nobody/trace_export").json()["status"], "failed")
    self.assertEqual(self.client.get("/api/v1/trace_files/0000").status_code, 404)


if __name__ == "__main__":
  unittest.main()
