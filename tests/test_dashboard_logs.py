import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server import gateway
from server.dashboard import data, logs


class RunLogsTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    patch = mock.patch.dict(os.environ, {"OPEN_RL_LOG_ARCHIVE": self.tmp.name + "/logs.sqlite", "OPEN_RL_DASHBOARD_DEMO": "0"})
    patch.start()
    self.addCleanup(patch.stop)
    self.pod = {
      "name": "trainer-a",
      "uid": "pod-uid",
      "node": "node-a",
      "labels": {"timeslice.io/job-id": "trainer-run-a"},
      "containers": [{"name": "trainer", "restart_count": 1}],
    }
    self.container = self.pod["containers"][0]
    self.client = TestClient(gateway.app)

  def ingest(self, text, run="run-a", previous=False):
    records = logs.parse_records(text, run, self.pod, self.container, previous)
    logs.retain(run, records, [])
    return records

  def test_structure_restarts_and_repeated_lines(self):
    line = '2026-09-09T12:00:00.123456789Z {"level":"error","rank":3,"request_id":"req-7","message":"failed"}'
    parsed = self.ingest(line + "\n" + line)
    self.ingest(line + "\n" + line)  # rereading tails must not duplicate them
    self.ingest(line, previous=True)
    self.assertEqual(len(logs.query("run-a")["records"]), 3)
    self.assertEqual(parsed[0]["severity"], "ERROR")
    self.assertEqual(parsed[0]["rank"], "3")
    self.assertEqual(parsed[0]["pod_uid"], "pod-uid")
    self.assertEqual(parsed[0]["attempt"], 1)
    self.assertEqual(len(logs.query("run-a", attempt=0)["records"]), 1)

  def test_stable_pagination_and_scope(self):
    self.ingest("\n".join(f"2026-09-09T12:00:0{i}Z INFO step {i}" for i in range(5)))
    self.ingest("2026-09-09T12:00:02Z ERROR another run", run="run-ab")
    first = logs.query("run-a", limit=2)
    self.ingest("2026-09-09T12:00:06Z INFO new record")
    second = logs.query("run-a", limit=2, cursor=first["next_cursor"])
    third = logs.query("run-a", limit=2, cursor=second["next_cursor"])
    self.assertEqual([row["message"] for page in (first, second, third) for row in page["records"]], [f"INFO step {i}" for i in range(4, -1, -1)])
    self.assertIsNone(third["next_cursor"])
    with self.assertRaises(ValueError):
      logs.query("run-ab", cursor=first["next_cursor"])
    with self.assertRaises(ValueError):
      logs.query("run-a", q="different", cursor=first["next_cursor"])

  def test_filters_and_retained_output_without_pods(self):
    self.ingest("2026-09-09T12:00:00Z INFO ready\n2026-09-09T12:01:00Z ERROR OOM\nno timestamp")
    records = logs.query(
      "run-a",
      since="2026-09-09T12:00:30Z",
      until="2026-09-09T12:02:00Z",
      q="oom",
      severity="ERROR",
      pod="trainer-a",
      node="node-a",
      container="trainer",
    )["records"]
    self.assertEqual(len(records), 1)
    with mock.patch.object(data, "k8s_snapshot", return_value={"pods": []}):
      response = self.client.get("/api/v1/dashboard/runs/run-a/logs")
    self.assertEqual(response.status_code, 200)
    self.assertEqual(len(response.json()["records"]), 3)
    self.assertFalse(response.json()["coverage"]["history_complete"])

  def test_collection_all_containers_previous_and_partial_failure(self):
    pod = {**self.pod, "containers": [self.container, {"name": "init", "restart_count": 0}]}
    api = mock.Mock()

    def read(name, namespace, **kwargs):
      if kwargs["container"] == "init":
        raise RuntimeError("sensitive error contents")
      return "2026-09-09T12:00:00Z INFO ready"

    api.read_namespaced_pod_log.side_effect = read
    with mock.patch.object(data, "k8s_core_v1", return_value=(api, None)):
      result = logs.collect("run-a", [pod])
    self.assertEqual(len(result["sources"]), 3)
    self.assertEqual(len(logs.query("run-a")["records"]), 2)
    self.assertEqual(result["sources"][-1]["status"], "unavailable")
    self.assertNotIn("sensitive", json.dumps(result))
    self.assertTrue(all(call.kwargs["timestamps"] for call in api.read_namespaced_pod_log.call_args_list))
    self.assertTrue(all(call.kwargs["limit_bytes"] == logs.MAX_BYTES for call in api.read_namespaced_pod_log.call_args_list))

  def test_large_run_source_rotation(self):
    api = mock.Mock()
    api.read_namespaced_pod_log.return_value = "2026-09-09T12:00:00Z INFO ready"
    with mock.patch.object(logs, "MAX_SOURCES", 1), mock.patch.object(data, "k8s_core_v1", return_value=(api, None)):
      first = logs.collect("run-a", [self.pod])
      second = logs.collect("run-a", [self.pod], source_offset=1)
    self.assertEqual(first["sources_omitted"], 1)
    self.assertFalse(first["sources"][0]["previous"])
    self.assertTrue(second["sources"][0]["previous"])
    self.assertEqual(len(logs.query("run-a")["records"]), 2)

  def test_requested_pod_limits_collection_and_exact_run_membership(self):
    same_run = {**self.pod, "name": "trainer-b"}
    other_run = {**self.pod, "name": "trainer-other", "labels": {"timeslice.io/job-id": "trainer-run-ab"}}
    with (
      mock.patch.object(data, "k8s_snapshot", return_value={"available": True, "pods": [self.pod, same_run, other_run]}),
      mock.patch.object(logs, "collect", return_value={"sources": [], "sources_omitted": 0}) as collect,
    ):
      response = self.client.get("/api/v1/dashboard/runs/run-a/logs?pod=trainer-b")
      self.assertEqual(response.status_code, 200)
      self.assertEqual([pod["name"] for pod in collect.call_args.args[1]], ["trainer-b"])
      response = self.client.get("/api/v1/dashboard/runs/run-a/logs?pod=trainer-other")
      self.assertEqual(response.status_code, 200)
      self.assertEqual(collect.call_args.args[1], [])

  def test_api_validation_before_collection(self):
    with mock.patch.object(data, "k8s_snapshot") as snapshot:
      for query in ["since=not-a-date", "since=2026-09-09T12:00:00", "cursor=bad", "since=2026-09-10T00:00:00Z&until=2026-09-09T00:00:00Z"]:
        self.assertEqual(self.client.get("/api/v1/dashboard/runs/run-a/logs?" + query).status_code, 400)
      snapshot.assert_not_called()
    self.assertEqual(self.client.get("/api/v1/dashboard/runs/run-a/logs?limit=1001").status_code, 422)
    self.assertEqual(self.client.get("/api/v1/dashboard/runs/run-a/logs?severity=madeup").status_code, 422)

  def test_large_messages_and_no_fabricated_timestamp(self):
    self.ingest("x" * 5000)
    record = logs.query("run-a")["records"][0]
    self.assertIsNone(record["timestamp"])
    self.assertEqual(record["severity"], "UNKNOWN")
    self.assertEqual(len(record["message"]), logs.MAX_MESSAGE)
    self.assertTrue(record["message_truncated"])
    self.assertEqual(logs.query("run-a", since="2020-01-01T00:00:00Z")["records"], [])

  def test_retention_and_archive_permissions(self):
    records = logs.parse_records("2026-09-09T12:00:00Z INFO old", "run-a", self.pod, self.container, False)
    records[0]["collected_at"] = "2000-01-01T00:00:00+00:00"
    logs.retain("run-a", records, [])
    result = logs.query("run-a")
    self.assertEqual(result["records"], [])
    self.assertIsNotNone(result["coverage"]["last_pruned_at"])
    self.assertEqual(os.stat(logs.archive_path()).st_mode & 0o777, 0o600)

  def test_demo_never_reads_real_cluster(self):
    with mock.patch.dict(os.environ, {"OPEN_RL_DASHBOARD_DEMO": "1"}), mock.patch.object(data, "k8s_core_v1") as api:
      response = self.client.get("/api/v1/dashboard/runs/demo-run-1/logs")
      api.assert_not_called()
      self.assertEqual(response.status_code, 200)
      self.assertTrue(response.json()["demo"])
      self.assertTrue(response.json()["records"])
