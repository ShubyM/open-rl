import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from server.telemetry import gke

CONFIG = {"project": "proj", "location": "us-central1", "cluster": "c1", "mode": "auto", "enabled": True, "configured": True}
SOURCE = {"pod": "orw-a", "node": "n1", "role": "trainer", "created_at": "2026-09-10T10:00:00+00:00", "until": "2026-09-10T11:00:00+00:00"}


def entry(pod="orw-a", at="2026-09-10T10:30:00Z", **extra):
  return {
    "timestamp": at,
    "insertId": "i1",
    "logName": "projects/proj/logs/stdout",
    "severity": "INFO",
    "resource": {"labels": {"pod_name": pod, "container_name": "worker"}},
    **extra,
  }


class GkeLogHelpersTest(unittest.TestCase):
  def setUp(self) -> None:
    for module, name, value in ((gke, "configuration", lambda: CONFIG), (gke.kubernetes, "namespace", lambda: "openrl-system")):
      patcher = patch.object(module, name, value)
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_filter_scopes_to_cluster_sources_window_and_search(self) -> None:
    text = gke.logs_filter([SOURCE], "2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00", "UNKNOWN", "oom")
    self.assertIn('resource.type="k8s_container"', text)
    self.assertIn('resource.labels.cluster_name="c1"', text)
    self.assertIn('resource.labels.namespace_name="openrl-system"', text)
    self.assertIn('resource.labels.pod_name="orw-a"', text)
    self.assertIn('severity="DEFAULT"', text)
    self.assertIn('textPayload:"oom" OR jsonPayload.message:"oom"', text)

  def test_entry_becomes_a_record_only_inside_a_source_lifetime(self) -> None:
    record = gke.entry_record(entry(textPayload="hello"), [SOURCE], "run-1")
    self.assertEqual((record["pod"], record["container"], record["role"], record["message"]), ("orw-a", "worker", "trainer", "hello"))
    self.assertIsNone(gke.entry_record(entry(at="2026-09-10T12:00:00Z", textPayload="late"), [SOURCE], "run-1"))
    self.assertIsNone(gke.entry_record(entry(pod="someone-else", textPayload="x"), [SOURCE], "run-1"))

    record = gke.entry_record(entry(jsonPayload={"message": "m", "rank": 2}), [SOURCE], "run-1")
    self.assertEqual(record["rank"], 2)
    self.assertIn('"message": "m"', record["message"])

  def test_stderr_lines_take_the_level_they_print(self) -> None:
    def stderr(text):
      return gke.entry_record(entry(logName="projects/proj/logs/stderr", severity="ERROR", textPayload=text), [SOURCE], "run-1")["severity"]

    for text, severity in (
      ("\rLoading weights:   0%|          | 0/427 [00:00<?, ?it/s]", "INFO"),
      ("(APIServer pid=54) INFO:     Started server process [54]", "INFO"),
      ("(EngineCore pid=825) 2026-10-14 18:44:50,632 - INFO - autotuner.py:972 - flashinfer.jit: [Autotuner]", "INFO"),
      ("/app/.venv/lib/python3.12/site-packages/peft/utils/save_and_load.py:356: UserWarning: Setting", "WARNING"),
      ("  warnings.warn(msg)", "WARNING"),
      ("Traceback (most recent call last):", "ERROR"),
      ("torch.OutOfMemoryError: CUDA out of memory.", "ERROR"),
    ):
      self.assertEqual(stderr(text), severity, text)
    progress = "\rFetching 4 files: 100%"
    self.assertEqual(gke.entry_record(entry(severity="ERROR", textPayload=progress), [SOURCE], "run-1")["severity"], "ERROR")

  def test_severity_filters_match_relabeled_stderr_lines(self) -> None:
    window = ("2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00")
    self.assertIn('(severity="INFO" OR (severity="ERROR" AND logName="projects/proj/logs/stderr"))', gke.logs_filter([SOURCE], *window, "INFO", ""))
    self.assertIn('severity="ERROR"', gke.logs_filter([SOURCE], *window, "ERROR", ""))
    entries = [
      entry(insertId=insert, logName="projects/proj/logs/stderr", severity="ERROR", textPayload=text)
      for insert, text in (("a", "\rLoading weights: 100%|"), ("b", "Traceback (most recent call last):"))
    ]
    with patch.object(gke, "request", AsyncMock(return_value={"entries": entries})):
      result = asyncio.run(gke.run_logs("r", [SOURCE], since="2026-09-10T10:00:00Z", until="2026-09-10T11:00:00Z", severity="ERROR"))
    self.assertEqual([record["message"] for record in result["records"]], ["Traceback (most recent call last):"])

  def test_logs_come_from_the_configured_views_or_else_the_project(self) -> None:
    view = "projects/proj/locations/global/buckets/openrl-runs/views/_AllLogs"
    for setting, expected in (("", ["projects/proj"]), (view, [view])):
      with self.subTest(setting=setting), patch.dict("os.environ", {"OPEN_RL_LOG_VIEWS": setting}):
        self.assertEqual(gke.log_views(), expected)

  def test_paging_preserves_the_query_and_lifetimes_after_inventory_changes(self) -> None:
    window = {"since": "2026-09-10T10:00:00Z", "until": "2026-09-10T11:00:00Z"}
    with patch.object(
      gke, "request", AsyncMock(side_effect=[{"nextPageToken": "provider-page-2"}, {"entries": [entry(textPayload="older")]}])
    ) as request:
      first = asyncio.run(gke.run_logs("r", [SOURCE], shared_runtime=True, **window))
      second = asyncio.run(gke.run_logs("r", [], cursor=first["next_cursor"]))
    before, after = [call.kwargs["json"] for call in request.await_args_list]
    self.assertEqual(after, {**before, "pageToken": "provider-page-2"})
    self.assertEqual(second["records"][0]["message"], "older")
    self.assertEqual(second["sources"], [SOURCE])
    self.assertEqual((second["since"], second["until"]), (first["since"], first["until"]))
    self.assertTrue(second["shared_runtime"])

  def test_cursor_rejects_tampering_expiry_and_changed_selection_before_querying(self) -> None:
    window = {"since": "2026-09-10T10:00:00Z", "until": "2026-09-10T11:00:00Z"}
    with patch.object(gke, "request", AsyncMock(return_value={"nextPageToken": "provider-page"})) as request:
      cursor = asyncio.run(gke.run_logs("r", [SOURCE], **window))["next_cursor"]
      request.reset_mock()
      for change in ({"run_id": "other"}, {"q": "different"}, {"limit": 50}, {"until": "2026-09-10T11:01:00Z"}, {"cursor": "bad" + cursor}):
        with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Invalid or expired"):
          asyncio.run(gke.run_logs(**{"run_id": "r", "sources": [SOURCE], "cursor": cursor, **window, **change}))
      with patch.object(gke.time, "time", return_value=10**12), self.assertRaisesRegex(ValueError, "Invalid or expired"):
        asyncio.run(gke.run_logs("r", [SOURCE], cursor=cursor, **window))
      request.assert_not_awaited()

  def test_cursor_has_bounded_encoded_and_decompressed_sizes(self) -> None:
    with self.assertRaisesRegex(ValueError, "too large"):
      gke.page_cursor({"value": "x" * gke.MAX_PLAN_BYTES})
    with patch.object(gke, "MAX_CURSOR", 10), self.assertRaisesRegex(ValueError, "too large"):
      gke.page_cursor({"value": "small"})
    cursor = gke.page_cursor({"value": "x" * 1000})
    with patch.object(gke, "MAX_PLAN_BYTES", 100), self.assertRaisesRegex(ValueError, "Invalid or expired"):
      gke.page_plan(cursor, [], None, None)

  def test_refused_credentials_back_off_without_repeating_the_request(self) -> None:
    response = httpx.Response(403, request=httpx.Request("POST", "https://logging.googleapis.com"))
    error = httpx.HTTPStatusError("denied", request=response.request, response=response)
    with (
      patch.object(gke, "_refused_until", 0),
      patch.object(gke.time, "monotonic", return_value=100),
      patch.object(gke, "request", AsyncMock(side_effect=error)) as request,
    ):
      first = asyncio.run(gke.run_logs("r", [SOURCE]))
      second = asyncio.run(gke.run_logs("r", [SOURCE]))
      self.assertEqual((first["error"], second["error"]), (gke.REFUSED, gke.REFUSED))
      self.assertEqual(request.await_count, 1)

  def test_same_name_pod_replacements_keep_separate_log_lifetimes(self) -> None:
    lanes = [
      {"pod": "worker", "pod_uid": "new", "node": "n2", "start": 300, "end": 400},
      {"pod": "worker", "pod_uid": "old", "node": "n1", "start": 100, "end": 150},
    ]
    run = {"run_id": "r1", "pods": []}
    sources = gke.pod_sources(run, lanes, gke.timestamp(500))
    self.assertEqual(len(sources), 2)
    for at, node in ((120, "n1"), (350, "n2")):
      self.assertEqual(gke.entry_record(entry("worker", gke.timestamp(at)), sources, "r1")["node"], node)
    self.assertIsNone(gke.entry_record(entry("worker", gke.timestamp(250)), sources, "r1"))
    run["pods"] = [{"name": "worker", "uid": "new", "node": "n2", "created_at": gke.timestamp(290)}]
    live = gke.pod_sources(run, lanes, gke.timestamp(500))
    self.assertEqual(
      [(s["node"], s["created_at"], s["until"]) for s in live],
      [("n2", gke.timestamp(290), gke.timestamp(500)), ("n1", gke.timestamp(100), gke.timestamp(210))],
    )
