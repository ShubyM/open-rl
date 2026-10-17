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

  def test_json_payload_is_rendered_when_there_is_no_text(self) -> None:
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
    window = {"since": "2026-09-10T10:00:00Z", "until": "2026-09-10T11:00:00Z"}
    view = "projects/proj/locations/global/buckets/openrl-runs/views/_AllLogs"
    for env, expected in (({}, ["projects/proj"]), ({"OPEN_RL_LOG_VIEWS": view}, [view])):
      with self.subTest(env=env), patch.dict("os.environ", env), patch.object(gke, "request", AsyncMock(return_value={"entries": []})) as request:
        asyncio.run(gke.run_logs("r", [SOURCE], **window))
        self.assertEqual(request.await_args.kwargs["json"]["resourceNames"], expected)

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
      self.assertEqual(gke._refused_until, 700)

  def test_pod_sources_cover_live_pods_and_the_runs_processes(self) -> None:
    run = {"run_id": "run-1", "pods": [{"name": "orw-live", "node": "n1", "role": "sampler", "created_at": "2026-09-10T10:00:00Z"}]}
    lanes = [{"pod": "orw-gone", "node": "n2", "role": "trainer", "start": 1789000000.0, "end": 1789003600.0}]
    sources = {s["pod"]: s for s in gke.pod_sources(run, lanes, "2026-09-10T11:00:00Z")}
    self.assertEqual(set(sources), {"orw-live", "orw-gone"})
    self.assertEqual(sources["orw-live"]["until"], "2026-09-10T11:00:00.000000+00:00")
    self.assertEqual(sources["orw-gone"]["role"], "trainer")
    self.assertEqual(sources["orw-gone"]["until"], gke.timestamp(1789003660.0))

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
