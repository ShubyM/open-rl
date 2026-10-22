import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from server.dashboard import sources

CONFIG = {"project": "proj", "location": "us-central1", "cluster": "c1", "mode": "auto", "enabled": True, "configured": True}
RUN = {"run_id": "r1", "runtime_id": "Qwen/Qwen3-8B", "pool": "sampler-qwen", "shared_runtime": True}
WINDOW = {"since": "2026-09-10T10:00:00Z", "until": "2026-09-10T11:00:00Z"}
BOUNDS = ("2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00")


def entry(pod="orw-a", at="2026-09-10T10:30:00Z", **extra):
  return {
    "timestamp": at,
    "insertId": "i1",
    "logName": "projects/proj/logs/stdout",
    "severity": "INFO",
    "resource": {"labels": {"pod_name": pod, "container_name": "worker"}},
    "labels": {"k8s-pod/openrl_io/role": "trainer", "compute.googleapis.com/resource_name": "n1"},
    **extra,
  }


class GkeLogHelpersTest(unittest.TestCase):
  def setUp(self) -> None:
    for module, name, value in ((sources, "configuration", lambda: CONFIG), (sources, "namespace", lambda: "openrl-system")):
      patcher = patch.object(module, name, value)
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_filter_takes_the_runs_workers_by_owner_label_its_pool_and_its_client(self) -> None:
    text = sources.logs_filter(RUN, *BOUNDS, "UNKNOWN", "oom")
    self.assertIn('resource.type="k8s_container"', text)
    self.assertIn('resource.labels.cluster_name="c1"', text)
    self.assertIn('resource.labels.namespace_name="openrl-system"', text)
    self.assertIn('labels."k8s-pod/timeslice_io/owner"="qwen-qwen3-8b"', text)
    self.assertIn('labels."k8s-pod/app_kubernetes_io/instance"="sampler-qwen"', text)
    self.assertIn('(labels.openrl_run_id="r1" AND labels.openrl_source="client")', text)
    self.assertIn(f'timestamp>="{BOUNDS[0]}"', text)
    self.assertIn('severity="DEFAULT"', text)
    self.assertIn('textPayload:"oom" OR jsonPayload.message:"oom"', text)
    one_pod = sources.logs_filter(RUN, *BOUNDS, None, "", pod="orw-a")
    self.assertIn('resource.labels.pod_name="orw-a"', one_pod)
    self.assertNotIn("timeslice_io/owner", one_pod)
    self.assertNotIn("timeslice_io/owner", sources.logs_filter(RUN, *BOUNDS, None, "", pod=sources.CLIENT))

  def test_an_entry_takes_its_role_and_node_from_the_pods_labels(self) -> None:
    record = sources.entry_record(entry(textPayload="hello"), "r1")
    self.assertEqual(
      (record["pod"], record["container"], record["role"], record["node"], record["message"]), ("orw-a", "worker", "trainer", "n1", "hello")
    )
    client = sources.entry_record(entry(pod="api-server", labels={"openrl_source": "client", "openrl_run_id": "r1"}, textPayload="step 3"), "r1")
    self.assertEqual((client["pod"], client["role"]), (sources.CLIENT, sources.CLIENT))
    record = sources.entry_record(entry(jsonPayload={"message": "m", "rank": 2}), "r1")
    self.assertEqual(record["rank"], 2)
    self.assertIn('"message": "m"', record["message"])

  def test_stderr_lines_take_the_level_they_print(self) -> None:
    def stderr(text):
      return sources.entry_record(entry(logName="projects/proj/logs/stderr", severity="ERROR", textPayload=text), "r1")["severity"]

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
    self.assertEqual(sources.entry_record(entry(severity="ERROR", textPayload=progress), "r1")["severity"], "ERROR")

  def test_severity_filters_match_relabeled_stderr_lines(self) -> None:
    self.assertIn('(severity="INFO" OR (severity="ERROR" AND logName="projects/proj/logs/stderr"))', sources.logs_filter(RUN, *BOUNDS, "INFO", ""))
    self.assertIn('severity="ERROR"', sources.logs_filter(RUN, *BOUNDS, "ERROR", ""))
    entries = [
      entry(insertId=insert, logName="projects/proj/logs/stderr", severity="ERROR", textPayload=text)
      for insert, text in (("a", "\rLoading weights: 100%|"), ("b", "Traceback (most recent call last):"))
    ]
    with patch.object(sources, "gcp_request", AsyncMock(return_value={"entries": entries})):
      result = asyncio.run(sources.run_logs(RUN, [], severity="ERROR", span=sources.window(**WINDOW)))
    self.assertEqual([record["message"] for record in result["records"]], ["Traceback (most recent call last):"])

  def test_logs_come_from_the_configured_views_or_else_the_project(self) -> None:
    view = "projects/proj/locations/global/buckets/openrl-runs/views/_AllLogs"
    for setting, expected in (("", ["projects/proj"]), (view, [view])):
      with self.subTest(setting=setting), patch.dict("os.environ", {"OPEN_RL_LOG_VIEWS": setting}):
        self.assertEqual(sources.log_views(), expected)

  def test_the_cursor_is_cloud_loggings_page_token_sent_with_the_same_query(self) -> None:
    pages = [{"nextPageToken": "provider-page-2"}, {"entries": [entry(textPayload="older")]}]
    with patch.object(sources, "gcp_request", AsyncMock(side_effect=pages)) as request:
      first = asyncio.run(sources.run_logs(RUN, [], span=sources.window(**WINDOW)))
      second = asyncio.run(sources.run_logs(RUN, [], cursor=first["next_cursor"], span=sources.window(**WINDOW)))
    before, after = [call.kwargs["json"] for call in request.await_args_list]
    self.assertEqual((first["next_cursor"], after), ("provider-page-2", {**before, "pageToken": "provider-page-2"}))
    self.assertEqual((second["records"][0]["message"], second["next_cursor"], second["shared_runtime"]), ("older", None, True))
    rejected = httpx.Response(400, request=httpx.Request("POST", "https://logging.googleapis.com"))
    error = httpx.HTTPStatusError("bad token", request=rejected.request, response=rejected)
    with patch.object(sources, "gcp_request", AsyncMock(side_effect=error)), self.assertRaisesRegex(ValueError, "Invalid or expired"):
      asyncio.run(sources.run_logs(RUN, [], cursor="stale", span=sources.window(**WINDOW)))

  def test_refused_credentials_back_off_without_repeating_the_request(self) -> None:
    response = httpx.Response(403, request=httpx.Request("POST", "https://logging.googleapis.com"))
    error = httpx.HTTPStatusError("denied", request=response.request, response=response)
    with (
      patch.object(sources, "refused_until", 0),
      patch.object(sources.time, "monotonic", return_value=100),
      patch.object(sources, "gcp_request", AsyncMock(side_effect=error)) as request,
    ):
      first = asyncio.run(sources.run_logs(RUN, [], span=sources.window(**WINDOW)))
      second = asyncio.run(sources.run_logs(RUN, [], span=sources.window(**WINDOW)))
      self.assertEqual((first["error"], second["error"]), (sources.REFUSED, sources.REFUSED))
      self.assertEqual(request.await_count, 1)
