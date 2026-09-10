import unittest
from unittest.mock import patch

from server.dashboard import gke

CONFIG = {"project": "proj", "location": "us-central1", "cluster": "c1", "mode": "auto", "enabled": True, "configured": True, "discovery": "explicit"}
SOURCE = {
  "pod": "orw-a",
  "pod_uid": "u1",
  "container": "worker",
  "node": "n1",
  "role": "trainer",
  "created_at": "2026-09-10T10:00:00Z",
  "until": "2026-09-10T11:00:00Z",
}


def entry(pod="orw-a", container="worker", at="2026-09-10T10:30:00Z", **extra):
  return {
    "timestamp": at,
    "insertId": "i1",
    "logName": "projects/proj/logs/stdout",
    "severity": "INFO",
    "resource": {
      "labels": {
        "project_id": "proj",
        "location": "us-central1",
        "cluster_name": "c1",
        "namespace_name": "openrl-system",
        "pod_name": pod,
        "container_name": container,
      }
    },
    **extra,
  }


class GkeLogHelpersTest(unittest.TestCase):
  def setUp(self) -> None:
    for target, value in (("configuration", lambda: CONFIG), ("kubernetes.k8s_namespace", lambda: "openrl-system")):
      patcher = patch.object(gke if "." not in target else gke.kubernetes, target.split(".")[-1], value)
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_filter_scopes_to_cluster_sources_window_and_search(self) -> None:
    text = gke.logs_filter([SOURCE], "2026-09-10T10:00:00+00:00", "2026-09-10T11:00:00+00:00", "UNKNOWN", "oom")
    self.assertIn('resource.type="k8s_container"', text)
    self.assertIn('resource.labels.cluster_name="c1"', text)
    self.assertIn('resource.labels.pod_name="orw-a"', text)
    self.assertIn('severity="DEFAULT"', text)
    self.assertIn('textPayload:"oom" OR jsonPayload.message:"oom"', text)

  def test_entry_becomes_a_record_only_inside_an_observed_lifetime(self) -> None:
    record = gke.entry_record(entry(textPayload="hello"), [SOURCE], "run-1")
    self.assertEqual((record["pod"], record["role"], record["message"], record["severity"]), ("orw-a", "trainer", "hello", "INFO"))
    self.assertIsNone(gke.entry_record(entry(at="2026-09-10T12:00:00Z", textPayload="late"), [SOURCE], "run-1"))
    self.assertIsNone(gke.entry_record(entry(pod="someone-else", textPayload="x"), [SOURCE], "run-1"))

  def test_json_payload_is_rendered_when_there_is_no_text(self) -> None:
    record = gke.entry_record(entry(jsonPayload={"message": "m", "rank": 2}), [SOURCE], "run-1")
    self.assertEqual(record["rank"], 2)
    self.assertIn('"message": "m"', record["message"])

  def test_select_sources_applies_only_the_given_filters(self) -> None:
    other = {**SOURCE, "pod": "orw-b", "node": "n2"}
    self.assertEqual(gke.select_sources([SOURCE, other], None, None, None), [SOURCE, other])
    self.assertEqual(gke.select_sources([SOURCE, other], None, None, "n2"), [other])

  def test_cursor_round_trip_and_rejection(self) -> None:
    fingerprint = gke.query_fingerprint("run-1", None, None, "", None, None, None, None, 200)
    cursor = gke.remember_page(fingerprint, "s", "e", "page-2", [SOURCE])
    self.assertEqual(gke.resume_page(cursor, fingerprint), ("s", "e", "page-2", [SOURCE]))
    with self.assertRaises(ValueError):
      gke.resume_page(cursor, "another-scope")
    with self.assertRaises(ValueError):
      gke.resume_page("gke.nope", fingerprint)
