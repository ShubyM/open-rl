import asyncio
import copy
import json
import os
import tempfile
import unittest
from unittest import mock

import httpx
from fastapi.testclient import TestClient

from server import api_server
from server.model_metadata import TrainingModelMetadata
from server.observability import client_runs, cloud_logging, fleet
from server.store import InMemoryStateStore
from server.worker_manager import FAILED, READY, STARTING, STOPPED, WAITING_FOR_PLACEMENT

NO_CLUSTER = {"available": True, "pods": [], "nodes": [], "workloads": None, "errors": []}


def meta(**fields) -> str:
  return TrainingModelMetadata(base_model="Qwen/Qwen3-8B", **fields).model_dump_json()


def cluster(*states: str) -> dict:
  return {"pods": [], "workloads": [{"model_id": "r1", "training_kind": "fft", "state": s} for s in states]}


class ApiTest(unittest.TestCase):
  def setUp(self):
    self.store = InMemoryStateStore()
    fleet.cache.clear()
    self.enterContext(mock.patch.object(fleet, "get_state_store", return_value=self.store))
    self.enterContext(mock.patch("server.observability.router.get_state_store", return_value=self.store))

  def get(self, path: str, cluster: dict = NO_CLUSTER, **records: str) -> dict:
    for key, value in records.items():
      asyncio.run(self.store.set_value(f"open_rl:{key}", value))
    self.enterContext(mock.patch.object(fleet, "kubernetes", mock.AsyncMock(return_value=copy.deepcopy(cluster))))
    return TestClient(api_server.app).get(path).json()


class InventoryTest(ApiTest):
  def test_a_run_is_active_only_while_the_client_has_a_live_session(self):
    body = self.get(
      "/api/v1/inspect/inventory",
      **{
        "model_meta:live": meta(session_ids=["s1"], created_at=3),
        "model_meta:idle": meta(session_ids=["s2"], created_at=2),
        "model_meta:done": meta(status="completed", session_ids=["s1"], created_at=1),
        "session:s1": "{}",
      },
    )

    runs = [(r["run_id"], r["lifecycle"], r["client_status"]) for r in body["runs"]]
    self.assertEqual(runs, [("live", "active", "active"), ("idle", "ended", "active"), ("done", "ended", "completed")])
    self.assertEqual(body["errors"], [])

  def test_runs_come_back_from_redis_when_kubernetes_is_unavailable(self):
    down = NO_CLUSTER | {"available": False, "errors": ["Kubernetes unavailable: no kubeconfig"]}
    crashing = {"name": "orw-x", "node": "n1", "owner": "x", "problem": "CrashLoopBackOff", "restarts": 4}
    healthy = crashing | {"name": "orw-y", "problem": None, "restarts": 0}
    records = {"model_meta:r1": meta(session_ids=["s1"]), "session:s1": "{}"}
    body = self.get("/api/v1/inspect/inventory", down | {"pods": [crashing, healthy]}, **records)

    self.assertEqual([(r["run_id"], r["lifecycle"]) for r in body["runs"]], [("r1", "active")])
    self.assertEqual(body["cluster"], {"available": False, "nodes": [], "pods": [{"name": "orw-x", "problem": "CrashLoopBackOff", "restarts": 4}]})
    self.assertEqual(body["errors"], ["Kubernetes unavailable: no kubeconfig"])

  def test_a_malformed_model_record_is_skipped_with_an_error(self):
    body = self.get("/api/v1/inspect/inventory", **{"model_meta:good": meta(), "model_meta:bad": "not json"})

    self.assertEqual([r["run_id"] for r in body["runs"]], ["good"])
    self.assertTrue(any("bad" in error for error in body["errors"]), body["errors"])


class JoinTest(unittest.TestCase):
  def test_lifecycle_follows_the_runs_workloads(self):
    live, raw = {"s1"}, meta(fine_tuning_type="full", session_ids=["s1"])
    self.assertEqual(fleet.join("r1", raw, live, cluster(STOPPED, STOPPED))["lifecycle"], "ended")
    self.assertEqual(fleet.join("r1", raw, live, cluster(WAITING_FOR_PLACEMENT, READY))["lifecycle"], "waiting")
    self.assertEqual(fleet.join("r1", raw, live, cluster(STARTING, READY))["lifecycle"], "placed")
    self.assertEqual(fleet.join("r1", raw, live, cluster(FAILED, READY))["lifecycle"], "failed")

  def test_a_finished_run_on_a_shared_runtime_does_not_claim_its_pods(self):
    pod = {"name": "orw-qwen-qwen3-8b-trainer", "node": "n1", "owner": "qwen-qwen3-8b-a82fd547", "problem": None, "restarts": 0}
    shared = {"pods": [pod], "workloads": [{"model_id": "Qwen/Qwen3-8B", "training_kind": "lora", "state": READY}]}
    finished = fleet.join("r2", meta(status="completed"), set(), shared)
    active = fleet.join("r3", meta(), set(), shared)
    self.assertEqual((finished["lifecycle"], finished["pods"], finished["nodes"]), ("ended", [], []))
    self.assertEqual((active["lifecycle"], active["pods"], active["nodes"]), ("placed", [{"name": pod["name"], "problem": None}], ["n1"]))
    self.assertTrue(finished["shared_runtime"] and active["shared_runtime"])


GKE = {"OPEN_RL_GKE_PROJECT": "p", "OPEN_RL_GKE_CLUSTER": "c", "OPEN_RL_GKE_LOCATION": "l", "OPEN_RL_WORKER_NAMESPACE": "ns"}


class LogsTest(ApiTest):
  def test_logs_read_only_the_runs_pods_and_page_with_cloud_loggings_cursor(self):
    sent = []

    async def send(method: str, url: str, json: dict, headers: dict) -> httpx.Response:
      sent.append(json)
      entry = {"textPayload": "INFO loading", "resource": {"labels": {"pod_name": "orw-r1-t"}}}
      page = {"entries": [entry], "nextPageToken": "t2"} if "pageToken" not in json else {"entries": []}
      return httpx.Response(200, json=page, request=httpx.Request(method, url))

    self.enterContext(mock.patch.dict(os.environ, GKE))
    self.enterContext(mock.patch.object(cloud_logging, "send", send))
    self.enterContext(mock.patch.object(cloud_logging, "gcp_headers", mock.AsyncMock(return_value={})))
    window = "since=2026-10-01T00:00:00Z&until=2026-10-01T01:00:00Z"
    first = self.get(f"/api/v1/inspect/logs?run_id=r1&pod=orw-r1-t&{window}", **{"model_meta:r1": meta(fine_tuning_type="full")})
    second = self.get(f"/api/v1/inspect/logs?run_id=r1&pod=orw-r1-t&{window}&cursor=t2")

    self.assertEqual([(r["pod"], r["message"]) for r in first["records"]], [("orw-r1-t", "INFO loading")])
    self.assertEqual((first["next_cursor"], second["records"], second["next_cursor"]), ("t2", [], None))
    self.assertEqual([s.get("pageToken") for s in sent], [None, "t2"])
    clauses = sent[0]["filter"].split(" AND ")
    self.assertIn('labels."k8s-pod/timeslice_io/owner"="r1"', clauses)
    self.assertIn('resource.labels.pod_name="orw-r1-t"', clauses)
    self.assertIn('resource.labels.namespace_name="ns"', clauses)

  def test_the_client_log_is_found_in_the_e2e_layout_and_paged_newest_first(self):
    root = self.enterContext(tempfile.TemporaryDirectory())
    run_dir = os.path.join(root, "open-rl", "runs", "rl_loop", "open-rl-tmp", "fft_gsm8k_rl")
    other = os.path.join(root, "open-rl", "runs", "rl_loop", "open-rl-tmp", "lora_gsm8k_rl")
    for directory, run_id in ((run_dir, "r1"), (other, "r9")):
      os.makedirs(directory)
      with open(os.path.join(directory, "checkpoints.jsonl"), "w") as f:
        f.write(json.dumps({"state_path": f"tinker://{run_id}/weights/000010"}) + "\n")
    with open(os.path.join(run_dir, "logs.log"), "w") as f:
      f.write("train:10 [INFO] step 1\ntrain:20 [ERROR] step 2 failed\nTraceback (most recent call last)\n")
    self.enterContext(mock.patch.dict(os.environ, {"OPEN_RL_CLIENT_RUNS_ROOT": root}))
    client_runs.listings.clear()

    newest = self.get("/api/v1/inspect/logs?run_id=r1&pod=client&limit=1", **{"model_meta:r1": meta()})
    older = self.get(f"/api/v1/inspect/logs?run_id=r1&pod=client&limit=1&cursor={newest['next_cursor']}")

    self.assertEqual(newest["source"], os.path.join(run_dir, "logs.log"))
    self.assertEqual([r["message"] for r in newest["records"]], ["train:20 [ERROR] step 2 failed\nTraceback (most recent call last)"])
    self.assertEqual(([r["message"] for r in older["records"]], older["next_cursor"]), (["train:10 [INFO] step 1"], None))


if __name__ == "__main__":
  unittest.main()
