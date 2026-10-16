import copy
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server.dashboard import router
from server.dashboard.__main__ import app
from tests.test_dashboard_inspection import PLACEMENT, operation, state

WINDOW = {"since": "1970-01-01T00:02:00Z", "until": "1970-01-01T00:03:10Z"}


class InspectionApiTest(unittest.TestCase):
  def setUp(self):
    self.client = TestClient(app)
    self.state = state()
    self.state["cluster"]["nodes"] = []
    self.state["runs"][0].update(shared_runtime=False)
    self.store = mock.Mock(read_samples=mock.AsyncMock(return_value=[]))
    patches = [
      mock.patch.object(router.snapshot, "current", mock.AsyncMock(return_value=self.state)),
      mock.patch.object(router.gke, "discover", mock.AsyncMock()),
      mock.patch.object(router, "get_state_store", return_value=self.store),
    ]
    for patch in patches:
      patch.start()
      self.addCleanup(patch.stop)

  def test_inventory_is_the_snapshot_without_history_and_the_api_is_read_only(self):
    response = self.client.get("/api/v1/inspect/inventory")
    self.assertEqual(response.status_code, 200)
    inventory = response.json()
    self.assertEqual(inventory["placements"][0]["devices"], PLACEMENT["devices"])
    self.assertNotIn("history", inventory)
    paths = self.client.get("/openapi.json").json()["paths"]
    self.assertIn("/api/v1/inspect/activity", paths)
    self.assertTrue(all(set(methods) == {"get"} for methods in paths.values()))

  def test_activity_batches_shared_workers_and_keeps_partial_source_errors(self):
    self.state["placements"][0] = {**PLACEMENT, "run_ids": ["r1", "r2"]}
    old = {**copy.deepcopy(PLACEMENT), "id": "expired", "name": "expired", "first_seen": 1, "last_seen": 10}
    self.state["history"].append(old)

    async def read_samples(key):
      if key == router.ops.turn_key("trainer"):
        raise RuntimeError("offline")
      return [operation(110, 130)]

    self.store.read_samples.side_effect = read_samples
    response = self.client.get("/api/v1/inspect/activity", params=WINDOW)
    self.assertEqual(response.status_code, 200)
    keys = sorted(call.args[0] for call in self.store.read_samples.await_args_list)
    self.assertEqual(keys, sorted([router.ops.key("r1"), router.ops.key("r2"), router.ops.turn_key("trainer")]))
    lane = next(p for p in response.json()["placements"] if p["id"] == "w1:r1")
    self.assertEqual(lane["intervals"], [[120, 130]])
    self.assertFalse(lane["exact"])
    self.assertIn("unavailable", lane["warning"])

  def test_queries_validate_windows_without_touching_sources(self):
    for path in ("activity", "series"):
      response = self.client.get(f"/api/v1/inspect/{path}", params={"since": "invalid"})
      self.assertEqual(response.status_code, 400)
    self.store.read_samples.assert_not_awaited()

  def test_run_activity_keeps_metrics_and_incidents_without_placement_history(self):
    self.state.update(placements=[], history=[])
    self.state["runs"][0]["pods"] = [{"name": "p", "events": [{"reason": "OOMKilled", "last_seen_at": "1970-01-01T00:02:30Z", "message": "OOM"}]}]
    self.store.read_samples.return_value = [operation(125, 140), operation(150, 200)]
    response = self.client.get("/api/v1/inspect/activity", params={**WINDOW, "run_id": "r1"})
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertEqual(data["run"], self.state["runs"][0])
    self.assertEqual(data["placements"], [])
    self.assertEqual([s["at"] for s in data["samples"]], [140])
    self.assertEqual(data["incidents"][0]["reason"], "OOMKilled")
    self.store.read_samples.assert_awaited_once_with(router.ops.key("r1"))

  def test_series_is_independently_callable_and_keeps_partial_data(self):
    backend = mock.Mock(name="hardware")
    backend.name = "fake"
    backend.devices = mock.AsyncMock(side_effect=RuntimeError("offline"))
    backend.workers_series = mock.AsyncMock(return_value={"trainer-0": {"cpu_cores": [[150, 2]], "memory_bytes": []}})
    with mock.patch.object(router.metrics, "hardware", return_value=(backend, None)):
      response = self.client.get("/api/v1/inspect/series", params={**WINDOW, "node_id": "n1"})
    self.assertEqual(response.status_code, 200)
    data = response.json()
    self.assertTrue(data["available"])
    self.assertEqual(data["workers"][0]["cpu_cores"], [[150, 2]])
    self.assertIn("unavailable", data["reason"])
    router.gke.discover.assert_awaited_once()

  def test_logs_keep_pagination_and_shared_membership(self):
    result = {"run_id": "r1", "available": True, "records": [], "next_cursor": "page2"}
    with (
      mock.patch.object(router.gke, "pod_sources", return_value=[]),
      mock.patch.object(router.gke, "run_logs", mock.AsyncMock(return_value=result)),
    ):
      response = self.client.get("/api/v1/inspect/logs", params={"run_id": "r1", **WINDOW})
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["next_cursor"], "page2")
    self.assertIs(response.json()["shared_runtime"], False)

  def test_openapi_describes_every_response_and_nested_diagnostic_contract(self):
    spec = self.client.get("/openapi.json").json()
    for path, methods in spec["paths"].items():
      schema = methods["get"]["responses"]["200"]["content"]["application/json"]["schema"]
      self.assertIn("$ref", schema, path)
      model = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[1]]
      self.assertTrue(model["properties"], path)
    models = spec["components"]["schemas"]
    for name, fields in {
      "Run": ["display_status", "session_live", "workers", "workers_error", "settings", "shared_runtime"],
      "Workload": ["placed_message", "placed_reason", "requested_memory"],
      "PlacementActivity": ["exact", "warning", "error", "intervals"],
      "Series": ["reason", "available"],
      "Logs": ["next_cursor", "error"],
      "PodLogs": ["previous", "text"],
    }.items():
      for field in fields:
        self.assertIn(field, models[name]["properties"])

  def test_schema_preserves_unknown_fields_null_false_zero_and_empty_collections(self):
    import random

    rng = random.Random(1739)
    for _ in range(100):
      extra = {"null": None, "false": False, "zero": 0, "array": [], "nested": {"value": rng.choice([None, False, 0, "λ<script>"])}}
      self.state["runs"][0].update(settings=extra, diagnostic=extra, session_live=None, workers_error="Store unavailable")
      self.state["cluster"]["scheduler"].update(error="Scheduler unavailable", available=False)
      data = self.client.get("/api/v1/inspect/inventory").json()
      self.assertEqual(data["runs"], self.state["runs"])
      self.assertEqual(data["cluster"], self.state["cluster"])
      self.assertNotIn("history", data)

  def test_fuzz_invalid_query_parameters_never_read_sources(self):
    import random

    rng = random.Random(42)
    cases = [("since", "".join(rng.choices("!xyz?<>&/λ", k=rng.randint(1, 60)))) for _ in range(100)]
    cases += [("until", "0001-01-01T00:00:00Z"), ("since", "9999-12-31T23:59:59-23:59")]
    for path in ("activity", "series", "logs"):
      for key, value in cases:
        with self.subTest(path=path, value=value):
          response = self.client.get(f"/api/v1/inspect/{path}", params={"run_id": "r1", key: value})
          self.assertIn(response.status_code, (400, 422))
    router.snapshot.current.assert_not_awaited()
    router.gke.discover.assert_not_awaited()
    self.store.read_samples.assert_not_awaited()

  def test_corrupt_telemetry_keeps_valid_records_and_reports_partial_failure(self):
    bad = [
      None,
      [],
      {},
      {"at": "oops"},
      operation(140, float("nan")),
      operation(140, float("inf")),
      operation(160, 150),
      operation(140, 150, metrics={"loss": float("inf")}),
      operation(140, 150, role=[]),
    ]
    self.store.read_samples.side_effect = lambda key: [operation(125, 140), *bad] if key == router.ops.key("r1") else []
    response = self.client.get("/api/v1/inspect/activity", params={**WINDOW, "run_id": "r1"})
    self.assertEqual(response.status_code, 200)
    result = response.json()
    self.assertEqual([s["at"] for s in result["samples"]], [140])
    self.assertIn("invalid", result["error"].lower())
    self.assertEqual(result["placements"][0]["intervals"], [[125, 140]])

  def test_pod_logs_preserve_previous_and_unknown_diagnostics(self):
    result = {"pod": "p", "container": None, "previous": True, "text": "untrusted <script> false 0", "future": {"flag": False}}
    with mock.patch.object(router.kubernetes, "pod_logs", return_value=result):
      response = self.client.get("/api/v1/inspect/pods/p/logs?previous=true")
    self.assertEqual(response.json(), result)

  def test_omitted_bounds_anchor_to_snapshot_and_report_the_resolved_window(self):
    self.store.read_samples.side_effect = lambda key: [operation(125, 140)] if key == router.ops.key("r1") else []
    for params in ({"run_id": "r1"}, {"run_id": "r1", "since": WINDOW["since"]}):
      data = self.client.get("/api/v1/inspect/activity", params=params).json()
      self.assertEqual(data["until"], "1970-01-01T00:03:20.000000+00:00")
      self.assertEqual(data["observed_at"], self.state["observed_at"])
      self.assertIn("Missing records do not prove no requests arrived", data["coverage"])
      self.assertEqual(data["placements"][0]["intervals"], [[125, 140]])
      self.assertEqual(len(data["samples"]), 1)

  def test_random_corrupt_ops_and_turns_preserve_good_samples(self):
    import random

    rng = random.Random(1739)
    invalid = [None, False, "bad", [], {}, float("nan"), float("inf")]
    for case in range(200):
      bad_op, bad_turn = operation(125, 140), {"started_at": 120, "at": 150, "ops": []}
      bad_op[rng.choice(["at", "started_at", "operation", "metrics"])] = rng.choice(invalid)
      bad_turn[rng.choice(["at", "started_at", "ops"])] = rng.choice(invalid[:-2])
      # Some mutations (empty metrics / ops) are valid; all must remain serializable.
      records = {router.ops.key("r1"): [bad_op, operation(155, 165)], router.ops.turn_key("trainer"): [bad_turn]}
      self.store.read_samples.side_effect = records.__getitem__
      response = self.client.get("/api/v1/inspect/activity", params={**WINDOW, "run_id": "r1"})
      self.assertEqual(response.status_code, 200, case)
      self.assertIn(165, [sample["at"] for sample in response.json()["samples"]], case)
      for lane in response.json()["placements"]:
        self.assertTrue(all(120 <= low < high <= 190 for low, high in lane["intervals"]), case)
