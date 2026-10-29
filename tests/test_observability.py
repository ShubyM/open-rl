import asyncio
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from server import api_server
from server.model_metadata import TrainingModelMetadata
from server.observability import fleet
from server.store import InMemoryStateStore


def meta(**fields) -> str:
  return TrainingModelMetadata(base_model="Qwen/Qwen3-8B", **fields).model_dump_json()


class InventoryTest(unittest.TestCase):
  def inventory(self, **records: str) -> dict:
    store = InMemoryStateStore()
    for key, value in records.items():
      asyncio.run(store.set_value(f"open_rl:{key}", value))
    fleet.cache.clear()
    self.enterContext(mock.patch.object(fleet, "get_state_store", return_value=store))
    return TestClient(api_server.app).get("/api/v1/inspect/inventory").json()

  def test_a_run_is_active_only_while_the_client_has_a_live_session(self):
    body = self.inventory(
      **{
        "model_meta:live": meta(session_ids=["s1"], created_at=3),
        "model_meta:idle": meta(session_ids=["s2"], created_at=2),
        "model_meta:done": meta(status="completed", session_ids=["s1"], created_at=1),
        "session:s1": "{}",
      }
    )

    runs = [(r["run_id"], r["lifecycle"], r["client_status"]) for r in body["runs"]]
    self.assertEqual(runs, [("live", "active", "active"), ("idle", "ended", "active"), ("done", "ended", "completed")])
    self.assertEqual(body["errors"], [])

  def test_a_malformed_model_record_is_skipped_with_an_error(self):
    body = self.inventory(**{"model_meta:good": meta(), "model_meta:bad": "not json"})

    self.assertEqual([r["run_id"] for r in body["runs"]], ["good"])
    self.assertTrue(any("bad" in error for error in body["errors"]), body["errors"])


if __name__ == "__main__":
  unittest.main()
