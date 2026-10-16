import asyncio
import unittest

from server.dashboard import history
from server.store import InMemoryStateStore

PLACEMENT = {
  "id": "w-1",
  "name": "fft-w-1",
  "label": "Qwen3-8B · abc",
  "run_ids": ["abc"],
  "runtime_id": "abc",
  "node": "n1",
  "pod": "orw-w-1",
  "role": "trainer",
  "owner_id": "abc",
  "device_count": 1,
  "devices": ["gpu-0"],
}


class PlacementHistoryTest(unittest.TestCase):
  def test_first_seen_sticks_and_last_seen_advances(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    asyncio.run(history.record(store, [PLACEMENT, {**PLACEMENT, "id": "w-2", "node": None}], now=130.0))
    entries = asyncio.run(history.read(store, since=0))
    self.assertEqual([e["id"] for e in entries], ["w-1"])
    self.assertEqual((entries[0]["first_seen"], entries[0]["last_seen"], entries[0]["devices"]), (100.0, 130.0, ["gpu-0"]))

  def test_a_shared_worker_keeps_the_runs_it_served(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    # The run finished; the shared worker is still placed but matched to no run.
    asyncio.run(history.record(store, [{**PLACEMENT, "run_ids": [], "label": "Qwen/Qwen3-8B"}], now=110.0))
    asyncio.run(history.record(store, [{**PLACEMENT, "run_ids": ["def"], "label": "Qwen3-8B · def"}], now=120.0))
    entry = asyncio.run(history.read(store, since=0))[0]
    self.assertEqual(entry["run_ids"], ["abc", "def"])
    self.assertEqual(entry["label"], "Qwen3-8B · def")

  def test_a_finished_run_keeps_its_label(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    asyncio.run(history.record(store, [{**PLACEMENT, "run_ids": [], "label": "Qwen/Qwen3-8B"}], now=110.0))
    self.assertEqual(asyncio.run(history.read(store, since=0))[0]["label"], "Qwen3-8B · abc")

  def test_devices_survive_the_claim_being_deleted_first(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    asyncio.run(history.record(store, [{**PLACEMENT, "devices": []}], now=110.0))
    self.assertEqual(asyncio.run(history.read(store, since=0))[0]["devices"], ["gpu-0"])

  def test_read_filters_by_last_seen(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    self.assertEqual(asyncio.run(history.read(store, since=200.0)), [])
    self.assertEqual(len(asyncio.run(history.read(store, since=100.0))), 1)

  def test_replacements_retain_the_previous_interval_and_physical_devices(self) -> None:
    original = {**PLACEMENT, "pod_uid": "p1", "device_uuids": {"gpu-0": "GPU-old"}}
    for change in ({"node": "n2"}, {"pod_uid": "p2"}, {"devices": ["gpu-1"]}, {"device_uuids": {"gpu-0": "GPU-new"}}):
      with self.subTest(change=change):
        store = InMemoryStateStore()
        asyncio.run(history.record(store, [original], now=100.0))
        asyncio.run(history.record(store, [original], now=110.0))
        asyncio.run(history.record(store, [{**original, **change}], now=120.0))
        current, past = asyncio.run(history.read(store, since=0))
        self.assertEqual((current["first_seen"], current["last_seen"]), (120.0, 120.0))
        self.assertEqual((past["first_seen"], past["last_seen"]), (100.0, 110.0))
        self.assertEqual((past["node"], past["pod_uid"], past["device_uuids"]), ("n1", "p1", {"gpu-0": "GPU-old"}))
        self.assertEqual(current["worker_id"], past["worker_id"])
        self.assertNotEqual(current["id"], past["id"])

  def test_a_new_pod_does_not_inherit_a_deleted_claim(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [{**PLACEMENT, "pod_uid": "p1"}], now=100.0))
    asyncio.run(history.record(store, [{**PLACEMENT, "pod_uid": "p2", "devices": []}], now=110.0))
    self.assertFalse(asyncio.run(history.read(store, since=0))[0]["devices"])

  def test_identity_enrichment_and_temporary_missing_devices_keep_the_interval(self) -> None:
    store = InMemoryStateStore()
    asyncio.run(history.record(store, [PLACEMENT], now=100.0))
    identified = {**PLACEMENT, "pod_uid": "p1", "device_uuids": {"gpu-0": "GPU-1"}}
    asyncio.run(history.record(store, [identified], now=110.0))
    asyncio.run(history.record(store, [{**PLACEMENT, "devices": []}], now=120.0))
    entries = asyncio.run(history.read(store, since=0))
    self.assertEqual(len(entries), 1)
    self.assertEqual((entries[0]["first_seen"], entries[0]["last_seen"]), (100.0, 120.0))
    self.assertEqual((entries[0]["pod_uid"], entries[0]["device_uuids"]), ("p1", {"gpu-0": "GPU-1"}))
