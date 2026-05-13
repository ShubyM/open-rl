from __future__ import annotations

import asyncio
import sys
import unittest

from tests._server_fixture import SERVER_DIR

sys.path.insert(0, str(SERVER_DIR))
import clock_cycle  # noqa: E402
from store import InMemoryStore  # noqa: E402


class FakeEngine:
  def __init__(self):
    self.adapters = set()
    self.loaded = []
    self.active = []

  def has_adapter(self, adapter_id):
    return adapter_id in self.adapters

  def load_from_state(self, adapter_id, checkpoint_ref, restore_optimizer=False):
    self.loaded.append((adapter_id, checkpoint_ref, restore_optimizer))
    self.adapters.add(adapter_id)

  def set_active_adapter(self, adapter_id):
    if adapter_id not in self.adapters:
      raise ValueError(f"missing adapter {adapter_id}")
    self.active.append(adapter_id)


class TestClockCycleCheckpointRecovery(unittest.TestCase):
  def test_ensure_adapter_ready_hydrates_missing_adapter_from_latest_checkpoint(self) -> None:
    async def run() -> None:
      original_engine = clock_cycle.engine
      fake_engine = FakeEngine()
      request_store = InMemoryStore()
      await request_store.publish_checkpoint("adapter-a", "/tmp/open-rl/checkpoints/a-1", {"restore_optimizer": True})

      try:
        clock_cycle.engine = fake_engine
        await clock_cycle.ensure_adapter_ready(request_store, "adapter-a")
      finally:
        clock_cycle.engine = original_engine

      self.assertEqual(fake_engine.loaded, [("adapter-a", "/tmp/open-rl/checkpoints/a-1", True)])
      self.assertEqual(fake_engine.active, ["adapter-a"])

    asyncio.run(run())


if __name__ == "__main__":
  unittest.main()
