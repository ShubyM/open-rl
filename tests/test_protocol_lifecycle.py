import tempfile
import time
import unittest
from unittest.mock import patch

from server import gateway
from server.protocol import SamplerSnapshot
from server.store import (
  InMemoryStore,
  acquire_sampler_snapshot,
  bump_model_revision,
  get_client_session,
  get_sampler_snapshot,
  prune_sampler_snapshots,
  put_sampler_artifact,
  put_sampler_snapshot,
  release_sampler_snapshot,
)
from server.training_requests_processor import save_sampler_snapshot


class RecordingWorker:
  def __init__(self):
    self.saved_states = []

  def save_state(self, model_id, state_path, include_optimizer=False, kind="state"):
    self.saved_states.append((model_id, state_path, include_optimizer, kind))
    return {"path": state_path}


class ClientSessionTest(unittest.IsolatedAsyncioTestCase):
  async def test_sessions_are_unique_and_heartbeat_refreshes_them(self) -> None:
    store = InMemoryStore()
    with patch.object(gateway, "get_store", return_value=store), patch.dict("os.environ", {"OPEN_RL_SESSION_TTL_SECONDS": "60"}):
      first = await gateway.create_session({"tags": ["dev"], "sdk_version": "1.2.3"})
      second = await gateway.create_session({})
      before = await get_client_session(store, first["session_id"])
      await gateway.session_heartbeat({"session_id": first["session_id"]})
      after = await get_client_session(store, first["session_id"])

    self.assertNotEqual(first["session_id"], second["session_id"])
    self.assertEqual(before.tags, ["dev"])
    self.assertEqual(before.sdk_version, "1.2.3")
    self.assertGreaterEqual(after.last_heartbeat, before.last_heartbeat)


class SamplerSnapshotLifecycleTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    self.store = InMemoryStore()
    self.worker = RecordingWorker()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)

  async def save(self, session_id: str, *, alias: str | None = None, ttl_seconds: int | None = None):
    with patch.dict(
      "os.environ",
      {
        "OPEN_RL_TMP_DIR": self.temp_dir.name,
        "OPEN_RL_SAMPLER_SNAPSHOT_RETENTION": "2",
      },
    ):
      return await save_sampler_snapshot(
        self.store,
        self.worker,
        {
          "alias": alias,
          "path": session_id if alias else None,
          "sampling_session_id": session_id,
          "ttl_seconds": ttl_seconds,
        },
        "model-a",
      )

  async def test_optimizer_revision_changes_only_when_explicitly_bumped(self) -> None:
    first = await self.save("session-0")
    revision = await bump_model_revision(self.store, "model-a")
    second = await self.save("session-1")

    self.assertEqual(first["revision"], 0)
    self.assertEqual(revision, 1)
    self.assertEqual(second["revision"], 1)

  async def test_repeated_save_of_one_revision_reuses_the_artifact(self) -> None:
    first = await self.save("session-a")
    second = await self.save("session-b")
    first_snapshot = await get_sampler_snapshot(self.store, "session-a")
    second_snapshot = await get_sampler_snapshot(self.store, "session-b")

    self.assertTrue(first["checkpoint_created"])
    self.assertFalse(second["checkpoint_created"])
    self.assertEqual(len(self.worker.saved_states), 1)
    self.assertEqual(first_snapshot.storage_path, second_snapshot.storage_path)

  async def test_ephemeral_retention_expires_old_sessions_without_retargeting(self) -> None:
    await self.save("session-0")
    await bump_model_revision(self.store, "model-a")
    await self.save("session-1")
    await bump_model_revision(self.store, "model-a")
    await self.save("session-2")

    self.assertIsNone(await get_sampler_snapshot(self.store, "session-0"))
    self.assertEqual((await get_sampler_snapshot(self.store, "session-1")).revision, 1)
    self.assertEqual((await get_sampler_snapshot(self.store, "session-2")).revision, 2)

  async def test_in_flight_snapshot_is_not_pruned(self) -> None:
    now = time.time()
    old = SamplerSnapshot(
      sampling_session_id="old",
      model_id="model-a",
      revision=0,
      storage_path="artifact-0",
      created_at=now - 10,
    )
    new = SamplerSnapshot(
      sampling_session_id="new",
      model_id="model-a",
      revision=1,
      storage_path="artifact-1",
      created_at=now,
    )
    await put_sampler_snapshot(self.store, old)
    await put_sampler_snapshot(self.store, new)
    await acquire_sampler_snapshot(self.store, "old")

    await prune_sampler_snapshots(self.store, "model-a", keep_ephemeral=1, now=now)
    self.assertIsNotNone(await get_sampler_snapshot(self.store, "old"))

    await release_sampler_snapshot(self.store, "old")
    await prune_sampler_snapshots(self.store, "model-a", keep_ephemeral=1, now=now)
    self.assertIsNone(await get_sampler_snapshot(self.store, "old"))

  async def test_named_snapshot_uses_ttl_instead_of_count_retention(self) -> None:
    now = time.time()
    named = SamplerSnapshot(
      sampling_session_id="named",
      model_id="model-a",
      revision=0,
      storage_path="artifact-0",
      named=True,
      created_at=now - 10,
      expires_at=now + 5,
    )
    ephemeral = SamplerSnapshot(
      sampling_session_id="ephemeral",
      model_id="model-a",
      revision=1,
      storage_path="artifact-1",
      created_at=now,
    )
    await put_sampler_snapshot(self.store, named)
    await put_sampler_snapshot(self.store, ephemeral)
    await put_sampler_artifact(self.store, "model-a", 0, "artifact-0")
    await put_sampler_artifact(self.store, "model-a", 1, "artifact-1")

    await prune_sampler_snapshots(self.store, "model-a", keep_ephemeral=1, now=now)
    self.assertIsNotNone(await get_sampler_snapshot(self.store, "named"))

    orphaned = await prune_sampler_snapshots(self.store, "model-a", keep_ephemeral=1, now=now + 6)
    self.assertIsNone(await get_sampler_snapshot(self.store, "named"))
    self.assertEqual(orphaned, ["artifact-0"])


if __name__ == "__main__":
  unittest.main()
