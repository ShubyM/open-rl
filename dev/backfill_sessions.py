"""Write session records for sessions from before the API server recorded them, so older runs keep their sessions.

A session's models are those whose metadata names it. It started when the first of them was created and was last heard
from at their last recorded op. Records are marked derived: the client's heartbeats were never stored, so checks do not
read a stale last_seen_at as a client that stopped. Records already in the store are kept, so a rerun adds nothing.

  OPEN_RL_OPS_TTL_DAYS=0 PYTHONPATH=src REDIS_URL=redis://HOST:PORT/0 python dev/backfill_sessions.py
"""

import asyncio
import json

from server import telemetry
from server.dashboard.snapshot import METADATA_PREFIX
from server.model_metadata import decode_model_metadata
from server.store import get_state_store



async def main() -> None:
  store = get_state_store()
  models = {key.removeprefix(METADATA_PREFIX): meta for key, value in (await store.scan_values(METADATA_PREFIX)).items() if (meta := decode_model_metadata(value))}
  sessions: dict[str, list[str]] = {}
  for model_id, meta in models.items():
    for session_id in meta.session_ids:
      sessions.setdefault(session_id, []).append(model_id)
  added = []
  for session_id, model_ids in sorted(sessions.items()):
    if await store.get_value(telemetry.session_key(session_id)) is not None:
      continue
    last = []
    for model_id in model_ids:
      for key in (telemetry.ops_key(model_id), telemetry.samples_key(model_id)):
        last += [s["at"] for s in await store.read_samples(key, last=1)]
    created = min(models[m].created_at for m in model_ids)
    record = {"session_id": session_id, "created_at": created, "last_seen_at": max(last, default=created), "model_ids": model_ids, "derived": True}
    await store.set_value(telemetry.session_key(session_id), json.dumps(record), ttl_seconds=telemetry.TTL_SECONDS)
    added.append(session_id)
  print(json.dumps({"sessions": len(sessions), "added": added}, indent=2))


if __name__ == "__main__":
  asyncio.run(main())
