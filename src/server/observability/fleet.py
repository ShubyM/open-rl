import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from server import openrl_status
from server.model_metadata import decode_model_metadata
from server.store import get_state_store

CACHE_SECONDS, STALE_SECONDS, READ_TIMEOUT_SECONDS = 5, 60, 6
Lifecycle = Literal["ended", "active"]


@dataclass
class Entry:
  at: float = 0.0
  task: asyncio.Future | None = None
  last_good: asyncio.Future | None = None
  last_good_at: float = 0.0


cache: dict[str, Entry] = {}


def succeeded(task: asyncio.Future | None) -> bool:
  return task is not None and task.done() and not task.cancelled() and task.exception() is None


async def cached(name: str, seconds: float, read) -> object:
  entry = cache.setdefault(name, Entry())
  task = entry.task
  if succeeded(task):
    entry.last_good, entry.last_good_at = task, entry.at
  if task is None or (task.done() and (not succeeded(task) or time.monotonic() - entry.at > seconds)):
    entry.at, entry.task = time.monotonic(), asyncio.ensure_future(read())
    entry.task.add_done_callback(lambda t: t.cancelled() or t.exception())  # retrieved, so asyncio does not log it
  if succeeded(entry.task):
    return entry.task.result()
  if entry.last_good is not None and time.monotonic() - entry.last_good_at <= STALE_SECONDS:
    return entry.last_good.result()
  return await asyncio.shield(entry.task)


async def snapshot() -> dict:
  return await cached("inventory", CACHE_SECONDS, read)


def lifecycle(status: str, live: bool) -> Lifecycle:
  return "active" if status == "active" and live else "ended"


def join(run_id: str, raw: str, live: set[str]) -> dict:
  row, meta = json.loads(raw), decode_model_metadata(raw)
  name = row.get("name") or row.get("run_name")
  run = {"run_id": run_id, "name": name or f"{(meta.base_model or 'Run').split('/')[-1]} · {run_id[:8]}", "display_name": name}
  run.update(model=meta.base_model, recipe_name=row.get("recipe_name"), fine_tuning_type=meta.fine_tuning_type, client_status=meta.status)
  run["lifecycle"] = lifecycle(meta.status, any(s in live for s in meta.session_ids))
  run.update(created_at=meta.created_at, completed_at=meta.completed_at, session_ids=meta.session_ids, settings=openrl_status.settings_of(meta))
  return run | {"shared_runtime": meta.runtime(run_id) != run_id}


async def read() -> dict:
  store, observed_at = get_state_store(), datetime.now(UTC).isoformat(timespec="seconds")
  scans = [asyncio.wait_for(store.scan_values(prefix), READ_TIMEOUT_SECONDS) for prefix in ("open_rl:model_meta:", "open_rl:session:")]
  metas, sessions = await asyncio.gather(*scans, return_exceptions=True)
  errors, runs = [], []
  if failed := next((r for r in (metas, sessions) if isinstance(r, Exception)), None):
    errors.append(f"Redis read failed: {f'no reply in {READ_TIMEOUT_SECONDS} s' if isinstance(failed, TimeoutError) else failed}")
    metas, sessions = {}, {}
  live = {name.removeprefix("open_rl:session:") for name in sessions}
  for name, raw in metas.items():
    run_id = name.removeprefix("open_rl:model_meta:")
    try:
      runs.append(join(run_id, raw, live))
    except Exception as exc:
      errors.append(f"model record {run_id} unreadable: {exc}")
  runs.sort(key=lambda r: -r["created_at"])
  return {"observed_at": observed_at, "errors": errors, "runs": runs}
