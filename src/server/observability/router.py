import asyncio
import os
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query

from server.model_metadata import decode_model_metadata
from server.observability import client_runs, cloud_logging, fleet
from server.store import get_state_store

router = APIRouter(tags=["inspection"])
BOUNDS = "ISO-8601 or Unix seconds; since also takes a span before until, e.g. 1h. Default: the 30 minutes ending now; at most 7 days."
Bound = Annotated[str | None, Query(max_length=64, description=BOUNDS)]
CURSOR_EXPIRED = "Invalid or expired log cursor; repeat the query without it"


@router.get("/api/v1/inspect/inventory")
async def inventory():
  """Runs with their lifecycle (ended, failed, waiting, placed, active), pods and nodes; the cluster's nodes with their GPUs
  and its pods with problems; and an error per unavailable source or unreadable run record. client_status is what the
  client last reported, active until it deletes the model, so a crashed run stays active there; lifecycle is what the
  cluster shows."""
  return await fleet.snapshot()


@router.get("/api/v1/inspect/logs")
async def logs(
  run_id: str,
  pod: Annotated[str | None, Query(description="One of the run's pods, or 'client' for its client's logs.log")] = None,
  q: str = Query("", max_length=1024),
  severity: str | None = Query(None, pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|UNKNOWN)$"),
  since: Bound = None,
  until: Bound = None,
  limit: int = Query(200, ge=1, le=1000),
  cursor: Annotated[str | None, Query(max_length=16384, description="next_cursor of a page with the same filters and window")] = None,
):
  """Newest first: the run's worker pods by owner label from Cloud Logging, or with pod=client its client's logs.log. That
  file's lines carry no time, so since and until do not apply to it and its cursor is a byte offset."""
  try:
    since, until = cloud_logging.window(since, until)
  except (ValueError, OverflowError, OSError) as exc:
    raise HTTPException(400, "Use ISO-8601 or Unix-second since/until and a positive window of at most seven days") from exc
  if (raw := await get_state_store().get_value(f"open_rl:model_meta:{run_id}")) is None:
    raise HTTPException(404, f"No run {run_id}")
  meta = decode_model_metadata(raw)
  result = {"run_id": run_id, "source": "gke", "since": since, "until": until, "records": [], "next_cursor": None}
  if pod == "client":
    run = {"run_id": run_id, "created_at": meta.created_at, "model": meta.base_model}
    candidates = await fleet.cached("client directories", 30, lambda: asyncio.to_thread(client_runs.directories))
    if not (directory := await asyncio.to_thread(client_runs.match, run, candidates)):
      return {**result, "source": "client", "available": False, "error": f"No client directory under {client_runs.root()} matches this run"}
    if cursor and not cursor.isdigit():
      raise HTTPException(400, CURSOR_EXPIRED)
    result["source"] = path = os.path.join(directory, "logs.log")
    try:
      page = await asyncio.to_thread(client_runs.log_page, directory, int(cursor) if cursor else None, limit, q, severity)
      return {**result, **page, "available": True}
    except OSError as exc:
      return {**result, "available": False, "error": f"{path} unreadable: {type(exc).__name__}"}
  await cloud_logging.discover()
  owner = fleet.owner(meta.runtime(run_id))
  try:
    page = await cloud_logging.log_page(owner, since, until, cursor=cursor, limit=limit, q=q, pod=pod, severity=severity)
    return {**result, **page, "available": True}
  except httpx.HTTPStatusError as exc:
    if exc.response.status_code == 400 and cursor:
      raise HTTPException(400, CURSOR_EXPIRED) from None
    return {**result, "available": False, "error": f"Cloud Logging refused the read (HTTP {exc.response.status_code}); check credentials and IAM"}
  except Exception as exc:
    return {**result, "available": False, "error": str(exc) if isinstance(exc, LookupError) else "Cloud Logging unavailable"}
