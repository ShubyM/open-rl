"""One connection pool for cloud telemetry, owned by the server lifespan."""

import asyncio
from contextlib import asynccontextmanager

import httpx

_client: httpx.AsyncClient | None = None
_inflight: dict[tuple, asyncio.Task] = {}
MAX_INFLIGHT = 128
REQUEST_SECONDS = 10


@asynccontextmanager
async def lifespan(_app=None):
  global _client
  async with httpx.AsyncClient(timeout=10) as connection:
    _client = connection
    try:
      yield
    finally:
      pending = list(_inflight.values())
      for task in pending:
        task.cancel()
      await asyncio.gather(*pending, return_exceptions=True)
      _inflight.clear()
      _client = None


def client() -> httpx.AsyncClient:
  if _client is None:
    raise RuntimeError("Telemetry HTTP client requires the server lifespan")
  return _client


async def _get_json(url: str, params: dict, headers: dict) -> dict:
  # A shared request can outlive its callers, so bound the whole operation.
  async with asyncio.timeout(REQUEST_SECONDS):
    response = await client().get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


async def get_json(url: str, *, params: dict, headers: dict) -> dict:
  """Share only overlapping identical requests, including their actual credentials."""
  key = (url, tuple(sorted(params.items())), tuple(sorted(headers.items())))
  task = _inflight.get(key)
  if task is None:
    if len(_inflight) >= MAX_INFLIGHT:
      return await _get_json(url, params, headers)
    task = _inflight[key] = asyncio.create_task(_get_json(url, params, headers))

    def finished(task):
      _inflight.pop(key, None)
      if not task.cancelled():
        task.exception()  # Observe failures even when every waiter has left.

    task.add_done_callback(finished)
  return await asyncio.shield(task)
