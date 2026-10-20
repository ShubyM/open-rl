"""One connection pool for cloud telemetry, owned by the server lifespan."""

import asyncio
from contextlib import asynccontextmanager

import httpx

_client: httpx.AsyncClient | None = None
# Used outside the server lifespan, by a script or a test, created on first use instead.
_fallback: httpx.AsyncClient | None = None
_inflight: dict[tuple, asyncio.Task] = {}
MAX_INFLIGHT = 128
REQUEST_SECONDS = 10
# A quota refusal or server error is usually momentary; one retry spares the page an error.
RETRY_SECONDS = 1


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
  global _fallback
  if _client is not None:
    return _client
  if _fallback is None or _fallback.is_closed:
    _fallback = httpx.AsyncClient(timeout=10)
  return _fallback


def transient(response: httpx.Response) -> bool:
  return response.status_code == 429 or response.status_code >= 500


async def send(method: str, url: str, **kwargs) -> httpx.Response:
  """One request, retried once after a transient refusal."""
  call = client().get if method == "GET" else lambda url, **kw: client().request(method, url, **kw)
  response = await call(url, **kwargs)
  if transient(response):
    await asyncio.sleep(RETRY_SECONDS)
    response = await call(url, **kwargs)
  response.raise_for_status()
  return response


async def _get_json(url: str, params: dict, headers: dict) -> dict:
  # A shared request can outlive its callers, so bound the whole operation.
  async with asyncio.timeout(REQUEST_SECONDS + RETRY_SECONDS):
    return (await send("GET", url, params=params, headers=headers)).json()


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
