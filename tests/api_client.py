"""An httpx client bound to an ASGI FastAPI app, for async tests."""

import httpx

from server import api_server


def asgi_client(app=None) -> httpx.AsyncClient:
  return httpx.AsyncClient(transport=httpx.ASGITransport(app=api_server.app if app is None else app), base_url="http://test")


async def post_json(client: httpx.AsyncClient, path: str, body: dict, **kwargs) -> dict:
  response = await client.post(f"/api/v1/{path}", json=body, **kwargs)
  response.raise_for_status()
  return response.json()
