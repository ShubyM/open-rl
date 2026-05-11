from __future__ import annotations

import json
import os

DEFAULT_VLLM_URL = "http://127.0.0.1:8001"
VLLM_ROUTES_ENV = "OPEN_RL_VLLM_ROUTES"


def parse_vllm_routes(raw: str | None = None) -> dict[str, str]:
  raw = raw if raw is not None else os.getenv(VLLM_ROUTES_ENV)
  if not raw:
    return {}
  raw = raw.strip()
  if not raw:
    return {}
  if raw.startswith("{"):
    routes = json.loads(raw)
    if not isinstance(routes, dict):
      raise ValueError(f"{VLLM_ROUTES_ENV} must be a JSON object")
    return {str(model): str(url) for model, url in routes.items() if url}

  routes: dict[str, str] = {}
  for item in raw.split(","):
    if not item.strip():
      continue
    model, sep, url = item.partition("=")
    if not sep or not model.strip() or not url.strip():
      raise ValueError(f"{VLLM_ROUTES_ENV} entries must use model_id=url")
    routes[model.strip()] = url.strip()
  return routes


def vllm_url_for_base_model(base_model_id: str | None, default_url: str | None = None) -> str:
  if base_model_id:
    route = parse_vllm_routes().get(base_model_id)
    if route:
      return route
  return default_url or os.getenv("VLLM_URL", DEFAULT_VLLM_URL)
