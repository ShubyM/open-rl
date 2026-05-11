from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from tests._server_fixture import SERVER_DIR


def _load_gateway_module():
  sys.path.insert(0, str(SERVER_DIR))
  spec = importlib.util.spec_from_file_location("gateway_under_test", Path(SERVER_DIR) / "gateway.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class _StoreStub:
  def __init__(self, state: dict | None = None):
    self.state = state
    self.futures: dict[str, dict] = {}

  async def get_model_state(self, state_id: str) -> dict | None:
    return self.state if self.state and self.state.get("state_id") == state_id else None

  async def set_future(self, req_id: str, result: dict) -> None:
    self.futures[req_id] = result


class _ResponseStub:
  def raise_for_status(self) -> None:
    return None

  def json(self) -> dict:
    return {"sequences": [{"tokens": [1], "logprobs": [], "stop_reason": "length"}]}


class _AsyncClientStub:
  calls: list[tuple[str, dict]] = []

  def __init__(self, *args, **kwargs):
    pass

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    return False

  async def post(self, url: str, json: dict, headers: dict):
    self.__class__.calls.append((url, json))
    return _ResponseStub()


class TestGatewayVLLMRouting(unittest.TestCase):
  def test_parse_vllm_routes_accepts_json_and_pairs(self) -> None:
    gateway = _load_gateway_module()

    self.assertEqual(
      gateway.parse_vllm_routes('{"Qwen/Qwen3-0.6B":"http://qwen:8001"}'),
      {"Qwen/Qwen3-0.6B": "http://qwen:8001"},
    )
    self.assertEqual(
      gateway.parse_vllm_routes("Qwen/Qwen3-0.6B=http://qwen:8001,google/gemma-3-1b-it=http://gemma:8001"),
      {"Qwen/Qwen3-0.6B": "http://qwen:8001", "google/gemma-3-1b-it": "http://gemma:8001"},
    )

  def test_asample_routes_model_state_to_matching_vllm_worker(self) -> None:
    gateway = _load_gateway_module()
    state = {
      "state_id": "tinker://run-a/sampler_weights/current",
      "base_model": "google/gemma-3-1b-it",
      "adapter_name": "current",
      "adapter_ref": "/mnt/open-rl/peft/run-a/current",
      "version": 4,
    }
    store = _StoreStub(state)
    routes = {
      "Qwen/Qwen3-0.6B": "http://qwen-vllm:8001",
      "google/gemma-3-1b-it": "http://gemma-vllm:8001",
    }

    with (
      patch.object(gateway, "store", store),
      patch.object(gateway.httpx, "AsyncClient", _AsyncClientStub),
      patch.dict(os.environ, {"SAMPLING_BACKEND": "vllm", "OPEN_RL_VLLM_ROUTES": json.dumps(routes)}, clear=False),
    ):
      _AsyncClientStub.calls.clear()
      result = asyncio.run(
        gateway.asample(
          {
            "sampling_session_id": "tinker://run-a/sampler_weights/current",
            "prompt": {"chunks": [{"tokens": [10, 11]}]},
            "sampling_params": {"max_tokens": 8},
          }
        )
      )

    self.assertIn("request_id", result)
    self.assertEqual(len(_AsyncClientStub.calls), 1)
    url, payload = _AsyncClientStub.calls[0]
    self.assertEqual(url, "http://gemma-vllm:8001/generate")
    self.assertEqual(payload["base_model_id"], "google/gemma-3-1b-it")
    self.assertEqual(payload["lora_id"], "current")
    self.assertEqual(payload["lora_path"], "/mnt/open-rl/peft/run-a/current")
    self.assertEqual(payload["lora_version"], 4)
    self.assertEqual(payload["prompt_token_ids"], [10, 11])
    self.assertEqual(store.futures[result["request_id"]]["type"], "sample")


if __name__ == "__main__":
  unittest.main()
