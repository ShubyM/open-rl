import json
import unittest
from unittest.mock import patch

import httpx

from server import external_sampler


def _mock_vllm_serve(state: dict):
  """A stock-vllm-serve stand-in: /v1/models, /v1/load_lora_adapter, /v1/completions."""

  def handler(request: httpx.Request) -> httpx.Response:
    state.setdefault("requests", []).append((request.url.path, request.content))
    if request.url.path == "/v1/models":
      return httpx.Response(200, json={"data": [{"id": "base-model", "max_model_len": state.get("max_model_len")}]})
    if request.url.path == "/v1/load_lora_adapter":
      state.setdefault("loaded", []).append(json.loads(request.content)["lora_name"])
      return httpx.Response(200, json={})
    if request.url.path == "/v1/unload_lora_adapter":
      state.setdefault("unloaded", []).append(json.loads(request.content)["lora_name"])
      return httpx.Response(200, json={})
    if request.url.path == "/v1/completions":
      body = json.loads(request.content)
      state["completion_body"] = body
      if body["model"] in state.get("forgotten_adapters", set()) and body["model"] not in state.get("loaded", []):
        return httpx.Response(404, text=f"The model `{body['model']}` does not exist.")
      return httpx.Response(
        200,
        json={
          "choices": [
            {
              "finish_reason": "stop",
              "logprobs": {"tokens": ["token_id:11", "token_id:7"], "token_logprobs": [-0.5, -1.25]},
            }
          ]
        },
      )
    return httpx.Response(404, text="not found")

  return httpx.MockTransport(handler)


class ExternalSamplerTest(unittest.IsolatedAsyncioTestCase):
  def setUp(self) -> None:
    external_sampler._served_model = None
    external_sampler._served_max_model_len = None
    external_sampler._loaded_adapters.clear()
    external_sampler._latest_adapter_for_model.clear()
    self.state: dict = {}
    transport = _mock_vllm_serve(self.state)
    self._client_patch = patch.object(external_sampler, "_client", lambda: httpx.AsyncClient(transport=transport))
    self._client_patch.start()
    self.addCleanup(self._client_patch.stop)
    self._env_patch = patch.dict("os.environ", {"SAMPLER_BASE_URL": "http://sampler:8001"})
    self._env_patch.start()
    self.addCleanup(self._env_patch.stop)

  async def test_lora_request_loads_adapter_and_parses_token_ids(self) -> None:
    result = await external_sampler.sample(
      {
        "prompt_token_ids": [1, 2, 3],
        "max_tokens": 8,
        "temperature": 0.0,
        "num_samples": 1,
        "lora_id": "tinker://m1/sampler_weights/sampler-1",
        "lora_path": "/tmp/open-rl/peft/m1/m1",
        "model_id": "m1",
      }
    )
    self.assertEqual(self.state["loaded"], ["tinker://m1/sampler_weights/sampler-1"])
    self.assertEqual(self.state["completion_body"]["model"], "tinker://m1/sampler_weights/sampler-1")
    self.assertTrue(self.state["completion_body"]["return_tokens_as_token_ids"])
    self.assertEqual(result["type"], "sample")
    self.assertEqual(result["sequences"][0]["tokens"], [11, 7])
    self.assertEqual(result["sequences"][0]["logprobs"], [-0.5, -1.25])
    self.assertEqual(result["sequences"][0]["stop_reason"], "stop")

  async def test_new_snapshot_retires_previous_adapter(self) -> None:
    base = {"prompt_token_ids": [1], "max_tokens": 4, "model_id": "m1", "lora_path": "/tmp/open-rl/peft/m1/m1"}
    await external_sampler.sample({**base, "lora_id": "snap-1"})
    await external_sampler.sample({**base, "lora_id": "snap-2"})
    self.assertEqual(self.state["loaded"], ["snap-1", "snap-2"])
    self.assertEqual(self.state["unloaded"], ["snap-1"])

  async def test_stale_adapter_cache_reregisters_and_retries(self) -> None:
    """A restarted vllm serve forgets dynamically loaded adapters; the client
    must re-register and retry instead of failing the sample."""
    external_sampler._loaded_adapters.add("snap-1")  # stale: server never saw it
    self.state["forgotten_adapters"] = {"snap-1"}
    result = await external_sampler.sample(
      {"prompt_token_ids": [1], "max_tokens": 4, "lora_id": "snap-1", "lora_path": "/tmp/open-rl/peft/m1/m1", "model_id": "m1"}
    )
    self.assertEqual(self.state["loaded"], ["snap-1"])
    self.assertEqual(result["sequences"][0]["tokens"], [11, 7])

  async def test_overhanging_max_tokens_is_clamped_to_model_len(self) -> None:
    self.state["max_model_len"] = 10
    await external_sampler.sample({"prompt_token_ids": [1] * 6, "max_tokens": 100})
    self.assertEqual(self.state["completion_body"]["max_tokens"], 4)

  async def test_overflowing_prompt_returns_length_stop_without_calling_vllm(self) -> None:
    self.state["max_model_len"] = 4
    result = await external_sampler.sample({"prompt_token_ids": [1] * 8, "max_tokens": 16, "num_samples": 2})
    self.assertNotIn("completion_body", self.state)
    self.assertEqual(
      result["sequences"],
      [{"tokens": [], "logprobs": [], "stop_reason": "length"}, {"tokens": [], "logprobs": [], "stop_reason": "length"}],
    )

  async def test_base_model_request_skips_adapter_machinery(self) -> None:
    result = await external_sampler.sample({"prompt_token_ids": [1, 2], "max_tokens": 4, "lora_id": "m1", "lora_path": None})
    self.assertNotIn("loaded", self.state)
    self.assertEqual(self.state["completion_body"]["model"], "base-model")
    self.assertEqual(result["sequences"][0]["tokens"], [11, 7])


if __name__ == "__main__":
  unittest.main()
