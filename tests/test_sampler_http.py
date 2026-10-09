"""The sampler's HTTP contract, which the set's dispatcher relies on to decide
between a final result and another attempt. The real app; only vLLM's generate
is replaced."""

import asyncio
import time
import unittest
from types import SimpleNamespace

import httpx

from server import sampler_http


class Engine:
  def __init__(self, error: Exception | None = None, hang: bool = False) -> None:
    self.engine = SimpleNamespace(errored=False)
    self.error = error
    self.hang = hang
    self.cancelled = False

  async def generate(self, request: dict) -> dict:
    try:
      if self.hang:
        await asyncio.sleep(3600)
      if self.error is not None:
        raise self.error
      return {"sequences": [{"tokens": request["prompt_token_ids"][-1:], "logprobs": [-0.5], "stop_reason": "length"}]}
    except asyncio.CancelledError:
      self.cancelled = True
      raise


class SamplerHttpTest(unittest.IsolatedAsyncioTestCase):
  async def post(self, engine: Engine, **openrl) -> httpx.Response:
    request = {"request_id": "r#1", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "num_samples": 1, **openrl}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=sampler_http.http_app(engine)), base_url="http://sampler") as client:
      return await client.post("/v1/completions", json={"model": "m", "prompt": [1, 2, 3], "max_tokens": 2, "openrl": request})

  async def test_a_sample_is_returned_with_its_type(self) -> None:
    response = await self.post(Engine())
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()["type"], "sample")
    self.assertEqual(response.json()["sequences"][0]["tokens"], [3])

  async def test_a_request_that_can_never_succeed_is_a_400(self) -> None:
    response = await self.post(Engine(error=ValueError("prompt too long")))
    self.assertEqual(response.status_code, 400)
    self.assertIn("prompt too long", response.json()["error_message"])

  async def test_a_temporary_failure_is_retryable(self) -> None:
    self.assertEqual((await self.post(Engine(error=RuntimeError("engine hiccup")))).status_code, 500)
    dead = Engine()
    dead.engine.errored = True
    self.assertEqual((await self.post(dead)).status_code, 503)

  async def test_the_attempt_deadline_cancels_generation(self) -> None:
    engine = Engine(hang=True)
    started = time.monotonic()
    response = await self.post(engine, deadline=time.time() + 0.5)
    self.assertEqual(response.status_code, 504)
    self.assertLess(time.monotonic() - started, 5)
    self.assertTrue(engine.cancelled)


if __name__ == "__main__":
  unittest.main()
