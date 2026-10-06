"""Sample LoRA models through an llm-d router in front of a shared vLLM pool.

The router sends each request to the replica that already holds the longest
cached prefix, so an agent's turns keep landing where its history is cached.
Replicas load adapters by name with vLLM's filesystem LoRA resolver, which
looks for <cache dir>/<name>/adapter_config.json.
"""

import asyncio
import hashlib
import json
import os
from typing import Any

import httpx

from server.vllm_options import split_stop

RETRIES = 3


def routers() -> dict[str, str]:
  """Base model to router URL, from OPEN_RL_LLMD_ROUTERS (a JSON object)."""
  raw = os.getenv("OPEN_RL_LLMD_ROUTERS", "")
  return json.loads(raw) if raw else {}


def router_for(base_model: str | None) -> str | None:
  return routers().get(base_model or "")


def adapter_name(model_id: str) -> str:
  """A sampling ref like tinker://<id>/sampler_weights/000031 as one path segment.
  Each saved version gets its own name, so replicas never serve a stale adapter."""
  digest = hashlib.sha256(model_id.encode()).hexdigest()[:16]
  return f"{model_id.split('://')[-1].split('/')[0]}-{digest}"


def publish_adapter(peft_dir: str, name: str, cache_dir: str) -> bool:
  """Point <cache_dir>/<name> at the saved adapter. False before the first save."""
  if not os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
    return False
  link = os.path.join(cache_dir, name)
  if not os.path.lexists(link):
    os.makedirs(cache_dir, exist_ok=True)
    try:
      os.symlink(peft_dir, link)
    except FileExistsError:
      pass
  return True


def completion_body(model: str, request: dict[str, Any]) -> dict[str, Any]:
  stop, stop_token_ids = split_stop(request.get("stop"))
  body = {
    "model": model,
    "prompt": request["prompt_token_ids"],
    "max_tokens": request["max_tokens"],
    "n": request.get("num_samples", 1),
    "temperature": request.get("temperature", 1.0),
    "top_p": request.get("top_p", 1.0),
    "top_k": request.get("top_k", -1),
    "logprobs": 1,
    "return_token_ids": True,
  }
  if stop:
    body["stop"] = stop
  if stop_token_ids:
    body["stop_token_ids"] = stop_token_ids
  # Asking for prompt logprobs makes vLLM skip the prefix cache.
  if request.get("include_prompt_logprobs"):
    body["prompt_logprobs"] = 1
  return body


def sample_response(completion: dict[str, Any], prompt_token_ids: list[int]) -> dict[str, Any]:
  """The worker's sample result shape from a /v1/completions response."""
  choices = sorted(completion["choices"], key=lambda choice: choice["index"])
  result: dict[str, Any] = {
    "type": "sample",
    "sequences": [
      {"tokens": choice["token_ids"], "logprobs": choice["logprobs"]["token_logprobs"], "stop_reason": choice["finish_reason"]} for choice in choices
    ],
  }
  prompt_logprobs = choices[0].get("prompt_logprobs") if choices else None
  if prompt_logprobs is not None:
    result["prompt_logprobs"] = [
      None if candidates is None or str(token) not in candidates else candidates[str(token)]["logprob"]
      for token, candidates in zip(prompt_token_ids, prompt_logprobs)
    ]
  return result


async def sample(client: httpx.AsyncClient, url: str, body: dict[str, Any]) -> dict[str, Any]:
  """Retries cover a replica restarting or the router briefly having no endpoint."""
  for attempt in range(RETRIES):
    try:
      response = await client.post(f"{url}/v1/completions", json=body)
      if response.status_code < 400:
        return sample_response(response.json(), body["prompt"])
      error: Exception = RuntimeError(f"llm-d router returned {response.status_code}: {response.text[:500]}")
      if response.status_code < 500:
        raise error
    except httpx.TransportError as exc:
      error = exc
    if attempt < RETRIES - 1:
      await asyncio.sleep(2**attempt)
  raise error
