"""Routed sampling (openrl.sampler_router=llmd): a LoRA job's samples go through
its sampler set's llm-d router, which picks the replica by load and cached prefix.
The set's samplers are stock `vllm serve`; `route` turns a sampling request into
an OpenAI completion and the answer back into a sample."""

import asyncio
import hashlib
import logging
import os
import random
import time
from typing import Any

import httpx

from server.vllm_options import split_stop

logger = logging.getLogger(__name__)

# The label every sampler pod of a set carries, which the set's router selects by.
SAMPLER_SET_LABEL = "openrl.io/sampler-set"
# The set's router pod and Service.
ROUTER_PREFIX = "openrl-router-"
ROUTER_PORT = 8081
ATTEMPT_TIMEOUT = float(os.getenv("OPEN_RL_SAMPLE_ATTEMPT_TIMEOUT", "1800"))
BACKOFF_CAP = 60


def sampler_set(runtime: str) -> str:
  """A label-safe name for a runtime's samplers."""
  return "set-" + hashlib.sha256(runtime.encode()).hexdigest()[:16]


def failed(message: str, category: str = "server") -> dict[str, Any]:
  return {"type": "RequestFailedResponse", "error_message": message, "category": category}


async def route(http: httpx.AsyncClient, url: str, request_id: str, request: dict[str, Any], deadline: float) -> dict[str, Any]:
  """Send a request through the router until it succeeds, fails for good, or
  reaches its deadline. `request["model"]` names the adapter's snapshot
  directory, which vLLM's filesystem resolver loads, or the base model."""
  stop, stop_token_ids = split_stop(request.get("stop"))
  body = {
    "model": request["model"],
    "prompt": request["prompt_token_ids"],
    "n": request["num_samples"],
    "max_tokens": request["max_tokens"],
    # The queue's samplers' defaults, never the model's generation config.
    "temperature": request.get("temperature", 1.0),
    "top_p": request.get("top_p") or 1.0,
    "top_k": request.get("top_k") or -1,
    "stop": stop,
    "stop_token_ids": stop_token_ids,
    "return_token_ids": True,
    "logprobs": 0,
    "prompt_logprobs": 0 if request.get("include_prompt_logprobs") else None,
  }
  attempt = 0
  while (remaining := deadline - time.time()) > 0:
    attempt += 1
    timeout = min(ATTEMPT_TIMEOUT, remaining)
    try:
      # vLLM aborts the generation when the call is dropped.
      response = await http.post(f"{url}/v1/completions", json=body, timeout=timeout)
      if response.status_code == 200:
        return sample(response.json(), request["prompt_token_ids"])
      if response.status_code in (400, 404):
        # A bad request, or an adapter no sampler can find: retrying cannot help.
        return failed(f"Sampler rejected the request: {response.text[:500]}", "user")
      error = f"{response.status_code}: {response.text[:500]}"
    except httpx.HTTPError as exc:
      error = f"{type(exc).__name__}: {exc}"
    delay = random.uniform(0.5, 1) * min(BACKOFF_CAP, 2 ** (attempt - 1))
    logger.info("Retrying %s in %.1fs: %s", request_id, delay, error)
    await asyncio.sleep(min(delay, max(deadline - time.time(), 0)))
  return failed(f"Sampling deadline passed after {attempt} attempt(s)", "user")


def sample(completion: dict[str, Any], prompt: list[int]) -> dict[str, Any]:
  """An OpenAI completion as the sample the queue's samplers return."""
  choices = sorted(completion["choices"], key=lambda choice: choice["index"])
  result = {
    "type": "sample",
    "sequences": [{"tokens": c["token_ids"], "logprobs": c["logprobs"]["token_logprobs"], "stop_reason": c["finish_reason"]} for c in choices],
  }
  if entries := choices[0].get("prompt_logprobs"):
    # Each position maps token IDs to logprobs; the first has none.
    result["prompt_logprobs"] = [None] + [(entry or {}).get(str(token), {}).get("logprob") for token, entry in zip(prompt[1:], entries[1:])]
  return result
