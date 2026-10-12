"""Routed sampling: a LoRA job's samples go through its sampler set's llm-d router,
which picks the replica by load and cached prefix. The set's samplers are stock
`vllm serve`; `route` turns a sampling request into an OpenAI completion and the
answer back into a sample.

openrl.sampler_router=llmd: the API server calls `route` itself.

openrl.sampler_router=llmd-stream: requests wait in the set's Redis Stream, and
the set's dispatcher (`python -m server.sampler_routing`) calls `route`. The API
appends each request with XADD; the entry's ID is in the request ID. The
dispatcher reads entries through a consumer group, so each stays pending until its
result is saved in one MULTI: SET NX the result, XACK, XDEL. The first result wins
and the stream holds only unfinished requests. A set has one dispatcher, always
under one consumer name, so a restarted one first reads back the entries it took
but never finished. The worker manager creates the stream with the set and
deletes it with the set, and XADD never creates it, so a removed set refuses new
requests. The Redis clients decode responses.
"""

import asyncio
import hashlib
import json
import logging
import os
import random
import time
from typing import Any

import httpx
import redis.asyncio as redis
from redis.exceptions import ResponseError

from server.vllm_options import split_stop

logger = logging.getLogger(__name__)

# The label every sampler pod of a set carries, which the set's router selects by.
SAMPLER_SET_LABEL = "openrl.io/sampler-set"
# The set's router pod and Service.
ROUTER_PREFIX = "openrl-router-"
ROUTER_PORT = 8081
ATTEMPT_TIMEOUT = float(os.getenv("OPEN_RL_SAMPLE_ATTEMPT_TIMEOUT", "1800"))
BACKOFF_CAP = 60

GROUP = "dispatch"
CONSUMER = "dispatcher"
ID_PREFIX = "sampled"
RESULT_TTL = int(os.getenv("OPEN_RL_SAMPLE_RESULT_TTL", "1800"))
ACTIVE = int(os.getenv("OPEN_RL_DISPATCH_ACTIVE", "256"))


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


# *** llmd-stream ***


# All of a set's keys share the hash tag {set}, so each MULTI stays on one node.
def stream_key(set_id: str) -> str:
  return f"openrl:{{{set_id}}}:stream"


def result_key(set_id: str, entry_id: str) -> str:
  return f"openrl:{{{set_id}}}:result:{entry_id}"


def parse(request_id: str) -> tuple[str, str] | None:
  """The set and stream entry of a stream request ID, or None for any other ID."""
  parts = request_id.split(":")
  return (parts[1], parts[2]) if len(parts) == 3 and parts[0] == ID_PREFIX and parts[1] and parts[2] else None


def create_set(client: Any, set_id: str) -> None:
  """From the worker manager, with a synchronous client. The group reads from the
  start, so nothing accepted before the dispatcher starts is missed."""
  try:
    client.xgroup_create(stream_key(set_id), GROUP, id="0", mkstream=True)
  except ResponseError as exc:
    if "BUSYGROUP" not in str(exc):
      raise


def remove_set(client: Any, set_id: str) -> None:
  """Drop the set's unfinished requests; their IDs then read as not found."""
  client.delete(stream_key(set_id))


async def accept(client: Any, set_id: str, request: dict[str, Any], deadline_seconds: float) -> str | None:
  """Store a request and return its ID, or None if the set was removed."""
  stored = json.dumps({**request, "deadline": time.time() + deadline_seconds})
  entry_id = await client.xadd(stream_key(set_id), {"input": stored}, nomkstream=True)
  return f"{ID_PREFIX}:{set_id}:{entry_id}" if entry_id else None


async def result(client: Any, request_id: str, wait: float = 50) -> dict[str, Any]:
  """The saved result, or a failure for an unknown or expired ID. A pending poll
  is held for most of the SDK's 60 s request timeout, then answered try_again."""
  set_id, entry_id = parse(request_id)
  deadline = time.monotonic() + wait
  while True:
    # The stream first: finishing saves the result and deletes the entry at once,
    # so a missing entry means the result, if any, is already readable.
    pending = await client.xrange(stream_key(set_id), entry_id, entry_id)
    if (raw := await client.get(result_key(set_id, entry_id))) is not None:
      return json.loads(raw)
    if not pending:
      return failed("Sample request not found, or its result expired", "user")
    if time.monotonic() > deadline:
      return {"type": "try_again"}
    await asyncio.sleep(0.25)


async def dispatch(client: Any, set_id: str, router_url: str, http: httpx.AsyncClient, active: int = ACTIVE) -> None:
  """Send the set's requests, at most `active` at once, counting those waiting to retry."""
  running: set[asyncio.Task] = set()
  start = "0"  # this consumer's unfinished entries first, then new ones
  try:
    while True:
      if len(running) >= active:
        await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
        continue
      try:
        if start == "0":
          reply = await client.xreadgroup(GROUP, CONSUMER, {stream_key(set_id): "0"})
        else:
          reply = await client.xreadgroup(GROUP, CONSUMER, {stream_key(set_id): ">"}, count=active - len(running), block=1000)
      except redis.RedisError as exc:
        logger.warning("Cannot read set %s: %s", set_id, exc)
        await asyncio.sleep(1)
        continue
      start = ">"
      for _, entries in reply or []:
        for entry_id, fields in entries:
          task = asyncio.create_task(send(client, set_id, entry_id, json.loads(fields["input"]), router_url, http))
          running.add(task)
          task.add_done_callback(running.discard)
  finally:
    for task in list(running):
      task.cancel()


async def send(client: Any, set_id: str, entry_id: str, request: dict[str, Any], router_url: str, http: httpx.AsyncClient) -> None:
  outcome = await route(http, router_url, f"{ID_PREFIX}:{set_id}:{entry_id}", request, request["deadline"])
  while True:
    try:
      async with client.pipeline(transaction=True) as pipe:
        pipe.set(result_key(set_id, entry_id), json.dumps(outcome), nx=True, ex=RESULT_TTL)
        pipe.xack(stream_key(set_id), GROUP, entry_id)
        pipe.xdel(stream_key(set_id), entry_id)
        await pipe.execute()
      return
    except redis.RedisError as exc:
      logger.warning("Cannot save the result of %s in %s: %s", entry_id, set_id, exc)
      await asyncio.sleep(1)


async def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
  # One line per sample request is noise; failures and retries are logged above.
  logging.getLogger("httpx").setLevel(logging.WARNING)
  client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
  async with httpx.AsyncClient(limits=httpx.Limits(max_connections=ACTIVE)) as http:
    await dispatch(client, os.environ["OPEN_RL_SAMPLER_SET"], os.getenv("OPEN_RL_ROUTER_URL", "http://127.0.0.1:8081"), http)


if __name__ == "__main__":
  asyncio.run(main())
