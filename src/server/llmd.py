"""Sample LoRA models through an llm-d router in front of a shared vLLM pool.

The router sends each request to the replica that already holds the longest
cached prefix, so an agent's turns keep landing where its history is cached.
Replicas load adapters by name with vLLM's filesystem LoRA resolver, which
looks for <cache dir>/<name>/adapter_config.json. The trainer overwrites one
PEFT directory on every save, so each sampler version is copied out under its
own name and deleted once a newer version is in use and it has no requests.
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
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


def job_of(ref: str) -> str:
  """The training job a model id or sampling ref belongs to."""
  return ref.split("://")[-1].split("/")[0]


def adapter_name(model_id: str) -> str:
  """A sampling ref like tinker://<id>/sampler_weights/000031 as one path segment.
  Each saved version gets its own name, so replicas never serve a stale adapter."""
  digest = hashlib.sha256(model_id.encode()).hexdigest()[:16]
  return f"{job_of(model_id)}-{digest}"


def snapshot_adapter(peft_dir: str, cache_dir: str, name: str, aliases: tuple[str, ...] = ()) -> bool:
  """Copy the saved adapter to <cache_dir>/<name>, and link each alias to it.
  False before the first save. The copy is renamed into place, so a replica
  never sees a partial one."""
  if not os.path.exists(os.path.join(peft_dir, "adapter_config.json")):
    return False
  target = os.path.join(cache_dir, name)
  if not os.path.exists(target):
    os.makedirs(cache_dir, exist_ok=True)
    staging = tempfile.mkdtemp(prefix=f".{name}-", dir=cache_dir)
    try:
      shutil.copytree(peft_dir, staging, dirs_exist_ok=True)
      os.rename(staging, target)
    except OSError:
      shutil.rmtree(staging, ignore_errors=True)
      if not os.path.exists(target):
        raise
  for alias in aliases:
    link = os.path.join(cache_dir, alias)
    if alias != name and not os.path.lexists(link):
      try:
        os.symlink(name, link)
      except FileExistsError:
        pass
  return True


def remove_adapter(cache_dir: str, names: list[str]) -> None:
  for name in names:
    path = os.path.join(cache_dir, name)
    if os.path.islink(path):
      os.unlink(path)
    else:
      shutil.rmtree(path, ignore_errors=True)


class AdapterVersions:
  """Each job's sampler versions in save order, and the requests in flight on
  each. A version can go once a newer one exists and nothing samples it. Kept
  in the gateway's memory, which assumes one gateway replica."""

  def __init__(self) -> None:
    # job -> versions in save order; a version is its names, the first one the directory.
    self.versions: dict[str, list[list[str]]] = defaultdict(list)
    self.in_flight: dict[str, int] = defaultdict(int)
    self.deleted_jobs: set[str] = set()

  def known(self, name: str) -> bool:
    return any(name in version for versions in self.versions.values() for version in versions)

  def add(self, job: str, names: list[str]) -> list[str]:
    """Record a newly saved version. Returns names that can be deleted now."""
    self.deleted_jobs.discard(job)
    if not any(set(names) & set(version) for version in self.versions[job]):
      self.versions[job].append(list(names))
    return self.collect(job)

  def start(self, name: str) -> None:
    self.in_flight[name] += 1

  def finish(self, job: str, name: str) -> list[str]:
    self.in_flight[name] -= 1
    if self.in_flight[name] <= 0:
      del self.in_flight[name]
    return self.collect(job)

  def remove_job(self, job: str) -> list[str]:
    """The job is deleted: every version can go once idle."""
    self.deleted_jobs.add(job)
    return self.collect(job)

  def collect(self, job: str) -> list[str]:
    versions = self.versions.get(job, [])
    keep_latest = 0 if job in self.deleted_jobs else 1
    removable, kept = [], []
    for index, version in enumerate(versions):
      superseded = index < len(versions) - keep_latest
      if superseded and not any(self.in_flight.get(name) for name in version):
        removable.extend(version[1:] + version[:1])  # links before the directory they point at
      else:
        kept.append(version)
    if kept:
      self.versions[job] = kept
    else:
      self.versions.pop(job, None)
    return removable


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
