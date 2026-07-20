# Sampling through an externally managed, stock `vllm serve` endpoint.
#
# Set SAMPLER_BASE_URL on the gateway and launch vLLM yourself:
#
#   VLLM_ALLOW_RUNTIME_LORA_UPDATING=true vllm serve <base-model> \
#     --port 8001 --enable-lora --max-lora-rank 64
#
# The gateway then never spawns sampler workers; asample calls translate to
# the OpenAI-compatible API. LoRA adapters are registered on demand through
# /v1/load_lora_adapter using the snapshot's unique session name, so a
# long-lived server never serves stale adapter weights. The vLLM process must
# see the same filesystem paths as the trainer (adapters live under
# $OPEN_RL_TMP_DIR/peft). Full fine-tuning is NOT supported through this path:
# stock vLLM has no checkpoint hot-reload; FFT keeps the managed queue workers.

import asyncio
import os

import httpx

_served_model: str | None = None
_served_max_model_len: int | None = None
_loaded_adapters: set[str] = set()
_latest_adapter_for_model: dict[str, str] = {}
_adapter_lock = asyncio.Lock()


def get_sampler_base_url() -> str | None:
  url = os.getenv("SAMPLER_BASE_URL")
  return url.rstrip("/") if url else None


def _client() -> httpx.AsyncClient:
  # Generation at long context is slow; only connecting has a tight deadline.
  return httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0))


async def preflight(base_url: str) -> None:
  """Fail fast at startup if SAMPLER_BASE_URL points at nothing usable."""
  try:
    async with _client() as client:
      resp = await client.get(f"{base_url}/health", timeout=5.0)
      resp.raise_for_status()
      # Resolve the served model now so a broken /v1/models (or an adapter-only
      # listing) fails at startup instead of on the first sample call.
      await _served_model_name(client, base_url)
  except Exception as exc:
    raise RuntimeError(
      f"SAMPLER_BASE_URL={base_url} is set but no vLLM server responds on {base_url}/health. "
      "Launch one first, e.g.: VLLM_ALLOW_RUNTIME_LORA_UPDATING=true "
      "vllm serve <base-model> --port 8001 --enable-lora --max-lora-rank 64"
    ) from exc


async def _served_model_name(client: httpx.AsyncClient, base_url: str) -> str:
  global _served_model, _served_max_model_len
  if _served_model is None:
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    models = resp.json()["data"]
    # Runtime-loaded LoRA adapters appear as extra entries without a
    # max_model_len; the base model is the one that has it.
    base_entries = [m for m in models if m.get("max_model_len")]
    model_data = base_entries[0] if base_entries else models[0]
    _served_model = model_data["id"]
    _served_max_model_len = model_data.get("max_model_len")
  return _served_model


async def _ensure_adapter_loaded(client: httpx.AsyncClient, base_url: str, name: str, path: str, base_model_id: str | None) -> None:
  if name in _loaded_adapters:
    return
  async with _adapter_lock:
    if name in _loaded_adapters:
      return
    resp = await client.post(f"{base_url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
    if resp.status_code != 200 and "already been loaded" not in resp.text:
      raise RuntimeError(
        f"Loading LoRA adapter {name!r} from {path!r} failed ({resp.status_code}): {resp.text}. "
        "Is the vLLM server running with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=true, "
        "and does it share the trainer's filesystem?"
      )
    _loaded_adapters.add(name)

    # Each sampler snapshot registers under a fresh name; retire the previous
    # snapshot's adapter so a long training run does not accumulate them.
    if base_model_id:
      previous = _latest_adapter_for_model.get(base_model_id)
      if previous and previous != name:
        try:
          await client.post(f"{base_url}/v1/unload_lora_adapter", json={"lora_name": previous})
        except Exception:
          pass
        _loaded_adapters.discard(previous)
      _latest_adapter_for_model[base_model_id] = name


def _token_id(token: str) -> int:
  # With return_tokens_as_token_ids, tokens arrive as "token_id:<n>".
  return int(token.rsplit(":", 1)[-1])


async def sample(req: dict) -> dict:
  """Serve one internal sampling request via the OpenAI-compatible API.

  Takes the same request dict the queue workers consume and returns the same
  result shape ({"type": "sample", "sequences": [...]}), so callers cannot
  tell which sampler backend ran.
  """
  base_url = get_sampler_base_url()
  assert base_url, "external_sampler.sample requires SAMPLER_BASE_URL"

  async with _client() as client:
    model = await _served_model_name(client, base_url)
    lora_id, lora_path = req.get("lora_id"), req.get("lora_path")
    if lora_id and lora_path:
      await _ensure_adapter_loaded(client, base_url, lora_id, lora_path, req.get("model_id"))
      model = lora_id

    # Same context-ceiling handling as the queue workers: a prompt that leaves
    # no room returns an empty length-stop completion (the cookbook scores it
    # as context overflow), and an overhanging max_tokens is clamped — stock
    # vLLM would otherwise reject either with a 400.
    prompt_token_ids = req.get("prompt_token_ids", [])
    max_tokens = req.get("max_tokens", 20)
    num_samples = req.get("num_samples", 1)
    if _served_max_model_len is not None:
      prompt_len = len(prompt_token_ids)
      if prompt_len >= _served_max_model_len or max_tokens <= 0:
        print(
          f"[external-sampler] Prompt of {prompt_len} tokens leaves no room in max_model_len={_served_max_model_len} "
          f"(max_tokens={max_tokens}); returning empty length-stop truncation."
        )
        return {"type": "sample", "sequences": [{"tokens": [], "logprobs": [], "stop_reason": "length"} for _ in range(num_samples)]}
      if prompt_len + max_tokens > _served_max_model_len:
        max_tokens = _served_max_model_len - prompt_len

    body: dict = {
      "model": model,
      "prompt": prompt_token_ids,
      "max_tokens": max_tokens,
      "temperature": req.get("temperature", 1.0),
      "top_p": req.get("top_p", 1.0),
      "n": req.get("num_samples", 1),
      "logprobs": 1,
      "return_tokens_as_token_ids": True,
    }
    top_k = req.get("top_k", -1)
    if top_k not in (None, -1):
      body["top_k"] = top_k
    if req.get("stop"):
      body["stop_token_ids"] = req["stop"]
    if req.get("include_prompt_logprobs"):
      body["prompt_logprobs"] = 0

    resp = await client.post(f"{base_url}/v1/completions", json=body)
    if resp.status_code in (400, 404) and lora_id and lora_path and lora_id in resp.text:
      # The server no longer knows this adapter (e.g. vllm serve restarted and
      # our loaded-adapter cache went stale). Re-register once and retry.
      _loaded_adapters.discard(lora_id)
      await _ensure_adapter_loaded(client, base_url, lora_id, lora_path, req.get("model_id"))
      resp = await client.post(f"{base_url}/v1/completions", json=body)
    if resp.status_code != 200:
      raise RuntimeError(f"External sampler /v1/completions returned {resp.status_code}: {resp.text}")
    data = resp.json()

    sequences = []
    prompt_logprobs_out = None
    for choice in data.get("choices", []):
      logprobs = choice.get("logprobs") or {}
      sequences.append(
        {
          "tokens": [_token_id(t) for t in logprobs.get("tokens", [])],
          "logprobs": logprobs.get("token_logprobs") or [],
          "stop_reason": choice.get("finish_reason"),
        }
      )
      if choice.get("prompt_logprobs") is not None and prompt_logprobs_out is None:
        prompt_ids = req.get("prompt_token_ids", [])
        prompt_logprobs_out = [
          (entry.get(str(token_id)) or {}).get("logprob") if isinstance(entry, dict) else None
          for token_id, entry in zip(prompt_ids, choice["prompt_logprobs"])
        ]

    result: dict = {"type": "sample", "sequences": sequences}
    if prompt_logprobs_out is not None:
      result["prompt_logprobs"] = prompt_logprobs_out
    return result
