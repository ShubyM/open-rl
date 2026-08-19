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
# OPEN_RL_SNAPSHOT_DIR, which defaults to $OPEN_RL_TMP_DIR/peft and may be
# tmpfs -- see src/training/paths.py, and note that tmpfs is node-local, so a
# vLLM server off-box needs that root on a shared filesystem instead).
# Full fine-tuning is NOT supported through this path:
# stock vLLM has no checkpoint hot-reload; FFT keeps the managed queue workers.

import array
import asyncio
import os
import zlib

import httpx

_served_model: dict[str, str] = {}
_served_max_model_len: dict[str, int | None] = {}
_loaded_adapters: dict[str, set[str]] = {}
_latest_adapter_for_model: dict[str, dict[str, str]] = {}
_adapter_lock = asyncio.Lock()


def get_sampler_base_urls() -> list[str]:
  """All external sampler instances. SAMPLER_BASE_URLS (comma-separated,
  one single-GPU server each) enables prefix-affinity routing; a lone
  SAMPLER_BASE_URL keeps the single-endpoint behavior."""
  urls = os.getenv("SAMPLER_BASE_URLS") or os.getenv("SAMPLER_BASE_URL") or ""
  return [u.strip().rstrip("/") for u in urls.split(",") if u.strip()]


def get_sampler_base_url() -> str | None:
  urls = get_sampler_base_urls()
  return urls[0] if urls else None


# Episode routing table, registered at RESPONSE time: after an instance
# serves a turn, (len, crc) of prompt+completion — exactly the prefix the
# rollout's next turn starts with — maps to that instance. Registering on the
# request side is wrong twice over: GRPO groups have byte-identical first
# prompts (all rollouts of a group would collapse onto one instance), and no
# fixed hash window separates tasks either — measured on the LAB recipe,
# tasks share up to 6,413 of their ~6,434 first-turn tokens. First turns
# therefore match nothing and spread least-loaded; every later turn follows
# the instance holding its own rollout's KV cache.
_episode_routes: dict[int, dict[int, str]] = {}  # length -> {crc -> url}
_route_order: list[tuple[int, int]] = []  # insertion order for eviction
_inflight: dict[str, int] = {}
_placement_counter = 0
_MAX_ROUTES = 4096


# Hash the token ids as fixed-width binary, not as their repr. The old
# `bytes(str(token_ids[:length]), "utf-8")` built a decimal string of the whole
# prefix on every call: ~1.9 ms at 40k tokens, and pick_base_url called it once
# per registered prefix length. With a few hundred lengths in the table that is
# 100-500 ms of synchronous CPU per routing decision, and _MAX_ROUTES=4096
# allows ~3 s. It runs on the gateway's event loop, so 48 concurrent rollouts
# serialize behind it -- which is what made AFFINITY=1 unstable: redis reads and
# 30 s TCP connects to vLLM both timed out while redis and vLLM were each
# answering in ~1 ms, and it worsened as the table grew. The single-URL path
# returns above, so AFFINITY=0 never hit this.
_TOKEN_WIDTH = array.array("q").itemsize


def _token_bytes(token_ids: list[int]) -> bytes:
  return array.array("q", token_ids).tobytes()


def _crc(token_ids: list[int], length: int) -> int:
  return zlib.crc32(_token_bytes(token_ids[:length]))


def pick_base_url(prompt_token_ids: list[int]) -> str:
  """Route a turn to the instance that served this rollout's previous turn
  (KV/prefix cache reuse); unrecognized prompts go to the least-loaded
  instance."""
  urls = get_sampler_base_urls()
  assert urls, "external sampler requires SAMPLER_BASE_URL(S)"
  if len(urls) == 1:
    return urls[0]
  length = len(prompt_token_ids)
  # One pass over the prompt covers every candidate length: crc32 chains, so
  # extending the running crc by the next slice equals hashing that whole
  # prefix from scratch. Cost is O(len(prompt)) for the table instead of
  # O(sum of all registered lengths).
  candidates_by_len = sorted(prefix_len for prefix_len in _episode_routes if prefix_len <= length)
  if candidates_by_len:
    buf = _token_bytes(prompt_token_ids)
    running = 0
    consumed = 0
    crc_at: list[tuple[int, int]] = []
    for prefix_len in candidates_by_len:
      running = zlib.crc32(buf[consumed * _TOKEN_WIDTH : prefix_len * _TOKEN_WIDTH], running)
      crc_at.append((prefix_len, running))
      consumed = prefix_len
    # Longest match wins. The old code took whichever length dict iteration
    # reached first, which on this recipe is a real hazard: LAB tasks share up
    # to 6,413 of their ~6,434 first-turn tokens, so a short prefix can match a
    # different rollout and send the turn to an instance holding none of its KV.
    for prefix_len, crc in reversed(crc_at):
      url = _episode_routes[prefix_len].get(crc)
      if url is not None and url in urls:
        return url
  global _placement_counter
  low = min(_inflight.get(u, 0) for u in urls)
  candidates = [u for u in urls if _inflight.get(u, 0) == low]
  url = candidates[_placement_counter % len(candidates)]
  _placement_counter += 1
  return url


def register_route(prefix_token_ids: list[int], url: str) -> None:
  """Remember that a rollout whose next turn starts with this prefix should
  return to `url`."""
  length = len(prefix_token_ids)
  key = _crc(prefix_token_ids, length)
  _episode_routes.setdefault(length, {})[key] = url
  _route_order.append((length, key))
  if len(_route_order) > _MAX_ROUTES:
    old_len, old_key = _route_order.pop(0)
    _episode_routes.get(old_len, {}).pop(old_key, None)


def drop_routes_for(url: str) -> None:
  """Forget every route pinned to an instance that just failed a request, so
  its episodes re-place onto healthy instances instead of retrying into it."""
  for length, by_crc in _episode_routes.items():
    stale = [k for k, u in by_crc.items() if u == url]
    for k in stale:
      del by_crc[k]
  _route_order[:] = [(length, k) for length, k in _route_order if k in _episode_routes.get(length, {})]


def _client() -> httpx.AsyncClient:
  # Generation at long context is slow; only connecting has a tight deadline.
  read_timeout = float(os.getenv("OPEN_RL_SAMPLER_TIMEOUT", "3600"))
  return httpx.AsyncClient(timeout=httpx.Timeout(read_timeout, connect=30.0))


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
  if base_url not in _served_model:
    resp = await client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    models = resp.json()["data"]
    # Runtime-loaded LoRA adapters appear as extra entries without a
    # max_model_len; the base model is the one that has it.
    base_entries = [m for m in models if m.get("max_model_len")]
    model_data = base_entries[0] if base_entries else models[0]
    _served_model[base_url] = model_data["id"]
    _served_max_model_len[base_url] = model_data.get("max_model_len")
  return _served_model[base_url]


async def _ensure_adapter_loaded(client: httpx.AsyncClient, base_url: str, name: str, path: str, base_model_id: str | None) -> None:
  loaded = _loaded_adapters.setdefault(base_url, set())
  if name in loaded:
    return
  async with _adapter_lock:
    if name in loaded:
      return
    resp = await client.post(f"{base_url}/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
    if resp.status_code != 200 and "already been loaded" not in resp.text:
      raise RuntimeError(
        f"Loading LoRA adapter {name!r} from {path!r} failed ({resp.status_code}): {resp.text}. "
        "Is the vLLM server running with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=true, "
        "and does it share the trainer's filesystem?"
      )
    loaded.add(name)

    # Each sampler snapshot registers under a fresh name; retire the previous
    # snapshot's adapter so a long training run does not accumulate them.
    if base_model_id:
      latest = _latest_adapter_for_model.setdefault(base_url, {})
      previous = latest.get(base_model_id)
      if previous and previous != name:
        try:
          await client.post(f"{base_url}/v1/unload_lora_adapter", json={"lora_name": previous})
        except Exception:
          pass
        loaded.discard(previous)
      latest[base_model_id] = name


def _token_id(token: str) -> int:
  # With return_tokens_as_token_ids, tokens arrive as "token_id:<n>".
  return int(token.rsplit(":", 1)[-1])


async def sample(req: dict) -> dict:
  """Serve one internal sampling request via the OpenAI-compatible API.

  Takes the same request dict the queue workers consume and returns the same
  result shape ({"type": "sample", "sequences": [...]}), so callers cannot
  tell which sampler backend ran.
  """
  prompt_token_ids = req.get("prompt_token_ids", [])
  base_url = pick_base_url(prompt_token_ids)
  _inflight[base_url] = _inflight.get(base_url, 0) + 1
  try:
    result = await _sample_on(base_url, req)
  except Exception:
    drop_routes_for(base_url)
    raise
  finally:
    _inflight[base_url] = _inflight.get(base_url, 1) - 1
  for seq in result.get("sequences", []):
    if seq.get("tokens"):
      register_route(prompt_token_ids + seq["tokens"], base_url)
  return result


async def _sample_on(base_url: str, req: dict) -> dict:
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
    max_model_len = _served_max_model_len.get(base_url)
    if max_model_len is not None:
      prompt_len = len(prompt_token_ids)
      if prompt_len >= max_model_len or max_tokens <= 0:
        print(
          f"[external-sampler] Prompt of {prompt_len} tokens leaves no room in max_model_len={max_model_len} "
          f"(max_tokens={max_tokens}); returning empty length-stop truncation."
        )
        return {"type": "sample", "sequences": [{"tokens": [], "logprobs": [], "stop_reason": "length"} for _ in range(num_samples)]}
      if prompt_len + max_tokens > max_model_len:
        max_tokens = max_model_len - prompt_len

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
      _loaded_adapters.setdefault(base_url, set()).discard(lora_id)
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
