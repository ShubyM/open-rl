# This file contains the vLLM worker implementation for high-throughput inference in Open-RL.

import asyncio
import base64
import hashlib
import json
import os
import sys
from contextlib import asynccontextmanager
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from safetensors.torch import load as load_safetensors
from safetensors.torch import save_file as save_safetensors_file
from state_delta import StateDeltaManifest, hash_adapter_config, tensor_checksum, validate_delta_manifest
from telemetry import get_tracer, instrument_fastapi

try:
  from vllm import SamplingParams
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.engine.async_llm_engine import AsyncLLMEngine
  from vllm.lora.request import LoRARequest
  from vllm.sampling_params import RequestOutputKind

  VLLM_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
  VLLM_IMPORT_ERROR = exc

  class SamplingParams:
    def __init__(self, **kwargs):
      self.kwargs = kwargs

  class AsyncEngineArgs:
    def __init__(self, **kwargs):
      self.kwargs = kwargs

  class AsyncLLMEngine:
    @classmethod
    def from_engine_args(cls, engine_args):
      raise RuntimeError(f"vLLM is not installed: {VLLM_IMPORT_ERROR}")

  class LoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path, load_inplace=False):
      self.lora_name = lora_name
      self.lora_int_id = lora_int_id
      self.lora_path = lora_path
      self.load_inplace = load_inplace

  class RequestOutputKind:
    FINAL_ONLY = "final_only"


tracer = get_tracer("openrl.vllm_sampler")

engine: Any = None
synced_lora_adapters: dict[str, dict[str, Any]] = {}
FLOAT_DTYPE_NAMES = {"float16", "bfloat16", "float32", "float64"}


def loaded_base_model_id() -> str | None:
  return os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL")


def lora_state_key(base_model_id: str | None, adapter_name: str) -> str:
  return f"{base_model_id}::{adapter_name}" if base_model_id else adapter_name


def assert_base_model_compatible(base_model_id: str | None) -> None:
  loaded = loaded_base_model_id()
  if base_model_id and loaded and base_model_id != loaded:
    raise ValueError(f"vLLM worker loaded {loaded}, cannot apply state for {base_model_id}")


class ShmTensorRef(BaseModel):
  name: str
  size: int


class TransportReceiptPayload(BaseModel):
  transport: str
  delta_id: str
  version: int
  locations: dict[str, str] = Field(default_factory=dict)
  expires_at: float | None = None


class LoraTensorSyncRequest(BaseModel):
  run_id: str
  version: int
  adapter_name: str
  base_model_id: str | None = None
  adapter_config: dict[str, Any]
  manifest: dict[str, Any]
  transport_receipt: TransportReceiptPayload | None = None
  adapter_path: str | None = None
  manifest_path: str | None = None
  tensors_safetensors_shm: ShmTensorRef | None = None
  tensors_safetensors_b64: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
  global engine

  print("\n" + "=" * 50)
  print("        Open-RL vLLM Inference Engine")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  model_name = os.getenv("BASE_MODEL") or os.getenv("VLLM_MODEL")
  print(f"-> Hardware     : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> Model        : {model_name or 'Not Set'}\n")

  mock_vllm = os.getenv("MOCK_VLLM", "0") == "1"
  if mock_vllm:
    print("[vLLM Subprocess] MOCK_VLLM=1, bypassing real engine init for local dev.")
  elif VLLM_IMPORT_ERROR is not None:
    print(f"[vLLM Subprocess] Error: vLLM import failed: {VLLM_IMPORT_ERROR}")
    sys.exit(1)
  elif not model_name:
    print("[vLLM Subprocess] Error: BASE_MODEL environment variable is required.")
    sys.exit(1)
  else:
    hf_overrides: dict = {}
    arch_override = os.getenv("VLLM_ARCHITECTURE_OVERRIDE")
    if arch_override:
      hf_overrides["architectures"] = [arch_override]

    engine_kwargs = {
      "model": model_name,
      "enable_lora": True,
      "max_loras": 8,
      "max_lora_rank": 64,
      "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "8192")),
      "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "64")),
      "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
      "enable_prefix_caching": False,
      "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "0") == "1",
    }
    if hf_overrides:
      engine_kwargs["hf_overrides"] = hf_overrides

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    print("[vLLM Subprocess] Engine initialized and ready to serve IPC requests.")

  yield


app = FastAPI(title="Open-RL vLLM Subprocess", lifespan=lifespan)
instrument_fastapi(app, excluded_urls="/healthz")


@app.get("/healthz")
async def healthz():
  return {"status": "ok", "mock": engine is None}


@app.post("/sync_lora_adapter")
async def sync_lora_adapter(req: Request):
  data = await req.json()
  adapter_name = data.get("adapter_name")
  base_model_id = data.get("base_model_id")
  version = data.get("version")
  adapter_path = data.get("adapter_path")
  if not adapter_name:
    return {"type": "RequestFailedResponse", "error_message": "adapter_name is required"}
  if version is None:
    return {"type": "RequestFailedResponse", "error_message": "version is required"}
  version = int(version)
  try:
    assert_base_model_compatible(base_model_id)
  except ValueError as exc:
    return {"type": "RequestFailedResponse", "error_message": str(exc)}
  state_key = lora_state_key(base_model_id, adapter_name)
  current = synced_lora_adapters.get(state_key)
  if current and int(current.get("version", 0)) >= version:
    return {"type": "RequestFailedResponse", "error_message": f"stale adapter version {version} for {adapter_name}"}

  synced_lora_adapters[state_key] = {
    "adapter_path": adapter_path,
    "base_model_id": base_model_id,
    "manifest_path": data.get("manifest_path"),
    "run_id": data.get("run_id"),
    "version": version,
  }
  return {"adapter_name": adapter_name, "version": version, "type": "sync_lora_adapter"}


def read_lora_safetensors_bytes(payload: LoraTensorSyncRequest) -> bytes:
  if payload.tensors_safetensors_shm is not None:
    shm = shared_memory.SharedMemory(name=payload.tensors_safetensors_shm.name)
    try:
      return bytes(shm.buf[: payload.tensors_safetensors_shm.size])
    finally:
      shm.close()
  if payload.tensors_safetensors_b64:
    return base64.b64decode(payload.tensors_safetensors_b64)
  raise ValueError("tensors_safetensors_shm or tensors_safetensors_b64 is required")


def verify_lora_delta_manifest(payload: LoraTensorSyncRequest, tensors: dict[str, Any]) -> StateDeltaManifest:
  manifest = StateDeltaManifest.from_dict(payload.manifest)
  validate_delta_manifest(manifest)

  if manifest.apply_target != "vllm_lora":
    raise ValueError(f"unsupported apply_target for vLLM sampler: {manifest.apply_target}")
  if manifest.run_id != payload.run_id:
    raise ValueError("manifest run_id does not match request")
  if manifest.version != payload.version:
    raise ValueError("manifest version does not match request")
  expected_config_hash = hash_adapter_config(payload.adapter_config)
  if manifest.adapter_config_hash != expected_config_hash:
    raise ValueError("manifest adapter_config_hash does not match adapter_config")

  tensor_names = set(tensors)
  expected_names = {entry.normalized_name for entry in manifest.tensors}
  if tensor_names != expected_names:
    raise ValueError("manifest tensor names do not match tensor payload")

  for entry in manifest.tensors:
    tensor = tensors[entry.normalized_name]
    if tuple(tensor.shape) != tuple(entry.shape):
      raise ValueError(f"tensor shape mismatch for {entry.normalized_name}")
    received_dtype = str(tensor.dtype).removeprefix("torch.")
    if received_dtype != entry.dtype and (received_dtype not in FLOAT_DTYPE_NAMES or entry.dtype not in FLOAT_DTYPE_NAMES):
      raise ValueError(f"tensor dtype mismatch for {entry.normalized_name}")
    if entry.checksum is not None and tensor_checksum(tensor) != entry.checksum:
      raise ValueError(f"tensor checksum mismatch for {entry.normalized_name}")

  if payload.transport_receipt and (payload.transport_receipt.delta_id != manifest.delta_id or payload.transport_receipt.version != manifest.version):
    raise ValueError("transport_receipt does not match manifest")

  return manifest


@app.post("/sync_lora_tensors")
async def sync_lora_tensors(req: Request):
  try:
    payload = LoraTensorSyncRequest.model_validate(await req.json())
    assert_base_model_compatible(payload.base_model_id)
    tensors = load_safetensors(read_lora_safetensors_bytes(payload))
    manifest = verify_lora_delta_manifest(payload, tensors)
    state_key = lora_state_key(payload.base_model_id, payload.adapter_name)
    current = synced_lora_adapters.get(state_key)
    if current and int(current.get("version", 0)) >= payload.version:
      raise ValueError(f"stale adapter version {payload.version} for {payload.adapter_name}")
    sync_root = Path(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")) / "vllm_lora_sync"
    adapter_key = hashlib.sha256(f"{payload.adapter_name}@{payload.version}:{manifest.delta_id}".encode()).hexdigest()[:24]
    adapter_dir = sync_root / adapter_key
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors_file(tensors, adapter_dir / "adapter_model.safetensors")
    with (adapter_dir / "adapter_config.json").open("w") as f:
      json.dump(payload.adapter_config, f)

    current_engine = engine
    if current_engine is not None:
      lora_cache_key = f"{state_key}@{payload.version}"
      lora_int_id = int(hashlib.md5(lora_cache_key.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
      lora_request = LoRARequest(lora_cache_key, lora_int_id, str(adapter_dir), load_inplace=True)
      await current_engine.add_lora(lora_request)

    synced_lora_adapters[state_key] = {
      "adapter_path": str(adapter_dir),
      "base_model_id": payload.base_model_id,
      "manifest_path": payload.manifest_path,
      "run_id": payload.run_id,
      "version": payload.version,
    }
    return {
      "adapter_name": payload.adapter_name,
      "adapter_path": str(adapter_dir),
      "tensor_count": len(tensors),
      "delta_id": manifest.delta_id,
      "type": "sync_lora_tensors",
      "version": payload.version,
    }
  except ValueError as e:
    return {"type": "RequestFailedResponse", "error_message": f"vLLM LoRA tensor sync error: {str(e)}"}
  except Exception as e:
    import traceback

    traceback.print_exc()
    return {"type": "RequestFailedResponse", "error_message": f"vLLM LoRA tensor sync error: {str(e)}"}


@app.post("/generate")
async def generate(req: Request):
  try:
    data = await req.json()

    request_id = data.get("request_id")
    prompt_token_ids = data.get("prompt_token_ids")
    max_tokens = data.get("max_tokens", 20)
    temperature = data.get("temperature", 1.0)
    stop = data.get("stop", None)
    top_p = data.get("top_p", 1.0)
    top_k = data.get("top_k", -1)
    num_samples = data.get("num_samples", 1)

    lora_id = data.get("lora_id", None)
    lora_path = data.get("lora_path", None)
    base_model_id = data.get("base_model_id")
    lora_version = int(data.get("lora_version", 0) or 0)
    include_prompt_logprobs = data.get("include_prompt_logprobs", False)
    assert_base_model_compatible(base_model_id)

    current_engine = engine
    if current_engine is None:
      # Mocking for local Mac dev
      await asyncio.sleep(0.1)
      # return dummy tokens locally
      return {"sequences": [{"tokens": [0] * max_tokens, "logprobs": [-0.1] * max_tokens, "stop_reason": "length"}]}

    prompt_logprobs_val = 1 if include_prompt_logprobs else None
    sampling_params = SamplingParams(
      n=num_samples,
      temperature=temperature,
      max_tokens=max_tokens,
      stop_token_ids=stop,
      top_p=top_p,
      top_k=top_k,
      logprobs=1,  # return logprobs for TITO RL
      prompt_logprobs=prompt_logprobs_val,
      output_kind=RequestOutputKind.FINAL_ONLY,
    )

    lora_request = None
    state_key = lora_state_key(base_model_id, lora_id) if lora_id else None
    sync_state = synced_lora_adapters.get(state_key) if state_key else None
    if sync_state and sync_state.get("adapter_path"):
      lora_path = sync_state["adapter_path"]
      lora_version = int(sync_state["version"])
    if lora_id and lora_path:
      # vLLM natively relies on lora_int_id to track cached adapter weights.
      # Include the published version so repeated sampler aliases reload changed files.
      lora_cache_key = f"{state_key or lora_id}@{lora_version}"
      lora_int_id = int(hashlib.md5(lora_cache_key.encode("utf-8")).hexdigest(), 16) % (2**31 - 1) + 1
      lora_request = LoRARequest(lora_cache_key, lora_int_id, lora_path)

    results_generator = current_engine.generate(
      prompt={"prompt_token_ids": prompt_token_ids}, sampling_params=sampling_params, request_id=request_id, lora_request=lora_request
    )

    final_output = None
    with tracer.start_as_current_span("vllm_generate_tokens") as span:
      span.set_attribute("vllm.prompt_len", len(prompt_token_ids) if prompt_token_ids else 0)
      span.set_attribute("vllm.max_tokens", max_tokens)
      if lora_id:
        span.set_attribute("vllm.lora_id", lora_id)
        span.set_attribute("vllm.lora_version", lora_version if lora_id and lora_path else 0)
      async for request_output in results_generator:
        final_output = request_output

    outputs = final_output.outputs if final_output else []
    sequences_out = []
    for output in outputs:
      generated_token_ids = list(output.token_ids)
      logprobs = []
      if output.logprobs:
        for idx, token_logprobs in enumerate(output.logprobs):
          # token_logprobs is a dict of {token_id: Logprob}
          token_id = generated_token_ids[idx]
          if token_logprobs and token_id in token_logprobs:
            logprob = token_logprobs[token_id].logprob
          else:
            logprob = -9999.0
          logprobs.append(logprob)
      sequences_out.append({"tokens": generated_token_ids, "logprobs": logprobs, "stop_reason": output.finish_reason})

    prompt_logprobs_out = None
    if final_output and final_output.prompt_logprobs:
      prompt_logprobs_out = []
      for idx, token_logprobs in enumerate(final_output.prompt_logprobs):
        if token_logprobs is None:
          prompt_logprobs_out.append(None)
        else:
          token_id = prompt_token_ids[idx]
          if token_id in token_logprobs:
            prompt_logprobs_out.append(token_logprobs[token_id].logprob)
          else:
            prompt_logprobs_out.append(None)

    return {"sequences": sequences_out, "prompt_logprobs": prompt_logprobs_out}
  except Exception as e:
    import traceback

    traceback.print_exc()
    # Return explicit 500 so upstream client logs it
    return {"type": "RequestFailedResponse", "error_message": f"vLLM Worker Error: {str(e)}"}


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8001)
