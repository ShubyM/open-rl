import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Removed direct PyTorch engine import to keep Gateway stateless
from .checkpoints import (
  STATE_KIND_CHECKPOINT,
  STATE_KIND_SNAPSHOT,
  build_artifact_path,
  decode_tinker_path,
  encode_tinker_path,
  get_peft_root,
)
from .state import get_store

store = get_store()
import logging
import time
import traceback

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Initialize OpenTelemetry TracerProvider
provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.environ.get("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")
else:
  print("OpenTelemetry: No exporter configured (ENABLE_GCP_TRACE=0)")


async def enqueue_traced_request(store, payload: dict) -> None:
  carrier = {}
  from opentelemetry import propagate

  propagate.inject(carrier)
  payload["trace_context"] = carrier
  await store.put_request(payload)


class FilterNoisyEndpoints(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "retrieve_future" not in msg and "session_heartbeat" not in msg


logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())


def is_single_process_mode() -> bool:
  explicit = os.getenv("OPEN_RL_SINGLE_PROCESS")
  if explicit is not None:
    return explicit == "1"
  return bool(os.getenv("OPEN_RL_BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  explicit = os.getenv("SAMPLER_BACKEND")
  if explicit:
    return explicit.lower()
  return "engine" if is_single_process_mode() else "vllm"


def get_default_model_name() -> str | None:
  if is_single_process_mode():
    from . import engine as trainer_engine

    if trainer_engine.engine.base_model_name:
      return trainer_engine.engine.base_model_name
  return os.getenv("OPEN_RL_BASE_MODEL") or os.getenv("VLLM_MODEL")


@asynccontextmanager
async def lifespan(app: FastAPI):
  task = None
  if is_single_process_mode():
    from . import engine as trainer_engine

    base_model = os.getenv("OPEN_RL_BASE_MODEL")
    print("\n" + "=" * 50)
    print(" Open-RL Single-Process Mode")
    print("=" * 50)
    print(f"-> Base model: {base_model or 'unset'}")
    print("-> Backend   : gateway + engine loop in one process\n")
    if base_model:
      await asyncio.to_thread(trainer_engine.engine.preload_base_model, base_model)
    task = asyncio.create_task(trainer_engine.clock_cycle_loop())
  try:
    yield
  finally:
    if task is not None:
      task.cancel()


app = FastAPI(title="Open-RL Server MVP", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat,/v1/futures")

# Keep strong references to background tasks to prevent GC mid-flight.
background_tasks = set()
sampling_sessions: dict[str, dict[str, str | None]] = {}


def _extract_lora_config(config: dict) -> dict:
  lora_config = {}
  for key in ("rank", "lora_r", "lora_alpha", "lora_dropout", "train_attn", "train_mlp", "train_unembed"):
    if key in config:
      lora_config[key] = config[key]
  if "lora_r" in lora_config and "rank" not in lora_config:
    lora_config["rank"] = lora_config.pop("lora_r")
  return lora_config


def _register_background_task(task: asyncio.Task) -> None:
  background_tasks.add(task)
  task.add_done_callback(background_tasks.discard)


def _register_sampling_session(model_id: str, model_path: str | None) -> str:
  session_id = str(uuid.uuid4())
  sampling_sessions[session_id] = {"model_id": model_id, "model_path": model_path}
  return session_id


def _resolve_sampling_target(
  *,
  sampling_session_id: str | None = None,
  model_id: str | None = None,
  model_path: str | None = None,
) -> tuple[str | None, str | None, str | None]:
  if sampling_session_id:
    session = sampling_sessions.get(sampling_session_id)
    if session:
      return session.get("model_id"), session.get("model_path"), sampling_session_id
    return sampling_session_id, None, sampling_session_id

  parsed_model_id, parsed_path = decode_tinker_path(model_path)
  if parsed_model_id:
    return parsed_model_id, parsed_path, None

  return model_id, model_path, None


def _live_adapter_path(model_id: str) -> str:
  return os.path.join(get_peft_root(), model_id)


def _resolve_state_path(state_path: str | None) -> str | None:
  if state_path is None:
    return None
  model_id, decoded_path = decode_tinker_path(state_path)
  if model_id and decoded_path:
    return decoded_path
  return state_path


async def _await_resolved_future(req_id: str, timeout: float = 3600.0) -> dict:
  deadline = time.time() + timeout
  while True:
    remaining = max(0.1, min(30.0, deadline - time.time()))
    if remaining <= 0:
      return {
        "type": "RequestFailedResponse",
        "error_message": f"Timed out waiting for request {req_id}",
        "category": "timeout",
      }

    result = await store.get_future(req_id, timeout=remaining)
    if result is None:
      return {
        "type": "RequestFailedResponse",
        "error_message": f"Future {req_id} not found",
        "category": "unknown",
      }
    if isinstance(result, dict) and result.get("status") == "pending":
      continue
    if isinstance(result, dict) and result.get("type") == "try_again":
      if time.time() >= deadline:
        return {
          "type": "RequestFailedResponse",
          "error_message": f"Timed out waiting for request {req_id}",
          "category": "timeout",
        }
      continue
    return result


async def _enqueue_model_request(model_id: str, req_type: str, **payload) -> str:
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})
  request = {"req_id": req_id, "model_id": model_id, "type": req_type, **payload}
  await enqueue_traced_request(store, request)
  return req_id


async def _enqueue_create_or_restore_model(
  *,
  base_model: str | None,
  lora_config: dict,
  state_path: str | None = None,
  restore_optimizer: bool = False,
) -> str:
  model_id = str(uuid.uuid4())
  await store.set_future(model_id, {"status": "pending"})
  await enqueue_traced_request(
    store,
    {
      "req_id": model_id,
      "model_id": model_id,
      "type": "create_model",
      "base_model": base_model,
      "lora_config": lora_config,
      "state_path": state_path,
      "restore_optimizer": restore_optimizer,
    },
  )
  return model_id


async def _enqueue_sample_request(
  *,
  prompt_tokens: list[int],
  max_tokens: int,
  temperature: float,
  num_samples: int,
  model_id: str | None,
  model_path: str | None,
  sampling_session_id: str | None,
) -> str:
  if not model_id:
    raise ValueError("model_id is required for sampling")

  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  sampler_backend = get_sampler_backend()
  if sampler_backend == "engine":
    await enqueue_traced_request(
      store,
      {
        "req_id": req_id,
        "model_id": model_id,
        "type": "sample",
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_samples": num_samples,
        "adapter_path": model_path,
      },
    )
    return req_id

  async def _route_to_vllm():
    try:
      lora_path = model_path
      if lora_path is None:
        live_adapter_path = _live_adapter_path(model_id)
        lora_path = live_adapter_path if os.path.isdir(live_adapter_path) else None

      lora_id = sampling_session_id or model_id
      payload = {
        "request_id": req_id,
        "prompt_token_ids": prompt_tokens,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "num_samples": num_samples,
        "lora_id": lora_id,
        "lora_path": lora_path,
      }

      vllm_url = os.environ.get("VLLM_URL", "http://127.0.0.1:8001")
      vllm_generate_endpoint = f"{vllm_url.rstrip('/')}/generate"
      headers = {"Content-Type": "application/json"}
      from opentelemetry import propagate

      propagate.inject(headers)

      import httpx

      async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(vllm_generate_endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
      if data.get("type") != "RequestFailedResponse":
        data["type"] = "sample"
      await store.set_future(req_id, data)
    except Exception as exc:
      traceback.print_exc()
      await store.set_future(
        req_id,
        {"type": "RequestFailedResponse", "error_message": str(exc), "category": "server_error"},
      )

  _register_background_task(asyncio.create_task(_route_to_vllm()))
  return req_id


async def _enqueue_compute_logprobs_request(
  *,
  tokens: list[int],
  model_id: str | None,
  model_path: str | None,
  sampling_session_id: str | None,
) -> str:
  if not model_id:
    raise ValueError("model_id is required for compute_logprobs")

  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  await enqueue_traced_request(
    store,
    {
      "req_id": req_id,
      "model_id": model_id,
      "type": "compute_logprobs",
      "tokens": tokens,
      "adapter_path": model_path,
      "sampling_session_id": sampling_session_id,
    },
  )
  return req_id


def _v1_future_response(result: dict | None) -> dict:
  if result is None:
    return {"status": "not_found"}
  if result.get("status") == "pending" or result.get("type") == "try_again":
    return {"status": "pending"}
  if result.get("type") == "RequestFailedResponse":
    return {"status": "error", "error": result.get("error_message", "unknown error")}
  if result.get("type") == "sample":
    completions = [sequence.get("tokens", []) for sequence in result.get("sequences", [])]
    return {"status": "done", "result": {"completions": completions}}
  if result.get("type") == "compute_logprobs":
    return {"status": "done", "result": {"logprobs": result.get("logprobs", [])}}
  return {"status": "done", "result": result}


@app.get("/api/v1/healthz")
async def health_check():
  return {"status": "ok"}


@app.get("/v1/health")
async def v1_health():
  return {"status": "ok"}


@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
  model_name = get_default_model_name()
  return {
    "supported_models": [{"model_name": model_name}] if model_name else [],
    "default_model": model_name,
    "single_process": is_single_process_mode(),
  }


@app.get("/v1/server_capabilities")
async def v1_server_capabilities():
  return await get_server_capabilities()


@app.get("/v1/futures/{future_id}")
async def get_future_v1(future_id: str, timeout: float = 0.0):
  wait_timeout = max(0.0, timeout)
  result = await store.get_future(future_id, timeout=wait_timeout)
  return _v1_future_response(result)


@app.post("/api/v1/create_session")
async def create_session(req: dict):
  return {"session_id": "sess-real-123", "type": "create_session"}


@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(req: dict):
  return {"type": "session_heartbeat"}


@app.post("/api/v1/create_model")
async def create_model(req: dict):
  base_model = req.get("base_model")
  state_path = _resolve_state_path(req.get("path") or req.get("state_path"))
  if not base_model and not state_path:
    return JSONResponse(status_code=400, content={"error": "base_model is required"})

  raw_lora_config = req.get("lora_config")
  lora_config = _extract_lora_config(raw_lora_config if isinstance(raw_lora_config, dict) else req)
  req_id = await _enqueue_create_or_restore_model(
    base_model=base_model,
    lora_config=lora_config,
    state_path=state_path,
    restore_optimizer=bool(req.get("restore_optimizer") or req.get("with_optimizer")),
  )
  return {"request_id": req_id}


@app.post("/api/v1/get_info")
async def get_info(req: dict):
  model_name = get_default_model_name()
  if not model_name:
    return JSONResponse(status_code=404, content={"error": "No base model is configured for the current server session"})

  return {
    "model_data": {"arch": "unknown", "model_name": model_name, "tokenizer_id": model_name},
    "model_id": req.get("model_id", "model-live-123"),  # Will be whatever client passed
    "is_lora": True,
    "lora_rank": 16,
    "model_name": model_name,
    "type": "get_info",
  }


@app.post("/v1/training/create_lora", status_code=202)
async def v1_create_lora_training_client(body: dict):
  base_model = body.get("base_model")
  if not base_model:
    return JSONResponse(status_code=400, content={"error": "base_model is required"})

  future_id = await _enqueue_create_or_restore_model(
    base_model=base_model,
    lora_config=_extract_lora_config(body),
  )
  return {"future_id": future_id}


@app.post("/v1/training/from_state", status_code=202)
async def v1_create_training_from_state(body: dict):
  state_path = _resolve_state_path(body.get("path"))
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "path is required"})

  future_id = await _enqueue_create_or_restore_model(
    base_model=body.get("base_model"),
    lora_config=_extract_lora_config(body),
    state_path=state_path,
    restore_optimizer=bool(body.get("restore_optimizer") or body.get("with_optimizer")),
  )
  return {"future_id": future_id}


@app.post("/v1/training/from_state_with_optimizer", status_code=202)
async def v1_create_training_from_state_with_optimizer(body: dict):
  payload = dict(body)
  payload["restore_optimizer"] = True
  return await v1_create_training_from_state(payload)


@app.post("/v1/training/{model_id}/forward_backward", status_code=202)
async def v1_forward_backward(model_id: str, body: dict):
  future_id = await _enqueue_model_request(
    model_id,
    "forward_backward",
    data=body.get("data", []),
    loss_fn=body.get("loss_fn", "cross_entropy"),
    loss_config=body.get("loss_fn_config", {}) or {},
  )
  return {"future_id": future_id}


@app.post("/v1/training/{model_id}/forward", status_code=202)
async def v1_forward(model_id: str, body: dict):
  future_id = await _enqueue_model_request(
    model_id,
    "forward",
    data=body.get("data", []),
    loss_fn=body.get("loss_fn", "cross_entropy"),
    loss_config=body.get("loss_fn_config", {}) or {},
  )
  return {"future_id": future_id}


@app.post("/v1/training/{model_id}/optim_step", status_code=202)
async def v1_optim_step(model_id: str, body: dict):
  future_id = await _enqueue_model_request(
    model_id,
    "optim_step",
    adam_params=body.get("adam_params", {}),
  )
  return {"future_id": future_id}


@app.post("/v1/training/{model_id}/save_state", status_code=202)
async def v1_save_state(model_id: str, body: dict):
  state_path = build_artifact_path(body.get("name"), STATE_KIND_CHECKPOINT, model_id)
  future_id = await _enqueue_model_request(
    model_id,
    "save_state",
    state_path=state_path,
    include_optimizer=True,
    kind=STATE_KIND_CHECKPOINT,
  )
  return {"future_id": future_id}


@app.post("/v1/training/{model_id}/load_state", status_code=202)
async def v1_load_state(model_id: str, body: dict):
  state_path = _resolve_state_path(body.get("path"))
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "path is required"})

  future_id = await _enqueue_model_request(
    model_id,
    "load_state",
    state_path=state_path,
    restore_optimizer=bool(body.get("restore_optimizer") or body.get("with_optimizer")),
    base_model=body.get("base_model"),
  )
  return {"future_id": future_id}


@app.post("/v1/training/{model_id}/save_weights_and_get_sampling_client", status_code=202)
async def v1_save_weights_and_get_sampling_client(model_id: str, body: dict | None = None):
  body = body or {}
  future_id = str(uuid.uuid4())
  await store.set_future(future_id, {"status": "pending"})

  async def _save_and_register_session():
    try:
      snapshot_path = build_artifact_path(body.get("name"), STATE_KIND_SNAPSHOT, model_id)
      inner_req_id = await _enqueue_model_request(
        model_id,
        "save_state",
        state_path=snapshot_path,
        include_optimizer=False,
        kind=STATE_KIND_SNAPSHOT,
      )
      inner_result = await _await_resolved_future(inner_req_id)
      if inner_result.get("type") == "RequestFailedResponse":
        await store.set_future(future_id, inner_result)
        return

      resolved_path = inner_result.get("path", snapshot_path)
      session_id = _register_sampling_session(model_id, resolved_path)
      await store.set_future(future_id, {"session_id": session_id, "path": resolved_path})
    except Exception as exc:
      await store.set_future(
        future_id,
        {"type": "RequestFailedResponse", "error_message": str(exc), "category": "server_error"},
      )

  _register_background_task(asyncio.create_task(_save_and_register_session()))
  return {"future_id": future_id}


@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  fwd_input = req.get("forward_backward_input", {})
  data = fwd_input.get("data", [])
  loss_fn = fwd_input.get("loss_fn", "cross_entropy")
  loss_config = fwd_input.get("loss_fn_config", {})
  model_id = req.get("model_id")

  await enqueue_traced_request(
    store,
    {
      "req_id": req_id,
      "model_id": model_id,
      "type": "forward_backward",
      "data": data,
      "loss_fn": loss_fn,
      "loss_config": loss_config,
    },
  )

  return {"request_id": req_id}


@app.post("/api/v1/forward")
async def forward(req: dict):
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  fwd_input = req.get("forward_input", {})
  data = fwd_input.get("data", [])
  loss_fn = fwd_input.get("loss_fn", "cross_entropy")
  loss_config = fwd_input.get("loss_fn_config", {})
  model_id = req.get("model_id")

  await enqueue_traced_request(
    store,
    {
      "req_id": req_id,
      "model_id": model_id,
      "type": "forward",
      "data": data,
      "loss_fn": loss_fn,
      "loss_config": loss_config,
    },
  )

  return {"request_id": req_id}


@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})

  adam_params = req.get("adam_params", {})
  model_id = req.get("model_id")

  await enqueue_traced_request(
    store,
    {
      "req_id": req_id,
      "model_id": model_id,
      "type": "optim_step",
      "adam_params": adam_params,
    },
  )

  return {"request_id": req_id}


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})
  model_id = req.get("model_id")  # Client passes the TrainingClient's model_id
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  alias = req.get("name") or req.get("alias") or req.get("path")

  async def _save_snapshot():
    try:
      snapshot_path = build_artifact_path(alias, STATE_KIND_SNAPSHOT, model_id)
      inner_req_id = await _enqueue_model_request(
        model_id,
        "save_state",
        state_path=snapshot_path,
        include_optimizer=False,
        kind=STATE_KIND_SNAPSHOT,
      )
      inner_result = await _await_resolved_future(inner_req_id)
      if inner_result.get("type") == "RequestFailedResponse":
        await store.set_future(req_id, inner_result)
        return

      resolved_path = inner_result.get("path", snapshot_path)
      session_id = _register_sampling_session(model_id, resolved_path)
      await store.set_future(
        req_id,
        {
          "path": encode_tinker_path(model_id, resolved_path),
          "sampling_session_id": session_id,
          "type": "save_weights_for_sampler",
        },
      )
    except Exception as exc:
      await store.set_future(
        req_id,
        {"type": "RequestFailedResponse", "error_message": str(exc), "category": "server_error"},
      )

  _register_background_task(asyncio.create_task(_save_snapshot()))
  return {"request_id": req_id}


@app.post("/api/v1/save_weights")
async def save_weights(req: dict):
  req_id = str(uuid.uuid4())
  await store.set_future(req_id, {"status": "pending"})
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  alias = req.get("path")

  async def _save_weights_only():
    try:
      state_path = build_artifact_path(alias, STATE_KIND_CHECKPOINT, model_id)
      inner_req_id = await _enqueue_model_request(
        model_id,
        "save_state",
        state_path=state_path,
        include_optimizer=False,
        kind=STATE_KIND_CHECKPOINT,
      )
      inner_result = await _await_resolved_future(inner_req_id)
      if inner_result.get("type") == "RequestFailedResponse":
        await store.set_future(req_id, inner_result)
        return

      resolved_path = inner_result.get("path", state_path)
      await store.set_future(req_id, {"path": resolved_path, "type": "save_weights"})
    except Exception as exc:
      await store.set_future(
        req_id,
        {"type": "RequestFailedResponse", "error_message": str(exc), "category": "server_error"},
      )

  _register_background_task(asyncio.create_task(_save_weights_only()))
  return {"request_id": req_id}


@app.post("/api/v1/load_weights")
async def load_weights(req: dict):
  model_id = req.get("model_id")
  state_path = _resolve_state_path(req.get("path"))
  if not model_id or not state_path:
    return JSONResponse(status_code=400, content={"error": "model_id and path are required"})

  request_id = await _enqueue_model_request(
    model_id,
    "load_state",
    state_path=state_path,
    restore_optimizer=bool(req.get("restore_optimizer") or req.get("with_optimizer")),
    base_model=req.get("base_model"),
  )
  return {"request_id": request_id}


@app.get("/api/v1/list_adapters")
async def list_adapters():
  """
  Scans the local temporary directory (used for RAM-disk sync) for available PEFT adapters.
  Returns a list of adapters with metadata (creation time, alias) if available.
  """
  tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
  peft_dir = os.path.join(tmp_dir, "peft")

  adapters = []

  if os.path.exists(peft_dir):
    # Scan all directories in peft_dir
    with os.scandir(peft_dir) as entries:
      for entry in entries:
        if entry.is_dir():
          model_id = entry.name
          metadata_path = os.path.join(entry.path, "metadata.json")

          adapter_info = {
            "model_id": model_id,
            "created_at": entry.stat().st_ctime,  # Fallback to filesystem time
            "timestamp": entry.stat().st_ctime,
            "alias": None,
          }

          # Try to read metadata.json
          if os.path.exists(metadata_path):
            try:
              with open(metadata_path) as f:
                meta = json.load(f)
                adapter_info.update(meta)
            except Exception:
              pass

          adapters.append(adapter_info)

  # Sort by creation time descending (newest first)
  # Prefer 'timestamp' (float) if available, otherwise 'created_at' if it's a number
  def get_sort_key(x):
    ts = x.get("timestamp")
    if isinstance(ts, (int, float)):
      return ts
    ca = x.get("created_at")
    if isinstance(ca, (int, float)):
      return ca
    return 0

  adapters.sort(key=get_sort_key, reverse=True)

  return {"adapters": adapters}


@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
  model_path = req.get("model_path")
  model_id = req.get("model_id")
  parsed_model_id, parsed_model_path = decode_tinker_path(model_path)

  if parsed_model_id and parsed_model_path:
    session_id = _register_sampling_session(parsed_model_id, parsed_model_path)
  elif model_path and model_path.startswith("tinker://"):
    session_id = model_path[len("tinker://") :]
  else:
    resolved_model_id = parsed_model_id or model_id
    if not resolved_model_id:
      return JSONResponse(status_code=400, content={"error": "model_id is required"})
    session_id = _register_sampling_session(resolved_model_id, parsed_model_path)

  return {"sampling_session_id": session_id, "type": "create_sampling_session"}


@app.post("/v1/sampling/create", status_code=202)
async def v1_create_sampling_client(body: dict):
  model_id, model_path, _ = _resolve_sampling_target(
    model_id=body.get("model_id"),
    model_path=body.get("model_path"),
  )
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  future_id = str(uuid.uuid4())
  await store.set_future(future_id, {"session_id": _register_sampling_session(model_id, model_path)})
  return {"future_id": future_id}


@app.post("/v1/sampling/{session_id}/sample", status_code=202)
async def v1_sample(session_id: str, body: dict):
  prompt = body.get("prompt", {}).get("tokens", [])
  sampling_params = body.get("sampling_params", {})
  max_tokens = sampling_params.get("max_tokens", 20)
  temperature = sampling_params.get("temperature", 1.0)
  num_samples = body.get("num_samples", 1)

  model_id, model_path, resolved_session_id = _resolve_sampling_target(sampling_session_id=session_id)
  try:
    future_id = await _enqueue_sample_request(
      prompt_tokens=prompt,
      max_tokens=max_tokens,
      temperature=temperature,
      num_samples=num_samples,
      model_id=model_id,
      model_path=model_path,
      sampling_session_id=resolved_session_id,
    )
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  return {"future_id": future_id}


@app.post("/v1/sampling/{session_id}/compute_logprobs", status_code=202)
async def v1_compute_logprobs(session_id: str, body: dict):
  tokens = body.get("prompt", {}).get("tokens", [])
  model_id, model_path, resolved_session_id = _resolve_sampling_target(sampling_session_id=session_id)
  try:
    future_id = await _enqueue_compute_logprobs_request(
      tokens=tokens,
      model_id=model_id,
      model_path=model_path,
      sampling_session_id=resolved_session_id,
    )
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  return {"future_id": future_id}


@app.post("/api/v1/asample")
async def asample(req: dict):
  prompt = req.get("prompt", {}).get("chunks", [])[0].get("tokens", [])
  params = req.get("sampling_params", {})
  max_tokens = params.get("max_tokens", 20)
  temperature = params.get("temperature", 1.0)
  num_samples = req.get("num_samples", 1)
  model_id, model_path, sampling_session_id = _resolve_sampling_target(
    sampling_session_id=req.get("sampling_session_id"),
    model_id=req.get("model_id"),
    model_path=req.get("model_path"),
  )
  try:
    req_id = await _enqueue_sample_request(
      prompt_tokens=prompt,
      max_tokens=max_tokens,
      temperature=temperature,
      num_samples=num_samples,
      model_id=model_id,
      model_path=model_path,
      sampling_session_id=sampling_session_id,
    )
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  return {"request_id": req_id}


@app.post("/api/v1/retrieve_future")
async def retrieve_future(req: dict):
  request_id = req.get("request_id")
  if not request_id:
    return JSONResponse(status_code=400, content={"error": "request_id is required"})

  result = await store.get_future(request_id, timeout=60.0)
  if result is None:
    return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})

  if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
    return JSONResponse(status_code=400, content=result)

  return result


@app.post("/api/v1/telemetry")
async def telemetry(req: dict):
  return {"status": "accepted"}
