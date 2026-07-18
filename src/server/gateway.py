# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import json
import logging
import math
import os
import shutil
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from server.control_plane import TRACKER_URL_FIELDS, safe_tracker_url
from server.control_plane import router as control_router
from server.protocol import ClientSession, CreateSessionRequest, SessionHeartbeatRequest, TrainingCommand
from server.run_metadata import MODEL_META_PREFIX, create_run_metadata, update_run_metadata
from server.store import (
  acquire_sampler_snapshot,
  claim_request,
  get_client_session,
  get_sampler_snapshot,
  get_store,
  model_revision_key,
  prune_sampler_snapshots,
  put_client_session,
  release_request_claim,
  release_sampler_snapshot,
  report_control_event,
)
from server.worker_manager import WorkerManager, create_fft_worker_manager


@dataclass
class TrainingModelMetadata:
  base_model: str | None
  created_at: float
  training_kind: str
  weight_sync_strategy: str | None = None
  full_config: dict[str, Any] | None = None
  lora_config: dict[str, Any] | None = None
  name: str | None = None
  tracker_url: str | None = None


store = get_store()
fft_worker_manager: WorkerManager | None = None

provider = TracerProvider()
trace.set_tracer_provider(provider)

if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    exporter = CloudTraceSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    print("OpenTelemetry: Configured GCP CloudTraceSpanExporter")
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")
else:
  print("OpenTelemetry: No exporter configured (ENABLE_GCP_TRACE=0)")


class FilterNoisyEndpoints(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    return "retrieve_future" not in msg and "session_heartbeat" not in msg


logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8001")


# *** Helpers ***


def is_single_process_mode() -> bool:
  return bool(os.getenv("BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  if sampling_backend := os.getenv("SAMPLING_BACKEND"):
    return sampling_backend.lower()
  return "torch" if is_single_process_mode() else "vllm"


def get_default_model_name() -> str | None:
  return os.getenv("BASE_MODEL")


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def is_mock_sampler() -> bool:
  return os.getenv("OPEN_RL_MOCK_SAMPLER", "0") == "1"


def tracker_url_from_request(req: dict) -> str | None:
  user_metadata = req.get("user_metadata")
  if not isinstance(user_metadata, dict):
    return None
  for field in TRACKER_URL_FIELDS:
    if tracker_url := safe_tracker_url(user_metadata.get(field)):
      return tracker_url
  return None


def run_name_from_request(req: dict) -> str | None:
  user_metadata = req.get("user_metadata")
  sources = (user_metadata, req) if isinstance(user_metadata, dict) else (req,)
  for source in sources:
    for field in ("name", "run_name", "display_name"):
      value = source.get(field)
      if not isinstance(value, str):
        continue
      name = value.strip()
      if name and len(name) <= 256 and not any(ord(character) < 32 for character in name):
        return name
  return None


def sampler_session_id(model_id: str, seq_id: int | str) -> str:
  return f"tinker://{model_id}/sampler_weights/sampler-{seq_id}"


def sampler_weights_path(model_id: str, name: str) -> str:
  return f"tinker://{model_id}/sampler_weights/{name}"


def checkpoint_state_path(model_id: str, name: str) -> str:
  if os.path.isabs(name):
    return name
  return os.path.join(TMP_DIR, "checkpoints", model_id, "weights", name)


def base_model_id_from_sampling_ref(model_id: str | None) -> str | None:
  if not model_id:
    return None

  if model_id.startswith("tinker://"):
    path = model_id[len("tinker://") :]
    parts = path.split("/")
    if len(parts) >= 3 and parts[1] == "sampler_weights":
      return parts[0]
    return path

  return model_id.split("-samp-")[0]


def is_sampler_weights_ref(model_id: str | None) -> bool:
  if not model_id or not model_id.startswith("tinker://"):
    return False

  path = model_id[len("tinker://") :]
  parts = path.split("/")
  return len(parts) >= 3 and parts[1] == "sampler_weights"


async def _extract_and_persist_model_metadata(
  req: dict[str, Any],
  request: Request | None = None,
  default_training_kind: str = "full",
) -> str:
  """Extract and normalize model configuration from headers and payload, persisting TrainingModelMetadata exactly once."""
  request_store = get_store()
  base_model = req.get("base_model")
  if not base_model and default_training_kind != "restored":
    raise ValueError("base_model is required in request payload")

  full_config = dict(req.get("full_config") or {})
  lora_config = dict(req.get("lora_config") or {})

  weight_sync_strategy = None
  training_kind = default_training_kind
  if request and hasattr(request, "headers"):
    weight_sync_strategy = request.headers.get("x-open-rl-weight-sync-strategy")
    if "x-open-rl-training-kind" in request.headers:
      training_kind = request.headers.get("x-open-rl-training-kind", default_training_kind)

  if weight_sync_strategy in ("full", "delta"):
    full_config["weight_sync_strategy"] = weight_sync_strategy

  model_id = requested_model_id(req)
  meta_obj = TrainingModelMetadata(
    base_model=base_model,
    created_at=time.time(),
    training_kind=training_kind,
    weight_sync_strategy=weight_sync_strategy,
    full_config=full_config,
    lora_config=lora_config,
    name=run_name_from_request(req),
    tracker_url=tracker_url_from_request(req),
  )
  await create_run_metadata(request_store, model_id, asdict(meta_obj))
  await request_store.set_value(model_revision_key(model_id), "0")
  restoring = default_training_kind == "restored"
  await report_control_event(
    request_store,
    model_id,
    component="gateway",
    phase="restoring" if restoring else "submitted",
    status="queued",
    message="Run restoration submitted" if restoring else f"Run submitted for {base_model}",
    details={"base_model": base_model, "training_kind": training_kind},
  )

  return model_id


def make_training_request(
  op: str,
  model_id: str | None,
  payload: dict,
  request_id: str | None = None,
) -> dict:
  return TrainingCommand(
    request_id=request_id or str(uuid.uuid4()),
    op=op,
    payload=payload,
    model_id=model_id,
  ).model_dump(exclude_none=True)


def ordered_request_id(req: dict, operation: str, model_id: str | None) -> str | None:
  sequence = req.get("seq_id")
  if sequence is None or model_id is None:
    return None
  return f"{model_id}:{operation}:{sequence}"


def requested_model_id(req: dict) -> str:
  session_id = req.get("session_id")
  sequence = req.get("model_seq_id")
  if session_id is not None and sequence is not None:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"open-rl:{session_id}:model:{sequence}"))
  return str(uuid.uuid4())


async def enqueue(request: dict, *, claim: bool = True) -> str:
  """Create a pending future, inject trace context, push to store. Returns req_id."""
  request_id = request["request_id"]
  if claim and not await claim_request(store, request_id):
    return request_id
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(request_id, {"status": "pending"})
  try:
    await store.put_request({**request, "trace_context": carrier})
  except Exception:
    await release_request_claim(store, request_id)
    raise
  op = str(request.get("op") or "request")
  await report_control_event(
    store,
    request.get("model_id"),
    component="trainer",
    phase=f"{op}_queued",
    status="queued",
    message=f"Queued {op.replace('_', ' ')} request",
    details={"request_id": request_id, "operation": op},
  )
  return request_id


async def launch_worker_and_enqueue(request: dict) -> str:
  """Ensure the model's dedicated trainer worker exists, then enqueue onto its queue.

  The launcher is idempotent per model_id, and Kubernetes (or the local process
  table) owns the worker's lifecycle from here; there is no separate launch
  queue. Launch failures resolve the future immediately so clients don't long-poll
  a request that can never be served.
  """
  assert fft_worker_manager is not None, "FFT worker manager is initialized by the app lifespan when FFT is enabled"
  request_id = request["request_id"]
  if not await claim_request(store, request_id):
    return request_id
  base_model = request.get("payload", {}).get("base_model")
  await store.set_future(request_id, {"status": "pending"})
  await report_control_event(
    store,
    request.get("model_id"),
    component="scheduler",
    phase="scheduling_trainer",
    status="starting",
    message="Creating or locating the trainer worker",
  )
  try:
    instance_id = await asyncio.to_thread(fft_worker_manager.launch_trainer, request["model_id"], base_model)
  except Exception as exc:
    traceback.print_exc()
    await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
    await report_control_event(
      store,
      request.get("model_id"),
      component="scheduler",
      phase="trainer_launch_failed",
      status="failed",
      level="error",
      message=str(exc),
    )
    return request_id
  await report_control_event(
    store,
    request.get("model_id"),
    component="scheduler",
    phase="trainer_scheduled",
    status="running",
    message="Trainer worker scheduled",
    details={"instance_id": instance_id},
  )
  return await enqueue(request, claim=False)


async def ensure_sampler_launched(model_id: str, base_model: str | None = None) -> str | None:
  if is_fft_enabled() and fft_worker_manager is not None and get_sampler_backend() == "vllm":
    if not base_model:
      request_store = get_store()
      value = await request_store.get_value(f"open_rl:model_meta:{model_id}") or await request_store.get_value(f"open_rl:model_base:{model_id}")
      if value:
        try:
          metadata = json.loads(value)
          base_model = metadata.get("base_model") if isinstance(metadata, dict) else value
        except (TypeError, json.JSONDecodeError):
          base_model = value
    return await asyncio.to_thread(fft_worker_manager.launch_sampler, model_id, base_model)
  return None


def decode_redis_value(value: object) -> str | None:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return str(value) if value is not None else None


async def wait_for_sampler_ready(model_id: str, base_model: str | None = None) -> None:
  request_store = get_store()
  if request_store.redis is None:
    return

  timeout = float(os.getenv("OPEN_RL_SAMPLER_READY_TIMEOUT_SECONDS", "600"))
  started = time.monotonic()
  expected_instance = await ensure_sampler_launched(model_id, base_model)
  ready_key = f"open_rl:sampler_ready:{model_id}"
  last_ready_value = None
  await report_control_event(
    request_store,
    model_id,
    component="sampler",
    phase="waiting_for_sampler",
    status="starting",
    message="Waiting for the sampler engine to initialize",
    details={"instance_id": expected_instance},
  )

  while time.monotonic() - started <= timeout:
    last_ready_value = decode_redis_value(await request_store.redis.get(ready_key))
    if last_ready_value and (expected_instance is None or last_ready_value == expected_instance):
      await report_control_event(
        request_store,
        model_id,
        component="sampler",
        phase="ready",
        status="ready",
        message="Sampler is ready",
        duration_seconds=time.monotonic() - started,
        details={"instance_id": expected_instance},
      )
      return

    if is_fft_enabled() and fft_worker_manager is not None:
      current_instance = await ensure_sampler_launched(model_id, base_model)
      if current_instance != expected_instance:
        expected_instance = current_instance
    await asyncio.sleep(1)

  message = (
    f"vLLM sampler {model_id} did not become ready within {timeout:.0f}s "
    f"(expected instance={expected_instance or 'legacy'}, readiness value={last_ready_value!r})"
  )
  await report_control_event(
    request_store,
    model_id,
    component="sampler",
    phase="initialization_timed_out",
    status="failed",
    level="error",
    message=message,
    duration_seconds=time.monotonic() - started,
  )
  raise TimeoutError(message)


async def preflight_vllm() -> None:
  """If SAMPLING_BACKEND=vllm, verify the vLLM worker is reachable at VLLM_URL.

  Prints a clear, actionable error instead of letting the first asample
  request fall through with a raw httpx connection refused.
  """
  if get_sampler_backend() != "vllm":
    return
  healthz = f"{VLLM_URL.rstrip('/')}/healthz"
  try:
    async with httpx.AsyncClient(timeout=3.0) as client:
      resp = await client.get(healthz)
      resp.raise_for_status()
  except Exception as exc:
    raise RuntimeError(
      f"SAMPLING_BACKEND=vllm but no vLLM worker is reachable at {VLLM_URL}.\n"
      f"Start it first with:  BASE_MODEL={os.getenv('BASE_MODEL') or '<model-id>'} "
      "uv run --no-sync python -m server.vllm_sampler"
    ) from exc


def translate_future_result(result: dict) -> dict:
  result_type = result.get("type")
  if result_type in {"model_created", "model_loaded_from_state"}:
    # SDK compatibility: the public client currently expects LoRA-shaped training metadata,
    # even for full fine-tuning jobs.
    response = {
      "model_id": result["model_id"],
      "is_lora": True,
      "type": "create_model" if result_type == "model_created" else "create_model_from_state",
    }
    if "rank" in result:
      response["lora_rank"] = result["rank"]
    elif result.get("training_kind") == "full":
      response["lora_rank"] = 16
    if result.get("base_model"):
      response["base_model"] = result["base_model"]
    return response

  public_type_by_internal_type = {
    "forward_backward_completed": "forward_backward",
    "optim_step_completed": "optim_step",
    "sample_completed": "sample",
    "state_saved": "save_weights",
    "weights_loaded": "load_weights",
    "sampler_weights_saved": "save_weights_for_sampler",
    "weights_saved": "save_weights",
  }
  if result_type in public_type_by_internal_type:
    response = dict(result)
    response["type"] = public_type_by_internal_type[result_type]
    return response

  return result


async def update_metadata_from_future(result: dict) -> None:
  """Fill metadata learned by a trainer without reviving a cleaned-up model."""
  if result.get("type") not in {"model_created", "model_loaded_from_state"}:
    return
  model_id = result.get("model_id")
  if not model_id:
    return
  updates = {field: result[field] for field in ("base_model", "training_kind") if result.get(field) is not None}
  if updates:
    await update_run_metadata(get_store(), str(model_id), updates, update_active=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
  global fft_worker_manager
  task = None
  if is_fft_enabled():
    fft_worker_manager = create_fft_worker_manager()
  if is_single_process_mode():
    base_model = os.getenv("BASE_MODEL")
    print("\n" + "=" * 50)
    print(" Open-RL Single-Process Mode")
    print("=" * 50)
    print(f"-> Base model: {base_model or 'unset'}")
    print(f"-> Sampling backend: {get_sampler_backend()}")
    print(f"-> FFT enabled     : {is_fft_enabled()}")
    print("-> Server mode     : API server + worker loop in one process\n")
    await preflight_vllm()
    if not is_fft_enabled():
      from server import training_requests_processor

      worker = training_requests_processor.LoraTrainingWorker()
      if base_model:
        await asyncio.to_thread(worker.load_base_model, base_model)
      task = asyncio.create_task(training_requests_processor.run_training_requests_processor(worker))
  try:
    yield
  finally:
    if task is not None:
      task.cancel()
    if fft_worker_manager is not None:
      fft_worker_manager.shutdown_all()
      fft_worker_manager = None


app = FastAPI(title="Open-RL Server MVP", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")
app.include_router(control_router)

CONTROL_UI_DIR = os.path.join(os.path.dirname(__file__), "static", "control")
if os.path.isdir(CONTROL_UI_DIR):
  app.mount("/control", StaticFiles(directory=CONTROL_UI_DIR, html=True), name="control")


# *** ServiceClient endpoints ***
@app.get("/api/v1/healthz")
async def health_check():
  return {"status": "ok"}


@app.get("/api/v1/get_server_capabilities")
async def get_server_capabilities():
  model_name = get_default_model_name()
  return {
    "supported_models": [{"model_name": model_name}] if model_name else [],
    "default_model": model_name,
    "single_process": is_single_process_mode(),
  }


@app.post("/api/v1/client/config")
async def client_config(_: dict):
  return {
    "pjwt_auth_enabled": False,
    "credential_default_source": "api_key",
    "sample_dispatch_bytes_semaphore_size": 10 * 1024 * 1024,
    "inflight_response_bytes_semaphore_size": 50 * 1024 * 1024,
  }


@app.post("/api/v1/create_session")
async def create_session(req: dict):
  request = CreateSessionRequest.model_validate(req)
  now = time.time()
  session = ClientSession(
    session_id=f"sess-{uuid.uuid4().hex}",
    created_at=now,
    last_heartbeat=now,
    tags=request.tags,
    user_metadata=request.user_metadata,
    sdk_version=request.sdk_version,
    project_id=request.project_id,
  )
  ttl_seconds = int(os.getenv("OPEN_RL_SESSION_TTL_SECONDS", "300"))
  if ttl_seconds < 1:
    return JSONResponse(status_code=500, content={"error": "OPEN_RL_SESSION_TTL_SECONDS must be positive"})
  await put_client_session(get_store(), session, ttl_seconds)
  return {"session_id": session.session_id, "type": "create_session"}


@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(req: dict):
  request = SessionHeartbeatRequest.model_validate(req)
  s = get_store()
  session = await get_client_session(s, request.session_id)
  if session is None:
    return JSONResponse(status_code=404, content={"error": "session not found"})
  session.last_heartbeat = time.time()
  ttl_seconds = int(os.getenv("OPEN_RL_SESSION_TTL_SECONDS", "300"))
  await put_client_session(s, session, ttl_seconds)
  return {"type": "session_heartbeat"}


def _get_request(request: Request) -> Request:
  return request


@app.post("/api/v1/create_model")
async def create_model(
  req: dict[str, Any],
  request: Request | None = Depends(_get_request),  # noqa: B008
) -> dict[str, Any]:
  """ServiceClient.create_lora_training_client_async()"""
  try:
    model_id = await _extract_and_persist_model_metadata(req, request, default_training_kind="full")
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  command = make_training_request(
    "create_model",
    model_id,
    {},
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/delete_model")
async def delete_model(req: dict):
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  await request_model_stop(model_id, request_store=store, preserve_metadata=False)
  return {"status": "ok"}


async def request_model_stop(
  model_id: str,
  *,
  request_store=None,
  preserve_metadata: bool,
) -> None:
  """Gracefully stop a run's workers while optionally retaining UI history."""
  target_store = request_store or store
  if is_fft_enabled():
    print(f"[GATEWAY] Requesting shutdown of workers for model {model_id}...")
    await target_store.put_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id, "op": "shutdown_workers"})
    await target_store.put_sampling_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id})
  stopped_at = time.time()
  await update_run_metadata(
    target_store,
    model_id,
    {"stopped_at": stopped_at},
    update_active=preserve_metadata,
  )
  await report_control_event(
    target_store,
    model_id,
    component="gateway",
    phase="stopped",
    status="stopped",
    message="Run workers were asked to stop",
  )
  if not preserve_metadata:
    await target_store.delete_values(f"{MODEL_META_PREFIX}{model_id}", f"open_rl:model_base:{model_id}", model_revision_key(model_id))
  for storage_ref in await prune_sampler_snapshots(target_store, model_id, keep_ephemeral=0):
    relative = storage_ref[len("tinker://") :] if storage_ref.startswith("tinker://") else storage_ref.lstrip("/")
    local_path = os.path.join(TMP_DIR, "sampler_full", relative)
    if os.path.isdir(local_path):
      await asyncio.to_thread(shutil.rmtree, local_path)


@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(
  req: dict[str, Any],
  request: Request | None = Depends(_get_request),  # noqa: B008
) -> dict[str, Any]:
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.get("state_path")
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "state_path is required"})
  # Resolve relative names under TMP_DIR/checkpoints, leave absolute paths alone.
  resolved_path = state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path)
  try:
    model_id = await _extract_and_persist_model_metadata(req, request, default_training_kind="restored")
  except ValueError as exc:
    return JSONResponse(status_code=400, content={"error": str(exc)})

  command = make_training_request(
    "create_model_from_state",
    model_id,
    {
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("restore_optimizer", False)),
    },
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/get_info")
async def get_info(req: dict):
  """ServiceClient — model metadata for the training client."""
  model_name = get_default_model_name()
  if not model_name:
    return JSONResponse(status_code=404, content={"error": "No base model is configured"})
  # SDK compatibility: the public client currently expects LoRA-shaped training metadata,
  # even when this process is running a full fine-tuning worker.
  result = {
    "model_data": {"arch": "unknown", "model_name": model_name, "tokenizer_id": model_name},
    "model_id": req.get("model_id", "model-live-123"),
    "is_lora": True,
    "lora_rank": 16,
    "model_name": model_name,
    "type": "get_info",
  }
  return result


@app.post("/api/v1/retrieve_future")
async def retrieve_future(req: dict):
  """ServiceClient — poll for async request results."""
  request_id = req.get("request_id")
  if not request_id:
    return JSONResponse(status_code=400, content={"error": "request_id is required"})

  result = await store.get_future(request_id, timeout=60.0)
  if result is None:
    return JSONResponse(status_code=400, content={"type": "RequestFailedResponse", "error_message": "Future not found"})
  if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
    return JSONResponse(status_code=400, content=result)
  if isinstance(result, dict):
    await update_metadata_from_future(result)
    return translate_future_result(result)
  return result


# *** TrainingClient endpoints ***
@app.post("/api/v1/forward")
async def forward(req: dict):
  """TrainingClient.forward_async()"""
  fwd_input = req.get("forward_input") or req.get("forward_backward_input") or {}
  req_id = await enqueue(
    make_training_request(
      "forward_backward",
      req.get("model_id"),
      {
        "data": fwd_input.get("data", []),
        "loss_fn": fwd_input.get("loss_fn", "cross_entropy"),
        "loss_config": fwd_input.get("loss_fn_config", {}),
      },
      request_id=ordered_request_id(req, "forward_backward", req.get("model_id")),
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
  """TrainingClient.forward_backward_async()"""
  fwd_input = req.get("forward_backward_input", {})
  req_id = await enqueue(
    make_training_request(
      "forward_backward",
      req.get("model_id"),
      {
        "data": fwd_input.get("data", []),
        "loss_fn": fwd_input.get("loss_fn", "cross_entropy"),
        "loss_config": fwd_input.get("loss_fn_config", {}),
      },
      request_id=ordered_request_id(req, "forward_backward", req.get("model_id")),
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
  """TrainingClient.optim_step_async()"""
  req_id = await enqueue(
    make_training_request(
      "optim_step",
      req.get("model_id"),
      {"adam_params": req.get("adam_params", {})},
      request_id=ordered_request_id(req, "optim_step", req.get("model_id")),
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights_for_sampler")
async def save_weights_for_sampler(req: dict):
  """TrainingClient.save_weights_for_sampler().

  The SDK uses this for both named sampler checkpoints and ephemeral
  save_weights_and_get_sampling_client() snapshots. Route it through the training
  queue so the sampler always sees weights saved after prior training requests.
  """
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  await ensure_sampler_launched(model_id)
  seq_id = req.get("sampling_session_seq_id")
  if seq_id is None:
    seq_id = uuid.uuid4().hex
  alias = req.get("name") or req.get("alias") or req.get("path")

  if alias:
    session_id = alias if alias.startswith("tinker://") else sampler_weights_path(model_id, alias)
  else:
    session_id = sampler_session_id(model_id, seq_id)
  req_id = await enqueue(
    make_training_request(
      "save_weights_for_sampler",
      model_id,
      {
        "alias": alias,
        "path": session_id if alias else None,
        "sampling_session_id": session_id,
        "ttl_seconds": req.get("ttl_seconds"),
        "skip_checkpoint": is_mock_sampler(),
      },
      request_id=ordered_request_id(req, "save_weights_for_sampler", model_id),
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/save_weights")
async def save_weights(req: dict):
  """TrainingClient.save_weights() / save_state().

  This is the endpoint the tinker SDK hits for both save_weights() and save_state().
  The SDK sends save_state(name) as `path`; we resolve that checkpoint name to
  TMP_DIR/checkpoints/<model_id>/weights/<path> so separate training jobs do not
  overwrite each other's named checkpoints.
  """
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})

  seq_id = req.get("seq_id") or int(time.time() * 1000)
  alias = req.get("path") or f"{model_id}-samp-{seq_id}"
  state_path = checkpoint_state_path(model_id, alias)

  req_id = ordered_request_id(req, "save_state", model_id) or str(uuid.uuid4())
  await enqueue(
    make_training_request(
      "save_state",
      model_id,
      {
        "state_path": state_path,
        "include_optimizer": bool(req.get("include_optimizer", False)),
        "kind": "weights",
      },
      request_id=req_id,
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/load_weights")
async def load_weights(req: dict):
  """TrainingClient.load_state() / load_state_with_optimizer()."""
  model_id = req.get("model_id")
  state_path = req.get("path")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "path is required"})

  resolved_path = checkpoint_state_path(model_id, state_path)
  req_id = await enqueue(
    make_training_request(
      "load_weights",
      model_id,
      {
        "state_path": resolved_path,
        "restore_optimizer": bool(req.get("optimizer", False)),
      },
      request_id=ordered_request_id(req, "load_weights", model_id),
    )
  )
  return {"request_id": req_id}


# *** SamplingClient endpoints ***
@app.post("/api/v1/create_sampling_session")
async def create_sampling_session(req: dict):
  """ServiceClient.create_sampling_client()"""
  model_path = req.get("model_path")
  base_model = req.get("base_model")
  model_id = req.get("model_id")

  if model_path and model_path.startswith("tinker://"):
    sess_id = model_path
    path = model_path[len("tinker://") :]
    parts = path.split("/")
    target_model_id = parts[0]
  elif base_model:
    sess_id = base_model
    target_model_id = base_model
  else:
    sess_id = model_id or "samp-session-live-123"
    target_model_id = sess_id

  if is_sampler_weights_ref(sess_id):
    snapshot = await get_sampler_snapshot(get_store(), sess_id)
    if snapshot is None or (snapshot.expires_at is not None and snapshot.expires_at <= time.time()):
      return JSONResponse(status_code=404, content={"error": "sampling checkpoint is expired or unknown"})

  if get_sampler_backend() == "vllm" and target_model_id:
    await wait_for_sampler_ready(target_model_id, base_model)

  return {"sampling_session_id": sess_id, "type": "create_sampling_session"}


@app.post("/api/v1/asample")
async def asample(req: dict):
  """SamplingClient.sample_async()"""
  chunks = req.get("prompt", {}).get("chunks", [])
  prompt = []
  for chunk in chunks:
    prompt.extend(chunk.get("tokens", []))
  params = req.get("sampling_params", {})
  max_tokens = params.get("max_tokens", 20)
  temperature = params.get("temperature", 1.0)
  stop = params.get("stop")
  top_p = params.get("top_p", 1.0)
  top_k = params.get("top_k", -1)
  num_samples = req.get("num_samples", 1)
  include_prompt_logprobs = req.get("prompt_logprobs", req.get("include_prompt_logprobs", False))

  model_id = req.get("model_id") or req.get("sampling_session_id")
  base_model_id = base_model_id_from_sampling_ref(model_id)

  if get_sampler_backend() == "torch":
    req_id = await enqueue(
      make_training_request(
        "sample",
        base_model_id or model_id,
        {
          "prompt_tokens": prompt,
          "max_tokens": max_tokens,
          "temperature": temperature,
          "num_samples": num_samples,
          "prompt_logprobs": bool(include_prompt_logprobs),
        },
        request_id=ordered_request_id(req, "sample", model_id),
      )
    )
    return {"request_id": req_id}

  # vLLM backend
  req_id = ordered_request_id(req, "sample", model_id) or str(uuid.uuid4())
  if not await claim_request(store, req_id):
    return {"request_id": req_id}
  carrier: dict = {}
  propagate.inject(carrier)

  acquired_snapshot_id = None
  if is_sampler_weights_ref(model_id):
    snapshot = await acquire_sampler_snapshot(store, model_id)
    if snapshot is None:
      await release_request_claim(store, req_id)
      return JSONResponse(status_code=404, content={"error": "sampling session is expired or unknown"})
    acquired_snapshot_id = snapshot.sampling_session_id
    storage_ref = snapshot.storage_path
    rel_path = storage_ref[len("tinker://") :] if storage_ref.startswith("tinker://") else storage_ref.lstrip("/")
    local_path = os.path.join(TMP_DIR, "sampler_full", rel_path)
    if is_fft_enabled():
      weights_path = local_path
      weights_revision = f"{snapshot.model_id}:{snapshot.revision}"
      lora_id = None
      lora_path = None
    else:
      weights_path = None
      weights_revision = None
      lora_id = snapshot.sampling_session_id
      lora_path = local_path
  elif is_fft_enabled():
    weights_path = None
    weights_revision = None
    lora_id = None
    lora_path = None
  else:
    weights_path = None
    weights_revision = None
    lora_id = model_id
    lora_path = None

  sampling_req = {
    "request_id": req_id,
    "prompt_token_ids": prompt,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "stop": stop,
    "top_p": top_p,
    "top_k": top_k,
    "num_samples": num_samples,
    "lora_id": lora_id,
    "lora_path": lora_path,
    "weights_path": weights_path,
    "weights_revision": weights_revision,
    "sampling_session_id": acquired_snapshot_id,
    "include_prompt_logprobs": include_prompt_logprobs,
    "model_id": base_model_id or model_id,
    "trace_context": carrier,
  }

  await store.set_future(req_id, {"status": "pending"})
  try:
    await store.put_sampling_request(sampling_req)
  except Exception:
    if acquired_snapshot_id is not None:
      await release_sampler_snapshot(store, acquired_snapshot_id)
    await release_request_claim(store, req_id)
    raise
  await report_control_event(
    store,
    base_model_id or model_id,
    component="sampler",
    phase="sample_queued",
    status="queued",
    message=f"Queued sampling batch ({num_samples} completion{'s' if num_samples != 1 else ''})",
    details={"request_id": req_id, "num_samples": num_samples, "max_tokens": max_tokens},
  )
  return {"request_id": req_id}


# *** CLI endpoints ***


@app.get("/api/v1/list_adapters")
async def list_adapters():
  """CLI `list` — scan the peft directory for saved adapters."""
  import json

  peft_dir = os.path.join(TMP_DIR, "peft")
  adapters = []

  if os.path.exists(peft_dir):
    for entry in sorted(os.scandir(peft_dir), key=lambda e: e.stat().st_ctime, reverse=True):
      if not entry.is_dir():
        continue
      info = {"model_id": entry.name, "created_at": entry.stat().st_ctime, "timestamp": entry.stat().st_ctime, "alias": None}
      metadata_path = os.path.join(entry.path, "metadata.json")
      if os.path.exists(metadata_path):
        try:
          with open(metadata_path) as f:
            info.update(json.load(f))
        except Exception:
          pass
      adapters.append(info)

  return {"adapters": adapters}


# *** Internal ***


@app.post("/api/v1/telemetry")
async def telemetry(payload: dict):
  raw_run_id = payload.get("run_id") or payload.get("model_id")
  if not raw_run_id:
    return {"status": "accepted"}
  run_id = str(raw_run_id)[:256]
  known_run = await store.get_value(f"open_rl:model_meta:{run_id}") is not None
  if not known_run:
    known_run = bool(await store.get_control_events(run_id, limit=1))
  if not known_run:
    # Preserve the fire-and-forget telemetry contract without allowing callers
    # to create an unbounded number of phantom control-plane runs.
    return {"status": "accepted"}

  raw_metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
  metrics: dict[str, float] = {}
  for raw_name, raw_value in raw_metrics.items():
    if len(metrics) >= 128:
      break
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float) or not math.isfinite(float(raw_value)):
      continue
    name = str(raw_name)[:128]
    if name:
      metrics[name] = float(raw_value)
  # Telemetry is an external client input. Keep it from impersonating worker
  # lifecycle events or storing unbounded strings in the control-event log.
  status = str(payload.get("status") or "running").lower()
  if status not in {"queued", "starting", "waiting", "running", "ready", "completed", "failed", "stopped"}:
    status = "running"
  await report_control_event(
    store,
    run_id,
    component="client",
    phase=str(payload.get("phase") or payload.get("event") or "telemetry")[:128],
    status=status,
    message=str(payload.get("message") or "Client telemetry received")[:2048],
    details={"metrics": metrics},
  )
  return {"status": "accepted"}
