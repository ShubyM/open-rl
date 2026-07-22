# This file contains the FastAPI server entry point and request handlers for the Open-RL API backend.

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from server import external_sampler
from server.external_sampler import get_sampler_base_url
from server.store import get_store
from server.worker_manager import WorkerManager, create_fft_worker_manager


@dataclass
class TrainingModelMetadata:
  base_model: str
  created_at: float
  training_kind: str
  user_metadata: dict | None = None


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
  """Drop per-poll and per-turn access-log lines; anything non-200 stays."""

  NOISY = ("retrieve_future", "session_heartbeat", "asample")

  def filter(self, record: logging.LogRecord) -> bool:
    msg = record.getMessage()
    if not any(endpoint in msg for endpoint in self.NOISY):
      return True
    return " 200" not in msg


logging.getLogger("uvicorn.access").addFilter(FilterNoisyEndpoints())

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


# *** Helpers ***


def is_single_process_mode() -> bool:
  return bool(os.getenv("BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  if sampling_backend := os.getenv("SAMPLING_BACKEND"):
    return sampling_backend.lower()
  if get_sampler_base_url():
    return "vllm"
  return "torch" if is_single_process_mode() else "vllm"


def get_default_model_name() -> str | None:
  return os.getenv("BASE_MODEL")


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def sampler_session_id(model_id: str, seq_id: int | str) -> str:
  return f"tinker://{model_id}/sampler_weights/sampler-{seq_id}"


def sampler_weights_path(model_id: str, name: str) -> str:
  return f"tinker://{model_id}/sampler_weights/{name}"


def tinker_state_path(model_id: str, name: str) -> str:
  """Public, resume-stable form of a training checkpoint path."""
  if name.startswith("tinker://") or os.path.isabs(name):
    return name
  return f"tinker://{model_id}/weights/{name}"


def resolve_state_ref(ref: str | None) -> str | None:
  """Resolve a tinker://<model_id>/weights/<name> checkpoint ref to its local path."""
  if not ref or not ref.startswith("tinker://"):
    return None
  parts = ref[len("tinker://") :].split("/")
  if len(parts) >= 3 and parts[1] == "weights":
    return os.path.join(TMP_DIR, "checkpoints", parts[0], "weights", *parts[2:])
  return None


def checkpoint_state_path(model_id: str, name: str) -> str:
  resolved = resolve_state_ref(name)
  if resolved:
    return resolved
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


def sampler_adapter_path(session_ref: str) -> str:
  """Immutable adapter dir for one sampler snapshot: peft/<model>/<label>.

  Each save_weights_for_sampler writes a fresh directory named after the
  session's last segment, so concurrent rollouts on an older session keep a
  stable, fully written adapter dir.
  """
  parts = session_ref[len("tinker://") :].split("/")
  return os.path.join(TMP_DIR, "peft", parts[0], parts[2])


def is_sampler_weights_ref(model_id: str | None) -> bool:
  if not model_id or not model_id.startswith("tinker://"):
    return False

  path = model_id[len("tinker://") :]
  parts = path.split("/")
  return len(parts) >= 3 and parts[1] == "sampler_weights"


def make_training_request(
  op: str,
  model_id: str | None,
  payload: dict,
  request_id: str | None = None,
) -> dict:
  request = {
    "request_id": request_id or str(uuid.uuid4()),
    "op": op,
    "payload": payload,
  }
  if model_id is not None:
    request["model_id"] = model_id
  return request


async def enqueue(request: dict) -> str:
  """Create a pending future, inject trace context, push to store. Returns req_id."""
  request_id = request["request_id"]
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(request_id, {"status": "pending"})
  await store.put_request({**request, "trace_context": carrier})
  return request_id


# The tinker SDK auto-retries mutating POSTs on timeouts and 5xx responses
# (tinker/lib/retry_handler.py). Enqueueing a retried request again would run
# it twice — double gradients on forward_backward, a doubled learning rate on
# optim_step. Mutating requests carry a per-training-client monotonically
# increasing seq_id, and model_ids are freshly minted UUIDs per session, so
# (op, model_id, seq_id) uniquely identifies one logical request; a repeat is
# always a client retry and gets the original request_id back. The TTL only
# bounds memory. Callers must remember_request() before any await follows
# absorb_retry(), so a concurrent duplicate cannot slip between the two.
RETRY_DEDUPE_TTL_SECONDS = 30 * 60.0
RETRY_DEDUPE_MAX_ENTRIES = 4096
enqueued_requests: OrderedDict[tuple[str, str, int], tuple[str, float]] = OrderedDict()


def retry_dedupe_key(op: str, req: dict) -> tuple[str, str, int] | None:
  model_id = req.get("model_id")
  seq_id = req.get("seq_id", req.get("sampling_session_seq_id"))
  if not model_id or seq_id is None:
    return None
  return (op, str(model_id), int(seq_id))


def absorb_retry(op: str, req: dict) -> str | None:
  """Return the original request_id if this (op, model_id, seq_id) was already enqueued."""
  key = retry_dedupe_key(op, req)
  if key is None:
    return None
  entry = enqueued_requests.get(key)
  if entry is None or time.monotonic() - entry[1] >= RETRY_DEDUPE_TTL_SECONDS:
    return None
  print(f"[Gateway] Absorbed retried {op} (model={key[1]}, seq_id={key[2]}); returning original request {entry[0]}")
  return entry[0]


def remember_request(op: str, req: dict, request_id: str) -> None:
  key = retry_dedupe_key(op, req)
  if key is None:
    return
  enqueued_requests[key] = (request_id, time.monotonic())
  enqueued_requests.move_to_end(key)
  while len(enqueued_requests) > RETRY_DEDUPE_MAX_ENTRIES:
    enqueued_requests.popitem(last=False)


async def launch_worker_and_enqueue(request: dict) -> str:
  """Ensure the model's dedicated trainer worker exists, then enqueue onto its queue.

  The launcher is idempotent per model_id, and Kubernetes (or the local process
  table) owns the worker's lifecycle from here; there is no separate launch
  queue. Launch failures resolve the future immediately so clients don't long-poll
  a request that can never be served.
  """
  assert fft_worker_manager is not None, "FFT worker manager is initialized by the app lifespan when FFT is enabled"
  request_id = request["request_id"]
  base_model = request.get("payload", {}).get("base_model")
  await store.set_future(request_id, {"status": "pending"})
  try:
    await asyncio.to_thread(fft_worker_manager.launch_trainer, request["model_id"], base_model)
  except Exception as exc:
    traceback.print_exc()
    await store.set_future(request_id, {"type": "RequestFailedResponse", "error_message": str(exc)})
    return request_id
  return await enqueue(request)


async def ensure_sampler_launched(model_id: str, base_model: str | None = None) -> None:
  if get_sampler_base_url():
    return  # externally managed vLLM server; nothing to launch
  if fft_worker_manager is not None and get_sampler_backend() == "vllm":
    if not base_model:
      s = get_store()
      val = await s.get_value(f"open_rl:model_meta:{model_id}") or await s.get_value(f"open_rl:model_base:{model_id}")
      if val:
        try:
          meta = json.loads(val)
          base_model = meta.get("base_model") if isinstance(meta, dict) else val
        except Exception:
          base_model = val
    try:
      await asyncio.to_thread(fft_worker_manager.launch_sampler, model_id, base_model)
    except Exception:
      traceback.print_exc()


async def preflight_vllm() -> None:
  """Reject unusable vllm-backend configurations in single-process mode.

  Without SAMPLER_BASE_URL, vLLM sampler workers are separate processes fed
  from the shared request store; the single-process in-memory store cannot
  reach them, so the first asample would hang forever. With SAMPLER_BASE_URL
  the gateway calls the external server directly, which works in any mode —
  just verify it is actually up.
  """
  if get_sampler_backend() != "vllm":
    return
  if base_url := get_sampler_base_url():
    await external_sampler.preflight(base_url)
    return
  raise RuntimeError(
    "SAMPLING_BACKEND=vllm requires queue mode (set REDIS_URL so the gateway can "
    "launch and feed per-model vLLM sampler workers) or SAMPLER_BASE_URL pointing "
    "at an externally launched `vllm serve`. Single-process mode without either "
    "samples with the in-process torch backend."
  )


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


@asynccontextmanager
async def lifespan(_: FastAPI):
  global fft_worker_manager
  task = None
  # The per-model vLLM sampler workers are launched on demand for FFT *and*
  # LoRA models alike; the manager just needs the shared queue store. With an
  # external SAMPLER_BASE_URL server there is nothing to launch (FFT still
  # needs the manager for its trainer workers).
  if is_fft_enabled() or (get_sampler_backend() == "vllm" and not is_single_process_mode() and not get_sampler_base_url()):
    fft_worker_manager = create_fft_worker_manager()
  if get_sampler_base_url() and get_sampler_backend() == "vllm" and not is_single_process_mode():
    for sampler_url in external_sampler.get_sampler_base_urls():
      await external_sampler.preflight(sampler_url)
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
  if os.getenv("OPEN_RL_EXTERNAL_TRAINER") == "1":
    # A dedicated trainer process (torchrun for OPEN_RL_FSDP_WORLD_SIZE > 1)
    # drains the training queue; running the in-gateway worker too would race
    # it for requests.
    if not os.getenv("REDIS_URL"):
      raise RuntimeError(
        "OPEN_RL_EXTERNAL_TRAINER=1 requires REDIS_URL: a dedicated trainer "
        "process can only share the queue through Redis."
      )
    print("-> Training: dedicated external trainer process (in-gateway worker disabled)")
  elif not is_fft_enabled():
    from server import training_requests_processor

    worker = training_requests_processor.LoraTrainingWorker()
    base_model = os.getenv("BASE_MODEL")
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
async def create_session(_: dict):
  return {"session_id": "sess-real-123", "type": "create_session"}


@app.post("/api/v1/session_heartbeat")
async def session_heartbeat(_: dict):
  return {"type": "session_heartbeat"}


@app.post("/api/v1/create_model")
async def create_model(req: dict):
  """ServiceClient.create_lora_training_client_async()"""
  base_model = req.get("base_model")
  if not base_model:
    return JSONResponse(status_code=400, content={"error": "base_model is required in request payload"})
  model_id = str(uuid.uuid4())
  s = get_store()
  meta_obj = TrainingModelMetadata(
    base_model=base_model, created_at=time.time(), training_kind="full", user_metadata=req.get("user_metadata") or None
  )
  await s.set_value(f"open_rl:model_meta:{model_id}", json.dumps(asdict(meta_obj)))
  await s.set_value(f"open_rl:model_base:{model_id}", base_model)
  command = make_training_request(
    "create_model",
    model_id,
    {
      "base_model": base_model,
      "lora_config": req.get("lora_config") or {},
      "full_config": req.get("full_config") or {},
    },
    request_id=model_id,
  )
  req_id = await launch_worker_and_enqueue(command) if is_fft_enabled() else await enqueue(command)
  return {"request_id": req_id}


@app.post("/api/v1/delete_model")
async def delete_model(req: dict):
  model_id = req.get("model_id")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  if is_fft_enabled():
    print(f"[GATEWAY] Requesting shutdown of workers for model {model_id}...")
    await store.put_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id, "op": "shutdown_workers"})
    await store.put_sampling_request({"request_id": "SHUTDOWN_SENTINEL", "model_id": model_id})
  await store.delete_values(f"open_rl:model_meta:{model_id}", f"open_rl:model_base:{model_id}")
  return {"status": "ok"}


@app.post("/api/v1/create_model_from_state")
async def create_model_from_state(req: dict):
  """ServiceClient.create_training_client_from_state_async()"""
  state_path = req.get("state_path")
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "state_path is required"})
  # Resolve relative names under TMP_DIR/checkpoints, leave absolute paths alone.
  resolved_path = resolve_state_ref(state_path) or (state_path if os.path.isabs(state_path) else os.path.join(TMP_DIR, "checkpoints", state_path))
  model_id = str(uuid.uuid4())
  base_model = req.get("base_model")
  s = get_store()
  meta_obj = TrainingModelMetadata(
    base_model=base_model or get_default_model_name() or "unknown",
    created_at=time.time(),
    training_kind="restored",
    user_metadata=req.get("user_metadata") or None,
  )
  await s.set_value(f"open_rl:model_meta:{model_id}", json.dumps(asdict(meta_obj)))
  if base_model:
    await s.set_value(f"open_rl:model_base:{model_id}", base_model)
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


@app.get("/api/v1/training_runs/{training_run_id}")
async def get_training_run(training_run_id: str):
  """RestClient.get_training_run / get_training_run_by_tinker_path.

  tinker-cookbook reads `user_metadata` from here on resume to verify the
  checkpoint's renderer matches the configured one ("renderer metadata").
  """
  raw = await get_store().get_value(f"open_rl:model_meta:{training_run_id}")
  if not raw:
    return JSONResponse(status_code=404, content={"error": f"training run {training_run_id} not found"})
  try:
    meta = json.loads(raw)
    if not isinstance(meta, dict):
      meta = {"base_model": str(meta)}
  except ValueError:
    meta = {"base_model": raw}

  created_at = float(meta.get("created_at") or time.time())
  return {
    "training_run_id": training_run_id,
    "base_model": meta.get("base_model") or "unknown",
    "model_owner": "local",
    "is_lora": not is_fft_enabled(),
    "corrupted": False,
    "lora_rank": None,
    "last_request_time": datetime.fromtimestamp(created_at, tz=UTC).isoformat(),
    "last_checkpoint": None,
    "last_sampler_checkpoint": None,
    "user_metadata": meta.get("user_metadata"),
  }


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
    return translate_future_result(result)
  return result


# *** TrainingClient endpoints ***
@app.post("/api/v1/forward")
async def forward(req: dict):
  """TrainingClient.forward_async() — forward WITHOUT gradient accumulation.

  The SDK's custom-loss flow (cookbook DPO) sends real weights here and
  expects no server-side gradients; the linearized gradient arrives as a
  separate forward_backward afterwards."""
  if (replayed := absorb_retry("forward", req)) is not None:
    return {"request_id": replayed}
  fwd_input = req.get("forward_input") or req.get("forward_backward_input") or {}
  request = make_training_request(
    "forward_backward",
    req.get("model_id"),
    {
      "data": fwd_input.get("data", []),
      "loss_fn": fwd_input.get("loss_fn", "cross_entropy"),
      "loss_config": fwd_input.get("loss_fn_config", {}),
      "forward_only": True,
    },
  )
  remember_request("forward", req, request["request_id"])
  return {"request_id": await enqueue(request)}


@app.post("/api/v1/forward_backward")
async def forward_backward(req: dict):
  """TrainingClient.forward_backward_async()"""
  if (replayed := absorb_retry("forward_backward", req)) is not None:
    return {"request_id": replayed}
  fwd_input = req.get("forward_backward_input", {})
  request = make_training_request(
    "forward_backward",
    req.get("model_id"),
    {
      "data": fwd_input.get("data", []),
      "loss_fn": fwd_input.get("loss_fn", "cross_entropy"),
      "loss_config": fwd_input.get("loss_fn_config", {}),
    },
  )
  remember_request("forward_backward", req, request["request_id"])
  return {"request_id": await enqueue(request)}


@app.post("/api/v1/optim_step")
async def optim_step(req: dict):
  """TrainingClient.optim_step_async()"""
  if (replayed := absorb_retry("optim_step", req)) is not None:
    return {"request_id": replayed}
  request = make_training_request(
    "optim_step",
    req.get("model_id"),
    {"adam_params": req.get("adam_params", {})},
  )
  remember_request("optim_step", req, request["request_id"])
  return {"request_id": await enqueue(request)}


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

  if (replayed := absorb_retry("save_weights_for_sampler", req)) is not None:
    return {"request_id": replayed}
  await ensure_sampler_launched(model_id)
  seq_id = req.get("sampling_session_seq_id") or int(time.time() * 1000)
  alias = req.get("name") or req.get("alias") or req.get("path")

  session_id = sampler_session_id(model_id, seq_id)
  request = make_training_request(
    "save_weights_for_sampler",
    model_id,
    {
      "alias": alias,
      "path": sampler_weights_path(model_id, alias) if alias else None,
      "sampling_session_id": session_id,
    },
  )
  remember_request("save_weights_for_sampler", req, request["request_id"])
  return {"request_id": await enqueue(request)}


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

  if (replayed := absorb_retry("save_weights", req)) is not None:
    return {"request_id": replayed}
  seq_id = req.get("seq_id") or int(time.time() * 1000)
  alias = req.get("path") or f"{model_id}-samp-{seq_id}"
  state_path = checkpoint_state_path(model_id, alias)

  req_id = str(uuid.uuid4())
  remember_request("save_weights", req, req_id)
  await enqueue(
    make_training_request(
      "save_state",
      model_id,
      {
        "state_path": state_path,
        # Returned to the client (and written to its checkpoints.jsonl); the
        # tinker:// form is what create_model_from_state/load_weights resolve
        # on resume, so it must round-trip.
        "public_path": tinker_state_path(model_id, alias),
        "include_optimizer": bool(req.get("include_optimizer", False)),
        "kind": "weights",
      },
      request_id=req_id,
    )
  )
  return {"request_id": req_id}


@app.post("/api/v1/weights_info")
async def weights_info(req: dict):
  """ServiceClient.create_training_client_from_state() resume flow.

  The SDK validates a tinker:// checkpoint ref here before loading it; the
  route was missing and the SDK's retry loop turned that into repeating 404s
  on every resume attempt.
  """
  tinker_path = req.get("tinker_path") or req.get("path")
  state_path = resolve_state_ref(tinker_path)
  if not state_path or not os.path.isdir(state_path):
    return JSONResponse(status_code=404, content={"error": f"No checkpoint at {tinker_path!r}"})

  metadata = {}
  metadata_file = os.path.join(state_path, "metadata.json")
  if os.path.isfile(metadata_file):
    with open(metadata_file, encoding="utf-8") as f:
      metadata = json.load(f)

  lora_rank = None
  train_unembed = None
  train_mlp = None
  for root, _dirs, files in os.walk(state_path):
    if "adapter_config.json" in files:
      with open(os.path.join(root, "adapter_config.json"), encoding="utf-8") as f:
        adapter_config = json.load(f)
      lora_rank = adapter_config.get("r")
      targets = adapter_config.get("target_modules") or []
      train_unembed = "lm_head" in targets
      train_mlp = any("gate_proj" in t or "up_proj" in t for t in targets)
      break

  base_model = metadata.get("base_model")
  if not base_model:
    return JSONResponse(status_code=404, content={"error": f"Checkpoint at {tinker_path!r} has no base_model metadata"})
  return {
    "base_model": base_model,
    "is_lora": lora_rank is not None,
    "lora_rank": lora_rank,
    "train_unembed": train_unembed,
    "train_mlp": train_mlp,
  }


@app.post("/api/v1/load_weights")
async def load_weights(req: dict):
  """TrainingClient.load_state() / load_state_with_optimizer()."""
  model_id = req.get("model_id")
  state_path = req.get("path")
  if not model_id:
    return JSONResponse(status_code=400, content={"error": "model_id is required"})
  if not state_path:
    return JSONResponse(status_code=400, content={"error": "path is required"})

  if (replayed := absorb_retry("load_weights", req)) is not None:
    return {"request_id": replayed}
  resolved_path = checkpoint_state_path(model_id, state_path)
  if is_fft_enabled() and fft_worker_manager is not None:
    # A resumed client can address a model whose trainer worker died with the
    # original run; without a worker the request sits in the model's queue
    # forever and the SDK polls "try again" indefinitely. launch_trainer is
    # idempotent per model_id, and a fresh worker restores itself from the
    # checkpoint this very request carries.
    base_model = await get_store().get_value(f"open_rl:model_base:{model_id}")
    try:
      await asyncio.to_thread(fft_worker_manager.launch_trainer, model_id, base_model)
    except Exception:
      traceback.print_exc()
  request = make_training_request(
    "load_weights",
    model_id,
    {
      "state_path": resolved_path,
      "restore_optimizer": bool(req.get("optimizer", False)),
    },
  )
  remember_request("load_weights", req, request["request_id"])
  return {"request_id": await enqueue(request)}


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

  if get_sampler_backend() == "vllm" and target_model_id:
    if is_fft_enabled():
      await ensure_sampler_launched(target_model_id, base_model)
    s = get_store()
    # Only managed queue workers (RedisStore) publish sampler_ready flags; the
    # in-memory store inherits redis=None, and an external SAMPLER_BASE_URL
    # server is preflighted at startup and never publishes one.
    if s.redis is not None and not get_sampler_base_url():
      print(f"[GATEWAY] Waiting for dynamic vLLM sampler worker to be ready for model {target_model_id}...")
      start_time = time.monotonic()
      while True:
        is_ready = await s.redis.get(f"open_rl:sampler_ready:{target_model_id}")
        if is_ready == "1" or is_ready == b"1":
          print(f"[GATEWAY] Dynamic vLLM sampler worker is ready! (took {time.monotonic() - start_time:.2f}s)")
          break
        if time.monotonic() - start_time > 300:
          raise TimeoutError("Timed out waiting for dynamic vLLM sampler worker to be ready")
        await asyncio.sleep(1)

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
      )
    )
    return {"request_id": req_id}

  # vLLM backend
  req_id = str(uuid.uuid4())
  carrier: dict = {}
  propagate.inject(carrier)
  await store.set_future(req_id, {"status": "pending"})

  if is_fft_enabled():
    rel_path = model_id[len("tinker://") :] if model_id.startswith("tinker://") else model_id.lstrip("/")
    local_path = os.path.join(TMP_DIR, "sampler_full", rel_path)
    weights_path = local_path
    lora_id = None
    lora_path = None
  else:
    weights_path = None
    lora_id = model_id
    lora_path = sampler_adapter_path(model_id) if is_sampler_weights_ref(model_id) else None

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
    "include_prompt_logprobs": include_prompt_logprobs,
    "model_id": base_model_id or model_id,
    "trace_context": carrier,
  }

  # Externally managed `vllm serve` (LoRA only — stock vLLM cannot hot-reload
  # FFT checkpoints, so FFT always uses the managed queue workers).
  if get_sampler_base_url() and not is_fft_enabled():
    task = asyncio.create_task(_sample_via_external_server(req_id, sampling_req))
    _external_sampler_tasks.add(task)
    task.add_done_callback(_external_sampler_tasks.discard)
    return {"request_id": req_id}

  await store.put_sampling_request(sampling_req)
  return {"request_id": req_id}


_external_sampler_tasks: set = set()


async def _sample_via_external_server(req_id: str, sampling_req: dict) -> None:
  try:
    result = await external_sampler.sample(sampling_req)
  except Exception as exc:
    traceback.print_exc()
    result = {"type": "RequestFailedResponse", "error_message": f"External vLLM sampler error: {exc}"}
  await store.set_future(req_id, result)


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
async def telemetry(_: dict):
  return {"status": "accepted"}
