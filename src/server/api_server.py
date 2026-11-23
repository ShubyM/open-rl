"""OpenRL HTTP API server with end-to-end binary protobuf passthrough."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from fastapi.utils import is_body_allowed_for_status_code
from opentelemetry import propagate, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import AliasChoices, BaseModel, BeforeValidator, Field, ValidationError, ValidationInfo, field_validator, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from server import openrl_status
from server import telemetry as run_records
from server.api import paths, settings, wire
from server.lora_snapshots import snapshot_name, snapshot_root
from server.model_metadata import TrainingModelMetadata, WeightSyncConfig, get_model_metadata, persist_model_metadata, pool_name
from server.observability.router import router as observability_router
from server.sampling_dispatcher import SAMPLE_DEADLINE_SECONDS
from server.sampling_store import SamplingStore
from server.session_registry import SessionRegistry
from server.store import RequestStore, StateStore, get_state_store, get_store
from server.vllm_options import split_stop
from server.worker_manager import LocalWorkerManager, WorkerManager, create_worker_manager, default_metadata, owner_id, owner_of, runtime_of
from training import commands
from training.commands import Command
from training.types import TRAINER_BACKENDS, FFTConfig, FineTuningType, LoraConfig

RELEASES_URL = "https://github.com/openrl-dev/openrl/releases"
EXAMPLE_MODELS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "google/gemma-4-e2b", "Qwen/Qwen3-8B"]
_META_SETTING_FIELDS = {"sampler_replicas", "exclusive", "trainer_backend", "trainer_device_count", "trainer_cp", "recipe", "client_log_id"}


# ***** requests *****


class CreateSessionRequest(BaseModel):
  tags: list[str] = Field(default_factory=list)
  user_metadata: dict[str, Any] | None = Field(default=None, description=settings.USER_METADATA_DOC)


_OR_EMPTY = BeforeValidator(lambda v: {} if v is None else v)


class _ModelBaseRequest(BaseModel):
  session_id: str | None = None
  user_metadata: dict[str, Any] | None = Field(default=None, description=settings.USER_METADATA_DOC)
  lora_config: Annotated[LoraConfig, _OR_EMPTY] = Field(default_factory=LoraConfig)
  full_config: Any = None  # removed; kept to reject loudly


class CreateModelRequest(_ModelBaseRequest):
  base_model: str


class CreateModelFromStateRequest(_ModelBaseRequest):
  state_path: str
  restore_optimizer: bool = False
  base_model: str | None = None


class ModelRequest(BaseModel):
  model_id: str


class GetInfoRequest(BaseModel):
  model_id: str | None = None


class _JsonDatum(BaseModel):
  model_input: list[int]
  loss_fn_inputs: dict[str, Any] = Field(default_factory=dict)

  @model_validator(mode="before")
  @classmethod
  def flatten_chunks(cls, raw: Any) -> Any:
    if isinstance(raw, dict) and isinstance(raw.get("model_input"), dict):
      return {**raw, "model_input": [t for ch in raw["model_input"].get("chunks", []) for t in ch.get("tokens", [])]}
    return raw


class ForwardBackwardInput(BaseModel):
  data: list[_JsonDatum] = Field(default_factory=list)
  loss_fn: str = "cross_entropy"
  loss_fn_config: Annotated[dict[str, Any], _OR_EMPTY] = Field(default_factory=dict)


class ForwardBackwardRequest(ModelRequest):
  seq_id: int = 0
  forward_backward_input: ForwardBackwardInput | None = None
  forward_input: ForwardBackwardInput | None = None
  forward_only: bool = False


class RetrieveFutureRequest(BaseModel):
  request_id: str


class OptimStepRequest(ModelRequest):
  adam_params: dict[str, Any] = {}


class SaveRequest(ModelRequest):
  seq_id: int | str | None = None
  sampling_session_seq_id: int | str | None = None
  name: str | None = None
  alias: str | None = None
  path: str | None = None


class LoadWeightsRequest(ModelRequest):
  path: str
  optimizer: bool = False


class WeightsInfoRequest(BaseModel):
  tinker_path: str


class SamplingParams(BaseModel):
  max_tokens: int = 20
  temperature: float = 1.0
  stop: Any = None
  top_p: float = 1.0
  top_k: int = -1
  seed: int | None = None

  @field_validator("max_tokens", "temperature", "top_p", "top_k", mode="before")
  @classmethod
  def default_sampling_param(cls, value: Any, info: ValidationInfo) -> Any:
    return cls.model_fields[info.field_name].default if value is None else value


class AsampleRequest(BaseModel):
  prompt: dict[str, Any] = {}
  sampling_params: SamplingParams = SamplingParams()
  num_samples: int = 1
  prompt_logprobs: bool = Field(default=False, validation_alias=AliasChoices("prompt_logprobs", "include_prompt_logprobs"))
  model_id: str | None = None
  sampling_session_id: str | None = None
  seq_id: int | None = None


# ***** api server *****


def runtime_jobs_key(runtime: str) -> str:
  return f"open_rl:runtime_jobs:{runtime}"


def _failed(message: str) -> dict[str, str]:
  return {"type": "RequestFailedResponse", "error_message": message}


async def _logged(coro: Any) -> None:
  try:
    await coro
  except Exception:
    traceback.print_exc()


async def http_error(_: Request, exc: StarletteHTTPException) -> Response:
  if not is_body_allowed_for_status_code(exc.status_code):
    return Response(status_code=exc.status_code, headers=exc.headers)
  return JSONResponse(status_code=exc.status_code, content={"error": exc.detail}, headers=exc.headers)


@dataclass
class ApiServer:
  """Dependency container for the OpenRL API server (zero mutable module globals)."""

  store: RequestStore = field(default_factory=get_store)
  state: StateStore = field(default_factory=get_state_store)
  worker_manager: WorkerManager | None = None
  session_registry: SessionRegistry | None = None
  tmp_dir: str | None = None
  sampler_backend: str | None = None
  open_rl_version: str = field(default_factory=lambda: os.getenv("OPEN_RL_VERSION") or "dev")
  saved_weights_wait_seconds: float = 2.0
  session_reap_interval_sec: float = 30.0
  reconcile_concurrency: int = 4
  owner_locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))
  slot_locks: defaultdict[str, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))

  def __post_init__(self) -> None:
    if self.session_registry is None:
      self.session_registry = SessionRegistry(self.state)

  def resolved_sampler_backend(self) -> str:
    return settings.get_sampler_backend() if self.sampler_backend is None else self.sampler_backend

  def worker_launcher_mode(self) -> str:
    if settings.is_single_process_mode():
      return "single_process"
    return (
      "none" if self.worker_manager is None else ("scheduler" if os.getenv("OPEN_RL_WORKER_MANAGER", "local").lower() == "scheduler" else "process")
    )

  def openrl_info(self) -> dict[str, Any]:
    ver, single = self.open_rl_version, settings.is_single_process_mode()
    limits = {"vllm_max_model_len": settings.env_int("VLLM_MAX_MODEL_LEN"), "train_token_budget": settings.env_int("OPEN_RL_TRAIN_TOKEN_BUDGET")}
    limits |= {"session_ttl_seconds": self.session_registry.ttl_seconds, "session_reap_interval_seconds": self.session_reap_interval_sec}
    info = {"version": ver, "worker_launcher": self.worker_launcher_mode(), "sampler_backend": self.resolved_sampler_backend()}
    info |= {"fft_enabled": settings.is_fft_enabled(), "settings_schema_url": "/openapi.json#/components/schemas/OpenRLSettings"}
    info |= {"models": {"any_huggingface_model": not single, "default": os.getenv("BASE_MODEL"), "examples": EXAMPLE_MODELS}, "limits": limits}
    info |= {"settings_howto": "Pass as user_metadata on create_session or create_model, or as TINKER_TAGS=openrl.key=value,..."}
    info |= {"skill_url": f"{RELEASES_URL}/download/{ver}/openrl-skill.tar.gz" if ver.startswith("v") else None}
    return info | {"getting_started_url": "https://github.com/openrl-dev/openrl/blob/main/docs/getting-started.md"}

  async def bind_session(self, session_id: str | None, model_id: str) -> None:
    if not session_id:
      return
    run_records.session_seen(session_id)
    owner = await asyncio.to_thread(owner_of, model_id)
    async with self.owner_locks[owner]:
      if self.worker_manager is not None:
        await self.session_registry.attach(session_id, owner)
      if (meta := await get_model_metadata(self.state, model_id)) is not None and session_id not in meta.session_ids:
        meta.session_ids.append(session_id)
        await persist_model_metadata(self.state, model_id, meta)

  async def reap_owner(self, owner: str) -> None:
    async with self.owner_locks[owner]:
      if await self.session_registry.in_use(owner) or self.worker_manager is None:
        return
      for model in await asyncio.to_thread(self.worker_manager.release_owner, owner):
        await self.state.delete_values(f"open_rl:trainer_ready:{model}")
        if self.resolved_sampler_backend() == "vllm":
          await SamplingStore(self.store.redis).delete_if_empty(pool_name(model))
      await self.session_registry.forget(owner)

  async def prune_runtime_jobs(self, runtime: str) -> int:
    key, live = runtime_jobs_key(runtime), 0
    for job in await self.state.set_members(key):
      meta = await get_model_metadata(self.state, job)
      sids = meta.session_ids if meta is not None else []
      if meta is not None and meta.status == "active" and (not sids or any([await self.session_registry.live(s) for s in sids])):
        live += 1
      else:
        await self.state.remove_from_set(key, job)
    return live

  async def persist_on_runtime_slot(self, model_id: str, meta: TrainingModelMetadata) -> None:
    cap = settings.env_int("OPEN_RL_LORA_JOBS_PER_RUNTIME") or 0
    if cap <= 0 or self.worker_manager is None or not meta.shares_runtime():
      await persist_model_metadata(self.state, model_id, meta)
      return
    async with self.slot_locks[meta.shared_runtime(0)]:
      slot = 0
      while await self.prune_runtime_jobs(meta.shared_runtime(slot)) >= cap:
        slot += 1
      meta.runtime_slot = slot
      await persist_model_metadata(self.state, model_id, meta)
      await self.state.add_to_set(runtime_jobs_key(meta.runtime(model_id)), model_id)

  async def extract_and_persist_model_metadata(
    self, req: CreateModelRequest | CreateModelFromStateRequest, request: Request | None = None, checkpoint_kind: FineTuningType | None = None
  ) -> tuple[str, TrainingModelMetadata]:
    if not req.base_model:
      raise ValueError("base_model is required in request payload")
    headers = request.headers if request is not None else {}
    for hdr, name in (("x-open-rl-fine-tuning-type", "fine_tuning_type"), ("x-open-rl-weight-sync-strategy", "weight_sync")):
      if hdr in headers:
        raise ValueError(f"The {hdr} header was removed; set {settings.SETTINGS_PREFIX}{name} in TINKER_TAGS or user_metadata")
    if req.full_config is not None:
      raise ValueError(f"full_config was removed; set {settings.SETTINGS_PREFIX}fft_seed and {settings.SETTINGS_PREFIX}fft_cpu_offload")
    cfg = settings.resolve_settings(req.user_metadata or {}, await self.session_registry.user_metadata(req.session_id))
    ft = cfg.fine_tuning_type or checkpoint_kind or "lora"
    fft_given = [n for n in ("weight_sync", "fft_seed", "fft_cpu_offload") if getattr(cfg, n) is not None]
    p, wm, is_local = settings.SETTINGS_PREFIX, self.worker_manager, isinstance(self.worker_manager, LocalWorkerManager)
    n_dev, cp, ck_ft = cfg.trainer_device_count, cfg.trainer_cp, cfg.fine_tuning_type
    for bad, msg in (
      (checkpoint_kind and ck_ft not in (None, checkpoint_kind), f"{p}fine_tuning_type={ck_ft} does not match the {checkpoint_kind} checkpoint"),
      (ft == "full" and not settings.is_fft_enabled(), "Full Fine-Tuning (FFT) is disabled on this Open-RL API server instance"),
      (ft != "full" and bool(fft_given), ", ".join(f"{p}{n}" for n in fft_given) + " applies to fine_tuning_type=full only"),
      (cfg.exclusive and wm is None, "openrl.exclusive needs a server that launches workers per model"),
      (cfg.sampler_replicas > 1 and is_local, "openrl.sampler_replicas above 1 needs a server that launches workers as pods"),
      (cfg.trainer_backend != "pytorch" and wm is None, f"openrl.trainer_backend={cfg.trainer_backend} needs a server that launches workers"),
      (cfg.trainer_backend not in TRAINER_BACKENDS and is_local, "A trainer image needs a server that launches workers as pods"),
      (cfg.trainer_backend == "automodel" and ft != "lora", "The automodel trainer supports LoRA only"),
      (n_dev > 1 and cfg.trainer_backend != "automodel", "openrl.trainer_device_count above 1 needs openrl.trainer_backend=automodel"),
      (n_dev > 1 and is_local, "openrl.trainer_device_count above 1 needs a server that launches workers as pods"),
      (bool(n_dev % cp), f"openrl.trainer_cp={cp} must divide openrl.trainer_device_count={n_dev}"),
    ):
      if bad:
        raise ValueError(msg)
    ws = WeightSyncConfig(strategy=cfg.weight_sync or "delta")
    cpu_offload = cfg.fft_cpu_offload if cfg.fft_cpu_offload is not None else not cfg.exclusive
    model_id = str(uuid.uuid4())
    full_cfg = FFTConfig(seed=cfg.fft_seed, cpu_offload=cpu_offload, weight_sync_strategy=ws.strategy).model_dump()
    meta = TrainingModelMetadata(
      base_model=req.base_model,
      created_at=time.time(),
      fine_tuning_type=ft,
      weight_sync_config=ws,
      full_config=full_cfg,
      lora_config=req.lora_config.model_dump(),
      **cfg.model_dump(include=_META_SETTING_FIELDS),
    )
    if meta.sampler_replicas > 1 and meta.shares_gpu():
      raise ValueError("openrl.sampler_replicas above 1 needs samplers on GPUs of their own; set openrl.exclusive=true")
    await self.persist_on_runtime_slot(model_id, meta)
    return model_id, meta

  async def resolve_active_set_id(self, model_id: str | None) -> str | None:
    meta = await get_model_metadata(self.state, model_id) if model_id else None
    return f"{meta.runtime(model_id)}-1" if (meta and meta.fine_tuning_type == "lora" and meta.base_model) else None

  async def enqueue(self, command: Command) -> str:
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    carrier[commands.ENQUEUED_AT_KEY] = str(time.time())
    active_set = await self.resolve_active_set_id(command.model_id)
    await self.store.put_request(commands.wire(command.model_copy(update={"trace_context": carrier})), active_set_id=active_set)
    run_records.enqueued(command.model_id)
    return command.request_id

  async def ensure_sampler_launched(self, model_id: str) -> None:
    if self.worker_manager is not None and self.resolved_sampler_backend() == "vllm":
      await _logged(asyncio.to_thread(self.worker_manager.ensure, model_id, "sampler"))

  async def launch_or_enqueue(self, command: Command, start_sampler: bool = False) -> str:
    if self.worker_manager is None:
      return await self.enqueue(command)
    try:
      await asyncio.to_thread(self.worker_manager.ensure, command.model_id, "trainer")
    except Exception as exc:
      traceback.print_exc()
      await self.store.set_result(command.request_id, _failed(str(exc)))
      return command.request_id
    if start_sampler:
      await self.ensure_sampler_launched(command.model_id)
    return await self.enqueue(command)

  async def accept_pool_sample(self, req: AsampleRequest, prompt: list[int]) -> str:
    ref = paths.sampling_ref(req.model_id or req.sampling_session_id or "")
    lookup_id = paths.base_model_id_from_sampling_ref(ref) or ref
    meta = await get_model_metadata(self.state, lookup_id)
    if meta is None:
      meta, revision, model, marker = default_metadata(lookup_id), "", lookup_id, None
    elif meta.fine_tuning_type == "full":
      revision, model = paths.resolve_sampler_weights_path(ref, base_dir=self.tmp_dir), meta.base_model
      marker = (Path(revision, "metadata.json"), f"No saved sampler weights for {ref}")
    else:
      revision, model = "", snapshot_name(ref)
      marker = (snapshot_root() / model / "adapter_config.json", f"No saved LoRA sampler weights for {ref}")
    if marker is not None:
      deadline = time.monotonic() + self.saved_weights_wait_seconds
      while not await asyncio.to_thread(paths.visible, marker[0]):
        if time.monotonic() >= deadline:
          raise HTTPException(status_code=400, detail=marker[1])
        await asyncio.sleep(0.2)
    params, (stop, stop_ids) = req.sampling_params, split_stop(req.sampling_params.stop)
    idem = f"{req.sampling_session_id}:{req.seq_id}" if req.sampling_session_id and req.seq_id is not None else None
    sp = {"n": req.num_samples, **params.model_dump(exclude={"stop"}), "stop": stop or [], "stop_token_ids": stop_ids or []}
    rec = {"request_id": str(uuid.uuid4()), "pool": meta.pool(lookup_id), "job": lookup_id, "revision": revision, "model": model}
    rec |= {"idempotency_key": idem, "prompt_token_ids": prompt, "sampling_params": sp, "include_prompt_logprobs": req.prompt_logprobs}
    return await SamplingStore(self.store.redis).accept(rec)

  async def reconcile_job(self, sampling: SamplingStore, pool: str, job: str) -> None:
    meta, runtime, _ = await asyncio.to_thread(runtime_of, job)
    if meta.pool(job) != pool or self.worker_manager is None:
      return
    owner = owner_id(runtime)
    async with self.owner_locks[owner]:
      if (cur := await get_model_metadata(self.state, job)) is not None and cur.status == "completed":
        await sampling.fail_job(pool, job, _failed(f"{job} was deleted"))
      elif not await asyncio.to_thread(self.worker_manager.sampler_exists, job):
        if await self.session_registry.in_use(owner):
          await asyncio.to_thread(self.worker_manager.ensure, job, "sampler")
        else:
          await sampling.fail_job(pool, job, _failed(f"The pool of {job} is gone and its session ended before a sampler read the request"))

  async def reconcile_pools(self) -> None:
    if self.worker_manager is None:
      return
    sampling, pairs, slots = SamplingStore(self.store.redis), [], asyncio.Semaphore(self.reconcile_concurrency)

    async def _collect(pool: str) -> None:
      if not (jobs := await sampling.pending_jobs(pool)):
        await sampling.delete_if_empty(pool)
      pairs.extend((pool, job) for job in jobs)

    for pool in await sampling.pools():
      await _logged(_collect(pool))

    async def _run(pool: str, job: str) -> None:
      async with slots:
        await _logged(self.reconcile_job(sampling, pool, job))

    await asyncio.gather(*(_run(p, j) for p, j in pairs))

  async def sweep_sample_deadlines(self) -> None:
    msg = f"No sampler read the request within {SAMPLE_DEADLINE_SECONDS}s"
    await SamplingStore(self.store.redis).fail_older_than(SAMPLE_DEADLINE_SECONDS, _failed(msg))

  @asynccontextmanager
  async def lifespan(self, _: FastAPI):
    task, created_wm = None, False
    settings.check_sampler_backend()
    settings.check_lora_jobs_per_runtime()
    run_records.start("api_server", worker_record=False)
    if self.worker_manager is None and (settings.is_fft_enabled() or os.getenv("REDIS_URL") or os.getenv("OPEN_RL_WORKER_MANAGER")):
      self.worker_manager, created_wm = create_worker_manager(), True
    if settings.is_single_process_mode() and not settings.is_fft_enabled():
      from server import training_requests_processor

      worker = training_requests_processor.LoraTrainingWorker()
      if base_model := os.getenv("BASE_MODEL"):
        await asyncio.to_thread(worker.load_base_model, base_model)
      task = asyncio.create_task(training_requests_processor.run_training_requests_processor(worker))

    async def _every(step) -> None:
      while True:
        await asyncio.sleep(self.session_reap_interval_sec)
        await _logged(step())

    async def _reap_all() -> None:
      for owner in await self.session_registry.owners():
        await _logged(self.reap_owner(owner))

    steps = ([_reap_all] if self.worker_manager is not None else []) + (
      [self.sweep_sample_deadlines, self.reconcile_pools] if self.resolved_sampler_backend() == "vllm" else []
    )
    bg = [asyncio.create_task(_every(s)) for s in steps]
    try:
      yield
    finally:
      for t in (*bg, task):
        if t is not None:
          t.cancel()
      if created_wm and self.worker_manager is not None:
        self.worker_manager.close()
        self.worker_manager = None
      await run_records.finish()


# ***** app factory *****


def create_app(*, instrument: bool = False, **server_kwargs: Any) -> FastAPI:
  srv = ApiServer(**{k: v for k, v in server_kwargs.items() if v is not None})
  api = FastAPI(title="OpenRL API server", version=srv.open_rl_version, description="OpenRL self-hosted Tinker API.", lifespan=srv.lifespan)
  api.state.server = srv

  def openapi_with_settings() -> dict[str, Any]:
    if not api.openapi_schema:
      schema = get_openapi(title=api.title, version=api.version, description=api.description, routes=api.routes)
      s_schema = settings.Settings.model_json_schema(ref_template="#/components/schemas/{model}")
      schemas = schema.setdefault("components", {}).setdefault("schemas", {})
      schemas.update({**s_schema.pop("$defs", {}), "OpenRLSettings": s_schema})
      api.openapi_schema = schema
    return api.openapi_schema

  api.openapi = openapi_with_settings
  if instrument:
    FastAPIInstrumentor.instrument_app(api, excluded_urls="/api/v1/retrieve_future,/api/v1/session_heartbeat")
  api.include_router(observability_router)
  api.add_exception_handler(StarletteHTTPException, http_error)

  @api.exception_handler(RequestValidationError)
  async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    detail = jsonable_encoder(exc.errors(), custom_encoder={bytes: lambda b: f"<{len(b)} bytes>"})
    return JSONResponse(status_code=422, content={"error": "invalid request", "detail": detail})

  # ***** service *****

  @api.get("/api/v1/healthz")
  async def health_check():
    return {"status": "ok"}

  @api.get("/api/v1/get_server_capabilities")
  async def get_server_capabilities():
    m, sp = os.getenv("BASE_MODEL"), settings.is_single_process_mode()
    return {"supported_models": [{"model_name": m}] if m else [], "default_model": m, "single_process": sp, "openrl": srv.openrl_info()}

  @api.get("/api/v1/openrl/models/{model_id}")
  async def openrl_model_status(model_id: str):
    ctx = openrl_status.StatusContext(srv.state, srv.session_registry, srv.worker_manager, srv.worker_launcher_mode())
    if (res := await openrl_status.model_status(ctx, model_id)) is None:
      raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    return res

  @api.post("/api/v1/client/config")
  async def client_config(_: dict):
    mb = 1024 * 1024
    return {"pjwt_auth_enabled": False, "credential_default_source": "api_key"} | {
      "sample_dispatch_bytes_semaphore_size": 10 * mb,
      "inflight_response_bytes_semaphore_size": 50 * mb,
    }

  @api.post("/api/v1/create_session")
  async def create_session(req: CreateSessionRequest):
    try:
      meta = {**settings.tag_metadata(req.tags), **(req.user_metadata or {})}
      settings.resolve_settings(meta)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    sid = f"sess-{uuid.uuid4().hex[:12]}"
    await srv.session_registry.update_metadata(sid, meta)
    run_records.session_seen(sid, meta)
    return {"session_id": sid, "type": "create_session"}

  @api.post("/api/v1/session_heartbeat")
  async def session_heartbeat(body: dict[str, Any]):
    if sid := body.get("session_id"):
      await srv.session_registry.heartbeat(sid)
      run_records.session_seen(sid)
    return {"type": "session_heartbeat"}

  @api.post("/api/v1/create_model")
  async def create_model(req: CreateModelRequest, request: Request):
    try:
      model_id, meta = await srv.extract_and_persist_model_metadata(req, request)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    await srv.bind_session(req.session_id, model_id)
    cmd = commands.CreateModel(
      request_id=model_id,
      model_id=model_id,
      base_model=meta.base_model,
      fine_tuning_type=meta.fine_tuning_type,
      lora_config=meta.lora_config or {},
      full_config=meta.full_config or {},
    )
    return {"request_id": await srv.launch_or_enqueue(cmd, start_sampler=meta.recipe == "rl")}

  @api.post("/api/v1/delete_model")
  async def delete_model(req: ModelRequest):
    if (meta := await get_model_metadata(srv.state, req.model_id)) is None:
      raise HTTPException(status_code=404, detail=f"Unknown model: {req.model_id}")
    active_set = await srv.resolve_active_set_id(req.model_id)
    if meta.shares_runtime():
      await srv.store.put_request(commands.wire(commands.DeleteModel(request_id=str(uuid.uuid4()), model_id=req.model_id)), active_set_id=active_set)
    elif srv.worker_manager is not None:
      await srv.store.put_request(commands.wire(commands.Shutdown(model_id=req.model_id)), active_set_id=active_set)
    async with srv.owner_locks[await asyncio.to_thread(owner_of, req.model_id)]:
      meta.status, meta.completed_at, meta.updated_at = "completed", time.time(), time.time()
      await persist_model_metadata(srv.state, req.model_id, meta)
      if srv.resolved_sampler_backend() == "vllm":
        await SamplingStore(srv.store.redis).fail_job(meta.pool(req.model_id), req.model_id, _failed(f"{req.model_id} was deleted"))
      if srv.worker_manager is not None and not meta.shares_runtime():
        await asyncio.to_thread(srv.worker_manager.release, req.model_id)
        if srv.resolved_sampler_backend() == "vllm":
          await SamplingStore(srv.store.redis).delete_if_empty(meta.pool(req.model_id))
    return {"status": "ok"}

  @api.post("/api/v1/create_model_from_state")
  async def create_model_from_state(req: CreateModelFromStateRequest, request: Request):
    resolved = paths.tinker_checkpoint_dir(req.state_path, base_dir=srv.tmp_dir)
    if resolved is None and req.state_path.startswith("tinker://"):
      raise HTTPException(status_code=400, detail="Invalid checkpoint URI")
    if resolved is None:
      resolved = req.state_path if os.path.isabs(req.state_path) else os.path.join(srv.tmp_dir or paths.TMP_DIR, "checkpoints", req.state_path)
    try:
      ckpt = await asyncio.to_thread(paths.checkpoint_info, resolved, base_dir=srv.tmp_dir)
      if req.base_model is not None and req.base_model != ckpt["base_model"]:
        raise ValueError("base_model does not match the checkpoint")
      upd = req.model_copy(update={"base_model": ckpt["base_model"]})
      model_id, meta = await srv.extract_and_persist_model_metadata(upd, request, checkpoint_kind="lora" if ckpt["is_lora"] else "full")
    except FileNotFoundError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc
    await srv.bind_session(req.session_id, model_id)
    ft: FineTuningType = "full" if meta.fine_tuning_type == "full" else "lora"
    cmd = commands.CreateModelFromState(
      request_id=model_id, model_id=model_id, state_path=resolved, restore_optimizer=req.restore_optimizer, fine_tuning_type=ft
    )
    return {"request_id": await srv.launch_or_enqueue(cmd, start_sampler=meta.recipe == "rl")}

  @api.post("/api/v1/get_info")
  async def get_info(req: GetInfoRequest):
    lookup = paths.base_model_id_from_sampling_ref(req.model_id) or req.model_id
    meta = await get_model_metadata(srv.state, lookup) if lookup else None
    if not (model := (meta.base_model if meta is not None else None) or os.getenv("BASE_MODEL")):
      raise HTTPException(status_code=404, detail="No base model is configured")
    rank = meta.lora_config.rank if meta is not None and meta.fine_tuning_type == "lora" else 16
    mdata = {"arch": "unknown", "model_name": model, "tokenizer_id": model}
    return {
      "model_data": mdata,
      "model_id": req.model_id or "model-live-123",
      "is_lora": True,
      "lora_rank": rank,
      "model_name": model,
      "type": "get_info",
    }

  @api.post("/api/v1/retrieve_future")
  async def retrieve_future(req: RetrieveFutureRequest, accept: str = Header(default="")):
    result = await srv.store.get_result(req.request_id, timeout=60.0)
    if result is None:
      return JSONResponse(status_code=400, content=_failed("Future not found"))
    if isinstance(result, dict) and result.get("type") == "RequestFailedResponse":
      return JSONResponse(status_code=400, content=result)
    if wire.is_proto_result(result):
      raw = bytes(result["proto_bytes"])
      if wire.PROTO in accept:
        return Response(content=raw, media_type=wire.PROTO)
      return wire.fb_output_to_json(raw) if result.get("type") == "forward_backward" else wire.sample_response_to_json(raw)
    return result

  # ***** training *****

  async def _enqueue_fb(request: Request, *, force_forward_only: bool = False) -> dict[str, str]:
    if (enc := request.headers.get("content-encoding", "identity").strip().lower()) not in ("", "identity"):
      raise HTTPException(status_code=415, detail=f"Content-Encoding {enc!r} is not supported; request compression is disabled")
    body, media = await request.body(), request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media == wire.PROTO:
      try:
        parsed = wire.parse(body)
      except wire.ProtoDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
      if not parsed.model_id:
        raise RequestValidationError([{"type": "missing", "loc": ("body", "model_id"), "msg": "Field required", "input": None}])
      fwd_only = True if force_forward_only else bool(parsed.forward_only)
      if force_forward_only and not parsed.forward_only:
        parsed.forward_only = True
        body = parsed.SerializeToString()
      model_id, proto_bytes, loss_fn, loss_cfg = parsed.model_id, body, parsed.loss_fn or "cross_entropy", wire.loss_config(parsed)
    elif media in (wire.JSON, ""):
      try:
        req = ForwardBackwardRequest.model_validate_json(body or b"{}")
      except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
      inp = req.forward_backward_input or req.forward_input or ForwardBackwardInput()
      fwd_only = True if force_forward_only else req.forward_only
      model_id, proto_bytes = req.model_id, wire.json_to_fb_proto(req.model_dump(), forward_only=fwd_only)
      loss_fn, loss_cfg = inp.loss_fn, inp.loss_fn_config
    else:
      raise HTTPException(status_code=415, detail=f"unsupported Content-Type {media!r}; send {wire.JSON} or {wire.PROTO}")
    cmd = commands.ForwardBackward(
      request_id=str(uuid.uuid4()), model_id=model_id, proto_bytes=proto_bytes, loss_fn=loss_fn, loss_config=loss_cfg, forward_only=fwd_only
    )
    return {"request_id": await srv.enqueue(cmd)}

  @api.post("/api/v1/forward")
  async def forward(request: Request):
    return await _enqueue_fb(request, force_forward_only=True)

  @api.post("/api/v1/forward_backward")
  async def forward_backward(request: Request):
    return await _enqueue_fb(request, force_forward_only=False)

  @api.post("/api/v1/optim_step")
  async def optim_step(req: OptimStepRequest):
    return {"request_id": await srv.enqueue(commands.OptimStep(request_id=str(uuid.uuid4()), model_id=req.model_id, adam_params=req.adam_params))}

  @api.post("/api/v1/save_weights_for_sampler")
  async def save_weights_for_sampler(req: SaveRequest):
    await srv.ensure_sampler_launched(req.model_id)
    seq_id = int(time.time() * 1000) if req.sampling_session_seq_id is None else req.sampling_session_seq_id
    alias = req.name or req.alias or req.path
    sw_path = f"tinker://{req.model_id}/sampler_weights/{alias}" if alias else None
    sid = f"tinker://{req.model_id}/sampler_weights/sampler-{seq_id}"
    cmd = commands.SaveWeightsForSampler(request_id=str(uuid.uuid4()), model_id=req.model_id, alias=alias, path=sw_path, sampling_session_id=sid)
    return {"request_id": await srv.enqueue(cmd)}

  @api.post("/api/v1/save_weights")
  async def save_weights(req: SaveRequest):
    seq_id = int(time.time() * 1000) if req.seq_id is None else req.seq_id
    state_path = paths.checkpoint_state_path(req.model_id, req.path or f"{req.model_id}-samp-{seq_id}", base_dir=srv.tmp_dir)
    cmd = commands.SaveState(request_id=str(uuid.uuid4()), model_id=req.model_id, state_path=state_path, include_optimizer=True, kind="weights")
    return {"request_id": await srv.enqueue(cmd)}

  @api.post("/api/v1/load_weights")
  async def load_weights(req: LoadWeightsRequest):
    if req.path.startswith("tinker://") and paths.tinker_checkpoint_dir(req.path, base_dir=srv.tmp_dir) is None:
      raise HTTPException(status_code=400, detail=f"{req.path} is not a tinker://<model>/weights/<name> path")
    resolved = paths.checkpoint_state_path(req.model_id, req.path, base_dir=srv.tmp_dir)
    cmd = commands.LoadWeights(request_id=str(uuid.uuid4()), model_id=req.model_id, state_path=resolved, restore_optimizer=req.optimizer)
    return {"request_id": await srv.enqueue(cmd)}

  @api.post("/api/v1/weights_info")
  async def weights_info(req: WeightsInfoRequest):
    try:
      return await asyncio.to_thread(paths.checkpoint_info, req.tinker_path, base_dir=srv.tmp_dir)
    except FileNotFoundError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc

  # ***** sampling *****

  @api.post("/api/v1/create_sampling_session")
  async def create_sampling_session(body: dict[str, Any]):
    mp = body.get("model_path")
    if mp and mp.startswith("tinker://"):
      sess_id, target = mp, mp[9:].split("/")[0]
    else:
      sess_id = target = body.get("base_model") or body.get("model_id") or "samp-session-live-123"
    await srv.bind_session(body.get("session_id"), target)
    if srv.resolved_sampler_backend() == "vllm":
      await srv.ensure_sampler_launched(target)
      return {"sampling_session_id": f"{sess_id}@{uuid.uuid4().hex}", "type": "create_sampling_session"}
    return {"sampling_session_id": sess_id, "type": "create_sampling_session"}

  @api.get("/api/v1/samplers/{sampler_id:path}")
  async def get_sampler(sampler_id: str):
    ref = paths.sampling_ref(sampler_id)
    base_id = paths.base_model_id_from_sampling_ref(ref)
    meta = await get_model_metadata(srv.state, base_id) if base_id else None
    if not (base_model := (meta.base_model if meta is not None else None) or base_id or os.getenv("BASE_MODEL")):
      raise HTTPException(status_code=404, detail=f"Unknown sampler {sampler_id}")
    return {"sampler_id": sampler_id, "base_model": base_model, "model_path": ref if ref.startswith("tinker://") else None}

  @api.post("/api/v1/asample")
  async def asample(req: AsampleRequest):
    prompt = [t for ch in req.prompt.get("chunks", []) for t in ch.get("tokens", [])]
    if srv.resolved_sampler_backend() == "torch":
      lookup_id = paths.base_model_id_from_sampling_ref(req.model_id or req.sampling_session_id) or req.model_id or req.sampling_session_id
      sp = req.sampling_params
      cmd = commands.Sample(
        request_id=str(uuid.uuid4()),
        model_id=lookup_id,
        prompt_tokens=prompt,
        max_tokens=sp.max_tokens,
        temperature=sp.temperature,
        num_samples=req.num_samples,
        prompt_logprobs=req.prompt_logprobs,
      )
      req_id = await srv.enqueue(cmd)
    else:
      req_id = await srv.accept_pool_sample(req, prompt)
    return {"request_id": req_id, "sample_sequence_ids": [f"{req_id}:{i}" for i in range(max(1, int(req.num_samples)))]}

  # ***** inspection *****

  @api.get("/api/v1/list_adapters")
  async def list_adapters():
    peft_dir, adapters = os.path.join(srv.tmp_dir or paths.TMP_DIR, "peft"), []
    if os.path.exists(peft_dir):
      for e in sorted(os.scandir(peft_dir), key=lambda x: x.stat().st_ctime, reverse=True):
        if not e.is_dir():
          continue
        info: dict[str, Any] = {"model_id": e.name, "created_at": e.stat().st_ctime, "timestamp": e.stat().st_ctime, "alias": None}
        if os.path.exists(mp := os.path.join(e.path, "metadata.json")):
          with suppress(Exception), open(mp) as f:
            info.update(json.load(f))
        adapters.append(info)
    return {"adapters": adapters}

  @api.post("/api/v1/telemetry")
  async def telemetry(body: dict):
    run_records.sdk_events(body)
    return {"status": "accepted"}

  return api


# ***** uvicorn entrypoint *****

provider = TracerProvider()
trace.set_tracer_provider(provider)
if os.getenv("ENABLE_GCP_TRACE", "0") == "1":
  try:
    from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

    provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
  except ImportError:
    print("OpenTelemetry: opentelemetry-exporter-gcp-trace is not installed")

logging.getLogger("uvicorn.access").addFilter(lambda r: "retrieve_future" not in r.getMessage() and "session_heartbeat" not in r.getMessage())
app = create_app(instrument=True)

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))
