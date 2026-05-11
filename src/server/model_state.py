"""Durable model state records shared by training and inference."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

TrainingMode = Literal["lora", "fft"]

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelState:
  state_id: str
  model_id: str
  base_model: str
  training_mode: TrainingMode
  version: int
  adapter_ref: str | None = None
  full_weights_ref: str | None = None
  optimizer_ref: str | None = None
  delta_ref: str | None = None
  runtime_backend: str | None = None
  runtime_key: str | None = None
  adapter_name: str | None = None
  tenant_id: str | None = None
  transport: str | None = None
  tensor_count: int = 0
  created_at: float = 0.0
  schema_version: int = SCHEMA_VERSION

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> ModelState:
    return cls(
      state_id=str(data["state_id"]),
      model_id=str(data["model_id"]),
      base_model=str(data["base_model"]),
      training_mode=data.get("training_mode", "lora"),
      version=int(data.get("version", 0)),
      adapter_ref=data.get("adapter_ref") or (data.get("checkpoint_ref") if data.get("training_mode", "lora") == "lora" else None),
      full_weights_ref=data.get("full_weights_ref"),
      optimizer_ref=data.get("optimizer_ref"),
      delta_ref=data.get("delta_ref") or data.get("state_delta_ref"),
      runtime_backend=data.get("runtime_backend") or data.get("inference_backend"),
      runtime_key=data.get("runtime_key"),
      adapter_name=data.get("adapter_name"),
      tenant_id=data.get("tenant_id"),
      transport=data.get("transport"),
      tensor_count=int(data.get("tensor_count", 0) or 0),
      created_at=float(data.get("created_at", 0.0) or 0.0),
      schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
    )


def state_ref(state: ModelState | Mapping[str, Any]) -> str | None:
  if isinstance(state, ModelState):
    return state.full_weights_ref or state.adapter_ref
  return state.get("full_weights_ref") or state.get("adapter_ref") or state.get("checkpoint_ref")


def latest_training_state_alias(model_id: str) -> str:
  return f"latest_training_state:{model_id}"


def lora_runtime_key(base_model: str | None, adapter_name: str | None, version: int) -> str | None:
  if not adapter_name:
    return None
  state_key = f"{base_model}::{adapter_name}" if base_model else adapter_name
  return f"{state_key}@{version}"


def lora_model_state(
  *,
  state_id: str,
  model_id: str,
  base_model: str,
  version: int,
  adapter_ref: str | None = None,
  checkpoint_ref: str | None = None,
  adapter_name: str | None = None,
  optimizer_ref: str | None = None,
  delta_ref: str | None = None,
  state_delta_ref: str | None = None,
  runtime_backend: str | None = None,
  inference_backend: str | None = None,
  runtime_key: str | None = None,
  tenant_id: str | None = None,
  transport: str | None = None,
  tensor_count: int = 0,
  created_at: float | None = None,
) -> ModelState:
  resolved_backend = runtime_backend or inference_backend
  resolved_adapter_ref = adapter_ref or checkpoint_ref
  return ModelState(
    state_id=state_id,
    model_id=model_id,
    base_model=base_model,
    training_mode="lora",
    version=version,
    adapter_ref=resolved_adapter_ref,
    adapter_name=adapter_name,
    optimizer_ref=optimizer_ref,
    delta_ref=delta_ref or state_delta_ref,
    runtime_backend=resolved_backend,
    runtime_key=runtime_key or lora_runtime_key(base_model, adapter_name or state_id, version),
    tenant_id=tenant_id,
    transport=transport,
    tensor_count=tensor_count,
    created_at=created_at if created_at is not None else time.time(),
  )
