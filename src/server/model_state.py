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
  checkpoint_ref: str
  adapter_ref: str | None = None
  full_weights_ref: str | None = None
  optimizer_ref: str | None = None
  state_delta_ref: str | None = None
  adapter_name: str | None = None
  inference_backend: str | None = None
  tenant_id: str | None = None
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
      checkpoint_ref=str(data["checkpoint_ref"]),
      adapter_ref=data.get("adapter_ref"),
      full_weights_ref=data.get("full_weights_ref"),
      optimizer_ref=data.get("optimizer_ref"),
      state_delta_ref=data.get("state_delta_ref"),
      adapter_name=data.get("adapter_name"),
      inference_backend=data.get("inference_backend"),
      tenant_id=data.get("tenant_id"),
      created_at=float(data.get("created_at", 0.0) or 0.0),
      schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
    )


def lora_model_state(
  *,
  state_id: str,
  model_id: str,
  base_model: str,
  version: int,
  checkpoint_ref: str,
  adapter_ref: str,
  adapter_name: str | None = None,
  optimizer_ref: str | None = None,
  state_delta_ref: str | None = None,
  inference_backend: str | None = None,
  tenant_id: str | None = None,
  created_at: float | None = None,
) -> ModelState:
  return ModelState(
    state_id=state_id,
    model_id=model_id,
    base_model=base_model,
    training_mode="lora",
    version=version,
    checkpoint_ref=checkpoint_ref,
    adapter_ref=adapter_ref,
    adapter_name=adapter_name,
    optimizer_ref=optimizer_ref,
    state_delta_ref=state_delta_ref,
    inference_backend=inference_backend,
    tenant_id=tenant_id,
    created_at=created_at if created_at is not None else time.time(),
  )
