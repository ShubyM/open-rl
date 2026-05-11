"""Durable Open-RL checkpoint envelopes."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from model_state import ModelState, lora_model_state

CheckpointTarget = Literal["trainer", "inference"]


@dataclass(frozen=True)
class CheckpointMetadata:
  base_model: str | None
  model_id: str
  state_id: str | None = None
  version: int = 0
  training_mode: str = "lora"
  kind: str = "state"
  is_lora: bool = True
  targets: tuple[CheckpointTarget, ...] = ("trainer", "inference")
  adapter_ref: str | None = None
  full_weights_ref: str | None = None
  optimizer_ref: str | None = None
  state_delta_ref: str | None = None
  adapter_name: str | None = None
  inference_backend: str | None = None
  has_optimizer: bool = False
  created_at: float = 0.0

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> CheckpointMetadata:
    try:
      created_at = float(data.get("created_at", data.get("timestamp", 0.0)) or 0.0)
    except (TypeError, ValueError):
      created_at = float(data.get("timestamp", 0.0) or 0.0)
    return cls(
      base_model=data.get("base_model"),
      model_id=str(data["model_id"]),
      state_id=data.get("state_id"),
      version=int(data.get("version", 0)),
      training_mode=str(data.get("training_mode", "lora")),
      kind=str(data.get("kind", "state")),
      is_lora=bool(data.get("is_lora", True)),
      targets=tuple(data.get("targets", ("trainer", "inference"))),
      adapter_ref=data.get("adapter_ref"),
      full_weights_ref=data.get("full_weights_ref"),
      optimizer_ref=data.get("optimizer_ref"),
      state_delta_ref=data.get("state_delta_ref"),
      adapter_name=data.get("adapter_name"),
      inference_backend=data.get("inference_backend"),
      has_optimizer=bool(data.get("has_optimizer", False)),
      created_at=created_at,
    )

  def to_model_state(self, checkpoint_ref: str | None = None) -> ModelState:
    if not self.base_model:
      raise ValueError("checkpoint metadata missing base_model")
    if not self.state_id:
      raise ValueError("checkpoint metadata missing state_id")
    if self.training_mode != "lora":
      raise ValueError(f"unsupported checkpoint training_mode={self.training_mode}")
    if not self.adapter_ref:
      raise ValueError("LoRA checkpoint metadata missing adapter_ref")
    return lora_model_state(
      state_id=self.state_id,
      model_id=self.model_id,
      base_model=self.base_model,
      version=self.version,
      checkpoint_ref=checkpoint_ref or self.adapter_ref,
      adapter_ref=self.adapter_ref,
      adapter_name=self.adapter_name,
      optimizer_ref=self.optimizer_ref,
      state_delta_ref=self.state_delta_ref,
      inference_backend=self.inference_backend,
      created_at=self.created_at,
    )


class FileCheckpointStore:
  def __init__(self, root_dir: str | os.PathLike[str] | None = None):
    tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    self.root_dir = Path(root_dir or Path(tmp_dir) / "checkpoints")

  def resolve(self, ref: str | os.PathLike[str]) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else self.root_dir / path

  def write_metadata(self, checkpoint_dir: str | os.PathLike[str], metadata: CheckpointMetadata) -> str:
    path = self.resolve(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "metadata.json"
    tmp_path = metadata_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))
    tmp_path.replace(metadata_path)
    return str(metadata_path)

  def read_metadata(self, checkpoint_dir: str | os.PathLike[str]) -> CheckpointMetadata:
    with (self.resolve(checkpoint_dir) / "metadata.json").open() as f:
      return CheckpointMetadata.from_dict(json.load(f))


def lora_checkpoint_metadata(
  *,
  base_model: str | None,
  model_id: str,
  checkpoint_dir: str | os.PathLike[str],
  kind: str,
  state_id: str | None = None,
  version: int = 0,
  adapter_name: str | None = None,
  inference_backend: str | None = None,
  optimizer_ref: str | None = None,
  state_delta_ref: str | None = None,
) -> CheckpointMetadata:
  return CheckpointMetadata(
    base_model=base_model,
    model_id=model_id,
    state_id=state_id,
    version=version,
    training_mode="lora",
    kind=kind,
    adapter_ref=str(checkpoint_dir),
    adapter_name=adapter_name,
    inference_backend=inference_backend,
    optimizer_ref=optimizer_ref,
    state_delta_ref=state_delta_ref,
    has_optimizer=optimizer_ref is not None,
    created_at=time.time(),
  )
