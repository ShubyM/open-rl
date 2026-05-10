"""Durable Open-RL checkpoint envelopes."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

CheckpointTarget = Literal["trainer", "inference"]


@dataclass(frozen=True)
class CheckpointMetadata:
  base_model: str | None
  model_id: str
  kind: str = "state"
  is_lora: bool = True
  targets: tuple[CheckpointTarget, ...] = ("trainer", "inference")
  adapter_ref: str | None = None
  optimizer_ref: str | None = None
  state_delta_ref: str | None = None
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
      kind=str(data.get("kind", "state")),
      is_lora=bool(data.get("is_lora", True)),
      targets=tuple(data.get("targets", ("trainer", "inference"))),
      adapter_ref=data.get("adapter_ref"),
      optimizer_ref=data.get("optimizer_ref"),
      state_delta_ref=data.get("state_delta_ref"),
      has_optimizer=bool(data.get("has_optimizer", False)),
      created_at=created_at,
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
  optimizer_ref: str | None = None,
  state_delta_ref: str | None = None,
) -> CheckpointMetadata:
  return CheckpointMetadata(
    base_model=base_model,
    model_id=model_id,
    kind=kind,
    adapter_ref=str(checkpoint_dir),
    optimizer_ref=optimizer_ref,
    state_delta_ref=state_delta_ref,
    has_optimizer=optimizer_ref is not None,
    created_at=time.time(),
  )
