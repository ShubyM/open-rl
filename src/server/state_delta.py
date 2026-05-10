"""Semantic state delta manifests for Open-RL runtime synchronization."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

TrainingMode = Literal["lora", "full"]
ApplyTarget = Literal["vllm_lora", "sglang_lora", "trainer_lora"]
TensorRole = Literal["lora_a", "lora_b", "embedding", "lm_head", "full_weight", "unknown"]

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TensorEntry:
  name: str
  normalized_name: str
  role: TensorRole
  dtype: str
  shape: tuple[int, ...]
  checksum: str | None
  storage_key: str

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> TensorEntry:
    return cls(
      str(data["name"]),
      str(data["normalized_name"]),
      data["role"],
      str(data["dtype"]),
      tuple(int(dim) for dim in data["shape"]),
      data.get("checksum"),
      str(data["storage_key"]),
    )


@dataclass(frozen=True)
class StateDeltaManifest:
  schema_version: int
  run_id: str
  version: int
  base_ref: str
  delta_id: str
  training_mode: TrainingMode
  apply_target: ApplyTarget
  adapter_config_hash: str | None
  tensors: tuple[TensorEntry, ...]
  created_at: float

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> StateDeltaManifest:
    return cls(
      int(data["schema_version"]),
      str(data["run_id"]),
      int(data["version"]),
      str(data["base_ref"]),
      str(data["delta_id"]),
      data["training_mode"],
      data["apply_target"],
      data.get("adapter_config_hash"),
      tuple(TensorEntry.from_dict(entry) for entry in data["tensors"]),
      float(data["created_at"]),
    )


@dataclass(frozen=True)
class ReceiverState:
  ref: str
  version: int


def hash_adapter_config(config: Mapping[str, Any] | None) -> str | None:
  if config is None:
    return None
  encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def tensor_checksum(tensor: torch.Tensor) -> str:
  cpu_tensor = tensor.detach().cpu().contiguous()
  return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def normalize_lora_tensor_name(name: str, adapter_id: str) -> str | None:
  parts = name.split(".")
  for marker in ("lora_A", "lora_B"):
    for idx, part in enumerate(parts):
      if part == marker and idx + 2 < len(parts) and parts[idx + 1] == adapter_id and parts[idx + 2] == "weight":
        return ".".join(parts[: idx + 1] + parts[idx + 2 :])

  for marker in ("lora_embedding_A", "lora_embedding_B"):
    for idx, part in enumerate(parts):
      if part == marker and idx + 1 < len(parts) and parts[idx + 1] == adapter_id:
        return ".".join(parts[: idx + 1] + parts[idx + 2 :])

  return None


def role_for_lora_tensor_name(name: str) -> TensorRole:
  if "lora_A" in name or "lora_embedding_A" in name:
    return "lora_a"
  if "lora_B" in name or "lora_embedding_B" in name:
    return "lora_b"
  if "embed" in name:
    return "embedding"
  if "lm_head" in name:
    return "lm_head"
  if "weight" in name:
    return "full_weight"
  return "unknown"


def build_lora_delta_manifest(
  *,
  run_id: str,
  version: int,
  tensors: Iterable[tuple[str, torch.Tensor]],
  apply_target: ApplyTarget,
  adapter_config: Mapping[str, Any] | None = None,
  base_ref: str | None = None,
  compute_checksums: bool = False,
  created_at: float | None = None,
) -> StateDeltaManifest:
  entries: list[TensorEntry] = []
  for name, tensor in tensors:
    normalized_name = normalize_lora_tensor_name(name, run_id)
    if normalized_name is None:
      continue
    storage_key = hashlib.sha256(f"{run_id}:{version}:{normalized_name}".encode()).hexdigest()
    entries.append(
      TensorEntry(
        name,
        normalized_name,
        role_for_lora_tensor_name(normalized_name),
        str(tensor.dtype).removeprefix("torch."),
        tuple(tensor.shape),
        tensor_checksum(tensor) if compute_checksums else None,
        storage_key,
      )
    )

  entries.sort(key=lambda entry: entry.normalized_name)
  resolved_base_ref = base_ref or f"{run_id}:{version - 1}"
  adapter_hash = hash_adapter_config(adapter_config)
  delta_input = {
    "run_id": run_id,
    "version": version,
    "base_ref": resolved_base_ref,
    "apply_target": apply_target,
    "adapter_config_hash": adapter_hash,
  }
  delta_input["storage_keys"] = [entry.storage_key for entry in entries]
  delta_id = hashlib.sha256(json.dumps(delta_input, sort_keys=True).encode("utf-8")).hexdigest()
  manifest = StateDeltaManifest(
    SCHEMA_VERSION,
    run_id,
    version,
    resolved_base_ref,
    delta_id,
    "lora",
    apply_target,
    adapter_hash,
    tuple(entries),
    created_at if created_at is not None else time.time(),
  )
  validate_delta_manifest(manifest)
  return manifest


def validate_delta_manifest(manifest: StateDeltaManifest) -> None:
  storage_keys = [entry.storage_key for entry in manifest.tensors]
  normalized_names = [entry.normalized_name for entry in manifest.tensors]
  if len(storage_keys) != len(set(storage_keys)) or len(normalized_names) != len(set(normalized_names)):
    raise ValueError("duplicate tensor entries")


def validate_for_apply(manifest: StateDeltaManifest, receiver_state: ReceiverState) -> None:
  validate_delta_manifest(manifest)
  if manifest.version <= receiver_state.version or manifest.base_ref != receiver_state.ref:
    raise ValueError("delta does not apply")
