"""Hot and durable backing stores for adapter snapshot payloads."""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Protocol

import torch
from safetensors.torch import save as save_safetensors
from state_delta import AdapterSnapshotManifest, TensorEntry


@dataclass(frozen=True)
class TransportReceipt:
  transport: str
  delta_id: str
  version: int
  locations: dict[str, str]
  expires_at: float | None = None

  def to_dict(self) -> dict:
    return asdict(self)


class DeltaWrite:
  def __init__(self, receipt: TransportReceipt, payload: dict, shm: shared_memory.SharedMemory | None = None):
    self.receipt = receipt
    self.payload = payload
    self.shm = shm

  def close(self) -> None:
    if self.shm is None:
      return
    self.shm.close()
    with contextlib.suppress(FileNotFoundError):
      self.shm.unlink()


class DeltaStore(Protocol):
  name: str

  def write_delta(self, manifest: AdapterSnapshotManifest, tensors: Iterable[tuple[TensorEntry, torch.Tensor]]) -> DeltaWrite: ...


@dataclass(frozen=True)
class DeltaRead:
  manifest: AdapterSnapshotManifest
  tensor_path: Path


def tensor_bytes(tensors: Iterable[tuple[TensorEntry, torch.Tensor]], dtype: torch.dtype | None = None) -> bytes:
  payload = {}
  for entry, tensor in tensors:
    item = tensor.detach().cpu().contiguous()
    if dtype is not None and item.is_floating_point():
      item = item.to(dtype)
    payload[entry.normalized_name] = item
  return save_safetensors(payload)


class SharedMemoryDeltaStore:
  name = "vllm_lora_tensor_shm"

  def __init__(self, tensor_dtype: torch.dtype | None = None):
    self.tensor_dtype = tensor_dtype

  def write_delta(self, manifest: AdapterSnapshotManifest, tensors: Iterable[tuple[TensorEntry, torch.Tensor]]) -> DeltaWrite:
    payload_bytes = tensor_bytes(tensors, self.tensor_dtype)
    shm = shared_memory.SharedMemory(create=True, size=len(payload_bytes))
    shm.buf[: len(payload_bytes)] = payload_bytes
    locations = {entry.storage_key: f"shm://{shm.name}" for entry in manifest.tensors}
    return DeltaWrite(
      receipt=TransportReceipt(self.name, manifest.delta_id, manifest.version, locations),
      payload={"tensors_safetensors_shm": {"name": shm.name, "size": len(payload_bytes)}},
      shm=shm,
    )


class HttpBodyDeltaStore:
  name = "vllm_lora_tensor_http"

  def __init__(self, tensor_dtype: torch.dtype | None = None):
    self.tensor_dtype = tensor_dtype

  def write_delta(self, manifest: AdapterSnapshotManifest, tensors: Iterable[tuple[TensorEntry, torch.Tensor]]) -> DeltaWrite:
    payload_bytes = tensor_bytes(tensors, self.tensor_dtype)
    locations = {entry.storage_key: "http_body" for entry in manifest.tensors}
    return DeltaWrite(
      receipt=TransportReceipt(self.name, manifest.delta_id, manifest.version, locations),
      payload={"tensors_safetensors_b64": base64.b64encode(payload_bytes).decode("ascii")},
    )


class FileDeltaStore:
  name = "file_delta"

  def __init__(self, root_dir: str | os.PathLike[str] | None = None, tensor_dtype: torch.dtype | None = None):
    tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    self.root_dir = Path(root_dir or Path(tmp_dir) / "deltas")
    self.tensor_dtype = tensor_dtype

  def write_delta(self, manifest: AdapterSnapshotManifest, tensors: Iterable[tuple[TensorEntry, torch.Tensor]]) -> DeltaWrite:
    delta_dir = self.root_dir / manifest.run_id / str(manifest.version)
    delta_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = delta_dir / "tensors.safetensors"
    manifest_path = delta_dir / "manifest.json"
    tensor_path.write_bytes(tensor_bytes(tensors, self.tensor_dtype))
    with manifest_path.open("w") as f:
      json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
    locations = {entry.storage_key: tensor_path.as_uri() for entry in manifest.tensors}
    return DeltaWrite(
      receipt=TransportReceipt(self.name, manifest.delta_id, manifest.version, locations),
      payload={"delta_ref": str(delta_dir), "manifest_path": str(manifest_path), "tensor_path": str(tensor_path)},
    )

  def read_delta(self, ref: str | os.PathLike[str]) -> DeltaRead:
    delta_dir = Path(ref)
    with (delta_dir / "manifest.json").open() as f:
      manifest = AdapterSnapshotManifest.from_dict(json.load(f))
    return DeltaRead(manifest=manifest, tensor_path=delta_dir / "tensors.safetensors")
