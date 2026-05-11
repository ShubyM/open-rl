"""Weight synchronization primitives for trainer-to-inference publication."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import torch
from delta_store import DeltaStore, FileDeltaStore, HttpBodyDeltaStore, SharedMemoryDeltaStore
from state_delta import StateDeltaManifest, TensorEntry, build_lora_delta_manifest

WeightRole = Literal["lora", "full_weight", "lm_head", "embedding"]


@dataclass(frozen=True)
class WeightTensor:
  name: str
  tensor: torch.Tensor
  dtype: str
  shape: tuple[int, ...]
  role: WeightRole


@dataclass(frozen=True)
class PublishedState:
  run_id: str
  version: int
  adapter_name: str
  inference_backend: str
  transport: str
  tensor_count: int
  durable_ref: str | None = None
  manifest_path: str | None = None
  base_model_id: str | None = None
  state_id: str | None = None
  adapter_ref: str | None = None


class TensorSelector(Protocol):
  def select(self, model: Any, adapter_id: str) -> list[WeightTensor]: ...


class TransferEngine(Protocol):
  name: str

  def publish(
    self,
    run_id: str,
    version: int,
    adapter_name: str,
    tensors: Sequence[WeightTensor],
    durable_ref: str | None,
    base_model_id: str | None = None,
  ) -> PublishedState: ...


class LoraTensorSelector:
  """Select only adapter tensors from a PEFT model state dict."""

  lora_markers = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")

  def select(self, model: Any, adapter_id: str) -> list[WeightTensor]:
    if model is None:
      raise ValueError("model is required")

    selected: list[WeightTensor] = []
    for name, tensor in model.state_dict().items():
      if not torch.is_tensor(tensor):
        continue
      if not self.is_adapter_tensor(name, adapter_id):
        continue
      selected.append(
        WeightTensor(
          name=name,
          tensor=tensor.detach(),
          dtype=str(tensor.dtype).removeprefix("torch."),
          shape=tuple(tensor.shape),
          role=self.role_for_name(name),
        )
      )
    selected.sort(key=lambda item: item.name)
    return selected

  def is_adapter_tensor(self, name: str, adapter_id: str) -> bool:
    if not any(marker in name for marker in self.lora_markers):
      return False
    return f".{adapter_id}." in name or name.endswith(f".{adapter_id}.weight")

  def role_for_name(self, name: str) -> WeightRole:
    if "lora_" in name:
      return "lora"
    if "lm_head" in name:
      return "lm_head"
    if "embed" in name:
      return "embedding"
    return "full_weight"


def build_vllm_lora_delta_manifest(
  run_id: str,
  version: int,
  tensors: Sequence[WeightTensor],
  adapter_config: dict[str, Any] | None,
  checksum: bool = False,
) -> StateDeltaManifest:
  return build_lora_delta_manifest(
    run_id=run_id,
    version=version,
    tensors=((item.name, item.tensor) for item in tensors),
    apply_target="vllm_lora",
    adapter_config=adapter_config,
    compute_checksums=checksum,
  )


def tensors_for_manifest(tensors: Sequence[WeightTensor], manifest: StateDeltaManifest) -> list[tuple[TensorEntry, torch.Tensor]]:
  tensors_by_name = {item.name: item.tensor for item in tensors}
  return [(entry, tensors_by_name[entry.name]) for entry in manifest.tensors]


def cast_weight_tensors(tensors: Sequence[WeightTensor], dtype: torch.dtype | None) -> list[WeightTensor]:
  if dtype is None:
    return list(tensors)
  casted: list[WeightTensor] = []
  for item in tensors:
    tensor = item.tensor
    if tensor.is_floating_point():
      tensor = tensor.to(dtype)
    casted.append(
      WeightTensor(
        name=item.name,
        tensor=tensor,
        dtype=str(tensor.dtype).removeprefix("torch."),
        shape=tuple(tensor.shape),
        role=item.role,
      )
    )
  return casted


def read_adapter_config(durable_ref: str | None) -> dict[str, Any] | None:
  if durable_ref is None:
    return None
  config_path = Path(durable_ref) / "adapter_config.json"
  if not config_path.exists():
    return None
  with config_path.open() as f:
    return json.load(f)


class FileAdapterTransferEngine:
  """Persist the semantic delta to a durable local file store."""

  name = "file_adapter_reload"

  def __init__(self, root_dir: str | os.PathLike[str] | None = None):
    self.store = FileDeltaStore(root_dir)

  def publish(
    self,
    run_id: str,
    version: int,
    adapter_name: str,
    tensors: Sequence[WeightTensor],
    durable_ref: str | None,
    base_model_id: str | None = None,
  ) -> PublishedState:
    manifest = build_vllm_lora_delta_manifest(run_id, version, tensors, read_adapter_config(durable_ref), checksum=True)
    write = self.store.write_delta(manifest, tensors_for_manifest(tensors, manifest))

    return PublishedState(run_id, version, adapter_name, "file", self.name, len(tensors), durable_ref, write.payload["manifest_path"], base_model_id)


class TorchRLVLLMTransferEngine:
  """Use a pre-initialized TorchRL VLLMWeightSender for NCCL transfer."""

  name = "torchrl_vllm_nccl"

  def __init__(self, sender: Any):
    self.sender = sender

  @staticmethod
  def model_metadata_for(tensors: Sequence[WeightTensor]) -> dict[str, tuple[torch.dtype, torch.Size]]:
    return {item.name: (item.tensor.dtype, torch.Size(item.shape)) for item in tensors}

  def publish(
    self,
    run_id: str,
    version: int,
    adapter_name: str,
    tensors: Sequence[WeightTensor],
    durable_ref: str | None,
    base_model_id: str | None = None,
  ) -> PublishedState:
    weights = {item.name: item.tensor for item in tensors}
    self.sender.update_weights(weights)
    return PublishedState(run_id, version, adapter_name, "vllm", self.name, len(tensors), durable_ref, base_model_id=base_model_id)


class VLLMAdapterTransferEngine:
  """Write the durable adapter delta, then notify vLLM of the versioned adapter path."""

  name = "vllm_lora_adapter_reload"

  def __init__(self, vllm_url: str, timeout: float = 30.0, fallback: TransferEngine | None = None, strict: bool = False):
    self.vllm_url = vllm_url.rstrip("/")
    self.timeout = timeout
    self.fallback = fallback or FileAdapterTransferEngine()
    self.strict = strict

  def publish(
    self,
    run_id: str,
    version: int,
    adapter_name: str,
    tensors: Sequence[WeightTensor],
    durable_ref: str | None,
    base_model_id: str | None = None,
  ) -> PublishedState:
    state = self.fallback.publish(run_id, version, adapter_name, tensors, durable_ref, base_model_id)
    try:
      post_vllm_adapter_sync(self.vllm_url, self.timeout, run_id, version, adapter_name, durable_ref, state.manifest_path, base_model_id)
    except Exception:
      if self.strict:
        raise
      print(f"[weight-sync] vLLM adapter sync skipped for {adapter_name}@{version}; durable adapter was written")
      return state
    return PublishedState(run_id, version, adapter_name, "vllm", self.name, len(tensors), durable_ref, state.manifest_path, base_model_id)


class VLLMLoraTensorTransferEngine:
  """Build a LoRA delta, write it to a store, then ask vLLM to apply it."""

  def __init__(
    self,
    vllm_url: str,
    store: DeltaStore | None = None,
    timeout: float = 30.0,
    fallback: TransferEngine | None = None,
    strict: bool = False,
    tensor_dtype: torch.dtype | None = None,
  ):
    self.vllm_url = vllm_url.rstrip("/")
    self.store = store or SharedMemoryDeltaStore()
    self.name = self.store.name
    self.timeout = timeout
    self.fallback = fallback or FileAdapterTransferEngine()
    self.strict = strict
    self.tensor_dtype = tensor_dtype

  def publish(
    self,
    run_id: str,
    version: int,
    adapter_name: str,
    tensors: Sequence[WeightTensor],
    durable_ref: str | None,
    base_model_id: str | None = None,
  ) -> PublishedState:
    try:
      adapter_config = read_adapter_config(durable_ref)
      checksum = os.getenv("OPEN_RL_WEIGHT_SYNC_CHECKSUM", "0") == "1"
      payload_tensors = cast_weight_tensors(tensors, self.tensor_dtype)
      manifest = build_vllm_lora_delta_manifest(run_id, version, payload_tensors, adapter_config, checksum=checksum)
      write = self.store.write_delta(manifest, tensors_for_manifest(payload_tensors, manifest))
      try:
        request = lora_delta_request(run_id, version, adapter_name, durable_ref, adapter_config, manifest, write, base_model_id)
        post_vllm_sync(self.vllm_url, self.timeout, request)
      finally:
        write.close()
      return PublishedState(run_id, version, adapter_name, "vllm", self.name, len(tensors), durable_ref, base_model_id=base_model_id)
    except Exception:
      if self.strict:
        raise
      print(f"[weight-sync] vLLM tensor sync skipped for {adapter_name}@{version}; falling back to file reload")
      state = self.fallback.publish(run_id, version, adapter_name, tensors, durable_ref, base_model_id)
      try:
        post_vllm_adapter_sync(self.vllm_url, self.timeout, run_id, version, adapter_name, durable_ref, state.manifest_path, base_model_id)
      except Exception:
        print(f"[weight-sync] vLLM adapter fallback sync skipped for {adapter_name}@{version}")
      return state


class VLLMLoraTensorHttpTransferEngine(VLLMLoraTensorTransferEngine):
  def __init__(self, vllm_url: str, **kwargs: Any):
    tensor_dtype = kwargs.pop("tensor_dtype", None)
    super().__init__(vllm_url, store=HttpBodyDeltaStore(), tensor_dtype=tensor_dtype, **kwargs)


class VLLMLoraTensorSharedMemoryTransferEngine(VLLMLoraTensorTransferEngine):
  def __init__(self, vllm_url: str, **kwargs: Any):
    tensor_dtype = kwargs.pop("tensor_dtype", None)
    super().__init__(vllm_url, store=SharedMemoryDeltaStore(), tensor_dtype=tensor_dtype, **kwargs)


def lora_delta_request(
  run_id: str,
  version: int,
  adapter_name: str,
  durable_ref: str | None,
  adapter_config: dict[str, Any] | None,
  manifest: StateDeltaManifest,
  write: Any,
  base_model_id: str | None = None,
) -> dict[str, Any]:
  return {
    "run_id": run_id,
    "version": version,
    "adapter_name": adapter_name,
    "base_model_id": base_model_id,
    "adapter_path": durable_ref,
    "adapter_config": adapter_config,
    "manifest": manifest.to_dict(),
    "transport_receipt": write.receipt.to_dict(),
    **write.payload,
  }


def post_vllm_sync(vllm_url: str, timeout: float, payload: dict[str, Any]) -> None:
  with httpx.Client(timeout=timeout) as client:
    response = client.post(f"{vllm_url}/sync_lora_tensors", json=payload)
    response.raise_for_status()
    data = response.json()
    if data.get("type") == "RequestFailedResponse":
      raise RuntimeError(data.get("error_message", "vLLM tensor sync failed"))


def post_vllm_adapter_sync(
  vllm_url: str,
  timeout: float,
  run_id: str,
  version: int,
  adapter_name: str,
  adapter_path: str | None,
  manifest_path: str | None = None,
  base_model_id: str | None = None,
) -> None:
  payload = {
    "run_id": run_id,
    "version": version,
    "adapter_name": adapter_name,
    "base_model_id": base_model_id,
    "adapter_path": adapter_path,
    "manifest_path": manifest_path,
  }
  with httpx.Client(timeout=timeout) as client:
    response = client.post(f"{vllm_url}/sync_lora_adapter", json=payload)
    response.raise_for_status()
    data = response.json()
    if data.get("type") == "RequestFailedResponse":
      raise RuntimeError(data.get("error_message", "vLLM adapter sync failed"))


def torch_dtype_from_name(name: str | None) -> torch.dtype | None:
  if not name:
    return None
  normalized = name.removeprefix("torch.").lower()
  dtypes = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
  }
  if normalized not in dtypes:
    raise ValueError(f"unsupported OPEN_RL_WEIGHT_SYNC_TENSOR_DTYPE={name}")
  return dtypes[normalized]


class WeightSyncBridge:
  def __init__(self, selector: TensorSelector | None = None, transfer_engine: TransferEngine | None = None):
    self.selector = selector or LoraTensorSelector()
    self.transfer_engine = transfer_engine or FileAdapterTransferEngine()
    self._versions: dict[str, int] = {}

  @classmethod
  def from_env(cls) -> WeightSyncBridge:
    if os.getenv("SAMPLING_BACKEND", "").lower() == "vllm" or "VLLM_URL" in os.environ:
      transport = os.getenv("OPEN_RL_WEIGHT_SYNC_TRANSPORT", "vllm_lora_adapter_reload")
      vllm_url = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
      kwargs = {
        "timeout": float(os.getenv("OPEN_RL_WEIGHT_SYNC_TIMEOUT", "30.0")),
        "strict": os.getenv("OPEN_RL_WEIGHT_SYNC_STRICT", "0") == "1",
        "tensor_dtype": torch_dtype_from_name(os.getenv("OPEN_RL_WEIGHT_SYNC_TENSOR_DTYPE")),
      }
      if transport in {"vllm_lora_adapter_reload", "file_adapter_reload"}:
        return cls(
          transfer_engine=VLLMAdapterTransferEngine(
            vllm_url,
            timeout=kwargs["timeout"],
            strict=kwargs["strict"],
          )
        )
      if transport in {"vllm_lora_tensors", "vllm_lora_tensors_http"}:
        return cls(transfer_engine=VLLMLoraTensorHttpTransferEngine(vllm_url, **kwargs))
      if transport in {"vllm_lora_tensors_shm", "vllm_lora_tensors_ipc"}:
        return cls(transfer_engine=VLLMLoraTensorSharedMemoryTransferEngine(vllm_url, **kwargs))
    return cls()

  def publish_for_inference(
    self,
    run_id: str,
    version: int | None,
    model: Any,
    adapter_name: str,
    durable_ref: str | None,
    base_model_id: str | None = None,
  ) -> PublishedState:
    resolved_version = version if version is not None else self.next_version(run_id)
    tensors = self.selector.select(model, run_id)
    state = self.transfer_engine.publish(run_id, resolved_version, adapter_name, tensors, durable_ref, base_model_id)
    return replace(state, base_model_id=base_model_id, state_id=adapter_name, adapter_ref=durable_ref)

  def next_version(self, run_id: str) -> int:
    version = self._versions.get(run_id, 0) + 1
    self._versions[run_id] = version
    return version
