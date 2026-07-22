"""Native vLLM WeightTransferEngine implementation for CPU snapshot weight sync.

Implements vLLM's abstract WeightTransferEngine contract to apply adaptive
sparse or full-tensor overwrites to a host CPU snapshot, then reload only the
changed Hugging Face tensors into GPU VRAM.
"""

import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load_file
from vllm.distributed.weight_transfer.base import (
  WeightTransferEngine,
  WeightTransferInitInfo,
  WeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import (
  download_weights_from_hf,
  safetensors_weights_iterator,
)

logger = init_logger("vllm.distributed.weight_transfer.delta_snapshot")
SPARSE_DELTA_FORMAT = "sparse_delta"
ABSOLUTE_TENSORS_FORMAT = "absolute_tensors"


@dataclass
class DeltaSnapshotInitInfo(WeightTransferInitInfo):
  """Initialization parameters for DeltaSnapshotWeightTransferEngine."""

  model_name_or_path: str = ""


@dataclass
class DeltaSnapshotUpdateInfo(WeightTransferUpdateInfo):
  """Update metadata specifying the target checkpoint or adaptive update path."""

  target_weights_path: str = ""
  is_checkpoint_format: bool = True
  base_model_path: str = ""


class DeltaSnapshotWeightTransferEngine(WeightTransferEngine):
  """Pull-based Delta Snapshot Weight Transfer Engine for vLLM.

  Applies adaptive .safetensors updates directly to an in-memory host CPU
  Hugging Face snapshot and feeds changed tensors into vLLM's native
  load_weights callback.
  """

  init_info_cls = DeltaSnapshotInitInfo
  update_info_cls = DeltaSnapshotUpdateInfo

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.current_weights_path: str | None = None
    self.cpu_snapshot: dict[str, torch.Tensor] = {}
    self.base_model: str = os.getenv("OPEN_RL_BASE_MODEL", os.getenv("BASE_MODEL", ""))
    if not self.base_model and self.model_config is not None:
      self.base_model = self.model_config.model

  def ensure_cpu_snapshot(self, base_model: str) -> None:
    if self.cpu_snapshot:
      return
    base_model = base_model or self.base_model or os.getenv("OPEN_RL_BASE_MODEL", os.getenv("BASE_MODEL", ""))
    logger.info(f"[DeltaSnapshotEngine] Initializing CPU weights snapshot (base model: '{base_model}')...")

    if base_model:
      start_t = time.perf_counter()
      if os.path.isdir(base_model):
        hf_folder = base_model
      else:
        hf_folder = download_weights_from_hf(base_model, cache_dir=None, allow_patterns=["*.safetensors"])

      hf_weights_files = sorted(
        os.path.join(hf_folder, filename) for filename in os.listdir(hf_folder) if filename.endswith(".safetensors") and "delta" not in filename
      )
      for name, tensor in safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load=False):
        if not name.endswith(".indices") and "delta" not in name:
          self.cpu_snapshot[name] = tensor.pin_memory() if torch.cuda.is_available() else tensor.clone()
      if self.cpu_snapshot:
        elapsed = (time.perf_counter() - start_t) * 1000.0
        logger.info(
          f"[DeltaSnapshotEngine] CPU weights snapshot initialized with {len(self.cpu_snapshot)} "
          f"HuggingFace tensors from base model '{base_model}' via vLLM weight iterator in {elapsed:.2f} ms."
        )
        return

    raise RuntimeError(f"Failed to initialize CPU weights snapshot from base model '{base_model}'.")

  def init_transfer_engine(self, init_info: DeltaSnapshotInitInfo) -> None:
    """No initialization is needed for pull-based checkpoint updates."""

  def start_weight_update(self) -> None:
    """Updates are applied in place, so no layerwise setup is needed."""

  @staticmethod
  def load_metadata(target_path: str) -> dict[str, Any]:
    if not os.path.isdir(target_path):
      return {}
    metadata_path = os.path.join(target_path, "metadata.json")
    if not os.path.exists(metadata_path):
      return {}
    with open(metadata_path) as f:
      return json.load(f)

  @staticmethod
  def layer_names(metadata: dict[str, Any], update_format: str) -> list[str]:
    names = metadata.get("layer_names")
    if names is None:
      raise ValueError(f"Missing 'layer_names' metadata in {update_format} update.")
    if isinstance(names, str):
      names = json.loads(names)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
      raise ValueError(f"Invalid 'layer_names' metadata in {update_format} update.")
    return names

  @staticmethod
  def update_file(target_path: str, update_format: str) -> str:
    update_file = os.path.join(target_path, "delta.safetensors")
    if not os.path.exists(update_file):
      raise ValueError(f"{update_format} metadata present but delta.safetensors not found at: {target_path}")
    return update_file

  def sparse_update_weights(
    self,
    target_path: str,
    metadata: dict[str, Any],
    base_model_path: str,
  ) -> list[tuple[str, torch.Tensor]]:
    start_t = time.perf_counter()
    update_file = self.update_file(target_path, SPARSE_DELTA_FORMAT)
    sparse_update = load_file(update_file, device="cpu")
    logger.info(f"[DeltaSnapshotEngine] Loaded sparse delta from {update_file} in {(time.perf_counter() - start_t) * 1000.0:.2f} ms")

    names = self.layer_names(metadata, SPARSE_DELTA_FORMAT)
    indices = sparse_update["delta.indices_flat"].to(torch.int64)
    values = sparse_update["delta.values_flat"]
    layer_lengths = sparse_update["delta.layer_lengths"].tolist()
    if len(names) != len(layer_lengths):
      raise ValueError(f"Mismatch between layer_names ({len(names)}) and layer_lengths ({len(layer_lengths)}) in sparse delta.")
    if any(length < 0 for length in layer_lengths):
      raise ValueError("Sparse delta layer lengths cannot be negative.")
    expected_elements = sum(layer_lengths)
    if expected_elements != indices.numel() or expected_elements != values.numel():
      raise ValueError("Sparse delta indices, values, and layer lengths do not contain the same number of elements.")
    if not names or not expected_elements:
      logger.info("[DeltaSnapshotEngine] Sparse update is a no-op; skipping GPU reload")
      return []

    self.ensure_cpu_snapshot(base_model_path)
    apply_start = time.perf_counter()
    index_slices = torch.split(indices, layer_lengths)
    value_slices = torch.split(values, layer_lengths)
    for name, layer_indices, layer_values in zip(names, index_slices, value_slices, strict=True):
      if name not in self.cpu_snapshot:
        raise KeyError(f"Parameter '{name}' found in sparse delta but missing from CPU snapshot.")
      self.cpu_snapshot[name].view(-1)[layer_indices] = layer_values

    changed_elements = indices.numel()
    total_elements = sum(tensor.numel() for tensor in self.cpu_snapshot.values())
    density_pct = 100.0 * changed_elements / max(1, total_elements)
    logger.info(
      f"[DeltaSnapshotEngine] Applied packed sparse delta ({len(names)} layers, "
      f"{changed_elements}/{total_elements} elements [{density_pct:.3f}% changed]) "
      f"in {(time.perf_counter() - apply_start) * 1000.0:.2f} ms"
    )
    return [(name, self.cpu_snapshot[name]) for name in names]

  def absolute_update_weights(
    self,
    target_path: str,
    metadata: dict[str, Any],
    base_model_path: str,
  ) -> list[tuple[str, torch.Tensor]]:
    start_t = time.perf_counter()
    update_file = self.update_file(target_path, ABSOLUTE_TENSORS_FORMAT)
    self.ensure_cpu_snapshot(base_model_path)
    incoming_tensors = load_file(update_file, device="cpu")
    names = self.layer_names(metadata, ABSOLUTE_TENSORS_FORMAT)
    if set(names) != set(incoming_tensors):
      raise ValueError("Absolute tensor names do not match metadata layer_names.")

    changed_weights = []
    for name, incoming_tensor in incoming_tensors.items():
      if name not in self.cpu_snapshot:
        raise KeyError(f"Parameter '{name}' found in absolute tensor update but missing from CPU snapshot.")
      previous_tensor = self.cpu_snapshot[name]
      if previous_tensor.shape != incoming_tensor.shape or previous_tensor.dtype != incoming_tensor.dtype:
        raise ValueError(f"Absolute tensor '{name}' does not match the CPU snapshot shape and dtype.")
      if not torch.equal(previous_tensor, incoming_tensor):
        self.cpu_snapshot[name] = incoming_tensor
        changed_weights.append((name, incoming_tensor))

    logger.info(
      f"[DeltaSnapshotEngine] Loaded adaptive absolute update ({len(changed_weights)} changed tensors) "
      f"from {update_file} in {(time.perf_counter() - start_t) * 1000.0:.2f} ms"
    )
    return changed_weights

  @staticmethod
  def checkpoint_weights(target_path: str) -> list[tuple[str, torch.Tensor]]:
    if target_path.endswith(".safetensors"):
      return list(load_file(target_path, device="cpu").items())
    if not os.path.isdir(target_path):
      raise ValueError(f"Unsupported weight path format: {target_path}")

    weights = []
    for root, _, files in os.walk(target_path):
      for filename in sorted(files):
        if filename.endswith(".safetensors"):
          weights.extend(load_file(os.path.join(root, filename), device="cpu").items())
    return weights

  def changed_checkpoint_weights(self, target_path: str) -> list[tuple[str, torch.Tensor]]:
    start_t = time.perf_counter()
    incoming_weights = self.checkpoint_weights(target_path)
    logger.info(
      f"[DeltaSnapshotEngine] Loaded {len(incoming_weights)} parameter tensors "
      f"from {target_path} in {(time.perf_counter() - start_t) * 1000.0:.2f} ms"
    )

    changed_weights = []
    for name, incoming_tensor in incoming_weights:
      previous_tensor = self.cpu_snapshot.get(name)
      if (
        previous_tensor is not None
        and previous_tensor.shape == incoming_tensor.shape
        and previous_tensor.dtype == incoming_tensor.dtype
        and torch.equal(previous_tensor, incoming_tensor)
      ):
        continue
      self.cpu_snapshot[name] = incoming_tensor
      changed_weights.append((name, incoming_tensor))

    logger.info(
      f"[DeltaSnapshotEngine] Verified checkpoint: {len(changed_weights)}/{len(incoming_weights)} tensors changed "
      f"({len(incoming_weights) - len(changed_weights)} no-op tensors skipped)"
    )
    return changed_weights

  def receive_weights(self, update_info: DeltaSnapshotUpdateInfo) -> None:
    """Receive adaptive updates and load changed Hugging Face tensors into vLLM."""
    target_path = update_info.target_weights_path
    if not target_path or not os.path.exists(target_path):
      raise ValueError(f"Target weights path does not exist: {target_path}")

    metadata = self.load_metadata(target_path)
    update_format = metadata.get("format")
    base_model_path = update_info.base_model_path or self.base_model
    if update_format == SPARSE_DELTA_FORMAT:
      weights_to_load = self.sparse_update_weights(target_path, metadata, base_model_path)
    elif update_format == ABSOLUTE_TENSORS_FORMAT:
      weights_to_load = self.absolute_update_weights(target_path, metadata, base_model_path)
    else:
      weights_to_load = self.changed_checkpoint_weights(target_path)

    if not weights_to_load:
      self.current_weights_path = target_path
      return

    # WeightTransferEngine's constructor supplies the model. Calling its loader
    # here follows the same boundary as vLLM's built-in IPC and NCCL engines.
    start_load = time.perf_counter()
    self.model.load_weights(weights_to_load)
    elapsed_load = (time.perf_counter() - start_load) * 1000.0
    self.current_weights_path = target_path
    logger.info(f"[DeltaSnapshotEngine] Incremental load_weights completed ({len(weights_to_load)} tensors) in {elapsed_load:.2f} ms")

  def finish_weight_update(self) -> None:
    """Updates are applied in place, so no layerwise finalization is needed."""

  def shutdown(self) -> None:
    """The pull-based engine owns no external resources."""

  @staticmethod
  def trainer_send_weights(
    iterator: Iterator[tuple[str, torch.Tensor]],
    trainer_args: dict[str, Any] | Any,
  ) -> None:
    """Static trainer-side hook for push engines (no-op for pull engines)."""


WeightTransferEngineFactory.register_engine(
  "delta_snapshot",
  DeltaSnapshotWeightTransferEngine,
)
