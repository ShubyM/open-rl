"""The sparse weight delta a full fine-tuning trainer hands its sampler.

After each optim_step the elements that changed are found against a host copy
of the last synced weights and kept as flat index/value lists, which the
sampler patches into its own copy in place. Under the vllm_fused format the HF
names and offsets are rewritten to vLLM's fused qkv_proj and gate_up_proj
tensors so the sampler can index them directly.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


class SparseWeightDelta:
  def __init__(self, weights: dict[str, torch.Tensor], hf_config: Any = None, delta_format: str = "vllm_fused"):
    self.hf_config = hf_config
    self.delta_format = delta_format
    self.previous: dict[str, torch.Tensor] = {}
    for name, tensor in weights.items():
      host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=torch.cuda.is_available())
      host.copy_(tensor.data)
      self.previous[name] = host
    self.total_elements = sum(tensor.numel() for tensor in weights.values())
    # Before the first step the delta is empty but still names every layer.
    self.names = list(weights)
    self.indices: list[torch.Tensor] = []
    self.values: list[torch.Tensor] = []
    self.lengths = [0] * len(self.names)
    self.changed = 0

  def record(self, weights: dict[str, torch.Tensor]) -> int:
    """Diff weights against the last synced copy and keep what changed.

    Call right after optim_step, while the weights are on the device. Several
    writes may follow one record, so nothing here is consumed by write().
    """
    started = time.perf_counter()
    names: list[str] = []
    indices: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    lengths: list[int] = []
    changed = 0
    for name, tensor in weights.items():
      previous = self.previous[name]
      mask = tensor.data.view(-1).ne(previous.to(tensor.device, non_blocking=True).view(-1))
      index = mask.nonzero(as_tuple=True)[0]
      if index.numel() == 0:
        continue
      index_cpu = index.to(torch.int64).contiguous().cpu()
      value_cpu = tensor.data.view(-1)[mask].contiguous().cpu()
      previous.view(-1)[index_cpu] = value_cpu
      names.append(name)
      indices.append(index_cpu)
      values.append(value_cpu)
      lengths.append(int(index_cpu.numel()))
      changed += int(index_cpu.numel())
    if self.delta_format == "vllm_fused":
      names, indices = fuse_for_vllm(names, indices, self.hf_config)
    self.names, self.indices, self.values, self.lengths, self.changed = names, indices, values, lengths, changed
    logger.info(
      f"[WEIGHT_DELTA] delta_compute_time={time.perf_counter() - started:.4f}s | "
      f"changed={changed}/{self.total_elements} ({100.0 * changed / max(1, self.total_elements):.2f}%) across {len(names)} layers"
    )
    return changed

  def write(self, path: str, *, base_model: str, model_id: str, kind: str = "sampler") -> dict[str, Any]:
    os.makedirs(path, exist_ok=True)
    # int64 indices: a flat index into a tensor with more than 2**31 elements
    # (Gemma 4's per-layer embedding table is 2.35e9) does not fit an int32, and
    # a wrapped negative index made the sampler's index_copy_ assert mid-run.
    if self.indices:
      indices_flat = torch.cat(self.indices).to(torch.int64).contiguous()
      values_flat = torch.cat(self.values).contiguous()
    else:
      dtype = next(iter(self.previous.values())).dtype if self.previous else torch.float32
      indices_flat = torch.empty(0, dtype=torch.int64)
      values_flat = torch.empty(0, dtype=dtype)
    packed = {
      "delta.indices_flat": indices_flat,
      "delta.values_flat": values_flat,
      "delta.layer_lengths": torch.tensor(self.lengths, dtype=torch.int64),
    }
    started = time.perf_counter()
    save_file(packed, os.path.join(path, "delta.safetensors"), metadata={"layer_names": json.dumps(self.names)})
    density = round(100.0 * self.changed / max(1, self.total_elements), 3)
    metadata = {
      "base_model": base_model,
      "created_at": datetime.now().isoformat(),
      "format": "sparse_delta",
      "kind": kind,
      "model_id": model_id,
      "changed_elements": self.changed,
      "total_elements": self.total_elements,
      "layer_names": self.names,
      "density_pct": density,
      "timestamp": time.time(),
    }
    with open(os.path.join(path, "metadata.json"), "w") as f:
      json.dump(metadata, f)
    logger.info(f"[WEIGHT_DELTA] model_id={model_id} kind={kind} | safetensors_save_time={time.perf_counter() - started:.4f}s | density={density}%")
    print(f"Saved sparse delta ({density}% changed elements, {self.changed}/{self.total_elements}) to {path}")
    return {"path": path, "density_pct": density}


def fuse_for_vllm(layer_names: list[str], indices: list[torch.Tensor], config: Any) -> tuple[list[str], list[torch.Tensor]]:
  """Rewrite HF names (q_proj, k_proj, v_proj, gate_proj, up_proj) and offset their indices into vLLM's fused tensors."""
  if config is None:
    return layer_names, indices
  # Multimodal wrappers (e.g. gemma-4 ForConditionalGeneration) nest the LM
  # dims under text_config.
  if getattr(config, "hidden_size", None) is None and getattr(config, "text_config", None) is not None:
    config = config.text_config

  hidden_size = getattr(config, "hidden_size", None)
  num_heads = getattr(config, "num_attention_heads", None)
  num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
  head_dim = getattr(config, "head_dim", None)
  if head_dim is None and hidden_size is not None and num_heads is not None:
    head_dim = hidden_size // num_heads

  intermediate_size = getattr(config, "intermediate_size", None)

  q_numel = (num_heads * head_dim * hidden_size) if (hidden_size and num_heads and head_dim) else None
  k_numel = (num_kv_heads * head_dim * hidden_size) if (hidden_size and num_kv_heads and head_dim) else None
  gate_numel = (intermediate_size * hidden_size) if (hidden_size and intermediate_size) else None
  # Bias rows fuse with bias-sized offsets (Qwen2.5 attention has QKV
  # biases; using weight-sized offsets sent bias indices out of bounds).
  q_bias_numel = (num_heads * head_dim) if (num_heads and head_dim) else None
  k_bias_numel = (num_kv_heads * head_dim) if (num_kv_heads and head_dim) else None

  mapped_names: list[str] = []
  mapped_indices: list[torch.Tensor] = []

  for name, idx in zip(layer_names, indices, strict=True):
    is_bias = name.endswith(".bias")
    if (".q_proj." in name or ".k_proj." in name or ".v_proj." in name) and q_numel is not None and k_numel is not None:
      qkv_name = name.replace(".q_proj.", ".qkv_proj.").replace(".k_proj.", ".qkv_proj.").replace(".v_proj.", ".qkv_proj.")
      qn, kn = (q_bias_numel, k_bias_numel) if is_bias else (q_numel, k_numel)
      offset = 0 if ".q_proj." in name else (qn if ".k_proj." in name else qn + kn)
      mapped_names.append(qkv_name)
      mapped_indices.append(idx + offset)
      continue

    if (".gate_proj." in name or ".up_proj." in name) and gate_numel is not None:
      gate_up_name = name.replace(".gate_proj.", ".gate_up_proj.").replace(".up_proj.", ".gate_up_proj.")
      # (No known FFT target has MLP biases; if one appears, intermediate_size
      # is the bias-sized gate offset.)
      offset = 0 if ".gate_proj." in name else (intermediate_size if is_bias else gate_numel)
      mapped_names.append(gate_up_name)
      mapped_indices.append(idx + offset)
      continue

    mapped_names.append(name)
    mapped_indices.append(idx)

  return mapped_names, mapped_indices
