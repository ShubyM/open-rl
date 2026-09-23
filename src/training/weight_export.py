"""Explicit sampler artifacts; sparse patches require consumers to apply exports in order.

The baseline belongs to sampler publication, not optimizer steps or GPU residency.
A patch contains replacement values, not arithmetic differences. It is not a
standalone checkpoint and cannot be used to skip or reorder exported versions.
"""

import json
import os
import time
from datetime import datetime

import safetensors.torch
import torch


def capture_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
  """Take an independent CPU baseline, including when the model is already on CPU."""
  return {name: param.detach().to(device="cpu", copy=True).contiguous() for name, param in model.named_parameters() if param.requires_grad}


def remap_hf_to_vllm_fused(
  model: torch.nn.Module,
  layer_names_list: list[str],
  indices_list: list[torch.Tensor],
) -> tuple[list[str], list[torch.Tensor]]:
  """Remaps HF layer names (q_proj, k_proj, v_proj, gate_proj, up_proj) and offsets indices to vLLM fused names."""
  config = getattr(model, "config", None)
  if config is None:
    return layer_names_list, indices_list
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

  for name, idx in zip(layer_names_list, indices_list):
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


def write_sparse_delta(
  model: torch.nn.Module,
  baseline: dict[str, torch.Tensor],
  path: str,
  *,
  model_id: str,
  base_model: str | None,
  delta_format: str,
) -> dict[str, str | float]:
  """Write current weights relative to the last successful export, then advance it.

  Missing baseline entries send the entire parameter, so a restored checkpoint
  can replace a sampler's earlier weights without assuming its current version.
  Only one parameter is copied from the device at a time. Baseline updates use
  the same changed values retained for serialization, rather than another model
  snapshot. A failed write leaves the baseline untouched for retry.
  """
  names: list[str] = []
  indices: list[torch.Tensor] = []
  values: list[torch.Tensor] = []
  shapes: list[torch.Size] = []
  total_elements = 0
  fallback_dtype = next(model.parameters()).dtype
  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    current = param.detach().cpu().reshape(-1)
    total_elements += current.numel()
    previous = baseline.get(name)
    if previous is None:
      changed = torch.arange(current.numel(), dtype=torch.int64)
    else:
      changed = current.ne(previous.reshape(-1)).nonzero(as_tuple=True)[0]
    if changed.numel():
      names.append(name)
      indices.append(changed)
      values.append(current[changed])
      shapes.append(param.shape)

  export_names, export_indices = names, indices
  if delta_format == "vllm_fused":
    export_names, export_indices = remap_hf_to_vllm_fused(model, names, indices)
  packed = {
    "delta.indices_flat": torch.cat(export_indices).to(torch.int64).contiguous() if indices else torch.empty(0, dtype=torch.int64),
    "delta.values_flat": torch.cat(values).contiguous() if values else torch.empty(0, dtype=fallback_dtype),
    "delta.layer_lengths": torch.tensor([index.numel() for index in indices], dtype=torch.int64),
  }
  changed_elements = sum(index.numel() for index in indices)
  density = round(100.0 * changed_elements / max(1, total_elements), 3)
  metadata = {
    "base_model": base_model,
    "created_at": datetime.now().isoformat(),
    "format": "sparse_delta",
    "kind": "sampler",
    "model_id": model_id,
    "changed_elements": changed_elements,
    "total_elements": total_elements,
    "layer_names": export_names,
    "density_pct": density,
    "timestamp": time.time(),
  }
  os.makedirs(path, exist_ok=True)
  safetensors.torch.save_file(packed, os.path.join(path, "delta.safetensors"), metadata={"layer_names": json.dumps(export_names)})
  with open(os.path.join(path, "metadata.json"), "w") as file:
    json.dump(metadata, file)

  for name, index, value, shape in zip(names, indices, values, shapes, strict=True):
    if name in baseline:
      baseline[name].view(-1)[index] = value
    else:
      baseline[name] = value.reshape(shape).clone()
  return {"path": path, "density_pct": density}
