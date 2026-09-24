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

from server.model_metadata import SPARSE_DELTA_VERSION


def capture_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
  """Take an independent CPU baseline, including when the model is already on CPU."""
  return {name: param.detach().to(device="cpu", copy=True).contiguous() for name, param in model.named_parameters() if param.requires_grad}


def write_sparse_delta(
  model: torch.nn.Module,
  baseline: dict[str, torch.Tensor],
  path: str,
  *,
  model_id: str,
  base_model: str | None,
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
      indices.append(changed.to(torch.int32 if current.numel() <= 2**31 else torch.int64))
      values.append(current[changed])
      shapes.append(param.shape)

  # Native checkpoint coordinates let vLLM handle fusion and sharding. Separate
  # tensors preserve each parameter's dtype, including mixed BF16/FP32 models.
  packed = {}
  for i, (index, value) in enumerate(zip(indices, values, strict=True)):
    packed[f"{i}.indices"] = index.contiguous()
    packed[f"{i}.values"] = value.contiguous()
  changed_elements = sum(index.numel() for index in indices)
  density = round(100.0 * changed_elements / max(1, total_elements), 3)
  metadata = {
    "base_model": base_model,
    "created_at": datetime.now().isoformat(),
    "format": "sparse_delta",
    "format_version": SPARSE_DELTA_VERSION,
    "kind": "sampler",
    "model_id": model_id,
    "changed_elements": changed_elements,
    "total_elements": total_elements,
    "layer_names": names,
    "layer_shapes": [list(shape) for shape in shapes],
    "density_pct": density,
    "timestamp": time.time(),
  }
  os.makedirs(path, exist_ok=True)
  safetensors.torch.save_file(packed, os.path.join(path, "delta.safetensors"), metadata={"layer_names": json.dumps(names)})
  with open(os.path.join(path, "metadata.json"), "w") as file:
    json.dump(metadata, file)

  for name, index, value, shape in zip(names, indices, values, shapes, strict=True):
    if name in baseline:
      baseline[name].view(-1)[index] = value
    else:
      baseline[name] = value.reshape(shape).clone()
  return {"path": path, "density_pct": density}
