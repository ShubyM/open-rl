"""What one model_id owns, and how it moves between the accelerator, the host and disk.

TrainedWeights is plain tensors under names. Where they live is tensor.device
and nothing else. Parking them, writing a checkpoint and exporting to the
sampler are all the same walk over those tensors to a different destination.
"""

import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from training.types import LoraConfig

# PEFT's on-disk adapter layout, which vLLM and the LoRA sampler read.
PEFT_PREFIX = "base_model.model."


@dataclass
class TrainedWeights:
  """Everything one model_id owns.

  For LoRA the weights are the A/B matrices, keyed "<module>.lora_A" and
  "<module>.lora_B". For full fine-tuning they are the base module's own
  parameters. Gradients are summed by forward_backward and emptied by
  optim_step, so a model with no grads is between steps.
  """

  weights: dict[str, torch.Tensor]
  lora_config: LoraConfig | None = None
  grads: dict[str, torch.Tensor] = field(default_factory=dict)
  adam_m: dict[str, torch.Tensor] = field(default_factory=dict)
  adam_v: dict[str, torch.Tensor] = field(default_factory=dict)
  adam_step: int = 0

  def tensors(self) -> Iterable[torch.Tensor]:
    for group in (self.weights, self.grads, self.adam_m, self.adam_v):
      yield from group.values()


# -- parking -----------------------------------------------------------------------

# Pinned host buffers by tensor id, allocated on the first park and reused after.
# Pinning is slow and the tensors live as long as the process, so the buffers do too.
pinned: dict[int, tuple[torch.device, torch.Tensor]] = {}


def park(tensors: Iterable[torch.Tensor]) -> None:
  """Copy every accelerator tensor to pinned host memory and give its device memory back.

  The copies are issued together and synchronized once, and only then is the
  device storage dropped, so a failed copy never loses the tensor.
  """
  if not torch.cuda.is_available():
    return
  moved = []
  for tensor in tensors:
    if tensor.device.type != "cuda":
      continue
    entry = pinned.get(id(tensor))
    if entry is None or entry[1].shape != tensor.shape or entry[1].dtype != tensor.dtype:
      entry = pinned[id(tensor)] = (tensor.device, torch.empty_like(tensor, device="cpu", pin_memory=True))
    entry[1].copy_(tensor.data, non_blocking=True)
    moved.append((tensor, entry[1]))
  torch.cuda.synchronize()
  for tensor, host in moved:
    tensor.data = host
  torch.cuda.empty_cache()


def unpark(tensors: Iterable[torch.Tensor]) -> None:
  """Bring parked tensors back to the device each one came from."""
  if not torch.cuda.is_available():
    return
  for tensor in tensors:
    entry = pinned.get(id(tensor))
    if entry is not None and tensor.device.type == "cpu":
      tensor.data = tensor.data.to(entry[0], non_blocking=True)
  torch.cuda.synchronize()


# -- checkpoints -------------------------------------------------------------------


def write(
  trained: TrainedWeights,
  path: str,
  *,
  base_model: str,
  model_id: str,
  with_adam: bool = False,
  kind: str = "state",
  hf_config: Any = None,
) -> None:
  """Write a checkpoint the samplers, the API server and read() all understand.

  LoRA lands in PEFT's layout under <path>/<model_id>/ so vLLM can load it as
  an adapter. Full weights land in <path>/model.safetensors under their HF
  names, next to config.json when hf_config is given, so vLLM and transformers
  both load the directory as a model.
  """
  os.makedirs(path, exist_ok=True)
  weights = {name: tensor.detach().contiguous().cpu() for name, tensor in trained.weights.items()}
  if trained.lora_config is not None:
    adapter_dir = os.path.join(path, model_id)
    os.makedirs(adapter_dir, exist_ok=True)
    save_file({peft_key(name): tensor for name, tensor in weights.items()}, os.path.join(adapter_dir, "adapter_model.safetensors"))
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
      json.dump(adapter_config(trained, base_model), f, indent=2)
  else:
    save_file(weights, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    if hf_config is not None:
      hf_config.save_pretrained(path)

  has_adam = with_adam and bool(trained.adam_m)
  if has_adam:
    adam = {f"m.{name}": tensor.detach().contiguous().cpu() for name, tensor in trained.adam_m.items()}
    adam.update({f"v.{name}": tensor.detach().contiguous().cpu() for name, tensor in trained.adam_v.items()})
    save_file(adam, os.path.join(path, "adam.safetensors"), metadata={"step": str(trained.adam_step)})

  metadata = {
    "base_model": base_model,
    "created_at": datetime.now().isoformat(),
    "kind": kind,
    "has_optimizer": has_adam,
    "model_id": model_id,
    "timestamp": time.time(),
  }
  with open(os.path.join(path, "metadata.json"), "w") as f:
    json.dump(metadata, f)


def read_metadata(path: str) -> dict[str, Any]:
  metadata_path = os.path.join(path, "metadata.json")
  if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"No metadata.json found at {path}")
  with open(metadata_path) as f:
    metadata = json.load(f)
  if not metadata.get("base_model"):
    raise ValueError(f"metadata.json at {path} missing base_model")
  return metadata


def read(path: str, model_id: str, restore_optimizer: bool = False) -> TrainedWeights:
  """Load a checkpoint written by write(), or by the PEFT-based workers before it, onto the CPU."""
  adapter_dir = next((d for d in (os.path.join(path, model_id), path) if os.path.exists(os.path.join(d, "adapter_config.json"))), None)
  if adapter_dir is not None:
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
      config = json.load(f)
    lora_config = lora_config_from_adapter(config)
    weights = {lora_key(key): tensor for key, tensor in load_file(os.path.join(adapter_dir, "adapter_model.safetensors")).items()}
  else:
    lora_config = None
    weights = load_file(os.path.join(path, "model.safetensors"))
  trained = TrainedWeights(weights, lora_config)
  if restore_optimizer:
    read_adam(trained, path)
  return trained


def read_adam(trained: TrainedWeights, path: str) -> None:
  adam_path = os.path.join(path, "adam.safetensors")
  torch_path = os.path.join(path, "optimizer.pt")
  if os.path.exists(adam_path):
    from safetensors import safe_open

    with safe_open(adam_path, "pt") as f:
      trained.adam_step = int(f.metadata()["step"])
    for key, tensor in load_file(adam_path).items():
      kind, name = key.split(".", 1)
      (trained.adam_m if kind == "m" else trained.adam_v)[name] = tensor
  elif os.path.exists(torch_path):
    # torch.optim.AdamW state from the PEFT-based workers, indexed by parameter
    # position. new_weights yields A then B per target in module order, the
    # same order PEFT's parameters() had.
    state = torch.load(torch_path, map_location="cpu", weights_only=False)["state"]
    names = list(trained.weights)
    if len(state) != len(names):
      raise ValueError(f"{torch_path} holds {len(state)} parameter states for {len(names)} weights")
    for index, name in enumerate(names):
      trained.adam_m[name] = state[index]["exp_avg"]
      trained.adam_v[name] = state[index]["exp_avg_sq"]
      trained.adam_step = int(state[index]["step"])


def peft_key(name: str) -> str:
  module, matrix = name.rsplit(".", 1)
  return f"{PEFT_PREFIX}{module}.{matrix}.weight"


def lora_key(key: str) -> str:
  key = key.removeprefix(PEFT_PREFIX).removesuffix(".weight")
  # Adapters saved with a named adapter carry it between lora_A and weight.
  module, matrix = key.rsplit(".lora_", 1)
  return f"{module}.lora_{matrix[0]}"


def adapter_config(trained: TrainedWeights, base_model: str) -> dict[str, Any]:
  config = trained.lora_config
  return {
    "peft_type": "LORA",
    "task_type": "CAUSAL_LM",
    "base_model_name_or_path": base_model,
    "r": config.rank,
    "lora_alpha": config.lora_alpha,
    "lora_dropout": config.lora_dropout,
    "target_modules": sorted({name.rsplit(".lora_", 1)[0] for name in trained.weights}),
    "bias": "none",
    "fan_in_fan_out": False,
    "init_lora_weights": True,
    "use_rslora": False,
    "inference_mode": False,
  }


def lora_config_from_adapter(config: dict[str, Any]) -> LoraConfig:
  targets = {str(name).rsplit(".", 1)[-1] for name in config.get("target_modules", [])}
  return LoraConfig(
    rank=int(config["r"]),
    lora_alpha=int(config.get("lora_alpha", config["r"])),
    lora_dropout=float(config.get("lora_dropout", 0.0)),
    train_attn=bool(targets & {"q_proj", "k_proj", "v_proj", "o_proj"}),
    train_mlp=bool(targets & {"gate_proj", "up_proj", "down_proj"}),
    train_unembed="lm_head" in targets,
  )
