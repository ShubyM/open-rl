# Full fine-tuning trainer worker lifecycle.

import gc
import itertools
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from server.model_metadata import SPARSE_DELTA_VERSION, WeightSyncConfig
from training.trainer_worker import BaseTrainerWorker, Datum

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True
  weight_sync_strategy: Literal["full", "delta"] | None = None


@dataclass
class SparseDelta:
  """The elements one optimizer step changed, per parameter, in checkpoint coordinates."""

  total_elements: int
  names: list[str] = field(default_factory=list)
  shapes: list[list[int]] = field(default_factory=list)
  indices: list[torch.Tensor] = field(default_factory=list)
  values: list[torch.Tensor] = field(default_factory=list)

  @property
  def changed_elements(self) -> int:
    return sum(int(indices.numel()) for indices in self.indices)


def trainable_model_parameters(model: PreTrainedModel) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


# (the device a tensor lives on, its pinned host copy)
HostCopy = tuple[torch.device, torch.Tensor]


def host_copy(table: dict, key: object, like: torch.Tensor) -> torch.Tensor:
  """The pinned host copy under key, allocated to match `like` on first use or when its shape changes."""
  entry = table.get(key)
  if entry is None or entry[1].shape != like.shape or entry[1].dtype != like.dtype:
    entry = (like.device, torch.empty(like.shape, dtype=like.dtype, device="cpu", pin_memory=torch.cuda.is_available()))
    table[key] = entry
  return entry[1]


def per_param_state(optimizer: torch.optim.Optimizer | None) -> list[tuple[torch.Tensor, dict]]:
  if optimizer is None:
    return []
  return [(param, state) for param, state in optimizer.state.items() if isinstance(state, dict)]


class FFTTrainingWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.model: PreTrainedModel | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.cpu_offload: bool = True
    self.weight_sync_strategy: str = WeightSyncConfig.from_env().strategy
    # Pinned host copies of the training state, allocated once and reused: where
    # sleep() puts it between GPU leases and, in delta mode, the baseline the
    # next delta is taken against. Parameters and buffers are keyed by the tensor
    # itself; gradients by their parameter, since backward makes a new grad
    # tensor; optimizer state by (parameter, key), since the optimizer replaces
    # its state tensors.
    self.offloaded: bool = False
    self.host_weights: dict[torch.Tensor, HostCopy] = {}
    self.host_grads: dict[torch.Tensor, HostCopy] = {}
    self.host_optimizer_state: dict[tuple[torch.Tensor, str], HostCopy] = {}
    # What the last optim_step changed, published by save_state_delta once the
    # GPU lease is released.
    self.pending_delta: SparseDelta | None = None

  def load_base_model(self, base_model_name: str) -> None:
    """Load one full model for one fine-tuning job process."""
    if self.model is not None and self.base_model_name == base_model_name:
      print(f"Full fine-tuning model {base_model_name} already loaded.")
      return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_device = "auto" if num_gpus > 1 else self.device
    print(f"Loading full fine-tuning model {base_model_name} (target device map: {target_device}, visible GPUs: {num_gpus})...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    self.model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map=target_device)
    print("Successfully loaded full fine-tuning model.")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: FFTConfig | None = None) -> None:
    """Load the per-job model if needed, then prepare it for full fine-tuning."""
    if config is not None:
      self.cpu_offload = config.cpu_offload
      if config.weight_sync_strategy:
        self.weight_sync_strategy = config.weight_sync_strategy
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    self.prepare_model_for_training()

  def prepare_model_for_training(self) -> None:
    assert self.model is not None, "Model is not loaded. Call load_base_model first."

    for param in self.model.parameters():
      param.requires_grad_(True)
    self.trainable_params = trainable_model_parameters(self.model)
    if self.weight_sync_strategy == "delta":
      # The delta baseline: each trainable parameter's host copy equals its device value.
      for param in self.trainable_params:
        host_copy(self.host_weights, param, param.data).copy_(param.data, non_blocking=True)

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on full fine-tuning model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.model.train()

  def assert_off_gpu(self, action: str) -> None:
    """Saves run outside the time-slicer lease, so with offload on the state must already be on the host."""
    if self.cpu_offload and not self.offloaded:
      raise RuntimeError(f"Cannot {action} while the worker holds the GPU (cpu_offload=True, not offloaded); saves run outside the GPU lease.")

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    self.assert_off_gpu("save model")

    tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
    name = alias or "fft-model"
    save_path = name if os.path.isabs(name) else os.path.join(tmp_dir, "fft", name)
    os.makedirs(save_path, exist_ok=True)

    self.model.save_pretrained(save_path)
    if self.tokenizer is not None:
      self.tokenizer.save_pretrained(save_path)

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": "weights",
      "model_id": alias,
      "timestamp": time.time(),
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved full fine-tuning model to {save_path}")
    return {"path": save_path}

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    self.assert_off_gpu("save state")

    if self.weight_sync_strategy == "delta" and not include_optimizer:
      return self.save_state_delta(model_id=model_id, state_path=state_path, kind=kind)

    os.makedirs(state_path, exist_ok=True)
    self.model.save_pretrained(state_path)
    if self.tokenizer is not None:
      self.tokenizer.save_pretrained(state_path)

    if include_optimizer and self.optimizer is not None:
      torch.save(self.optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and self.optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved full fine-tuning state to {state_path}")
    return {"path": state_path}

  def save_state_delta(
    self,
    model_id: str,
    state_path: str,
    kind: str = "sampler",
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    self.assert_off_gpu("save state delta")

    os.makedirs(state_path, exist_ok=True)
    t_collect_start = time.perf_counter()
    # Before the first optim_step there is nothing to publish: an empty delta.
    delta = self.pending_delta if self.pending_delta is not None else SparseDelta(total_elements=sum(p.numel() for p in self.model.parameters()))
    # Separate values tensors preserve each parameter's dtype (e.g. FP32 norms
    # alongside BF16 projections). Names/indices stay in checkpoint coordinates.
    packed_delta = {}
    for i, (indices, values) in enumerate(zip(delta.indices, delta.values, strict=True)):
      packed_delta[f"{i}.indices"] = indices.contiguous()
      packed_delta[f"{i}.values"] = values.contiguous()

    t_collect_end = time.perf_counter()
    collect_time = t_collect_end - t_collect_start

    import safetensors.torch

    delta_path = os.path.join(state_path, "delta.safetensors")
    t_save_start = time.perf_counter()
    safetensors.torch.save_file(
      packed_delta,
      delta_path,
      metadata={"layer_names": json.dumps(delta.names)},
    )
    t_save_end = time.perf_counter()
    save_file_time = t_save_end - t_save_start

    density_pct = round(100.0 * delta.changed_elements / max(1, delta.total_elements), 3)
    logger.info(
      f"[SAVE_STATE_DELTA] model_id={model_id} kind={kind} | "
      f"collect_time={collect_time:.4f}s | "
      f"safetensors_save_time={save_file_time:.4f}s | "
      f"total_delta_save_time={collect_time + save_file_time:.4f}s | "
      f"changed={delta.changed_elements}/{delta.total_elements} ({density_pct:.2f}%) across {len(delta.names)} layers"
    )

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "format": "sparse_delta",
      "format_version": SPARSE_DELTA_VERSION,
      "layer_shapes": delta.shapes,
      "kind": kind,
      "model_id": model_id,
      "changed_elements": delta.changed_elements,
      "total_elements": delta.total_elements,
      "layer_names": delta.names,
      "density_pct": density_pct,
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved sparse delta ({density_pct}% changed elements, {delta.changed_elements}/{delta.total_elements}) to {state_path}")
    return {"path": state_path, "density_pct": density_pct}

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")

    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")

    self.base_model_name = base_model
    self.tokenizer = AutoTokenizer.from_pretrained(state_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    target_device = "auto" if num_gpus > 1 else self.device
    self.model = AutoModelForCausalLM.from_pretrained(state_path, dtype=dtype, device_map=target_device)
    # A new model means new tensors; the old model's host copies go with it.
    self.host_weights, self.host_grads, self.host_optimizer_state = {}, {}, {}
    self.pending_delta = None
    self.prepare_model_for_training()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, "optimizer.pt")
      if os.path.exists(optimizer_path):
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=1e-4)
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        print(f"Restored optimizer state from {optimizer_path}")

    print(f"Loaded full fine-tuning state from {state_path}")
    return {"model_id": model_id, "base_model": base_model}

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    res = super().forward_backward(self.model, data, loss_fn, loss_config)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def compute_weight_delta(self) -> SparseDelta:
    """Diff every trainable parameter against its host copy and advance the copy."""
    assert self.model is not None, "Model must be loaded first."
    delta = SparseDelta(total_elements=sum(p.numel() for p in self.model.parameters()))
    for name, param in self.model.named_parameters():
      if not param.requires_grad:
        continue
      baseline = self.host_weights[param][1].view(-1)  # seeded by prepare_model_for_training
      current = param.data.view(-1)
      changed = current.ne(baseline.to(param.device, non_blocking=True))
      indices = changed.nonzero(as_tuple=True)[0]
      if indices.numel() == 0:
        continue
      # int32 indices halve the file when they fit; the sampler accepts either.
      index_dtype = torch.int32 if param.numel() <= 2**31 else torch.int64
      indices = indices.to(index_dtype).cpu()
      values = current[changed].cpu()
      baseline[indices.to(torch.int64)] = values
      delta.names.append(name)
      delta.shapes.append(list(param.shape))
      delta.indices.append(indices)
      delta.values.append(values)
    return delta

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if not self.trainable_params:
      self.trainable_params = trainable_model_parameters(self.model)

    if self.optimizer is None:
      lr = adam_params.get("learning_rate", 1e-4)
      beta1 = adam_params.get("beta1", 0.9)
      beta2 = adam_params.get("beta2", 0.95)
      eps = adam_params.get("eps", 1e-12)
      weight_decay = adam_params.get("weight_decay", 0.0)

      print(f"Initializing AdamW optimizer for full fine-tuning model with lr={lr}")
      self.optimizer = torch.optim.AdamW(
        self.trainable_params,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
      )

    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    t_clip_start = time.perf_counter()
    total_norm = torch.nn.utils.clip_grad_norm_(
      self.trainable_params,
      max_grad_norm,
    )
    t_clip_end = time.perf_counter()
    clip_time = t_clip_end - t_clip_start

    t_step_start = time.perf_counter()
    self.optimizer.step()
    self.optimizer.zero_grad()
    t_step_end = time.perf_counter()
    step_time = t_step_end - t_step_start

    delta_compute_time = 0.0
    if self.weight_sync_strategy == "delta":
      t_delta_start = time.perf_counter()
      self.pending_delta = self.compute_weight_delta()
      delta_compute_time = time.perf_counter() - t_delta_start
      delta = self.pending_delta
      logger.info(
        f"[OPTIM_STEP] model_id={model_id} | delta_compute_time={delta_compute_time:.4f}s | "
        f"changed={delta.changed_elements}/{delta.total_elements} "
        f"({100.0 * delta.changed_elements / max(1, delta.total_elements):.2f}%) across {len(delta.names)} layers"
      )

    logger.info(
      f"[OPTIM_STEP] model_id={model_id} | clip_grad_time={clip_time:.4f}s | "
      f"optimizer_step_time={step_time:.4f}s | delta_compute_time={delta_compute_time:.4f}s | "
      f"total_optim_time={clip_time + step_time + delta_compute_time:.4f}s"
    )

    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(total_norm.item()),
        "time/compute_delta_diff": self.sanitize_float(delta_compute_time),
        "time/optimizer_step": self.sanitize_float(step_time),
        "time/clip_grad_norm": self.sanitize_float(clip_time),
      },
    }

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    return super().generate(self.model, prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)

  # The processor brackets each GPU lease with these.
  def sleep(self) -> None:
    """Copy the device-resident training state to the host and free it on the device.

    Parameters are left pointing at their host copies, so shapes stay right and
    save_pretrained reads them directly while the worker is off the GPU.
    """
    if not self.cpu_offload or self.model is None or self.offloaded or not torch.cuda.is_available():
      return
    start = time.perf_counter()

    # Queue every device-to-host copy, then wait once, then free: the copies
    # overlap and nothing is released before it has landed.
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor.device.type == "cuda":
        host_copy(self.host_weights, tensor, tensor.data).copy_(tensor.data, non_blocking=True)
      grad = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
      if grad is not None and grad.device.type == "cuda":
        host_copy(self.host_grads, tensor, grad).copy_(grad, non_blocking=True)
    for param, state in per_param_state(self.optimizer):
      for key, value in list(state.items()):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
          host_copy(self.host_optimizer_state, (param, key), value).copy_(value, non_blocking=True)

    torch.cuda.synchronize()

    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      entry = self.host_weights.get(tensor)
      if entry is not None:
        tensor.data = entry[1]
      grad = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
      if grad is not None and tensor in self.host_grads:
        grad.data = self.host_grads[tensor][1]
    for param, state in per_param_state(self.optimizer):
      for key in list(state):
        entry = self.host_optimizer_state.get((param, key))
        if entry is not None:
          state[key] = entry[1]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    self.offloaded = True
    print(f"[FFTTrainingWorker] Offloaded weights & states to pinned host memory in {(time.perf_counter() - start) * 1000:.1f} ms.")

  def wake_up(self) -> None:
    """Bring the training state back to the device. The host copies stay allocated for the next sleep."""
    if not self.offloaded:
      return
    assert self.model is not None, "Model must be loaded first."
    start = time.perf_counter()

    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      entry = self.host_weights.get(tensor)
      if entry is not None:
        tensor.data = entry[1].to(entry[0], non_blocking=True)
      grad = tensor.grad if isinstance(tensor, torch.nn.Parameter) else None
      if grad is not None and tensor in self.host_grads:
        device, host = self.host_grads[tensor]
        grad.data = host.to(device, non_blocking=True)
    for param, state in per_param_state(self.optimizer):
      for key in list(state):
        entry = self.host_optimizer_state.get((param, key))
        if entry is not None:
          state[key] = entry[1].to(entry[0], non_blocking=True)

    torch.cuda.synchronize()
    self.offloaded = False
    print(f"[FFTTrainingWorker] Reloaded weights & states to the device in {(time.perf_counter() - start) * 1000:.1f} ms.")
