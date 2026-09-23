"""One full fine-tuning model, its optimizer, and optional CUDA residency."""

import gc
import itertools
import json
import os
import time
from datetime import datetime
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from server.model_metadata import WeightSyncConfig
from training import hf_operations
from training.types import Datum, FFTConfig
from training.weight_export import capture_weights, write_sparse_delta

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


def trainable_model_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


class FFTTrainingWorker:
  def __init__(
    self,
    *,
    model: PreTrainedModel | None = None,
    tokenizer: Any = None,
    device: torch.device | str | None = None,
    base_model_name: str | None = None,
    cpu_offload: bool = True,
    weight_sync_cfg: WeightSyncConfig | None = None,
    token_budget: int | None = None,
  ):
    if device is not None:
      self.device = torch.device(device)
    elif model is not None:
      self.device = next(model.parameters()).device
    else:
      self.device = hf_operations.default_device()
    self.dtype: torch.dtype | None = next(model.parameters()).dtype if model is not None else None
    self.device_map: torch.device | str | None = None
    self.tokenizer = tokenizer
    self.model = model
    self.base_model_name = base_model_name
    self.params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.cpu_offload = cpu_offload
    self.token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0")) if token_budget is None else token_budget
    self.weight_sync_cfg = weight_sync_cfg if weight_sync_cfg is not None else WeightSyncConfig.from_env()
    self._is_offloaded = False
    self._param_shadow: dict[torch.Tensor, tuple[torch.device, torch.Tensor]] = {}
    self._grad_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._opt_shadow: dict[tuple[torch.nn.Parameter, str], tuple[torch.device, torch.Tensor]] = {}
    self._exported_weights: dict[str, torch.Tensor] = {}
    if model is not None:
      self.prepare_model_for_training()

  def set_weight_sync_strategy(self, strategy: str) -> None:
    if strategy not in ("full", "delta"):
      raise ValueError(f"Invalid weight_sync_strategy '{strategy}'. Must be 'full' or 'delta'.")
    self.weight_sync_cfg.strategy = strategy
    # A change to delta sends a full replacement on its first export; the
    # sampler may have received a full checkpoint since the last delta.
    self._exported_weights.clear()

  def load_base_model(self, base_model_name: str) -> None:
    if self.model is not None and self.base_model_name == base_model_name:
      return
    self.base_model_name = base_model_name
    self._load_hf_model(base_model_name)

  def _load_hf_model(self, path: str) -> None:
    # CUDA capability queries may initialize a context. Resolve loading
    # settings once, inside the same lease as the first model allocation.
    if self.dtype is None:
      self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    if self.device_map is None:
      self.device_map = "auto" if self.device.type == "cuda" and torch.cuda.device_count() > 1 else self.device
    self.tokenizer = AutoTokenizer.from_pretrained(path)
    self.model = AutoModelForCausalLM.from_pretrained(path, dtype=self.dtype, device_map=self.device_map)

  def create_model(self, base_model_name: str, model_id: str | None = None, config: FFTConfig | None = None) -> None:
    if config is not None:
      self.cpu_offload = config.cpu_offload
      if config.weight_sync_strategy:
        self.set_weight_sync_strategy(config.weight_sync_strategy)
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    self.prepare_model_for_training()

  def prepare_model_for_training(self, *, capture_export_baseline: bool = True) -> None:
    assert self.model is not None, "Model is not loaded. Call load_base_model first."
    for param in self.model.parameters():
      param.requires_grad_(True)
    self.params = trainable_model_parameters(self.model)
    self.optimizer = None
    self._param_shadow.clear()
    self._grad_shadow.clear()
    self._opt_shadow.clear()
    self._is_offloaded = False
    self._exported_weights = capture_weights(self.model) if capture_export_baseline and self.weight_sync_cfg.strategy == "delta" else {}
    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
      except (AttributeError, ValueError):
        pass
    self.model.train()

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    """Save a full HF checkpoint while awake, independently of sampler export."""
    assert self.model is not None, "Model must be loaded first."
    os.makedirs(state_path, exist_ok=True)
    self.model.save_pretrained(state_path)
    if self.tokenizer is not None:
      self.tokenizer.save_pretrained(state_path)
    has_optimizer = include_optimizer and self.optimizer is not None
    if has_optimizer:
      torch.save(self.optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": has_optimizer,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as file:
      json.dump(metadata, file)
    return {"path": state_path}

  def save_for_sampler(self, model_id: str, alias: str | None, ref: str | None) -> str:
    """Export explicitly. Delta consumers must apply successful exports in order."""
    assert self.model is not None, "Model must be loaded first."
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    rel_path = ref.removeprefix("tinker://").lstrip("/")
    path = os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "sampler_full", rel_path)
    if self.weight_sync_cfg.strategy == "delta":
      write_sparse_delta(
        self.model,
        self._exported_weights,
        path,
        model_id=model_id,
        base_model=self.base_model_name,
        delta_format=self.weight_sync_cfg.delta_format,
      )
    else:
      self.save_state(model_id, path, include_optimizer=False, kind="sampler")
    return path

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    with open(os.path.join(state_path, "metadata.json")) as file:
      metadata = json.load(file)
    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")
    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if restore_optimizer and (not metadata.get("has_optimizer") or not os.path.isfile(optimizer_path)):
      raise ValueError(f"Checkpoint {state_path} has no optimizer state")
    self._load_hf_model(state_path)
    self.base_model_name = base_model
    # The sampler may still hold an earlier model. The first post-restore
    # export replaces every parameter before incremental publication resumes.
    self.prepare_model_for_training(capture_export_baseline=False)
    if restore_optimizer:
      self.optimizer = hf_operations.build_optimizer(self.params, {})
      self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=True))
    return {"model_id": model_id, "base_model": base_model}

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    return hf_operations.forward_backward(
      self.model,
      data,
      loss_fn,
      loss_config,
      forward_only=forward_only,
      tokenizer=self.tokenizer,
      device=self.device,
      token_budget=self.token_budget,
    )

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.optimizer is None:
      self.optimizer = hf_operations.build_optimizer(self.params, adam_params)
    return {"metrics": hf_operations.optim_step(self.optimizer, adam_params)}

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    return hf_operations.generate(
      self.model, prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs, tokenizer=self.tokenizer, device=self.device
    )

  def sleep(self) -> None:
    """Offload GPU tensors to pinned host CPU memory and empty CUDA allocator cache."""
    if not self.cpu_offload or self.model is None or self._is_offloaded or self.device.type != "cuda":
      return
    start_t = time.perf_counter()

    # Phase 1: Launch Batched Asynchronous DMA copies WITHOUT freeing GPU tensors!
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor.device.type == "cuda":
        orig_device = tensor.device
        if tensor in self._param_shadow and self._param_shadow[tensor][1].shape == tensor.shape:
          cpu_buf = self._param_shadow[tensor][1]
        else:
          cpu_buf = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          self._param_shadow[tensor] = (orig_device, cpu_buf)
        cpu_buf.copy_(tensor.data, non_blocking=True)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor.grad.device.type == "cuda":
        orig_device = tensor.grad.device
        if tensor in self._grad_shadow and self._grad_shadow[tensor][1].shape == tensor.grad.shape:
          cpu_buf = self._grad_shadow[tensor][1]
        else:
          cpu_buf = torch.empty(tensor.grad.shape, dtype=tensor.grad.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          self._grad_shadow[tensor] = (orig_device, cpu_buf)
        cpu_buf.copy_(tensor.grad.data, non_blocking=True)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.device.type == "cuda":
              orig_device = v.device
              opt_key = (param, k)
              if opt_key in self._opt_shadow and self._opt_shadow[opt_key][1].shape == v.shape:
                cpu_buf = self._opt_shadow[opt_key][1]
              else:
                cpu_buf = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=torch.cuda.is_available())
                self._opt_shadow[opt_key] = (orig_device, cpu_buf)
              cpu_buf.copy_(v, non_blocking=True)

    # Phase 2: Single Barrier Synchronization point!
    if torch.cuda.is_available():
      torch.cuda.synchronize()

    # Phase 3: Now that DMA has finished, safely deallocate GPU VRAM!
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor in self._param_shadow:
        orig_device = self._param_shadow[tensor][0]
        tensor.data = torch.empty(0, dtype=tensor.dtype, device=orig_device)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor in self._grad_shadow:
        orig_device = self._grad_shadow[tensor][0]
        tensor.grad.data = torch.empty(0, dtype=tensor.grad.dtype, device=orig_device)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          for k in list(state.keys()):
            opt_key = (param, k)
            if opt_key in self._opt_shadow:
              orig_device, cpu_buf = self._opt_shadow[opt_key]
              state[k] = cpu_buf

    if torch.cuda.is_available():
      gc.collect()
      torch.cuda.empty_cache()
      if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()

    self._is_offloaded = True
    print(f"[FFT Worker] Offloaded weights & states to pinned CPU memory in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def wake_up(self) -> None:
    """Reload pinned CPU shadow tensors back to CUDA VRAM without destroying host shadow buffers."""
    if not self.cpu_offload or self.model is None or not self._is_offloaded or self.device.type != "cuda":
      return
    start_t = time.perf_counter()

    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor in self._param_shadow:
        orig_device, cpu_data = self._param_shadow[tensor]
        tensor.data = cpu_data.to(orig_device, non_blocking=True)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor in self._grad_shadow:
        orig_device, cpu_grad = self._grad_shadow[tensor]
        tensor.grad.data = cpu_grad.to(orig_device, non_blocking=True)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          state.pop("_orig_devices", None)
          target_device = param.device
          for k, v in list(state.items()):
            opt_key = (param, k)
            if opt_key in self._opt_shadow:
              orig_device, cpu_buf = self._opt_shadow[opt_key]
              state[k] = cpu_buf.to(orig_device, non_blocking=True)
            elif isinstance(v, torch.Tensor) and v.device.type == "cpu" and k != "step":
              state[k] = v.to(target_device, non_blocking=True)

    if torch.cuda.is_available():
      torch.cuda.synchronize()

    self._is_offloaded = False
    print(f"[FFT Worker] Reloaded weights & states to CUDA in {(time.perf_counter() - start_t) * 1000:.1f} ms.")
