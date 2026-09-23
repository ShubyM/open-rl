# Full fine-tuning trainer worker lifecycle.

import gc
import itertools
import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.trainer_worker import BaseTrainerWorker, Datum

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True
  weight_sync_strategy: str | None = None


def trainable_model_parameters(model: PreTrainedModel) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


from server.model_metadata import SPARSE_DELTA_VERSION, WeightSyncConfig


class FFTTrainingWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.model: PreTrainedModel | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.cpu_offload: bool = True
    self.weight_sync_cfg: WeightSyncConfig = WeightSyncConfig.from_env()
    self._prev_weights: dict[str, torch.Tensor] = {}
    self._saved_weights: dict[str, torch.Tensor] = {}

  def set_weight_sync_strategy(self, strategy: str) -> None:
    if strategy not in ("full", "delta"):
      raise ValueError(f"Invalid weight_sync_strategy '{strategy}'. Must be 'full' or 'delta'.")
    self.weight_sync_cfg.strategy = strategy
    if strategy == "full":
      self._prev_weights.clear()
      self._saved_weights.clear()
    elif self.model is not None and not self._prev_weights:
      self._prev_weights = {name: p.data.detach().cpu().clone() for name, p in self.model.named_parameters() if p.requires_grad}

  def _is_on_cuda(self) -> bool:
    if self.model is None:
      return False
    return any(t.device.type == "cuda" for t in itertools.chain(self.model.parameters(), self.model.buffers()))

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
      if hasattr(config, "weight_sync_strategy") and config.weight_sync_strategy:
        self.set_weight_sync_strategy(config.weight_sync_strategy)
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    self.prepare_model_for_training()

  def prepare_model_for_training(self) -> None:
    assert self.model is not None, "Model is not loaded. Call load_base_model first."

    for param in self.model.parameters():
      param.requires_grad_(True)
    self.trainable_params = trainable_model_parameters(self.model)
    self._saved_weights.clear()
    self._prev_weights = (
      {name: param.data.detach().cpu().clone() for name, param in self.model.named_parameters() if param.requires_grad}
      if self.weight_sync_cfg.strategy == "delta"
      else {}
    )

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on full fine-tuning model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.model.train()

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and self._is_on_cuda():
      raise RuntimeError("Cannot save model while worker tensors are on CUDA when cpu_offload=True. GPU time-slicer lock is not held during saves.")

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
    if self.cpu_offload and self._is_on_cuda():
      raise RuntimeError("Cannot save state while worker tensors are on CUDA when cpu_offload=True. GPU time-slicer lock is not held during saves.")

    if self.weight_sync_cfg.strategy == "delta" and not include_optimizer:
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
    if self.cpu_offload and self._is_on_cuda():
      raise RuntimeError(
        "Cannot save state delta while worker tensors are on CUDA when cpu_offload=True. GPU time-slicer lock is not held during saves."
      )

    os.makedirs(state_path, exist_ok=True)
    t_collect_start = time.perf_counter()
    layer_names_list: list[str] = []
    layer_shapes: list[list[int]] = []
    packed_delta: dict[str, torch.Tensor] = {}
    total_changed = 0
    total_elements = 0

    # Separate values tensors preserve each parameter's dtype (e.g. FP32 norms
    # alongside BF16 projections). Names/indices stay in checkpoint coordinates.
    record_saved = not self._saved_weights
    for name, param in self.model.named_parameters():
      total_elements += param.numel()
      if not param.requires_grad:
        continue
      cur_cpu = param.data.detach().cpu()
      if record_saved:
        self._saved_weights[name] = cur_cpu.clone()
      prev_cpu = self._prev_weights.get(name)
      if prev_cpu is None or prev_cpu.shape != cur_cpu.shape:
        self._prev_weights[name] = cur_cpu.clone()
        continue
      diff_mask = cur_cpu.view(-1).ne(prev_cpu.view(-1))
      indices = diff_mask.nonzero(as_tuple=True)[0]
      if indices.numel() > 0:
        index_dtype = torch.int32 if param.numel() <= 2**31 else torch.int64
        idx_cpu = indices.to(index_dtype).contiguous()
        val_cpu = cur_cpu.view(-1)[diff_mask].contiguous()
        layer_idx = len(layer_names_list)
        layer_names_list.append(name)
        layer_shapes.append(list(param.shape))
        packed_delta[f"{layer_idx}.indices"] = idx_cpu
        packed_delta[f"{layer_idx}.values"] = val_cpu
        total_changed += int(idx_cpu.numel())

    t_collect_end = time.perf_counter()
    collect_time = t_collect_end - t_collect_start

    import safetensors.torch

    delta_path = os.path.join(state_path, "delta.safetensors")
    t_save_start = time.perf_counter()
    safetensors.torch.save_file(
      packed_delta,
      delta_path,
      metadata={"layer_names": json.dumps(layer_names_list)},
    )
    t_save_end = time.perf_counter()
    save_file_time = t_save_end - t_save_start

    logger.info(
      f"[SAVE_STATE_DELTA] model_id={model_id} kind={kind} | "
      f"collect_time={collect_time:.4f}s | "
      f"safetensors_save_time={save_file_time:.4f}s | "
      f"total_delta_save_time={collect_time + save_file_time:.4f}s | "
      f"changed={total_changed}/{total_elements} ({100.0 * total_changed / max(1, total_elements):.2f}%) across {len(layer_names_list)} layers"
    )

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "format": "sparse_delta",
      "format_version": SPARSE_DELTA_VERSION,
      "layer_shapes": layer_shapes,
      "kind": kind,
      "model_id": model_id,
      "changed_elements": total_changed,
      "total_elements": total_elements,
      "layer_names": layer_names_list,
      "density_pct": round(100.0 * total_changed / max(1, total_elements), 3),
      "timestamp": time.time(),
    }
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved sparse delta ({metadata['density_pct']}% changed elements, {total_changed}/{total_elements}) to {state_path}")
    return {"path": state_path, "density_pct": metadata["density_pct"]}

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

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if not self.trainable_params:
      self.trainable_params = trainable_model_parameters(self.model)
    if self._saved_weights:
      self._prev_weights = self._saved_weights
      self._saved_weights = {}

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

    logger.info(
      f"[OPTIM_STEP] model_id={model_id} | clip_grad_time={clip_time:.4f}s | "
      f"optimizer_step_time={step_time:.4f}s | total_optim_time={clip_time + step_time:.4f}s"
    )

    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(total_norm.item()),
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

  def sleep(self) -> None:
    """Offload GPU tensors to pinned host CPU memory and empty CUDA allocator cache."""
    if not self.cpu_offload or not self._is_on_cuda():
      return
    start_t = time.perf_counter()

    # Phase 1: Launch batched asynchronous DMA copies while GPU tensors remain alive.
    staged: list[tuple[torch.Tensor, torch.Tensor]] = []
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor.device.type == "cuda":
        cpu_buf = torch.empty_like(tensor.data, device="cpu", pin_memory=True)
        cpu_buf.copy_(tensor.data, non_blocking=True)
        staged.append((tensor, cpu_buf))
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor.grad.device.type == "cuda":
        cpu_grad = torch.empty_like(tensor.grad.data, device="cpu", pin_memory=True)
        cpu_grad.copy_(tensor.grad.data, non_blocking=True)
        staged.append((tensor.grad, cpu_grad))

    opt_staged: list[tuple[dict[Any, Any], Any, torch.Tensor]] = []
    if self.optimizer is not None:
      for state in self.optimizer.state.values():
        if isinstance(state, dict):
          for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.device.type == "cuda" and k != "step":
              cpu_buf = torch.empty_like(v, device="cpu", pin_memory=True)
              cpu_buf.copy_(v, non_blocking=True)
              opt_staged.append((state, k, cpu_buf))

    # Phase 2: Single barrier synchronization point.
    torch.cuda.synchronize()

    # Phase 3: Point parameters, gradients, and optimizer states at the pinned CPU tensors.
    for target, cpu_buf in staged:
      target.data = cpu_buf
    for state, k, cpu_buf in opt_staged:
      state[k] = cpu_buf
    del staged, opt_staged

    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
      torch.cuda.ipc_collect()

    print(f"[FFT Worker] Offloaded weights & states to pinned CPU memory in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def wake_up(self) -> None:
    """Reload pinned CPU tensors back to CUDA VRAM."""
    if not self.cpu_offload or self.model is None or self._is_on_cuda() or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()

    # Keep pinned CPU source tensors alive until the async H2D DMA copies complete.
    cpu_refs: list[torch.Tensor] = []
    for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
      if tensor.device.type == "cpu":
        cpu_refs.append(tensor.data)
        tensor.data = tensor.data.to(self.device, non_blocking=True)
      if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None and tensor.grad.device.type == "cpu":
        cpu_refs.append(tensor.grad.data)
        tensor.grad.data = tensor.grad.data.to(self.device, non_blocking=True)

    if self.optimizer is not None:
      for param, state in self.optimizer.state.items():
        if isinstance(state, dict):
          for k, v in list(state.items()):
            if isinstance(v, torch.Tensor) and v.device.type == "cpu" and k != "step":
              cpu_refs.append(v)
              state[k] = v.to(param.device, non_blocking=True)

    torch.cuda.synchronize()
    del cpu_refs
    print(f"[FFT Worker] Reloaded weights & states to CUDA in {(time.perf_counter() - start_t) * 1000:.1f} ms.")
