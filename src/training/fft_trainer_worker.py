# Full fine-tuning: one model per process, trained whole and published as a
# whole checkpoint or a sparse weight delta.

import gc
import itertools
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from server.model_metadata import WeightSyncConfig
from training.commands import CreateModel, CreateModelFromState, LoadWeights
from training.distributed import default_device
from training.trainer_worker import Trainer, TrainingWorker, enable_gradient_checkpointing, sanitize_float, tmp_dir
from training.types import FFTConfig

__all__ = ["FFTConfig", "FFTTrainer", "FFTTrainingWorker", "trainable_model_parameters"]


def trainable_model_parameters(model: PreTrainedModel) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


class FFTTrainer(Trainer):
  """One full-parameter model. Every weight trains, so the optimizer, the CPU
  offload buffers and the sparse delta shadow all belong to this one trainer.

  optim_step runs the shared clip/step/zero and then, under the delta sync
  strategy, records which weights moved since the last step so the samplers can
  be patched instead of reloaded.
  """

  def __init__(
    self,
    model_id: str,
    model: PreTrainedModel,
    params: list[torch.nn.Parameter],
    tokenizer: Any,
    base_model_name: str,
    cpu_offload: bool,
    weight_sync_cfg: WeightSyncConfig,
  ):
    super().__init__(model_id, model, params, tokenizer=tokenizer, base_model_name=base_model_name)
    self.cpu_offload = cpu_offload
    self.weight_sync_cfg = weight_sync_cfg
    self.model_layer_names = [name for name, p in model.named_parameters() if p.requires_grad]
    self.total_model_elements = sum(p.numel() for p in model.parameters())
    self._is_offloaded = False
    self._latest_delta_tensors: dict[str, Any] = {}
    self._latest_total_changed = 0
    self._latest_total_elements = 0
    self._param_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._grad_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._opt_shadow: dict[tuple[torch.nn.Parameter, str], tuple[torch.device, torch.Tensor]] = {}
    if self.weight_sync_cfg.strategy == "delta":
      self.seed_delta_shadow()

  # -- delta shadow -----------------------------------------------------------

  def seed_delta_shadow(self) -> None:
    """Snapshot each trainable weight into a pinned CPU buffer so the next
    optim_step can diff against it."""
    for param in self.model.parameters():
      if param.requires_grad and param not in self._param_shadow:
        cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
        cpu_buf.copy_(param.data, non_blocking=True)
        self._param_shadow[param] = (param.device, cpu_buf)

  def prepare_model_for_training(self) -> None:
    """Recapture the trainable set after the model was built or reloaded."""
    for param in self.model.parameters():
      param.requires_grad_(True)
    self.params = trainable_model_parameters(self.model)
    self.model_layer_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
    self.total_model_elements = sum(p.numel() for p in self.model.parameters())
    if self.weight_sync_cfg.strategy == "delta":
      self.seed_delta_shadow()
    enable_gradient_checkpointing(self.model)
    self.model.train()

  # -- training ---------------------------------------------------------------

  def forward_backward(self, data: list, loss_fn: str, loss_config: dict | None = None, forward_only: bool = False) -> dict[str, Any]:
    res = super().forward_backward(data, loss_fn, loss_config, forward_only)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def optim_step(self, adam_params: dict[str, Any]) -> dict[str, Any]:
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    metrics = super().optim_step(adam_params)["metrics"]
    delta_time = self.compute_delta() if self.weight_sync_cfg.strategy == "delta" else 0.0
    metrics["time/compute_delta_diff"] = sanitize_float(delta_time)
    return {"metrics": metrics}

  def compute_delta(self) -> float:
    """Diff every trainable weight against its CPU shadow and record the changed
    elements for the next sampler delta. Returns the seconds it took."""
    start = time.perf_counter()
    self._latest_delta_tensors = {}
    self._latest_total_changed = 0
    self._latest_total_elements = self.total_model_elements

    layer_names_list: list[str] = []
    indices_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []
    layer_lengths_list: list[int] = []

    for name, param in self.model.named_parameters():
      if not param.requires_grad:
        continue
      prev_tensor = self.prev_cpu_weight(param)
      if prev_tensor is None:
        cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
        cpu_buf.copy_(param.data, non_blocking=True)
        self._param_shadow[param] = (param.device, cpu_buf)
        prev_tensor = cpu_buf

      prev_gpu = prev_tensor.to(param.device, non_blocking=True)
      diff_mask = param.data.view(-1).ne(prev_gpu.view(-1))
      indices = diff_mask.nonzero(as_tuple=True)[0]
      if indices.numel() > 0:
        idx_cpu = indices.to(torch.int64).contiguous().cpu()
        val_cpu = param.data.view(-1)[diff_mask].contiguous().cpu()
        layer_names_list.append(name)
        indices_list.append(idx_cpu)
        values_list.append(val_cpu)
        layer_lengths_list.append(int(idx_cpu.numel()))
        self._latest_total_changed += int(idx_cpu.numel())
        self.record_cpu_weight(param, idx_cpu, val_cpu)
      del prev_gpu, diff_mask, indices

    if self.weight_sync_cfg.delta_format == "vllm_fused":
      layer_names_list, indices_list = self.remap_hf_to_vllm_fused(layer_names_list, indices_list)

    self._latest_delta_tensors = {
      "names": layer_names_list,
      "indices_list": indices_list,
      "values_list": values_list,
      "layer_lengths_list": layer_lengths_list,
    }
    elapsed = time.perf_counter() - start
    logger.info(
      f"[OPTIM_STEP] model_id={self.model_id} | delta_compute_time={elapsed:.4f}s | "
      f"changed={self._latest_total_changed}/{self._latest_total_elements} "
      f"({100.0 * self._latest_total_changed / max(1, self._latest_total_elements):.2f}%) across {len(layer_names_list)} layers"
    )
    return elapsed

  def prev_cpu_weight(self, param: torch.nn.Parameter) -> torch.Tensor | None:
    if param in self._param_shadow:
      return self._param_shadow[param][1]
    return None

  def record_cpu_weight(self, param: torch.nn.Parameter, indices: torch.Tensor, values: torch.Tensor) -> None:
    if param in self._param_shadow:
      self._param_shadow[param][1].view(-1)[indices.to(torch.int64).cpu()] = values

  def remap_hf_to_vllm_fused(
    self,
    layer_names_list: list[str],
    indices_list: list[torch.Tensor],
  ) -> tuple[list[str], list[torch.Tensor]]:
    """Remaps HF layer names (q_proj, k_proj, v_proj, gate_proj, up_proj) and offsets indices to vLLM fused names."""
    config = getattr(self.model, "config", None)
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

  # -- offload gating and shadow views ----------------------------------------

  def require_offloaded_for_save(self, action: str) -> None:
    """A cpu_offload worker saves from its host shadow, which is only valid once
    sleep() has offloaded. Saving while the model is GPU-resident would race the
    lease it is not holding."""
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(f"Cannot {action} while the model is GPU-resident and cpu_offload=True; the GPU lease is not held during host-side saves.")

  def restore_from_shadow(self) -> bool:
    """Point the offloaded parameters at their host shadow so save_pretrained can
    read real weights. Returns whether the model was offloaded."""
    was_offloaded = self._is_offloaded
    if was_offloaded:
      for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
        if tensor in self._param_shadow:
          tensor.data = self._param_shadow[tensor][1]
    return was_offloaded

  def drop_shadow_view(self, was_offloaded: bool) -> None:
    if was_offloaded:
      for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
        if tensor in self._param_shadow:
          tensor.data = torch.empty(0, dtype=tensor.dtype, device=self._param_shadow[tensor][0])

  # -- checkpoints and publication --------------------------------------------

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    self.require_offloaded_for_save("save model")
    name = alias or "fft-model"
    save_path = name if os.path.isabs(name) else os.path.join(tmp_dir(), "fft", name)
    os.makedirs(save_path, exist_ok=True)

    was_offloaded = self.restore_from_shadow()
    try:
      self.model.save_pretrained(save_path)
      if self.tokenizer is not None:
        self.tokenizer.save_pretrained(save_path)
    finally:
      self.drop_shadow_view(was_offloaded)

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
      json.dump(self.checkpoint_metadata(kind="weights", alias=alias), f)

    print(f"Saved full fine-tuning model to {save_path}")
    return {"path": save_path}

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    self.require_offloaded_for_save("save state")
    if self.weight_sync_cfg.strategy == "delta":
      if kind != "sampler":
        logger.warning("save_state for %s under the delta strategy writes a delta, not a resumable checkpoint", self.model_id)
      return self.save_state_delta(model_id=self.model_id, state_path=state_path, kind=kind)

    os.makedirs(state_path, exist_ok=True)
    was_offloaded = self.restore_from_shadow()
    try:
      self.model.save_pretrained(state_path)
      if self.tokenizer is not None:
        self.tokenizer.save_pretrained(state_path)
    finally:
      self.drop_shadow_view(was_offloaded)

    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(self.checkpoint_metadata(kind=kind, has_optimizer=False), f)

    print(f"Saved full fine-tuning state to {state_path}")
    return {"path": state_path}

  def save_state_delta(self, model_id: str, state_path: str, kind: str = "sampler") -> dict[str, Any]:
    self.require_offloaded_for_save("save state delta")
    os.makedirs(state_path, exist_ok=True)

    t_collect_start = time.perf_counter()
    if self._latest_delta_tensors and "names" in self._latest_delta_tensors:
      layer_names_list = self._latest_delta_tensors["names"]
      indices_list = self._latest_delta_tensors["indices_list"]
      values_list = self._latest_delta_tensors["values_list"]
      layer_lengths_list = self._latest_delta_tensors["layer_lengths_list"]
      total_changed = self._latest_total_changed
      total_elements = self._latest_total_elements
    else:
      layer_names_list = self.model_layer_names
      layer_lengths_list = [0] * len(layer_names_list)
      total_changed = 0
      total_elements = self.total_model_elements
      indices_list = []
      values_list = []

    # int64 indices: a flat index into a tensor with more than 2**31 elements
    # (Gemma 4's per-layer embedding table is 2.35e9) does not fit an int32, and
    # a wrapped negative index made the sampler's index_copy_ assert mid-run.
    if indices_list:
      indices_flat = torch.cat(indices_list).to(torch.int64).contiguous()
      values_flat = torch.cat(values_list).contiguous()
    else:
      fallback_dtype = next(self.model.parameters()).dtype
      indices_flat = torch.empty(0, dtype=torch.int64, device="cpu")
      values_flat = torch.empty(0, dtype=fallback_dtype, device="cpu")

    layer_lengths_tensor = torch.tensor(layer_lengths_list, dtype=torch.int64, device="cpu")
    packed_delta = {
      "delta.indices_flat": indices_flat,
      "delta.values_flat": values_flat,
      "delta.layer_lengths": layer_lengths_tensor,
    }
    collect_time = time.perf_counter() - t_collect_start

    import safetensors.torch

    delta_path = os.path.join(state_path, "delta.safetensors")
    t_save_start = time.perf_counter()
    safetensors.torch.save_file(packed_delta, delta_path, metadata={"layer_names": json.dumps(layer_names_list)})
    save_file_time = time.perf_counter() - t_save_start

    logger.info(
      f"[SAVE_STATE_DELTA] model_id={model_id} kind={kind} | "
      f"collect_time={collect_time:.4f}s | safetensors_save_time={save_file_time:.4f}s | "
      f"total_delta_save_time={collect_time + save_file_time:.4f}s | "
      f"changed={total_changed}/{total_elements} ({100.0 * total_changed / max(1, total_elements):.2f}%) across {len(layer_names_list)} layers"
    )

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "format": "sparse_delta",
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

  # -- offload / wake ---------------------------------------------------------

  def sleep(self) -> None:
    """Offload GPU tensors to pinned host CPU memory and empty CUDA allocator cache."""
    if not self.cpu_offload or self._is_offloaded or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()

    # Phase 1: launch batched asynchronous DMA copies without freeing GPU tensors.
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

    # Phase 2: single barrier synchronization point.
    if torch.cuda.is_available():
      torch.cuda.synchronize()

    # Phase 3: now that DMA has finished, safely deallocate GPU VRAM.
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
              _, cpu_buf = self._opt_shadow[opt_key]
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
    if not self.cpu_offload or not self._is_offloaded or not torch.cuda.is_available():
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


class FFTTrainingWorker(TrainingWorker):
  """One full model for the life of the process. create() builds it from a
  create_model request, restore() from a checkpoint, and sleep/wake_up hand its
  GPU memory off around the lease by delegating to the one trainer."""

  single_model = True
  full_parameter = True

  def __init__(self):
    self.trainer: FFTTrainer | None = None
    self.device = torch.device("cpu")
    self.cpu_offload = True
    self.weight_sync_cfg = WeightSyncConfig.from_env()
    self.base_model_name: str | None = None

  def set_weight_sync_strategy(self, strategy: str) -> None:
    if strategy not in ("full", "delta"):
      raise ValueError(f"Invalid weight_sync_strategy '{strategy}'. Must be 'full' or 'delta'.")
    self.weight_sync_cfg.strategy = strategy

  def initialize(self, base_model: str | None = None) -> None:
    self.device = default_device()

  def create(self, command: CreateModel) -> FFTTrainer:
    config = command.full_config
    self.cpu_offload = config.cpu_offload
    if config.weight_sync_strategy:
      self.set_weight_sync_strategy(config.weight_sync_strategy)
    model, tokenizer = self.build_model(command.base_model)
    self.base_model_name = command.base_model
    if config.seed is not None:
      torch.manual_seed(config.seed)
    for param in model.parameters():
      param.requires_grad_(True)
    params = trainable_model_parameters(model)
    trainer = FFTTrainer(command.model_id, model, params, tokenizer, command.base_model, self.cpu_offload, self.weight_sync_cfg)
    trainer.input_device = self.device
    enable_gradient_checkpointing(model)
    model.train()
    self.trainer = trainer
    return trainer

  def restore(self, command: CreateModelFromState) -> FFTTrainer:
    model, tokenizer, base_model = self.build_from_state(command.state_path)
    self.base_model_name = base_model
    params = trainable_model_parameters(model)
    trainer = FFTTrainer(command.model_id, model, params, tokenizer, base_model, self.cpu_offload, self.weight_sync_cfg)
    trainer.input_device = self.device
    trainer.prepare_model_for_training()
    trainer.rebind(trainer.params)
    if command.restore_optimizer:
      metadata_path = os.path.join(command.state_path, "metadata.json")
      with open(metadata_path) as f:
        trainer.restore_optimizer(command.state_path, json.load(f), map_location=self.device)
    self.trainer = trainer
    return trainer

  def reload(self, trainer: FFTTrainer, command: LoadWeights) -> dict[str, Any]:
    """LoadWeights: rebuild the model from the checkpoint, swap it in and rebind
    the optimizer to the fresh parameter objects."""
    model, tokenizer, base_model = self.build_from_state(command.state_path)
    trainer.model = model
    trainer.tokenizer = tokenizer
    trainer.base_model_name = base_model
    self.base_model_name = base_model
    trainer._param_shadow = {}
    trainer._grad_shadow = {}
    trainer._opt_shadow = {}
    trainer._is_offloaded = False
    trainer.prepare_model_for_training()
    trainer.rebind(trainer.params)
    if command.restore_optimizer:
      metadata_path = os.path.join(command.state_path, "metadata.json")
      with open(metadata_path) as f:
        trainer.restore_optimizer(command.state_path, json.load(f), map_location=self.device)
    return {"model_id": trainer.model_id, "base_model": base_model}

  def remove(self, trainer: Trainer) -> None:
    trainer.close()
    self.trainer = None

  def target_device(self) -> Any:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return "auto" if num_gpus > 1 else self.device

  def build_model(self, base_model_name: str) -> tuple[PreTrainedModel, Any]:
    target_device = self.target_device()
    print(f"Loading full fine-tuning model {base_model_name} (target device map: {target_device})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map=target_device)
    print("Successfully loaded full fine-tuning model.")
    return model, tokenizer

  def build_from_state(self, state_path: str) -> tuple[PreTrainedModel, Any, str]:
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")
    with open(metadata_path) as f:
      base_model = json.load(f).get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")
    target_device = self.target_device()
    tokenizer = AutoTokenizer.from_pretrained(state_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(state_path, dtype=dtype, device_map=target_device)
    return model, tokenizer, base_model

  def save_needs_gpu(self) -> bool:
    return self.trainer is not None and not self.trainer.cpu_offload

  def sleep(self) -> None:
    if self.trainer is not None:
      self.trainer.sleep()

  def wake_up(self) -> None:
    if self.trainer is not None:
      self.trainer.wake_up()
