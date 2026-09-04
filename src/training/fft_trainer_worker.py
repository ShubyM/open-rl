# Full fine-tuning trainer worker lifecycle.

import gc
import itertools
import json
import logging
import math
import os
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedModel

from training import paths
from training.distributed import barrier, fsdp_group, is_distributed, is_primary
from training.model_loading import load_text_causal_lm
from training.trainer_worker import (
  BaseTrainerWorker,
  Datum,
  activation_offload_context,
  attention_forward_kwargs,
  project_target_logprobs,
)

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


def configure_fft_attention(attention_backend: str) -> None:
  if attention_backend == "sdpa" and torch.cuda.is_available() and os.getenv("OPEN_RL_SDPA_NO_MATH", "1") == "1":
    # Gradient checkpointing recomputes attention during backward, outside the
    # original forward context. Disable the quadratic math backend process-wide
    # so recomputation cannot silently materialize [batch, heads, seq, seq].
    torch.backends.cuda.enable_math_sdp(False)


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True
  weight_sync_strategy: str | None = None


class FSDPTargetLogprobModel(torch.nn.Module):
  """FSDP root whose forward avoids materializing [batch, sequence, vocabulary]."""

  def __init__(self, causal_lm: PreTrainedModel):
    super().__init__()
    self.causal_lm = causal_lm

  def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    model = self.causal_lm
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
    if backbone is None:
      raise RuntimeError(f"FSDP fused logprobs cannot resolve the backbone for {type(model).__name__}")

    backbone_mask = None if bool(attention_mask.all()) else attention_mask
    attention_context = nullcontext()
    if input_ids.is_cuda and os.getenv("OPEN_RL_SDPA_NO_MATH", "1") == "1":
      from torch.nn.attention import SDPBackend, sdpa_kernel

      attention_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    # activation_offload_context streams saved activations to pinned host RAM
    # (OPEN_RL_ACTIVATION_CPU_OFFLOAD) - activations are per-rank and unsharded
    # under FSDP, so long-context runs need it exactly like single-GPU ones.
    with activation_offload_context(input_ids), attention_context:
      # attention_forward_kwargs supplies the low-resource FlexAttention tiles
      # for wide heads (Gemma's 512-dim global heads): the default tiles need
      # 256KB of shared memory per block and sm_90 tops out at 227KB.
      outputs = backbone(
        input_ids=input_ids,
        attention_mask=backbone_mask,
        use_cache=False,
        return_dict=True,
        **attention_forward_kwargs(model.config),
      )

    seq_len = target_token_ids.shape[1]
    hidden = outputs.last_hidden_state[:, :seq_len, :]
    return project_target_logprobs(model, hidden, target_token_ids)


def trainable_model_parameters(model: PreTrainedModel) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


from server.model_metadata import WeightSyncConfig


class FFTTrainingWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.model: torch.nn.Module | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.fsdp_enabled = False
    self.cpu_offload: bool = True
    self.weight_sync_cfg: WeightSyncConfig = WeightSyncConfig.from_env()
    self._is_offloaded: bool = False
    self._latest_delta_tensors: dict[str, torch.Tensor] = {}
    self._latest_total_changed: int = 0
    self._latest_total_elements: int = 0
    self._param_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._grad_shadow: dict[torch.nn.Parameter, tuple[torch.device, torch.Tensor]] = {}
    self._opt_shadow: dict[tuple[torch.nn.Parameter, str], tuple[torch.device, torch.Tensor]] = {}
    self._prev_weights_shadow: dict[str, torch.Tensor] = {}
    self.model_layer_names: list[str] = []
    self.total_model_elements: int = 0

  def set_weight_sync_strategy(self, strategy: str) -> None:
    if strategy not in ("full", "delta"):
      raise ValueError(f"Invalid weight_sync_strategy '{strategy}'. Must be 'full' or 'delta'.")
    self.weight_sync_cfg.strategy = strategy

  def _get_prev_cpu_weight(self, name: str, param: torch.nn.Parameter) -> torch.Tensor | None:
    if param in self._param_shadow:
      return self._param_shadow[param][1]
    return None

  def _update_prev_cpu_weight(self, name: str, param: torch.nn.Parameter, indices: torch.Tensor, values: torch.Tensor) -> None:
    if param in self._param_shadow:
      self._param_shadow[param][1].view(-1)[indices.to(torch.int64).cpu()] = values

  def load_base_model(self, base_model_name: str) -> None:
    """Load one full model for one fine-tuning job process."""
    if self.model is not None and self.base_model_name == base_model_name:
      print(f"Full fine-tuning model {base_model_name} already loaded.")
      return

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    mode = f"FSDP rank {os.getenv('RANK', '0')}/{os.getenv('WORLD_SIZE', '1')}" if is_distributed() else str(self.device)
    print(f"Loading full fine-tuning model {base_model_name} ({mode}, visible GPUs: {num_gpus})...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    load_kwargs: dict[str, Any] = {"dtype": dtype}
    if attention_backend := os.getenv("OPEN_RL_ATTN_IMPLEMENTATION"):
      load_kwargs["attn_implementation"] = attention_backend
    if not is_distributed() and num_gpus > 1:
      load_kwargs["device_map"] = "auto"
    causal_lm = load_text_causal_lm(base_model_name, **load_kwargs)
    configure_fft_attention(causal_lm.config.get_text_config()._attn_implementation)
    if ENABLE_GRADIENT_CHECKPOINTING:
      causal_lm.gradient_checkpointing_enable()
      causal_lm.enable_input_require_grads()

    if is_distributed():
      from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
      from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
      from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

      decoder_classes = {type(module) for module in causal_lm.modules() if type(module).__name__.endswith("DecoderLayer")}
      if not decoder_classes:
        raise RuntimeError(f"Could not identify decoder layers for FSDP wrapping in {type(causal_lm).__name__}")
      wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=decoder_classes)
      mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
      self.model = FSDP(
        FSDPTargetLogprobModel(causal_lm),
        process_group=fsdp_group(),
        auto_wrap_policy=wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=self.device,
        use_orig_params=True,
        limit_all_gathers=True,
      )
      self.fsdp_enabled = True
      self.cpu_offload = False
      print(f"FSDP FULL_SHARD enabled across {os.getenv('WORLD_SIZE')} ranks; decoder classes: {[c.__name__ for c in decoder_classes]}")
    elif num_gpus > 1:
      self.model = causal_lm  # device_map="auto" already placed the layers
    else:
      self.model = causal_lm.to(self.device)
    print(f"Full fine-tuning attention backend: {causal_lm.config.get_text_config()._attn_implementation}")
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
    self.model_layer_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
    self.total_model_elements = sum(p.numel() for p in self.model.parameters())
    if self.weight_sync_cfg.strategy == "delta":
      if self.fsdp_enabled:
        raise RuntimeError("Delta weight sync is unsupported under FSDP; use weight_sync_strategy=full")
      for param in self.model.parameters():
        if param.requires_grad and param not in self._param_shadow:
          cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          cpu_buf.copy_(param.data, non_blocking=True)
          self._param_shadow[param] = (param.device, cpu_buf)

    self.model.train()

  def _prepare_for_save(self) -> bool:
    was_offloaded = self._is_offloaded
    if was_offloaded and self.model is not None:
      for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
        if tensor in self._param_shadow:
          tensor.data = self._param_shadow[tensor][1]
    return was_offloaded

  def _cleanup_after_save(self, was_offloaded: bool) -> None:
    if was_offloaded and self.model is not None:
      for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
        if tensor in self._param_shadow:
          tensor.data = torch.empty(0, dtype=tensor.dtype, device=self._param_shadow[tensor][0])

  def causal_model(self) -> PreTrainedModel:
    assert self.model is not None
    if self.fsdp_enabled:
      return self.model.module.causal_lm
    return self.model

  def full_model_state_dict(self) -> dict[str, torch.Tensor]:
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    assert self.fsdp_enabled
    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, config):
      wrapped_state = self.model.state_dict()
    if not is_primary():
      return {}
    prefix = "causal_lm."
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in wrapped_state.items()}

  def optimizer_state_dict(self) -> dict[str, Any]:
    assert self.optimizer is not None
    if not self.fsdp_enabled:
      return self.optimizer.state_dict()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    assert self.model is not None
    return FSDP.full_optim_state_dict(self.model, self.optimizer, rank0_only=True)

  def write_pretrained(self, save_path: str) -> None:
    state_dict = self.full_model_state_dict() if self.fsdp_enabled else None
    if is_primary():
      self.causal_model().save_pretrained(save_path, state_dict=state_dict)
      if self.tokenizer is not None:
        self.tokenizer.save_pretrained(save_path)
    barrier()

  def save_checkpoint(self, path: str, metadata: dict[str, Any], include_optimizer: bool = False) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    # Stage into a sibling directory and swap it in with atomic renames: a save
    # killed mid-write (OOM) must never leave a half-overwritten checkpoint
    # that vLLM or a resumed trainer can load as a mix of old and new shards.
    staging_path = f"{path}.staging-{os.getpid()}"
    previous_path = f"{path}.previous-{os.getpid()}"
    if is_primary():
      shutil.rmtree(staging_path, ignore_errors=True)
      os.makedirs(staging_path, exist_ok=True)
    barrier()
    was_offloaded = self._prepare_for_save()
    try:
      self.write_pretrained(staging_path)
      if include_optimizer and self.optimizer is not None:
        optimizer_state = self.optimizer_state_dict()
        if is_primary():
          torch.save(optimizer_state, os.path.join(staging_path, "optimizer.pt"))
    finally:
      self._cleanup_after_save(was_offloaded)
    if is_primary():
      with open(os.path.join(staging_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
      shutil.rmtree(previous_path, ignore_errors=True)
      if os.path.exists(path):
        os.rename(path, previous_path)
      os.rename(staging_path, path)
      shutil.rmtree(previous_path, ignore_errors=True)
    barrier()
    print(f"Saved full fine-tuning state to {path}")
    return {"path": path}

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(
        "Cannot save model while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
        "GPU time-slicer lock is not held during save operations."
      )

    tmp_dir = paths.tmp_dir()
    name = alias or "fft-model"
    save_path = name if os.path.isabs(name) else os.path.join(tmp_dir, "fft", name)
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": "weights",
      "model_id": alias,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(save_path, metadata)

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(
        "Cannot save state while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
        "GPU time-slicer lock is not held during save operations."
      )

    if self.weight_sync_cfg.strategy == "delta" and not include_optimizer:
      return self.save_state_delta(model_id=model_id, state_path=state_path, kind=kind)

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and self.optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(state_path, metadata, include_optimizer)

  def save_state_delta(
    self,
    model_id: str,
    state_path: str,
    kind: str = "sampler",
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.cpu_offload and not self._is_offloaded:
      raise RuntimeError(
        "Cannot save state delta while worker is not offloaded (self._is_offloaded is False) when cpu_offload=True. "
        "GPU time-slicer lock is not held during save operations."
      )

    os.makedirs(state_path, exist_ok=True)
    total_changed = 0
    total_elements = 0
    layer_names_list: list[str] = []
    indices_list: list[torch.Tensor] = []
    values_list: list[torch.Tensor] = []
    layer_lengths_list: list[int] = []

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

    if indices_list:
      indices_flat = torch.cat(indices_list).to(torch.int32).contiguous()
      values_flat = torch.cat(values_list).contiguous()
    else:
      fallback_dtype = next(self.model.parameters()).dtype if self.model else torch.float32
      indices_flat = torch.empty(0, dtype=torch.int32, device="cpu")
      values_flat = torch.empty(0, dtype=fallback_dtype, device="cpu")

    layer_lengths_tensor = torch.tensor(layer_lengths_list, dtype=torch.int64, device="cpu")
    packed_delta = {
      "delta.indices_flat": indices_flat,
      "delta.values_flat": values_flat,
      "delta.layer_lengths": layer_lengths_tensor,
    }

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

    self.load_base_model(state_path)
    self.base_model_name = base_model
    self.prepare_model_for_training()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, "optimizer.pt")
      if os.path.exists(optimizer_path):
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=1e-4, foreach=False)
        full_optimizer_state = torch.load(optimizer_path, map_location="cpu") if is_primary() else None
        if self.fsdp_enabled:
          from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

          sharded_state = FSDP.scatter_full_optim_state_dict(
            full_optimizer_state,
            self.model,
            optim=self.optimizer,
          )
          self.optimizer.load_state_dict(sharded_state)
        else:
          self.optimizer.load_state_dict(full_optimizer_state)
        print(f"Restored optimizer state from {optimizer_path}")

    print(f"Loaded full fine-tuning state from {state_path}")
    return {"model_id": model_id, "base_model": base_model}

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    res = super().forward_backward(self.model, data, loss_fn, loss_config, forward_only=forward_only)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def _remap_hf_to_vllm_fused(
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

  def compute_target_logprobs(
    self,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    if self.fsdp_enabled:
      return model(input_ids, attention_mask, target_token_ids)
    return super().compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if not self.trainable_params:
      self.trainable_params = trainable_model_parameters(self.model)

    # ZeRO-Offload-style step: params and grads move to the host so the AdamW
    # moments never occupy GPU memory. Frees 2x model size of VRAM for
    # activations at the cost of PCIe traffic (~30s/step for a 9B model).
    cpu_step = os.getenv("OPEN_RL_OPTIM_CPU_STEP", "0") == "1" and not self.fsdp_enabled
    if cpu_step:
      self.model.to("cpu")
      self.move_optimizer_state("cpu")
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        # Per-parameter updates cap the optimizer's transient VRAM/RAM at one
        # tensor instead of a fused batch; the measured seq-len ceilings and
        # CPU-step timings assume this. Do not "fix" to foreach=True.
        foreach=False,
      )

    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    t_clip_start = time.perf_counter()
    total_norm = (
      self.model.clip_grad_norm_(max_grad_norm)
      if self.fsdp_enabled
      else torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm, foreach=False)
    )
    t_clip_end = time.perf_counter()
    clip_time = t_clip_end - t_clip_start

    t_step_start = time.perf_counter()
    self.optimizer.step()
    self.optimizer.zero_grad()
    t_step_end = time.perf_counter()
    step_time = t_step_end - t_step_start

    if cpu_step:
      self.model.to(self.device)

    delta_compute_time = 0.0
    if self.weight_sync_cfg.strategy == "delta" and self.model is not None and hasattr(self.model, "named_parameters"):
      t_delta_start = time.perf_counter()
      self._latest_delta_tensors.clear()
      self._latest_total_changed = 0
      self._latest_total_elements = self.total_model_elements

      layer_names_list: list[str] = []
      indices_list: list[torch.Tensor] = []
      values_list: list[torch.Tensor] = []
      layer_lengths_list: list[int] = []

      for name, param in self.model.named_parameters():
        if not param.requires_grad:
          continue
        prev_tensor = self._get_prev_cpu_weight(name, param)
        if prev_tensor is None:
          cpu_buf = torch.empty(param.shape, dtype=param.dtype, device="cpu", pin_memory=torch.cuda.is_available())
          cpu_buf.copy_(param.data, non_blocking=True)
          self._param_shadow[param] = (param.device, cpu_buf)
          prev_tensor = cpu_buf

        prev_gpu = prev_tensor.to(param.device, non_blocking=True)

        diff_mask = param.data.view(-1).ne(prev_gpu.view(-1))
        indices = diff_mask.nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
          idx_cpu = indices.to(torch.int32).contiguous().cpu()
          val_cpu = param.data.view(-1)[diff_mask].contiguous().cpu()
          layer_names_list.append(name)
          indices_list.append(idx_cpu)
          values_list.append(val_cpu)
          layer_lengths_list.append(int(idx_cpu.numel()))
          self._latest_total_changed += int(idx_cpu.numel())
          self._update_prev_cpu_weight(name, param, idx_cpu, val_cpu)
        del prev_gpu, diff_mask, indices

      if self.weight_sync_cfg.delta_format == "vllm_fused":
        layer_names_list, indices_list = self._remap_hf_to_vllm_fused(layer_names_list, indices_list)

      self._latest_delta_tensors = {
        "names": layer_names_list,
        "indices_list": indices_list,
        "values_list": values_list,
        "layer_lengths_list": layer_lengths_list,
      }

      t_delta_end = time.perf_counter()
      delta_compute_time = t_delta_end - t_delta_start
      logger.info(
        f"[OPTIM_STEP] model_id={model_id} | delta_compute_time={delta_compute_time:.4f}s | "
        f"changed={self._latest_total_changed}/{self._latest_total_elements} "
        f"({100.0 * self._latest_total_changed / max(1, self._latest_total_elements):.2f}%) across {len(layer_names_list)} layers"
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
    if self.fsdp_enabled:
      raise RuntimeError("Sampling from an FSDP trainer is unsupported; use the vLLM sampler worker")
    return super().generate(self.model, prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)

  def sleep(self) -> None:
    """Offload GPU tensors to pinned host CPU memory and empty CUDA allocator cache."""
    if self.fsdp_enabled or not self.cpu_offload or self.model is None or self._is_offloaded or not torch.cuda.is_available():
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
    if self.fsdp_enabled or not self.cpu_offload or self.model is None or not self._is_offloaded or not torch.cuda.is_available():
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

  def move_optimizer_state(self, device: str | torch.device) -> None:
    if self.optimizer is None:
      return
    target = torch.device(device)
    for state in self.optimizer.state.values():
      for key, value in state.items():
        if isinstance(value, torch.Tensor) and (key != "step" or target.type == "cpu"):
          state[key] = value.to(target)
