# Full fine-tuning trainer worker lifecycle.

import gc
import json
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from typing import Any

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.distributed import barrier, fsdp_group, is_distributed, is_primary
from training.trainer_worker import BaseTrainerWorker, Datum, project_target_logprobs

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True


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
    with attention_context:
      outputs = backbone(input_ids=input_ids, attention_mask=backbone_mask, use_cache=False, return_dict=True)

    seq_len = target_token_ids.shape[1]
    hidden = outputs.last_hidden_state[:, :seq_len, :]
    return project_target_logprobs(model, hidden, target_token_ids)


def trainable_model_parameters(model: PreTrainedModel) -> list[torch.nn.Parameter]:
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError("No trainable parameters found for full fine-tuning model")
  return params


def configure_fft_attention() -> str:
  attention_backend = os.getenv("OPEN_RL_ATTN_IMPLEMENTATION", "sdpa")
  if torch.cuda.is_available() and os.getenv("OPEN_RL_SDPA_NO_MATH", "1") == "1":
    # Gradient checkpointing recomputes attention during backward, outside the
    # original forward context. Disable the quadratic math backend process-wide
    # so recomputation cannot silently materialize [batch, heads, seq, seq].
    torch.backends.cuda.enable_math_sdp(False)
  return attention_backend


class FFTTrainingWorker(BaseTrainerWorker):
  def __init__(self):
    super().__init__()
    self.model: torch.nn.Module | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.fsdp_enabled = False
    self.cpu_offload: bool = True
    self._is_offloaded: bool = False

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

    attention_backend = configure_fft_attention()
    causal_lm = AutoModelForCausalLM.from_pretrained(
      base_model_name,
      dtype=dtype,
      attn_implementation=attention_backend,
    )
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
    else:
      self.model = causal_lm.to(self.device)
    print(f"Full fine-tuning attention backend: {causal_lm.config.get_text_config()._attn_implementation}")
    print("Successfully loaded full fine-tuning model.")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: FFTConfig | None = None) -> None:
    """Load the per-job model if needed, then prepare it for full fine-tuning."""
    if config is not None:
      self.cpu_offload = config.cpu_offload
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    self.prepare_model_for_training()

  def prepare_model_for_training(self) -> None:
    assert self.model is not None, "Model is not loaded. Call load_base_model first."

    for param in self.model.parameters():
      param.requires_grad_(True)
    self.trainable_params = trainable_model_parameters(self.model)

    self.model.train()

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
    if is_primary():
      os.makedirs(path, exist_ok=True)
    barrier()
    self.write_pretrained(path)
    if include_optimizer and self.optimizer is not None:
      optimizer_state = self.optimizer_state_dict()
      if is_primary():
        torch.save(optimizer_state, os.path.join(path, "optimizer.pt"))
    if is_primary():
      with open(os.path.join(path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    barrier()
    print(f"Saved full fine-tuning state to {path}")
    return {"path": path}

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    tmp_dir = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
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
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and self.optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(state_path, metadata, include_optimizer)

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

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    res = super().forward_backward(self.model, data, loss_fn, loss_config)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

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
        foreach=False,
      )

    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    total_norm = (
      self.model.clip_grad_norm_(max_grad_norm)
      if self.fsdp_enabled
      else torch.nn.utils.clip_grad_norm_(self.trainable_params, max_grad_norm, foreach=False)
    )

    self.optimizer.step()
    self.optimizer.zero_grad()

    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(total_norm.item()),
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
    stop: list[int] | None = None,
    top_p: float = 1.0,
    top_k: int = -1,
  ) -> dict[str, Any]:
    if self.fsdp_enabled:
      raise RuntimeError("Sampling from an FSDP trainer is unsupported; use the vLLM sampler worker")
    return super().generate(
      model=self.model,
      prompt_tokens=prompt_tokens,
      max_tokens=max_tokens,
      num_samples=num_samples,
      temperature=temperature,
      include_prompt_logprobs=include_prompt_logprobs,
      stop=stop,
      top_p=top_p,
      top_k=top_k,
    )

  def sleep(self) -> None:
    """Move the single-GPU trainer to CPU. FSDP state is handled by llm-d."""
    if self.fsdp_enabled or not self.cpu_offload or self.model is None or self._is_offloaded or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()
    self.model.to("cpu")
    self.move_optimizer_state("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    self._is_offloaded = True
    print(f"[FFT Worker] Moved weights and optimizer to CPU in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def wake_up(self, include_optimizer: bool = True) -> None:
    if self.fsdp_enabled or not self.cpu_offload or self.model is None or not self._is_offloaded or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()
    self.model.to(self.device)
    if include_optimizer:
      self.move_optimizer_state(self.device)
    self._is_offloaded = False
    print(f"[FFT Worker] Restored weights and optimizer to CUDA in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def move_optimizer_state(self, device: str | torch.device) -> None:
    if self.optimizer is None:
      return
    target = torch.device(device)
    for state in self.optimizer.state.values():
      for key, value in state.items():
        if isinstance(value, torch.Tensor) and (key != "step" or target.type == "cpu"):
          state[key] = value.to(target)
