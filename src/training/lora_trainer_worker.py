# LoRA trainer worker lifecycle and adapter management.

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModelForCausalLM, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training import hf_operations
from training.types import Datum, LoraConfig

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


def active_adapter_parameters(model: PeftModelForCausalLM, adapter_id: str) -> list[torch.nn.Parameter]:
  model.set_adapter(adapter_id)
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError(f"No trainable parameters found for adapter '{adapter_id}'")
  return params


@dataclass
class AdapterState:
  params: list[torch.nn.Parameter]
  optimizer: torch.optim.Optimizer | None = None


class LoraTrainingWorker:
  def __init__(
    self,
    *,
    base_model: PreTrainedModel | None = None,
    tokenizer: Any = None,
    device: torch.device | str | None = None,
    base_model_name: str | None = None,
    token_budget: int | None = None,
  ):
    if device is None:
      device = next(base_model.parameters()).device if base_model is not None else hf_operations.default_device()
    self.device = torch.device(device)
    if base_model is not None:
      self.dtype = next(base_model.parameters()).dtype
    else:
      self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    self.tokenizer = tokenizer
    self.token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0")) if token_budget is None else token_budget
    self.base_model = base_model
    self.peft_model: PeftModelForCausalLM | None = None
    self.base_model_name = base_model_name
    self.adapters: dict[str, AdapterState] = {}
    self.lora_target_modules: dict[tuple[bool, bool, bool], list[str]] = {}

  def load_base_model(self, base_model_name: str) -> None:
    """Eagerly load the massive base model tensors into VRAM."""
    if self.base_model is not None and self.base_model_name == base_model_name:
      print(f"Base model {base_model_name} already loaded.")
      return

    print(f"Loading base model {base_model_name} to {self.device}...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=self.dtype, device_map=self.device)
    print("Successfully loaded.")

  def target_lora_modules(self, config: LoraConfig) -> list[str]:
    assert self.base_model is not None

    cache_key = (config.train_attn, config.train_mlp, config.train_unembed)
    if cache_key in self.lora_target_modules:
      return self.lora_target_modules[cache_key]

    target_suffixes: list[str] = []
    if config.train_attn:
      target_suffixes.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if config.train_mlp:
      # TODO: Revisit MLP targets for packed/MoE module names across supported backends.
      target_suffixes.extend(["gate_proj", "up_proj", "down_proj"])
    if config.train_unembed:
      # getattr because not every config defines tie_word_embeddings (transformers
      # v5 dropped it from the PretrainedConfig base class). When absent,
      # transformers performs no tying, so False matches the loaded model.
      if getattr(self.base_model.config, "tie_word_embeddings", False):
        # An adapter on a tied lm_head shares the embedding tensor: PEFT warns,
        # merging would corrupt embed_tokens, and vLLM refuses lm_head adapter
        # weights for tied models. Keep every produced adapter vLLM-loadable.
        print(
          f"[LoRA] Ignoring train_unembed=True: {self.base_model_name} ties lm_head to embed_tokens, "
          "and the resulting adapter could not be loaded by vLLM."
        )
      else:
        target_suffixes.append("lm_head")

    if not target_suffixes:
      raise ValueError("No trainable LoRA targets remain (train_unembed is ignored on tied-embeddings models; enable train_attn or train_mlp)")

    # Once an adapter exists, PEFT has wrapped the targeted layers and the
    # Linear sits under base_layer; a later config must still find it.
    target_names = set(target_suffixes)
    target_modules = [
      name
      for name, module in self.base_model.named_modules()
      if name.rsplit(".", 1)[-1] in target_names and isinstance(getattr(module, "base_layer", module), torch.nn.Linear)
    ]
    if not target_modules:
      raise ValueError(f"No supported LoRA target modules found for suffixes: {target_suffixes}")
    self.lora_target_modules[cache_key] = target_modules
    return target_modules

  def create_adapter(self, adapter_id: str, config: LoraConfig) -> None:
    """Create a new LoRA adapter on top of the loaded base model."""
    assert self.base_model is not None, "Base model is not loaded. Call load_base_model first."

    if not any([config.train_attn, config.train_mlp, config.train_unembed]):
      raise ValueError("At least one LoRA training target must be enabled.")

    print(f"Creating LoRA adapter '{adapter_id}'...")

    peft_config = PeftLoraConfig(
      task_type="CAUSAL_LM",
      r=config.rank,
      lora_alpha=config.lora_alpha,
      lora_dropout=config.lora_dropout,
      bias="none",
      target_modules=self.target_lora_modules(config),
      modules_to_save=None,
    )

    if config.seed is not None:
      torch.manual_seed(config.seed)
    if self.peft_model is None:
      self.peft_model = get_peft_model(self.base_model, peft_config, adapter_name=adapter_id)
    else:
      self.peft_model.add_adapter(adapter_id, peft_config)

    self.adapters[adapter_id] = AdapterState(active_adapter_parameters(self.peft_model, adapter_id))

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on PEFT model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.peft_model.train()
    print(f"LoRA adapter '{adapter_id}' created and set to active.")

  def create_model(self, base_model_name: str, model_id: str, config: LoraConfig) -> None:
    """Load the shared base model if needed, then create a trainable LoRA adapter."""
    self.load_base_model(base_model_name)
    self.create_adapter(model_id, config)

  def save_state(
    self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state", *, alias: str | None = None
  ) -> dict[str, Any]:
    """Save adapter weights (and optionally optimizer state) to a specific path."""
    assert self.peft_model is not None, "Model must be loaded first."

    adapter = self.adapter_for(model_id)
    os.makedirs(state_path, exist_ok=True)
    self.peft_model.save_pretrained(state_path, selected_adapters=[model_id])

    optimizer = adapter.optimizer
    if include_optimizer and optimizer is not None:
      torch.save(optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))

    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    if alias is not None:
      metadata["alias"] = alias
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved state for '{model_id}' to {state_path}")
    return {"path": state_path}

  def save_for_sampler(self, model_id: str, alias: str | None, ref: str | None) -> str | None:
    """Explicitly export to the shared adapter directory used by the sampler."""
    save_path = os.path.join(os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl"), "peft", model_id)
    self.save_state(model_id, save_path, kind="sampler", alias=alias)
    # LoRA references share this mutable directory; FFT version publication
    # and pruning must not treat sibling adapters as old versions.
    return None

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    """Create an adapter from a saved state directory.

    Expects the directory to contain a metadata.json describing base_model
    and (optionally) an adapter subdirectory with the saved LoRA weights.
    """
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")

    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")
    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if restore_optimizer and (not metadata.get("has_optimizer") or not os.path.isfile(optimizer_path)):
      raise ValueError(f"Checkpoint {state_path} has no optimizer state")

    src_adapter_id = metadata.get("model_id")
    adapter_dir = state_path
    if src_adapter_id and os.path.exists(os.path.join(state_path, src_adapter_id)):
      adapter_dir = os.path.join(state_path, src_adapter_id)

    self.load_base_model(base_model)
    assert self.base_model is not None

    if self.peft_model is None:
      self.peft_model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_dir, adapter_name=model_id, is_trainable=True)
    else:
      if model_id in self.peft_model.peft_config:
        self.peft_model.delete_adapter(model_id)
      self.peft_model.load_adapter(adapter_dir, adapter_name=model_id, is_trainable=True)

    params = active_adapter_parameters(self.peft_model, model_id)
    adapter = AdapterState(params)
    self.adapters[model_id] = adapter

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on PEFT model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.peft_model.train()

    if restore_optimizer:
      optimizer = hf_operations.build_optimizer(params, {})
      optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device, weights_only=True))
      adapter.optimizer = optimizer

    print(f"Loaded state for '{model_id}' from {state_path}")
    return {"model_id": model_id, "is_lora": True, "base_model": base_model}

  def adapter_for(self, model_id: str | None) -> AdapterState:
    if model_id is None:
      raise ValueError("model_id is required for a LoRA trainer")
    try:
      adapter = self.adapters[model_id]
    except KeyError:
      raise ValueError(f"No state for adapter '{model_id}'") from None
    self.peft_model.set_adapter(model_id)
    return adapter

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    self.adapter_for(model_id)
    return hf_operations.forward_backward(
      self.peft_model, data, loss_fn, loss_config, forward_only, tokenizer=self.tokenizer, device=self.device, token_budget=self.token_budget
    )

  def optim_step(self, adam_params: dict[str, Any], model_id: str) -> dict[str, Any]:
    adapter = self.adapter_for(model_id)
    if adapter.optimizer is None:
      adapter.optimizer = hf_operations.build_optimizer(adapter.params, adam_params)
    return {"metrics": hf_operations.optim_step(adapter.optimizer, adam_params)}

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    self.adapter_for(model_id)
    return hf_operations.generate(
      self.peft_model,
      prompt_tokens,
      max_tokens,
      num_samples,
      temperature,
      include_prompt_logprobs,
      tokenizer=self.tokenizer,
      device=self.device,
    )
