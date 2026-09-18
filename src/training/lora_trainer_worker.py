# LoRA host: one frozen base model serving many adapters, each its own Trainer.

import json
import os
import time
import traceback
from datetime import datetime
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModelForCausalLM, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training.commands import CreateModel, CreateModelFromState, SaveWeightsForSampler
from training.trainer_worker import Trainer, TrainingWorker, enable_gradient_checkpointing, tmp_dir
from training.types import LoraConfig, SamplerWeights

__all__ = ["LoraConfig", "LoraTrainer", "LoraTrainingWorker", "active_adapter_parameters"]


def active_adapter_parameters(model: PeftModelForCausalLM, adapter_id: str) -> list[torch.nn.Parameter]:
  model.set_adapter(adapter_id)
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError(f"No trainable parameters found for adapter '{adapter_id}'")
  return params


class LoraTrainer(Trainer):
  """One adapter on a shared PEFT model.

  Several adapters share one PeftModelForCausalLM but each LoraTrainer owns a
  distinct parameter list, optimizer and metrics. Every operation activates
  this adapter first, so only its parameters take a gradient and the base and
  the other adapters are untouched. The processor runs one command at a time,
  so an activation is never switched out from under a live graph.
  """

  def __init__(self, model_id: str, peft_model: PeftModelForCausalLM, params: list[torch.nn.Parameter], tokenizer: Any, base_model_name: str):
    super().__init__(model_id, peft_model, params, tokenizer=tokenizer, base_model_name=base_model_name)

  def activate(self) -> PeftModelForCausalLM:
    self.model.set_adapter(self.model_id)
    return self.model

  def forward_backward(self, data: list, loss_fn: str, loss_config: dict | None = None, forward_only: bool = False) -> dict[str, Any]:
    self.activate()
    return super().forward_backward(data, loss_fn, loss_config, forward_only)

  def optim_step(self, adam_params: dict[str, Any]) -> dict[str, Any]:
    self.activate()
    metrics = super().optim_step(adam_params)
    # The LoRA samplers hot-load the adapter directory, so publish it every step.
    self.save_adapter()
    return metrics

  def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    self.activate()
    return super().generate(*args, **kwargs)

  # -- checkpoints ------------------------------------------------------------

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    self.activate()
    os.makedirs(state_path, exist_ok=True)
    self.model.save_pretrained(state_path, selected_adapters=[self.model_id])

    has_optimizer = include_optimizer and self.optimizer is not None
    if has_optimizer:
      torch.save(self.optimizer.state_dict(), os.path.join(state_path, "optimizer.pt"))

    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(self.checkpoint_metadata(kind=kind, has_optimizer=has_optimizer), f)
    print(f"Saved state for '{self.model_id}' to {state_path}")
    return {"path": state_path}

  def adopt_reloaded(self, params: list[torch.nn.Parameter], state_path: str, metadata: dict[str, Any], restore_optimizer: bool) -> None:
    """Bind the adapter parameters a reload produced and, if asked, the
    optimizer saved beside them. The old optimizer belonged to the old
    parameters, so rebind drops it before the restore rebuilds it."""
    self.rebind(params)
    if restore_optimizer:
      self.restore_optimizer(state_path, metadata, map_location=self.input_device)

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    return {"path": self.save_adapter(alias).path}

  def publish_sampler_weights(self, command: SaveWeightsForSampler) -> SamplerWeights:
    return self.save_adapter(command.alias)

  def save_adapter(self, alias: str | None = None) -> SamplerWeights:
    """Write the adapter where the LoRA sampler workers load it: peft/<id>/<id>/."""
    adapter_id = self.model_id
    adapter_root = os.path.join(tmp_dir(), "peft", adapter_id)
    adapter_dir = os.path.join(adapter_root, adapter_id)
    try:
      os.makedirs(adapter_root, exist_ok=True)
      self.model.set_adapter(adapter_id)
      self.model.save_pretrained(adapter_root, selected_adapters=[adapter_id])

      metadata = {
        "model_id": adapter_id,
        "created_at": datetime.now().isoformat(),
        "timestamp": time.time(),
        **({"alias": alias} if alias is not None else {}),
      }
      with open(os.path.join(adapter_root, "metadata.json"), "w") as f:
        json.dump(metadata, f)
      print(f"Auto-saved adapter '{adapter_id}' to {adapter_root}")
    except Exception as e:
      print(f"[ERROR] Failed to auto-save weights for {adapter_id}: {e}")
      traceback.print_exc()
    return SamplerWeights(kind="adapter", path=adapter_dir)


class LoraTrainingWorker(TrainingWorker):
  """Hosts N LoRA adapters on one frozen base model. The base model, the PEFT
  wrapper and the tokenizer live for the worker's lifetime; create() and
  restore() build adapters on top of them and return a LoraTrainer each."""

  single_model = False
  full_parameter = False

  def __init__(self):
    self.base_model: PreTrainedModel | None = None
    self.peft_model: PeftModelForCausalLM | None = None
    self.tokenizer: Any = None
    self.base_model_name: str | None = None
    self.device = torch.device("cpu")
    self.lora_target_modules: dict[tuple[bool, bool, bool], list[str]] = {}

  def initialize(self, base_model: str | None = None) -> None:
    """Select this rank's device and optionally preload the shared base model
    so the first create() does not pay the cold-start."""
    from training.distributed import default_device

    self.device = default_device()
    if base_model:
      self.load_base_model(base_model)

  # -- worker API -------------------------------------------------------------

  def create(self, command: CreateModel) -> LoraTrainer:
    self.load_base_model(command.base_model)
    params = self.create_adapter(command.model_id, command.lora_config)
    return self.build_trainer(command.model_id, params)

  def restore(self, command: CreateModelFromState) -> LoraTrainer:
    params, base_model = self.load_adapter_state(command.model_id, command.state_path)
    trainer = self.build_trainer(command.model_id, params)
    if command.restore_optimizer:
      metadata_path = os.path.join(command.state_path, "metadata.json")
      with open(metadata_path) as f:
        trainer.restore_optimizer(command.state_path, json.load(f), map_location=self.device)
    return trainer

  def build_trainer(self, model_id: str, params: list[torch.nn.Parameter]) -> LoraTrainer:
    assert self.peft_model is not None
    trainer = LoraTrainer(model_id, self.peft_model, params, self.tokenizer, self.base_model_name)
    trainer.input_device = self.device
    return trainer

  def remove(self, trainer: Trainer) -> None:
    """Drop one adapter. The shared base and PEFT wrapper stay for the others."""
    if self.peft_model is not None and trainer.model_id in self.peft_model.peft_config:
      self.peft_model.delete_adapter(trainer.model_id)
    trainer.close()

  def reload(self, trainer: LoraTrainer, command) -> dict[str, Any]:
    """LoadWeights on an existing adapter: reload its weights in place and
    rebind the trainer to the new parameter objects."""
    params, base_model = self.load_adapter_state(trainer.model_id, command.state_path)
    metadata_path = os.path.join(command.state_path, "metadata.json")
    with open(metadata_path) as f:
      metadata = json.load(f)
    trainer.adopt_reloaded(params, command.state_path, metadata, command.restore_optimizer)
    return {"model_id": trainer.model_id, "base_model": base_model}

  # -- base model and adapters ------------------------------------------------

  def load_base_model(self, base_model_name: str) -> None:
    """Eagerly load the massive base model tensors into VRAM."""
    if self.base_model is not None and self.base_model_name == base_model_name:
      print(f"Base model {base_model_name} already loaded.")
      return
    if self.base_model is not None and self.base_model_name != base_model_name:
      raise ValueError(f"This LoRA worker hosts adapters on '{self.base_model_name}'; it cannot also serve '{base_model_name}'.")

    print(f"Loading base model {base_model_name} to {self.device}...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype, device_map=self.device)
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

  def create_adapter(self, adapter_id: str, config: LoraConfig) -> list[torch.nn.Parameter]:
    """Create a new LoRA adapter on the shared base model and capture its
    trainable parameters. On any failure the half-created adapter is removed so
    the shared PEFT model is left exactly as it was."""
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
    fresh_peft_model = self.peft_model is None
    try:
      if fresh_peft_model:
        self.peft_model = get_peft_model(self.base_model, peft_config, adapter_name=adapter_id)
      else:
        self.peft_model.add_adapter(adapter_id, peft_config)
      params = self.register_adapter(adapter_id)
    except Exception:
      # Undo a partial injection so the next request finds a clean model.
      if fresh_peft_model:
        self.peft_model = None
      elif self.peft_model is not None and adapter_id in self.peft_model.peft_config:
        self.peft_model.delete_adapter(adapter_id)
      raise

    print(f"LoRA adapter '{adapter_id}' created and set to active.")
    return params

  def register_adapter(self, adapter_id: str) -> list[torch.nn.Parameter]:
    assert self.peft_model is not None
    self.peft_model.set_adapter(adapter_id)
    params = active_adapter_parameters(self.peft_model, adapter_id)
    enable_gradient_checkpointing(self.peft_model)
    self.peft_model.train()
    return params

  def load_adapter_state(self, adapter_id: str, state_path: str) -> tuple[list[torch.nn.Parameter], str]:
    """Create (or replace) an adapter from a saved state directory. Returns its
    trainable parameters and the base model it was trained on."""
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")
    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")

    src_adapter_id = metadata.get("model_id")
    adapter_dir = (
      os.path.join(state_path, src_adapter_id) if src_adapter_id and os.path.exists(os.path.join(state_path, src_adapter_id)) else state_path
    )

    self.load_base_model(base_model)
    assert self.base_model is not None
    if self.peft_model is None:
      self.peft_model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_dir, adapter_name=adapter_id, is_trainable=True)
    else:
      if adapter_id in self.peft_model.peft_config:
        self.peft_model.delete_adapter(adapter_id)
      self.peft_model.load_adapter(adapter_dir, adapter_name=adapter_id, is_trainable=True)

    params = self.register_adapter(adapter_id)
    print(f"Loaded state for '{adapter_id}' from {state_path}")
    return params, base_model
