# LoRA trainer worker lifecycle and adapter management.

import json
import math
import os
import time
from datetime import datetime
from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModelForCausalLM, get_peft_model
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from training import paths
from training.distributed import all_reduce_gradients, barrier, broadcast_parameters, is_primary
from training.trainer_worker import BaseTrainerWorker, Datum

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"


class LoraConfig(BaseModel):
  rank: int = 16
  seed: int | None = None
  lora_alpha: int = 16
  # Default 0: dropout during logprob computation makes trainer logprobs
  # stochastic while the sampler's are deterministic, biasing every
  # importance-sampling ratio (and it costs ~1000 dropout kernels per step).
  lora_dropout: float = 0.0
  train_attn: bool = True
  train_mlp: bool = True
  train_unembed: bool = False


def active_adapter_parameters(model: PeftModelForCausalLM, adapter_id: str) -> list[torch.nn.Parameter]:
  model.set_adapter(adapter_id)
  params = [param for param in model.parameters() if param.requires_grad]
  if not params:
    raise ValueError(f"No trainable parameters found for adapter '{adapter_id}'")
  return params


class LoraTrainingWorker(BaseTrainerWorker):
  # Gradients meet only at optim_step's all_reduce; empty shards skip work.
  backward_runs_collectives = False

  def __init__(self):
    super().__init__()
    self.base_model: PreTrainedModel | None = None
    self.peft_model: PeftModelForCausalLM | None = None
    self.base_model_name: str | None = None
    self.adapter_states: dict[str, dict[str, Any]] = {}
    self.lora_target_modules: dict[tuple[bool, bool, bool], list[str]] = {}
    self.linear_module_names: list[str] | None = None

  def load_base_model(self, base_model_name: str) -> None:
    """Eagerly load the massive base model tensors into VRAM."""
    if self.base_model is not None and self.base_model_name == base_model_name:
      print(f"Base model {base_model_name} already loaded.")
      return

    print(f"Loading base model {base_model_name} to {self.device}...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # Multimodal checkpoints (Qwen3.5/3.6, Gemma) load as a text-only causal LM
    # here; saved adapters then need their keys mapped back to the hub layout
    # so the vLLM sampler can match them (see _remap_adapter_to_hub_layout).
    base_config = AutoConfig.from_pretrained(base_model_name)
    self.base_is_multimodal = getattr(base_config, "text_config", None) is not None

    # Gemma's global heads are 512-wide with GQA (8 query vs 2 KV heads), which
    # every fused SDPA kernel rejects — flash caps head_dim at 256, and the
    # dense path wants matching num_heads. That leaves only the quadratic math
    # fallback, which OPEN_RL_SDPA_NO_MATH blocks, so the step dies with
    # "No available kernel" rather than silently OOMing. FlexAttention handles
    # both (attention_forward_kwargs supplies the low-resource tiles for the
    # wide heads). This mirrors model_loading.load_text_causal_lm, which the
    # full-parameter trainers use; the LoRA path never got the same wiring.
    text_model_types = {"gemma3n", "gemma3n_text", "gemma4", "gemma4_text"}
    attn_implementation = os.getenv("OPEN_RL_ATTN_IMPLEMENTATION") or (
      "flex_attention" if base_config.model_type in text_model_types else "sdpa"
    )
    print(f"LoRA base attention backend: {attn_implementation}")

    self.base_model = AutoModelForCausalLM.from_pretrained(
      base_model_name, dtype=dtype, device_map=self.device, attn_implementation=attn_implementation
    )
    # Captured before any peft wrapping: get_peft_model mutates the module
    # tree in place, so later isinstance(nn.Linear) scans would miss every
    # already-adapted projection and a second adapter would silently train
    # only the still-unwrapped modules.
    self.linear_module_names = [name for name, module in self.base_model.named_modules() if isinstance(module, torch.nn.Linear)]
    print("Successfully loaded.")

  def target_lora_modules(self, config: LoraConfig) -> list[str]:
    assert self.base_model is not None

    cache_key = (config.train_attn, config.train_mlp, config.train_unembed)
    if cache_key in self.lora_target_modules:
      return self.lora_target_modules[cache_key]

    target_suffixes: list[str] = []
    if config.train_attn:
      target_suffixes.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
      # Hybrid-attention checkpoints (Qwen3.5/3.6) implement most layers as
      # gated-deltanet linear attention whose projections use these names —
      # q/k/v/o alone would leave 3 out of 4 attention layers untrained. vLLM
      # serves adapters on them: with --enable-lora it builds split
      # in_proj_qkv/in_proj_z modules (instead of fused in_proj_qkvz) and
      # packs in_proj_b/in_proj_a via its packed_modules_mapping. The GDN
      # conv1d is excluded below by the nn.Linear filter.
      target_suffixes.extend(["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"])
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
      elif os.getenv("OPEN_RL_LORA_TRAIN_UNEMBED", "") == "1":
        target_suffixes.append("lm_head")
      else:
        # The tinker SDK's LoraConfig defaults train_unembed=True, so nearly
        # every client asks for this. vLLM only serves lm_head adapters for
        # model classes that declare embedding_modules (Qwen3.5/Gemma do not),
        # so a trained lm_head either fails the adapter load outright or gets
        # silently dropped — a trained-vs-served policy divergence. Keep every
        # produced adapter vLLM-loadable by default.
        print(
          "[LoRA] Ignoring train_unembed=True: vLLM cannot apply lm_head adapters for this model "
          "family, so the sampler would reject the adapter. Set OPEN_RL_LORA_TRAIN_UNEMBED=1 to "
          "train it anyway (torch sampler only)."
        )

    if not target_suffixes:
      raise ValueError(
        "No trainable LoRA targets remain (train_unembed is ignored unless OPEN_RL_LORA_TRAIN_UNEMBED=1; enable train_attn or train_mlp)"
      )

    target_names = set(target_suffixes)
    module_names = self.linear_module_names
    if module_names is None:
      module_names = [name for name, module in self.base_model.named_modules() if isinstance(module, torch.nn.Linear)]
    target_modules = [name for name in module_names if name.rsplit(".", 1)[-1] in target_names]
    if not target_modules:
      raise ValueError(f"No supported LoRA target modules found for suffixes: {target_suffixes}")
    self.lora_target_modules[cache_key] = target_modules
    return target_modules

  def create_adapter(self, adapter_id: str, config: LoraConfig) -> None:
    """Create a new LoRA adapter on top of the loaded base model."""
    assert self.base_model is not None, "Base model is not loaded. Call load_base_model first."

    if adapter_id in self.adapter_states:
      del self.adapter_states[adapter_id]

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

    if "lm_head" in peft_config.target_modules:
      self.output_head_is_adapted = True

    if config.seed is not None:
      torch.manual_seed(config.seed)
    if self.peft_model is None:
      self.peft_model = get_peft_model(self.base_model, peft_config, adapter_name=adapter_id)
    else:
      self.peft_model.add_adapter(adapter_id, peft_config)

    self.peft_model.set_adapter(adapter_id)
    self.adapter_states[adapter_id] = {"trainable_params": active_adapter_parameters(self.peft_model, adapter_id), "optimizer": None}
    # lora_A initializes randomly per process; data-parallel ranks must train
    # rank 0's copy or their adapters silently diverge from step one.
    broadcast_parameters(self.adapter_states[adapter_id]["trainable_params"])

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on PEFT model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.peft_model.train()
    print(f"LoRA adapter '{adapter_id}' created and set to active.")

    self.save_adapter(adapter_id)

  def create_model(self, base_model_name: str, model_id: str, config: LoraConfig) -> None:
    """Load the shared base model if needed, then create a trainable LoRA adapter."""
    self.load_base_model(base_model_name)
    self.create_adapter(model_id, config)

  def _remap_adapter_to_hub_layout(self, adapter_dir: str) -> None:
    """Rewrite adapter keys from the text-only layout to the hub (multimodal) layout.

    The trainer holds a text-only causal LM, so peft writes module names like
    `base_model.model.model.layers.N...`. vLLM resolves adapter modules through
    the multimodal model's hf_to_vllm_mapper, which expects the hub layout
    (`model.language_model.layers.N...`); text-layout names match nothing and
    vLLM silently applies NO adapter — sampling then serves the base model.
    """
    if not getattr(self, "base_is_multimodal", False):
      return
    weights_file = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_file):
      return
    from safetensors.torch import load_file, save_file

    tensors = load_file(weights_file)
    if any(".language_model." in key for key in tensors):
      return
    prefix = "base_model.model.model."
    remapped = {key.replace(prefix, prefix + "language_model.", 1) if key.startswith(prefix) else key: value for key, value in tensors.items()}
    save_file(remapped, weights_file, metadata={"format": "pt"})
    print(f"Remapped adapter keys to the multimodal hub layout in {weights_file}")

  SNAPSHOT_KEEP = 4

  def save_adapter(self, adapter_id: str, alias: str | None = None, session_label: str | None = None) -> None:
    """Write the adapter snapshot on rank 0; other ranks wait at the barrier
    so no rank reports success while the sampler-visible dir is mid-write.
    Adapters are replicated across data-parallel ranks, so one copy is the
    whole truth."""
    if self.peft_model is None:
      print(f"[LoRA] Cannot save adapter '{adapter_id}': no active PEFT model initialized.")
      return
    if is_primary():
      self.write_adapter(adapter_id, alias, session_label)
    barrier()

  def write_adapter(self, adapter_id: str, alias: str | None = None, session_label: str | None = None) -> None:
    """Save adapter weights to disk for the sampler.

    Each sampler snapshot gets its own immutable directory
    (peft/<adapter_id>/<session_label>), written via staging + atomic rename.
    The previous behavior overwrote one directory in place on every save while
    samplers could be reading it concurrently — vLLM then found the directory
    mid-write and failed with "<dir> doesn't contain tensors". Failures now
    propagate to the caller (the training future) instead of logging a
    success-shaped response over a broken adapter dir.
    """
    adapter_root = os.path.join(paths.snapshot_root(), adapter_id)
    final_dir = os.path.join(adapter_root, session_label or adapter_id)
    staging_root = os.path.join(adapter_root, f".staging-{os.getpid()}-{time.time_ns()}")
    os.makedirs(staging_root, exist_ok=True)

    try:
      self.peft_model.set_adapter(adapter_id)
      self.peft_model.save_pretrained(staging_root, selected_adapters=[adapter_id])
      staged_adapter = os.path.join(staging_root, adapter_id)
      self._remap_adapter_to_hub_layout(staged_adapter)

      if os.path.exists(final_dir):
        # Only the legacy label (adapter_id itself) can collide; snapshot
        # labels are unique per save. Move the old dir aside, never delete
        # under a reader.
        os.replace(final_dir, os.path.join(staging_root, "replaced"))
      os.rename(staged_adapter, final_dir)

      if alias and alias != os.path.basename(final_dir):
        # Alias-named refs (e.g. tinker://<id>/sampler_weights/final) resolve
        # to peft/<id>/<alias>, but the adapter itself lives in the snapshot
        # dir — without this link the returned ref points at a directory that
        # was never written and every sample against it fails.
        alias_path = os.path.join(adapter_root, alias)
        staged_link = os.path.join(staging_root, "alias-link")
        os.symlink(os.path.basename(final_dir), staged_link)
        if os.path.isdir(alias_path) and not os.path.islink(alias_path):
          os.replace(alias_path, os.path.join(staging_root, "replaced-alias"))
        os.replace(staged_link, alias_path)
    finally:
      import shutil

      shutil.rmtree(staging_root, ignore_errors=True)

    metadata = {"model_id": adapter_id, "created_at": datetime.now().isoformat(), "timestamp": time.time()}
    if alias is not None:
      metadata["alias"] = alias
    with open(os.path.join(adapter_root, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    self._prune_snapshots(adapter_root, keep=self.SNAPSHOT_KEEP, current=final_dir)
    print(f"Auto-saved adapter '{adapter_id}' to {final_dir}")

  def _prune_snapshots(self, adapter_root: str, keep: int, current: str) -> None:
    """Delete all but the newest `keep` snapshot dirs (in-flight rollouts may
    still sample from a recent previous snapshot). Snapshots an alias symlink
    (e.g. "final") points at are kept regardless of age."""
    try:
      entries = os.listdir(adapter_root)
      alias_targets = {
        os.path.realpath(os.path.join(adapter_root, name)) for name in entries if os.path.islink(os.path.join(adapter_root, name))
      }
      snapshots = sorted(
        (
          os.path.join(adapter_root, name)
          for name in entries
          if name.startswith("sampler-") and os.path.isdir(os.path.join(adapter_root, name)) and not os.path.islink(os.path.join(adapter_root, name))
        ),
        key=os.path.getmtime,
        reverse=True,
      )
      import shutil

      for stale in snapshots[keep:]:
        if stale != current and os.path.realpath(stale) not in alias_targets:
          shutil.rmtree(stale, ignore_errors=True)
    except OSError:
      pass

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    """Save adapter weights (and optionally optimizer state) to a specific path."""
    assert self.peft_model is not None, "Model must be loaded first."
    if not is_primary():
      barrier()
      return {"path": state_path}

    self.peft_model.set_adapter(model_id)
    os.makedirs(state_path, exist_ok=True)
    self.peft_model.save_pretrained(state_path, selected_adapters=[model_id])

    adapter_state = self.adapter_states.get(model_id)
    optimizer = adapter_state.get("optimizer") if adapter_state is not None else None
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
    with open(os.path.join(state_path, "metadata.json"), "w") as f:
      json.dump(metadata, f)

    print(f"Saved state for '{model_id}' to {state_path}")
    barrier()
    return {"path": state_path}

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    """Create an adapter from a saved state directory.

    Expects the directory to contain a metadata.json describing base_model
    and (optionally) an adapter subdirectory with the saved LoRA weights.
    """
    metadata_path = os.path.join(state_path, "metadata.json")
    if os.path.exists(metadata_path):
      with open(metadata_path) as f:
        metadata = json.load(f)
    else:
      # Sampler snapshots carry no metadata.json; PEFT's adapter_config.json
      # names the base model, which is all a weights-only load needs.
      adapter_config_path = os.path.join(state_path, "adapter_config.json")
      if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
          f"{state_path} has neither metadata.json nor adapter_config.json — not a checkpoint or adapter snapshot"
        )
      with open(adapter_config_path) as f:
        metadata = {"base_model": json.load(f).get("base_model_name_or_path")}

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"{state_path} does not name its base model (metadata.json/adapter_config.json)")

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
        if model_id in self.adapter_states:
          del self.adapter_states[model_id]
      self.peft_model.load_adapter(adapter_dir, adapter_name=model_id, is_trainable=True)

    self.peft_model.set_adapter(model_id)
    if "lm_head" in (self.peft_model.peft_config[model_id].target_modules or ()):
      self.output_head_is_adapted = True
    params = active_adapter_parameters(self.peft_model, model_id)
    adapter_state = {"trainable_params": params, "optimizer": None}
    self.adapter_states[model_id] = adapter_state

    if ENABLE_GRADIENT_CHECKPOINTING:
      try:
        self.peft_model.gradient_checkpointing_enable()
        self.peft_model.enable_input_require_grads()
        print("Gradient checkpointing and input require grads enabled on PEFT model.")
      except Exception as e:
        print(f"Failed to enable gradient checkpointing: {e}")

    self.peft_model.train()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, "optimizer.pt")
      if os.path.exists(optimizer_path):
        lr = 1e-4
        optimizer = torch.optim.AdamW(params, lr=lr)
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        adapter_state["optimizer"] = optimizer
        print(f"Restored optimizer state for '{model_id}' from {optimizer_path}")

    print(f"Loaded state for '{model_id}' from {state_path}")
    return {"model_id": model_id, "is_lora": True, "base_model": base_model}

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.peft_model is not None, "Model must be loaded first."
    if model_id:
      self.peft_model.set_adapter(model_id)
    return super().forward_backward(self.peft_model, data, loss_fn, loss_config, forward_only=forward_only)

  def optim_step(self, adam_params: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Apply accumulated gradients and update model weights."""
    assert self.peft_model is not None, "Model must be loaded first."
    if not model_id:
      raise ValueError("model_id is required for optim_step")

    self.peft_model.set_adapter(model_id)
    try:
      adapter_state = self.adapter_states[model_id]
    except KeyError as e:
      raise ValueError(f"Adapter '{model_id}' has no cached trainable parameters") from e
    params = adapter_state["trainable_params"]

    if adapter_state.get("optimizer") is None:
      lr = adam_params.get("learning_rate", 1e-4)
      beta1 = adam_params.get("beta1", 0.9)
      beta2 = adam_params.get("beta2", 0.95)
      eps = adam_params.get("eps", 1e-12)
      weight_decay = adam_params.get("weight_decay", 0.0)

      print(f"Initializing AdamW optimizer for '{model_id}' with lr={lr}")
      adapter_state["optimizer"] = torch.optim.AdamW(
        params,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
      )

    optimizer = adapter_state["optimizer"]
    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    # Each data-parallel rank holds gradients for its datum shard; combine
    # them so the replicated optimizer steps identically on every rank.
    all_reduce_gradients(params)

    total_norm = torch.nn.utils.clip_grad_norm_(
      params,
      max_grad_norm,
    )

    optimizer.step()
    optimizer.zero_grad()

    self.save_adapter(model_id)

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
  ) -> dict[str, Any]:
    if model_id:
      self.peft_model.set_adapter(model_id)
    return super().generate(self.peft_model, prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)
