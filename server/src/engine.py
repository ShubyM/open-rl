import asyncio
import hashlib
import json
import os
import threading
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import torch
from fastapi import FastAPI
from opentelemetry import trace
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

tracer = trace.get_tracer(__name__)

from .checkpoints import OPTIMIZER_STATE_FILENAME, read_state_metadata, write_state_metadata
from .state import get_store

store = get_store()

import math

ATTN_TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGET_SUFFIXES = ["gate_proj", "up_proj", "down_proj"]


class TrainerEngine:
  def __init__(self):
    self.model = None
    self.model_obj = None
    self.tokenizer = None

    # Store optimizers per model_id
    self.optimizers: dict[str, torch.optim.Optimizer] = {}
    self.loaded_snapshot_adapters: dict[str, tuple[str, float]] = {}
    self._init_lock = threading.Lock()

    # Decide device
    if torch.cuda.is_available():
      self.device = "cuda"
    elif torch.backends.mps.is_available():
      self.device = "mps"
    else:
      self.device = "cpu"

    self.base_model_name = None

  def _ensure_base_model_loaded(self, base_model: str):
    if getattr(self, "model_obj", None) is not None and self.base_model_name == base_model:
      return

    if self.base_model_name != base_model:
      self.model = None
      self.optimizers.clear()
      self.loaded_snapshot_adapters.clear()

    self.base_model_name = base_model
    print(f"Loading base model {base_model} to {self.device}...")
    self.tokenizer = AutoTokenizer.from_pretrained(base_model)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    self.model_obj = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=self.device)

  def preload_base_model(self, base_model: str):
    """Eagerly load the massive base model tensors into VRAM."""
    with self._init_lock:
      if self.model is None or self.base_model_name != base_model:
        print(f"[EAGER INIT] Pre-loading heavy base model {base_model} to {self.device}...")
        self._ensure_base_model_loaded(base_model)
        print(f"[EAGER INIT] {base_model} successfully seated in VRAM.")

  def _delete_adapter_if_present(self, model_id: str):
    if self.model is None:
      return
    peft_config = getattr(self.model, "peft_config", {})
    if model_id in peft_config:
      self.model.delete_adapter(model_id)
    self.optimizers.pop(model_id, None)

  def _infer_base_model_from_state(self, state_path: str, explicit_base_model: str | None = None) -> str:
    if explicit_base_model:
      return explicit_base_model

    metadata = read_state_metadata(state_path)
    base_model = metadata.get("base_model")
    if isinstance(base_model, str) and base_model:
      return base_model

    adapter_config_path = os.path.join(state_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
      with open(adapter_config_path, encoding="utf-8") as handle:
        adapter_config = json.load(handle)
      inferred = adapter_config.get("base_model_name_or_path")
      if isinstance(inferred, str) and inferred:
        return inferred

    raise ValueError(f"Unable to infer base model from saved state: {state_path}")

  def _load_optimizer_state(self, model_id: str, state_path: str):
    optimizer_state_path = os.path.join(state_path, OPTIMIZER_STATE_FILENAME)
    if not os.path.exists(optimizer_state_path):
      raise FileNotFoundError(f"No optimizer state found at {optimizer_state_path}")

    params = [p for p in self.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params)
    optimizer_state = torch.load(optimizer_state_path, map_location=self.device)
    optimizer.load_state_dict(optimizer_state)

    for state in optimizer.state.values():
      for key, value in list(state.items()):
        if torch.is_tensor(value):
          state[key] = value.to(self.device)

    self.optimizers[model_id] = optimizer

  def _load_model_from_state(self, base_model: str | None, model_id: str, state_path: str, restore_optimizer: bool):
    if not os.path.isdir(state_path):
      raise FileNotFoundError(f"Saved state directory not found: {state_path}")

    resolved_base_model = self._infer_base_model_from_state(state_path, base_model)
    self._ensure_base_model_loaded(resolved_base_model)
    self._delete_adapter_if_present(model_id)

    if self.model is None:
      self.model = PeftModel.from_pretrained(self.model_obj, state_path, adapter_name=model_id, is_trainable=True)
    else:
      self.model.load_adapter(state_path, adapter_name=model_id, is_trainable=True)

    self.model.set_adapter(model_id)
    self.model.train()
    self.optimizers.pop(model_id, None)
    if restore_optimizer:
      self._load_optimizer_state(model_id, state_path)

    self._auto_save_adapter(model_id)

  def load_model(
    self,
    base_model: str | None,
    model_id: str,
    lora_config: dict[str, Any] | None = None,
    state_path: str | None = None,
    restore_optimizer: bool = False,
  ):
    with self._init_lock:
      if state_path:
        self._load_model_from_state(base_model, model_id, state_path, restore_optimizer)
        return

      if not base_model:
        raise ValueError("base_model is required when creating a new training client")

      # If the user asks for a different base model than what's loaded, we have to swap it.
      # But normally, eager init guarantees this matches.
      self._ensure_base_model_loaded(base_model)
      config = lora_config or {}
      rank = config.get("rank", 16)
      train_attn = config.get("train_attn", True)
      train_mlp = config.get("train_mlp", True)
      train_unembed = config.get("train_unembed", True)
      if not any([train_attn, train_mlp, train_unembed]):
        raise ValueError("At least one LoRA training target must be enabled.")

      # Tinker's LoRA config is intentionally coarse; PEFT still expects concrete target names here.
      # Keep the historical broad behavior by default, but constrain Gemma4 to the language tower.
      explicit_target_modules = os.getenv("OPEN_RL_TARGET_MODULES")
      use_text_tower_lora = getattr(self.model_obj.config, "model_type", None) == "gemma4"
      target_modules: str | list[str]
      layers_to_transform = None
      layers_pattern = None
      target_suffixes: list[str] = []
      if train_attn:
        target_suffixes.extend(ATTN_TARGET_SUFFIXES)
      if train_mlp:
        target_suffixes.extend(MLP_TARGET_SUFFIXES)
      if explicit_target_modules == "all-linear":
        target_modules = "all-linear"
        print("[engine] Forcing PEFT target_modules=all-linear via OPEN_RL_TARGET_MODULES")
      elif use_text_tower_lora:
        if target_suffixes:
          target_modules = target_suffixes
          model_config = getattr(self.model_obj.config, "text_config", self.model_obj.config)
          num_hidden_layers = model_config.num_hidden_layers
          layers_to_transform = list(range(num_hidden_layers))
          layers_pattern = "language_model.layers"
        else:
          target_modules = []
      elif train_attn and train_mlp and train_unembed:
        target_modules = "all-linear"
      else:
        target_modules = target_suffixes

      peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias="none",
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
        modules_to_save=["lm_head", "embed_tokens"] if train_unembed else None,
      )

      if getattr(self, "model", None) is None:
        # First time wrapping the base model with PEFT
        self.model = get_peft_model(self.model_obj, peft_config, adapter_name=model_id)
        self.model.train()
        print(f"Base model wrapped with initial LoRA adapter '{model_id}'.")
      else:
        print(f"Adding new LoRA adapter '{model_id}' to existing base model...")
        self.model.add_adapter(model_id, peft_config)
        self.model.train()

    # Reset/initialize optimizer for this new adapter
    if model_id in self.optimizers:
      del self.optimizers[model_id]

    self._auto_save_adapter(model_id)

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    with tracer.start_as_current_span("save_state") as span, self._init_lock:
      span.set_attribute("model_id", model_id or "unknown")
      span.set_attribute("state_path", state_path)
      if not model_id:
        raise ValueError("model_id is required for save_state")
      assert self.model is not None, "Model not loaded."

      self.model.set_adapter(model_id)
      os.makedirs(state_path, exist_ok=True)
      self.model.save_pretrained(state_path, selected_adapters=[model_id])

      optimizer_file = None
      optimizer_state_path = os.path.join(state_path, OPTIMIZER_STATE_FILENAME)
      optimizer = self.optimizers.get(model_id)
      if include_optimizer and optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_state_path)
        optimizer_file = OPTIMIZER_STATE_FILENAME
      elif os.path.exists(optimizer_state_path):
        os.remove(optimizer_state_path)

      metadata = {
        "base_model": self.base_model_name,
        "created_at": datetime.now().isoformat(),
        "kind": kind,
        "model_id": model_id,
        "optimizer_file": optimizer_file,
        "timestamp": time.time(),
      }
      write_state_metadata(state_path, metadata)
      return {"path": state_path}

  def _auto_save_adapter(self, model_id: str):
    try:
      tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
      ram_path = os.path.join(tmp_dir, "peft", model_id)
      os.makedirs(ram_path, exist_ok=True)

      with tracer.start_as_current_span(f"auto_save_weights_{model_id}"):
        self.model.save_pretrained(ram_path, selected_adapters=[model_id])

      metadata = {"model_id": model_id, "alias": None, "created_at": datetime.now().isoformat(), "timestamp": time.time()}
      with open(os.path.join(ram_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    except Exception as e:
      print(f"[ERROR] Failed to auto-save weights for {model_id}: {e}")
      traceback.print_exc()

  def set_active_adapter(self, model_id: str):
    with self._init_lock:
      if self.model is not None:
        self.model.set_adapter(model_id)

  def forward(self, data: list[dict[str, Any]], loss_fn: str, loss_fn_config: dict = None, model_id: str = None) -> dict[str, Any]:
    with tracer.start_as_current_span("forward") as span, self._init_lock:
      span.set_attribute("model_id", model_id or "unknown")
      span.set_attribute("batch_size", len(data))
      span.set_attribute("loss_fn", loss_fn)
      return self._forward_backward_internal(data, loss_fn, loss_fn_config, model_id, do_backward=False)

  def forward_backward(self, data: list[dict[str, Any]], loss_fn: str, loss_fn_config: dict = None, model_id: str = None) -> dict[str, Any]:
    with tracer.start_as_current_span("forward_backward") as span, self._init_lock:
      span.set_attribute("model_id", model_id or "unknown")
      span.set_attribute("batch_size", len(data))
      span.set_attribute("loss_fn", loss_fn)
      return self._forward_backward_internal(data, loss_fn, loss_fn_config, model_id, do_backward=True)

  def _forward_backward_internal(
    self,
    data: list[dict[str, Any]],
    loss_fn: str,
    loss_fn_config: dict = None,
    model_id: str = None,
    do_backward: bool = True,
  ) -> dict[str, Any]:
    """
    data: List of Datum objects
    """
    assert self.model is not None, "Model not loaded."
    if model_id:
      self.model.set_adapter(model_id)
    self.model.train()

    if model_id and model_id not in self.optimizers:
      # Ensuring optimizer exists is good, but we don't zero_grad here anymore
      pass

    total_loss = 0.0
    loss_fn_outputs = []

    for datum in data:
      # Extract inputs
      chunks = datum.get("model_input", {}).get("chunks", [])
      input_tokens = []
      for chunk in chunks:
        input_tokens.extend(chunk.get("tokens", []))

      inputs_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)

      # Extract loss inputs
      loss_inputs = datum.get("loss_fn_inputs", {})

      weights_data = loss_inputs.get("weights", {}).get("data", [])
      weights_tensor = torch.tensor(weights_data, dtype=torch.float32, device=self.device)

      targets_data = loss_inputs.get("target_tokens", {}).get("data", [])
      targets_tensor = torch.tensor(targets_data, dtype=torch.long, device=self.device)

      outputs = self.model(inputs_tensor, use_cache=False)
      logits = outputs.logits[0]  # Shape: (SeqLen, VocabSize)

      # gather logprobs for the target tokens
      # The client SDK already offsets inputs vs targets.
      seq_len = min(logits.size(0), targets_tensor.size(0))

      sliced_logits = logits[:seq_len]
      sliced_targets = targets_tensor[:seq_len]

      target_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1).gather(dim=-1, index=sliced_targets.unsqueeze(-1)).squeeze(-1)

      # Clamp weights tensor if it exists, otherwise ones
      if weights_tensor.numel() > 0:
        weights_tensor = weights_tensor[:seq_len]
      else:
        weights_tensor = torch.ones_like(target_logprobs)

      if loss_fn == "cross_entropy":
        elementwise_loss = -target_logprobs * weights_tensor
        loss = elementwise_loss.sum()
      elif loss_fn == "importance_sampling":
        ref_logprobs_raw = loss_inputs.get("logprobs")
        advs_raw = loss_inputs.get("advantages")

        ref_logprobs = ref_logprobs_raw.get("data") if isinstance(ref_logprobs_raw, dict) else ref_logprobs_raw
        advs = advs_raw.get("data") if isinstance(advs_raw, dict) else advs_raw
        if not ref_logprobs or not advs:
          raise ValueError("importance_sampling requires 'logprobs' and 'advantages' in loss_fn_inputs")

        ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
        advantages_tensor = torch.tensor(advs, dtype=target_logprobs.dtype, device=self.device)

        # Align reference logits and advantages to the generated right-aligned targets
        ref_tensor = ref_tensor[:seq_len]
        advantages_tensor = advantages_tensor[:seq_len]

        # Prevent overflow in exp() by explicitly clamping diff
        diff = target_logprobs - ref_tensor
        diff = torch.clamp(diff, min=-20.0, max=20.0)

        ratio = torch.exp(diff)

        elementwise_loss = -(ratio * advantages_tensor) * weights_tensor
        # Add nan_to_num to be absolutely sure no trailing NaNs poison the gradients
        elementwise_loss = torch.nan_to_num(elementwise_loss, nan=0.0, posinf=0.0, neginf=0.0)

        loss = elementwise_loss.sum()
      elif loss_fn == "ppo":
        ref_logprobs_raw = loss_inputs.get("logprobs")
        advs_raw = loss_inputs.get("advantages")

        ref_logprobs = ref_logprobs_raw.get("data") if isinstance(ref_logprobs_raw, dict) else ref_logprobs_raw
        advs = advs_raw.get("data") if isinstance(advs_raw, dict) else advs_raw
        if not ref_logprobs or not advs:
          raise ValueError("ppo requires 'logprobs' and 'advantages' in loss_fn_inputs")

        ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
        advantages_tensor = torch.tensor(advs, dtype=target_logprobs.dtype, device=self.device)

        ref_tensor = ref_tensor[:seq_len]
        advantages_tensor = advantages_tensor[:seq_len]

        # Prevent overflow in exp() by explicitly clamping diff
        diff = target_logprobs - ref_tensor
        diff = torch.clamp(diff, min=-20.0, max=20.0)
        ratio = torch.exp(diff)

        epsilon = loss_fn_config.get("clip_range", 0.2) if loss_fn_config else 0.2

        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_tensor

        # PPO Objective: maximize min(surr1, surr2)
        # Loss: minimize -min(surr1, surr2)
        elementwise_objective = torch.min(surr1, surr2)

        loss = -elementwise_objective.sum()
      else:
        raise NotImplementedError(f"Loss {loss_fn} not implemented in MVP yet.")

      if do_backward:
        loss.backward()
      total_loss += loss.item()

      logprobs_list = target_logprobs.detach().cpu().tolist()
      logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]

      # Construct loss_fn_output
      loss_fn_outputs.append({"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}})
    mean_loss = total_loss / max(1, len(data))

    return {
      "metrics": {"loss:mean": self._sanitize_float(mean_loss), "loss:sum": self._sanitize_float(total_loss)},
      "loss_fn_outputs": loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def _sanitize_float(self, val: float) -> float:
    if math.isinf(val):
      return -9999.0 if val < 0 else 9999.0
    if math.isnan(val):
      return 0.0
    return val

  def _apply_optimizer_params(self, optimizer: torch.optim.Optimizer, adam_params: dict[str, Any]):
    if "learning_rate" in adam_params:
      optimizer.defaults["lr"] = adam_params["learning_rate"]
    if "weight_decay" in adam_params:
      optimizer.defaults["weight_decay"] = adam_params["weight_decay"]
    if "eps" in adam_params:
      optimizer.defaults["eps"] = adam_params["eps"]
    if "beta1" in adam_params or "beta2" in adam_params:
      default_beta1, default_beta2 = optimizer.defaults.get("betas", (0.9, 0.95))
      optimizer.defaults["betas"] = (
        adam_params.get("beta1", default_beta1),
        adam_params.get("beta2", default_beta2),
      )

    for group in optimizer.param_groups:
      if "learning_rate" in adam_params:
        group["lr"] = adam_params["learning_rate"]
      if "weight_decay" in adam_params:
        group["weight_decay"] = adam_params["weight_decay"]
      if "eps" in adam_params:
        group["eps"] = adam_params["eps"]
      if "beta1" in adam_params or "beta2" in adam_params:
        beta1, beta2 = group.get("betas", optimizer.defaults.get("betas", (0.9, 0.95)))
        group["betas"] = (
          adam_params.get("beta1", beta1),
          adam_params.get("beta2", beta2),
        )

  def optim_step(self, adam_params: dict[str, Any], model_id: str = None):
    with tracer.start_as_current_span("optim_step") as span, self._init_lock:
      span.set_attribute("model_id", model_id or "unknown")
      return self._optim_step_internal(adam_params, model_id)

  def _optim_step_internal(self, adam_params: dict[str, Any], model_id: str = None):
    if not model_id:
      raise ValueError("model_id is required for optim_step")
    assert self.model is not None, "Model not loaded."

    self.model.set_adapter(model_id)
    self.model.train()

    if model_id not in self.optimizers:
      lr = adam_params.get("learning_rate", 1e-4)
      beta1 = adam_params.get("beta1", 0.9)
      beta2 = adam_params.get("beta2", 0.95)
      eps = adam_params.get("eps", 1e-12)
      weight_decay = adam_params.get("weight_decay", 0.0)

      # Ensure we only pass the parameters of the active adapter for this model_id
      # peft's model.parameters() works, but it's better to filter requires_grad
      params = [p for p in self.model.parameters() if p.requires_grad]

      self.optimizers[model_id] = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)

    optimizer = self.optimizers[model_id]
    self._apply_optimizer_params(optimizer, adam_params)

    # Compute grad norm BEFORE stepping
    total_norm = 0.0
    for p in self.model.parameters():
      if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm**0.5
    print(f"DEBUG: grad_norm={total_norm}")

    # Apply gradient clipping if requested
    clip_norm = adam_params.get("grad_clip_norm", 0.0)
    if clip_norm > 0.0:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)

    optimizer.step()
    optimizer.zero_grad()

    # Auto-save for Hardware Pipeline bypass
    self._auto_save_adapter(model_id)

    return {"metrics": {"grad_norm:mean": self._sanitize_float(total_norm)}}

  def _ensure_snapshot_adapter_loaded(self, snapshot_path: str) -> str:
    snapshot_mtime = os.path.getmtime(snapshot_path)
    cached = self.loaded_snapshot_adapters.get(snapshot_path)
    if cached and cached[1] == snapshot_mtime:
      return cached[0]

    base_model = self._infer_base_model_from_state(snapshot_path)
    self._ensure_base_model_loaded(base_model)

    if cached and self.model is not None:
      cached_adapter_name = cached[0]
      peft_config = getattr(self.model, "peft_config", {})
      if cached_adapter_name in peft_config:
        self.model.delete_adapter(cached_adapter_name)

    adapter_name = f"snapshot_{hashlib.md5(f'{snapshot_path}:{snapshot_mtime}'.encode('utf-8')).hexdigest()[:12]}"
    peft_config = getattr(self.model, "peft_config", {})
    if self.model is None:
      self.model = PeftModel.from_pretrained(self.model_obj, snapshot_path, adapter_name=adapter_name, is_trainable=False)
    elif adapter_name not in peft_config:
      self.model.load_adapter(snapshot_path, adapter_name=adapter_name, is_trainable=False)

    self.loaded_snapshot_adapters[snapshot_path] = (adapter_name, snapshot_mtime)
    return adapter_name

  def compute_logprobs(self, tokens: list[int], model_id: str = None, adapter_path: str | None = None) -> list[float | None]:
    with self._init_lock:
      if adapter_path:
        adapter_name = self._ensure_snapshot_adapter_loaded(adapter_path)
      else:
        adapter_name = model_id
      assert self.model is not None, "Model not loaded."
      if adapter_name:
        self.model.set_adapter(adapter_name)
      self.model.eval()

      if not tokens:
        return []
      if len(tokens) == 1:
        return [None]

      input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
      with torch.no_grad():
        outputs = self.model(input_tensor, use_cache=False)
        log_probs = torch.nn.functional.log_softmax(outputs.logits[0], dim=-1)

    result: list[float | None] = [None]
    for idx, token_id in enumerate(tokens[1:], start=0):
      result.append(self._sanitize_float(log_probs[idx, token_id].item()))
    return result

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str = None,
    adapter_path: str | None = None,
  ) -> dict[str, Any]:
    with self._init_lock:
      if adapter_path:
        adapter_name = self._ensure_snapshot_adapter_loaded(adapter_path)
      else:
        adapter_name = model_id
      assert self.model is not None, "Model not loaded."
      if adapter_name:
        self.model.set_adapter(adapter_name)
      self.model.eval()

      input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
      do_sample = (num_samples > 1) or (temperature and temperature > 0.0)

      with torch.no_grad():
        attention_mask = torch.ones_like(input_tensor)
        outputs = self.model.generate(
          input_tensor,
          attention_mask=attention_mask,
          max_new_tokens=max_tokens,
          pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
          do_sample=do_sample,
          temperature=temperature if do_sample else None,
          top_p=None,
          top_k=None,
          num_return_sequences=num_samples,
          output_scores=True,
          return_dict_in_generate=True,
        )

    sequences_out = []
    for seq_idx in range(num_samples):
      gen_sequences = outputs.sequences[seq_idx]
      generated_tokens = gen_sequences[len(prompt_tokens) :].cpu().tolist()

      logprobs = []
      for token_step_idx in range(len(generated_tokens)):
        score_tensor = outputs.scores[token_step_idx]
        logprob_dist = torch.nn.functional.log_softmax(score_tensor[seq_idx], dim=-1)
        token_id = generated_tokens[token_step_idx]
        logprob = logprob_dist[token_id].item()
        if logprob == float("-inf"):
          logprob = -9999.0
        elif logprob == float("inf"):
          logprob = 9999.0
        logprobs.append(logprob)

      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

    return {"sequences": sequences_out}


# Global singleton
engine = TrainerEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Gateway")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  vllm_url = os.getenv("VLLM_URL", "http://127.0.0.1:8001")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}")
  print(f"-> Inference: Routing completions to VLLM_URL={vllm_url}\n")

  # Start the clock cycle loop
  task = asyncio.create_task(clock_cycle_loop())
  yield

  # Cleanup if needed
  task.cancel()


async def clock_cycle_loop():
  global store
  while True:
    try:
      # Block until requests are available and drain the queue
      # With the new RR Queue logic, this batch is guaranteed to belong to ONE tenant
      batch = await store.get_requests()
      if not batch:
        await asyncio.sleep(0.1)
        continue

      m_id = batch[0].get("model_id", "default")

      with tracer.start_as_current_span("clock_cycle_batch") as batch_span:
        batch_span.set_attribute("batch_size", len(batch))
        batch_span.set_attribute("model_id", m_id)

        print(f"\n[CLOCK CYCLE] Popped {len(batch)} requests for tenant: {m_id}")

        with tracer.start_as_current_span("process_model_batch") as model_span:
          model_span.set_attribute("model_id", m_id)
          model_span.set_attribute("model_reqs", len(batch))
          print(f"  -> [TENSOR CORE] Hot-swapping to LoRA adapter: {m_id}")

          # Set active adapter, UNLESS the batch is trying to create it!
          has_create_model = any(r.get("type") == "create_model" for r in batch)
          if not has_create_model:
            try:
              with tracer.start_as_current_span("set_active_adapter"):
                await asyncio.to_thread(engine.set_active_adapter, m_id)
            except Exception as e:
              print(f"Failed to set adapter {m_id}: {e}")
              for r in batch:
                await store.set_future(r["req_id"], {"type": "RequestFailedResponse", "error_message": str(e)})
              continue

          print(f"     Executing {len(batch)} operations for {m_id}...")

          # Execute sequentially
          for r in batch:
            req_id = r["req_id"]
            req_type = r["type"]

            carrier = r.get("trace_context", {})
            from opentelemetry import context as otel_context
            from opentelemetry import propagate

            ctx = propagate.extract(carrier) if carrier else None
            token = otel_context.attach(ctx) if ctx else None

            try:
              if req_type == "forward_backward":
                data = r["data"]
                loss_fn = r["loss_fn"]
                loss_config = r["loss_config"]
                result = await asyncio.to_thread(engine.forward_backward, data, loss_fn, loss_config, m_id)
                result["type"] = "forward_backward"
                await store.set_future(req_id, result)
              elif req_type == "forward":
                data = r["data"]
                loss_fn = r["loss_fn"]
                loss_config = r["loss_config"]
                result = await asyncio.to_thread(engine.forward, data, loss_fn, loss_config, m_id)
                result["type"] = "forward"
                await store.set_future(req_id, result)
              elif req_type == "optim_step":
                adam_params = r["adam_params"]
                result = await asyncio.to_thread(engine.optim_step, adam_params, m_id)
                result["type"] = "optim_step"
                await store.set_future(req_id, result)
              elif req_type == "sample":
                prompt_tokens = r["prompt_tokens"]
                max_tokens = r["max_tokens"]
                num_samples = r["num_samples"]
                temperature = r.get("temperature", 0.0)
                adapter_path = r.get("adapter_path")
                result = await asyncio.to_thread(engine.generate, prompt_tokens, max_tokens, num_samples, temperature, m_id, adapter_path)
                result["type"] = "sample"
                await store.set_future(req_id, result)
              elif req_type == "compute_logprobs":
                tokens = r["tokens"]
                adapter_path = r.get("adapter_path")
                logprobs = await asyncio.to_thread(engine.compute_logprobs, tokens, m_id, adapter_path)
                await store.set_future(req_id, {"type": "compute_logprobs", "logprobs": logprobs})
              elif req_type == "save_state":
                state_path = r["state_path"]
                include_optimizer = bool(r.get("include_optimizer", False))
                kind = r.get("kind", "state")
                result = await asyncio.to_thread(engine.save_state, m_id, state_path, include_optimizer, kind)
                result["type"] = "save_state"
                await store.set_future(req_id, result)
              elif req_type == "load_state":
                state_path = r["state_path"]
                base_model = r.get("base_model")
                restore_optimizer = bool(r.get("restore_optimizer", False))
                await asyncio.to_thread(engine.load_model, base_model, m_id, None, state_path, restore_optimizer)
                await store.set_future(req_id, {"path": state_path, "type": "load_state"})
              elif req_type == "create_model":
                base_model = r["base_model"]
                lora_config = r.get("lora_config") or {}
                rank = lora_config.get("rank", 16)
                state_path = r.get("state_path")
                restore_optimizer = bool(r.get("restore_optimizer", False))
                await asyncio.to_thread(engine.load_model, base_model, m_id, lora_config, state_path, restore_optimizer)
                await store.set_future(req_id, {"model_id": m_id, "is_lora": True, "lora_rank": rank, "type": "create_model"})
            except Exception as e:
              traceback.print_exc()
              await store.set_future(req_id, {"type": "RequestFailedResponse", "error_message": str(e)})
            finally:
              if token:
                otel_context.detach(token)

    except asyncio.CancelledError:
      break
    except Exception as e:
      print(f"Error in clock cycle loop: {e}")
      traceback.print_exc()

      # If the background Redis pod was restarted, the asyncio connection pool gets stuck.
      # We forcefully destroy the singleton to create a fresh connection pool on the next loop.
      import redis

      if isinstance(e, redis.exceptions.ConnectionError):
        print("[engine] Destroying StateStore singleton to force Redis reconnection...")
        from . import state

        state._store_instance = None
        store = state.get_store()

      await asyncio.sleep(1)


if __name__ == "__main__":
  import threading

  import uvicorn
  from fastapi import FastAPI

  print("\n" + "=" * 50)
  print("      Open-RL PyTorch Training Worker")
  print("=" * 50)
  cuda_devs = os.getenv("CUDA_VISIBLE_DEVICES", "ALL")
  print(f"-> Hardware : CUDA_VISIBLE_DEVICES={cuda_devs}\n")

  # 1. Eagerly load the base model to bypass cold-start penalties
  preload_target = os.getenv("VLLM_MODEL")
  is_ready = False
  if preload_target:
    engine.preload_base_model(preload_target)
    is_ready = True
  else:
    print("[WARNING] VLLM_MODEL not provided. Cold-start penalty will apply on first request.")
    is_ready = True  # Nothing to load, so we are "ready" to receive the first request

  # 2. Stand up a lightweight ASGI app strictly for Kubernetes Readiness Probes
  probe_app = FastAPI()

  @probe_app.get("/healthz")
  def healthz():
    if is_ready:
      return {"status": "ready"}
    from fastapi import HTTPException

    raise HTTPException(status_code=503, detail="Model Loading")

  def run_probe_server():
    # Run on port 8000 so the Kubernetes deployment health check works
    uvicorn.run(probe_app, host="0.0.0.0", port=8000, log_level="warning")

  threading.Thread(target=run_probe_server, daemon=True).start()

  # 3. Start the infinite tensor crunching loop
  async def main():
    task = asyncio.create_task(clock_cycle_loop())
    try:
      await task
    except asyncio.CancelledError:
      pass

  asyncio.run(main())
