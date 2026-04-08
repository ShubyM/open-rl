import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM, LoraConfig as PeftLoraConfig, get_peft_model
from pydantic import BaseModel
from typing import Any, Optional
import os
import json
import time
import traceback
from datetime import datetime
import math

# Define our own simplified types
class TensorData(BaseModel):
  data: list[int] | list[float]

class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int] # Simplified to a flat list of tokens

class TrainerEngine:
  def __init__(self):
    # The raw pre-trained base model (e.g., Gemma, Qwen) loaded in VRAM
    self.base_model: PreTrainedModel | None = None 
    
    # The model wrapped with PEFT/LoRA adapters that we actually train
    self.peft_model: PeftModelForCausalLM | None = None 
    
    # The tokenizer associated with the base model
    self.tokenizer: PreTrainedTokenizerBase | None = None
    
    # String identifier of the currently loaded base model
    self.base_model_name: str | None = None 
    
    # Store optimizers per model_id (adapter ID)
    self.optimizers: dict[str, torch.optim.Optimizer] = {}

    # Decide device
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  def load_base_model(self, base_model_name: str) -> None:
    """Eagerly load the massive base model tensors into VRAM."""
    if self.base_model is not None and self.base_model_name == base_model_name:
      print(f"Base model {base_model_name} already loaded.")
      return

    print(f"Loading base model {base_model_name} to {self.device}...")
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype, device_map=self.device)
    print("Successfully loaded.")

  def create_adapter(self, adapter_id: str, config: Any) -> None:
    """Create a new LoRA adapter on top of the loaded base model."""
    assert self.base_model is not None, "Base model is not loaded. Call load_base_model first."

    # Reset/initialize optimizer for this new adapter
    if adapter_id in self.optimizers:
      del self.optimizers[adapter_id]

    print(f"Creating LoRA adapter '{adapter_id}'...")
    
    # Map config to PEFT LoraConfig
    target_modules = []
    if getattr(config, "train_attn", True):
      target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])
    if getattr(config, "train_mlp", True):
      target_modules.extend(["gate_proj", "up_proj", "down_proj"])
      
    if getattr(config, "train_attn", True) and getattr(config, "train_mlp", True) and getattr(config, "train_unembed", True):
      target_modules = "all-linear"
    
    peft_config = PeftLoraConfig(
      task_type="CAUSAL_LM",
      r=getattr(config, "rank", 16),
      lora_alpha=getattr(config, "rank", 16),
      lora_dropout=0.05,
      bias="none",
      target_modules=target_modules,
      modules_to_save=["lm_head", "embed_tokens"] if getattr(config, "train_unembed", True) else None,
    )

    if self.peft_model is None:
      self.peft_model = get_peft_model(self.base_model, peft_config, adapter_name=adapter_id)
    else:
      self.peft_model.add_adapter(adapter_id, peft_config)
      
    self.peft_model.train()
    print(f"LoRA adapter '{adapter_id}' created and set to active.")
    
    # Auto-save the adapter as required by design
    self.save_adapter(adapter_id)

  def save_adapter(self, adapter_id: str) -> None:
    """Save adapter weights to disk for reliability and sharing."""
    try:
      tmp_dir = os.environ.get("OPEN_RL_TMP_DIR", "/tmp/open-rl")
      save_path = os.path.join(tmp_dir, "peft", adapter_id)
      os.makedirs(save_path, exist_ok=True)

      # Save the adapter weights
      self.peft_model.save_pretrained(save_path, selected_adapters=[adapter_id])

      # Save minimal metadata
      metadata = {
        "model_id": adapter_id,
        "created_at": datetime.now().isoformat(),
        "timestamp": time.time()
      }
      with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
      print(f"Auto-saved adapter '{adapter_id}' to {save_path}")
    except Exception as e:
      print(f"[ERROR] Failed to auto-save weights for {adapter_id}: {e}")
      traceback.print_exc()

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: Optional[dict] = None, model_id: Optional[str] = None) -> dict[str, Any]:
    """Core training step: forward pass, loss computation, and backward pass."""
    assert self.peft_model is not None, "Model must be loaded first."
    
    total_loss = 0.0
    loss_fn_outputs = []
    
    # Ensure model is in train mode
    self.peft_model.train()
    
    for datum in data:
      # 1. Common Setup: Extract tokens and get logprobs
      target_logprobs, targets_tensor, weights_tensor = self._get_logprobs(datum)
      
      # 2. Specialized Loss Calculation
      match loss_fn:
        case "cross_entropy":
          loss = self._compute_cross_entropy_loss(target_logprobs, weights_tensor)
        case "importance_sampling":
          loss = self._compute_importance_sampling_loss(target_logprobs, weights_tensor, datum)
        case "ppo":
          loss = self._compute_ppo_loss(target_logprobs, targets_tensor, datum, loss_config)
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")
          
      # 3. Common Cleanup: Backward pass
      loss.backward()
      total_loss += loss.item()
      
      # Save logprobs for return
      logprobs_list = target_logprobs.detach().cpu().tolist()
      logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
      
      loss_fn_outputs.append({
        "logprobs": {
          "data": logprobs_list,
          "dtype": "float32",
          "shape": [len(logprobs_list)]
        }
      })
      
    mean_loss = total_loss / max(1, len(data))
    
    return {
      "metrics": {
        "loss:mean": self._sanitize_float(mean_loss),
        "loss:sum": self._sanitize_float(total_loss)
      },
      "loss_fn_outputs": loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def _get_logprobs(self, datum: Datum) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (target_logprobs, targets_tensor, weights_tensor)."""
    # model_input is now just a flat list of tokens!
    inputs_tensor = torch.tensor([datum.model_input], dtype=torch.long, device=self.device)

    # Extract targets
    targets_data = datum.loss_fn_inputs["target_tokens"].data
    targets_tensor = torch.tensor(targets_data, dtype=torch.long, device=self.device)

    # Extract weights with default fallback to 1.0
    if "weights" in datum.loss_fn_inputs:
      weights_data = datum.loss_fn_inputs["weights"].data
    else:
      weights_data = [1.0] * len(targets_data)
    
    weights_tensor = torch.tensor(weights_data, dtype=torch.float32, device=self.device)

    outputs = self.peft_model(inputs_tensor, use_cache=False)
    logits = outputs.logits[0]  # Shape: (SeqLen, VocabSize)

    seq_len = min(logits.size(0), targets_tensor.size(0))
    sliced_logits = logits[:seq_len]
    sliced_targets = targets_tensor[:seq_len]

    target_logprobs = torch.nn.functional.log_softmax(sliced_logits, dim=-1).gather(dim=-1, index=sliced_targets.unsqueeze(-1)).squeeze(-1)

    if weights_tensor.numel() > 0:
      weights_tensor = weights_tensor[:seq_len]

    return target_logprobs, targets_tensor, weights_tensor

  def _compute_cross_entropy_loss(self, target_logprobs: torch.Tensor, weights_tensor: torch.Tensor) -> torch.Tensor:
    """Simple cross entropy loss."""
    elementwise_loss = -target_logprobs * weights_tensor
    return elementwise_loss.sum()

  def _compute_importance_sampling_loss(self, target_logprobs: torch.Tensor, weights_tensor: torch.Tensor, datum: Datum) -> torch.Tensor:
    """Importance sampling loss for RL."""
    if "logprobs" not in datum.loss_fn_inputs:
      raise ValueError("importance_sampling requires logprobs in loss_fn_inputs")
      
    ref_logprobs = datum.loss_fn_inputs["logprobs"].data
    ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
    
    seq_len = min(target_logprobs.size(0), ref_tensor.size(0), weights_tensor.size(0))
    target_logprobs = target_logprobs[:seq_len]
    ref_tensor = ref_tensor[:seq_len]
    weights_tensor = weights_tensor[:seq_len]
    
    diff = target_logprobs - ref_tensor
    diff = torch.clamp(diff, min=-20.0, max=20.0)
    ratio = torch.exp(diff)
    
    elementwise_loss = -ratio * weights_tensor
    return elementwise_loss.sum()

  def _compute_ppo_loss(self, target_logprobs: torch.Tensor, targets_tensor: torch.Tensor, datum: Datum, loss_config: Optional[dict]) -> torch.Tensor:
    """PPO loss for RL."""
    if "logprobs" not in datum.loss_fn_inputs or "advantages" not in datum.loss_fn_inputs:
      raise ValueError("ppo requires 'logprobs' and 'advantages' in loss_fn_inputs")
      
    ref_logprobs = datum.loss_fn_inputs["logprobs"].data
    advantages = datum.loss_fn_inputs["advantages"].data
    
    ref_tensor = torch.tensor(ref_logprobs, dtype=target_logprobs.dtype, device=self.device)
    advantages_tensor = torch.tensor(advantages, dtype=target_logprobs.dtype, device=self.device)
    
    seq_len = min(target_logprobs.size(0), ref_tensor.size(0), advantages_tensor.size(0))
    target_logprobs = target_logprobs[:seq_len]
    ref_tensor = ref_tensor[:seq_len]
    advantages_tensor = advantages_tensor[:seq_len]
    
    diff = target_logprobs - ref_tensor
    diff = torch.clamp(diff, min=-20.0, max=20.0)
    ratio = torch.exp(diff)
    
    epsilon = loss_config.get("clip_range", 0.2) if loss_config else 0.2
    
    surr1 = ratio * advantages_tensor
    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages_tensor
    
    elementwise_objective = torch.min(surr1, surr2)
    return -elementwise_objective.sum()

  def _sanitize_float(self, val: float) -> float:
    if math.isinf(val):
      return -9999.0 if val < 0 else 9999.0
    if math.isnan(val):
      return 0.0
    return val

  def optim_step(self, model_id: str, learning_rate: float, grad_clip_norm: Optional[float] = None) -> dict[str, Any]:
    """Apply accumulated gradients and update model weights."""
    assert self.peft_model is not None, "Model must be loaded first."
    
    if model_id not in self.optimizers:
      print(f"Initializing AdamW optimizer for '{model_id}' with lr={learning_rate}")
      self.optimizers[model_id] = torch.optim.AdamW(self.peft_model.parameters(), lr=learning_rate)
      
    optimizer = self.optimizers[model_id]
    
    for param_group in optimizer.param_groups:
      param_group['lr'] = learning_rate
      
    # Compute grad norm and clip if needed
    total_norm = torch.nn.utils.clip_grad_norm_(self.peft_model.parameters(), grad_clip_norm or float('inf'))
      
    optimizer.step()
    optimizer.zero_grad()
    
    # Auto-save the adapter as required by design
    self.save_adapter(model_id)
    
    return {
      "status": "ok",
      "metrics": {
        "grad_norm:mean": self._sanitize_float(float(total_norm))
      }
    }
