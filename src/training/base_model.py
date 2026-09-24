"""What Tinker calls base_model: the HF module with its checkpoint loaded.

For LoRA the module stays frozen and a tenant's A/B matrices are added by
forward hooks on the target Linear layers, so any number of tenants share one
module and switching tenants touches nothing. For full fine-tuning a tenant's
weights are the module's own parameters.
"""

import contextlib
import math
import os
from collections.abc import Iterator
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from training.trained_weights import TrainedWeights
from training.types import LoraConfig

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"

LORA_TARGETS = {
  "train_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
  # TODO: Revisit MLP targets for packed/MoE module names across supported backends.
  "train_mlp": ("gate_proj", "up_proj", "down_proj"),
  "train_unembed": ("lm_head",),
}
LORA_SUFFIXES = {suffix for suffixes in LORA_TARGETS.values() for suffix in suffixes}


def default_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def sanitize_float(val: float) -> float:
  if math.isinf(val):
    return -9999.0 if val < 0 else 9999.0
  if math.isnan(val):
    return 0.0
  return val


class BaseModel:
  def __init__(self, name: str, module: torch.nn.Module, tokenizer: PreTrainedTokenizerBase | None = None, device: torch.device | None = None):
    self.name = name
    self.module = module
    self.tokenizer = tokenizer
    self.device = device or default_device()
    # The tenant whose LoRA the hooks add, set by using().
    self.active: TrainedWeights | None = None
    self.lora_targets: dict[str, torch.nn.Linear] = {
      module_name: child
      for module_name, child in module.named_modules()
      if module_name.rsplit(".", 1)[-1] in LORA_SUFFIXES and isinstance(child, torch.nn.Linear)
    }
    for module_name, child in self.lora_targets.items():
      child.register_forward_hook(self.lora_hook(module_name))

  @property
  def pad_token_id(self) -> int:
    if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
      return self.tokenizer.pad_token_id
    return 0

  def lora_hook(self, module_name: str) -> Any:
    a_key, b_key = f"{module_name}.lora_A", f"{module_name}.lora_B"

    def hook(linear: torch.nn.Linear, args: tuple, output: torch.Tensor) -> torch.Tensor:
      trained = self.active
      if trained is None or a_key not in trained.weights:
        return output
      a, b = trained.weights[a_key], trained.weights[b_key]
      x = torch.nn.functional.dropout(args[0].to(a.dtype), trained.lora_config.lora_dropout, training=linear.training)
      # PEFT's LoRA, B(A(dropout(x))) * alpha / r, computed in the adapter's dtype.
      return output + ((x @ a.T) @ b.T * (trained.lora_config.lora_alpha / a.shape[0])).to(output.dtype)

    return hook

  @contextlib.contextmanager
  def using(self, trained: TrainedWeights) -> Iterator[None]:
    """Add trained's LoRA to every forward inside, including the recompute a
    checkpointed layer runs during backward, which is why the whole step goes in here."""
    self.active = trained if trained.lora_config is not None else None
    try:
      yield
    finally:
      self.active = None

  def weight_device(self, name: str) -> torch.device:
    """Where a weight belongs, next to its layer when the module is spread over devices."""
    if ".lora_" in name:
      return self.lora_targets[name.rsplit(".lora_", 1)[0]].weight.device
    return self.device

  def lora_target_names(self, config: LoraConfig) -> list[str]:
    wanted = {suffix for flag, suffixes in LORA_TARGETS.items() if getattr(config, flag) for suffix in suffixes}
    if not wanted:
      raise ValueError("At least one LoRA training target must be enabled.")
    # getattr because not every config defines tie_word_embeddings (transformers
    # v5 dropped it from the PretrainedConfig base class). When absent,
    # transformers performs no tying, so False matches the loaded model.
    if "lm_head" in wanted and getattr(self.module.config, "tie_word_embeddings", False):
      # An adapter on a tied lm_head shares the embedding tensor, and vLLM
      # refuses lm_head adapter weights for tied models.
      print(f"[LoRA] Ignoring train_unembed=True: {self.name} ties lm_head to embed_tokens, and the resulting adapter could not be loaded by vLLM.")
      wanted.discard("lm_head")
    names = [module_name for module_name in self.lora_targets if module_name.rsplit(".", 1)[-1] in wanted]
    if not names:
      raise ValueError(f"No supported LoRA target modules found for suffixes: {sorted(wanted)}")
    return names

  def new_weights(self, config: LoraConfig | None) -> dict[str, torch.Tensor]:
    """Fresh weights for one model_id. A new adapter computes exactly the base model, since B starts at zero."""
    if config is None:
      for param in self.module.parameters():
        param.requires_grad_(True)
      return dict(self.module.named_parameters())
    for param in self.module.parameters():
      param.requires_grad_(False)
    weights = {}
    for module_name in self.lora_target_names(config):
      linear = self.lora_targets[module_name]
      a = torch.empty(config.rank, linear.in_features, dtype=torch.float32, device=linear.weight.device)
      torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5))
      b = torch.zeros(linear.out_features, config.rank, dtype=torch.float32, device=linear.weight.device)
      weights[f"{module_name}.lora_A"] = a.requires_grad_()
      weights[f"{module_name}.lora_B"] = b.requires_grad_()
    return weights

  def token_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Selected target logprobs with shape [batch, max_target_len]."""
    outputs = self.module(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, : target_token_ids.shape[1], :]
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    """Generate completions with HF generate."""
    self.module.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
    prompt_logprobs = self.prompt_logprobs(input_tensor) if include_prompt_logprobs else None

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = self.module.generate(
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
        logprobs.append(sanitize_float(logprob))

      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

    result = {"sequences": sequences_out}
    if prompt_logprobs is not None:
      result["prompt_logprobs"] = prompt_logprobs
    return result

  def prompt_logprobs(self, input_tensor: torch.Tensor) -> list[float | None]:
    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = self.module(input_tensor, attention_mask=attention_mask)
      logprob_dist = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)

    prompt_tokens = input_tensor[0].tolist()
    prompt_logprobs: list[float | None] = [None]
    for token_idx, token_id in enumerate(prompt_tokens[1:]):
      logprob = logprob_dist[token_idx, token_id].item()
      prompt_logprobs.append(sanitize_float(logprob))

    return prompt_logprobs


def load_base_model(name: str, device: torch.device | None = None) -> BaseModel:
  """Load name from the hub or a local directory onto the device, spread over every visible GPU when there are several."""
  device = device or default_device()
  num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
  device_map = "auto" if num_gpus > 1 else device
  print(f"Loading base model {name} (device map: {device_map}, visible GPUs: {num_gpus})...")
  tokenizer = AutoTokenizer.from_pretrained(name)
  dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
  module = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map)
  if ENABLE_GRADIENT_CHECKPOINTING:
    try:
      module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
      module.enable_input_require_grads()
      print("Gradient checkpointing and input require grads enabled.")
    except Exception as exc:
      print(f"Failed to enable gradient checkpointing: {exc}")
  print("Successfully loaded.")
  return BaseModel(name, module, tokenizer, device)
