"""LoRA tenants over one shared base module, through PEFT.

PEFT wraps a module tree in place, so every tenant gets a tree of its own
over the base's tensors and PEFT wraps that. Tenants then never switch
adapters, and the base module is never touched.
"""

import copy
import itertools

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModelForCausalLM, get_peft_model

from training.types import LoraConfig

LORA_TARGETS = {
  "train_attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
  # TODO: Revisit MLP targets for packed/MoE module names across supported backends.
  "train_mlp": ("gate_proj", "up_proj", "down_proj"),
  "train_unembed": ("lm_head",),
}


def own_tree(base: torch.nn.Module) -> torch.nn.Module:
  """A module tree of its own over base's tensors. Only the Python modules are copied; every parameter and buffer is shared."""
  return copy.deepcopy(base, memo={id(tensor): tensor for tensor in itertools.chain(base.parameters(), base.buffers())})


def target_modules(base: torch.nn.Module, config: LoraConfig) -> list[str]:
  wanted = {suffix for flag, suffixes in LORA_TARGETS.items() if getattr(config, flag) for suffix in suffixes}
  if not wanted:
    raise ValueError("At least one LoRA training target must be enabled.")
  # getattr because not every config defines tie_word_embeddings (transformers
  # v5 dropped it from the PretrainedConfig base class). When absent,
  # transformers performs no tying, so False matches the loaded model.
  if "lm_head" in wanted and getattr(base.config, "tie_word_embeddings", False):
    # An adapter on a tied lm_head shares the embedding tensor: PEFT warns,
    # merging would corrupt embed_tokens, and vLLM refuses lm_head adapter
    # weights for tied models. Keep every produced adapter vLLM-loadable.
    base_name = getattr(base, "name_or_path", "the base model")
    print(f"[LoRA] Ignoring train_unembed=True: {base_name} ties lm_head to embed_tokens, so vLLM could not load the adapter.")
    wanted.discard("lm_head")
  names = [name for name, module in base.named_modules() if name.rsplit(".", 1)[-1] in wanted and isinstance(module, torch.nn.Linear)]
  if not names:
    raise ValueError(f"No supported LoRA target modules found for suffixes: {sorted(wanted)}")
  return names


def lora_model(base: torch.nn.Module, model_id: str, config: LoraConfig) -> PeftModelForCausalLM:
  """A trainable adapter named model_id over base. Saved, PEFT nests it under <dir>/<model_id>/, the layout the samplers read."""
  peft_config = PeftLoraConfig(
    task_type="CAUSAL_LM",
    r=config.rank,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    bias="none",
    target_modules=target_modules(base, config),
  )
  return get_peft_model(own_tree(base), peft_config, adapter_name=model_id)
