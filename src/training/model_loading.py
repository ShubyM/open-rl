"""Model loading helpers shared by full-parameter trainers."""

from typing import Any


def load_text_causal_lm(model_name: str, **kwargs: Any):
  """Load only the language model from supported multimodal checkpoints."""
  from transformers import AutoConfig, AutoModelForCausalLM

  config = AutoConfig.from_pretrained(model_name)
  text_model_types = {"gemma3n", "gemma3n_text", "gemma4", "gemma4_text"}
  kwargs.setdefault("attn_implementation", "flex_attention" if config.model_type in text_model_types else "sdpa")
  if config.model_type in {"gemma3n", "gemma4"}:
    print(f"Loading text-only {config.text_config.model_type} weights from {model_name}")
    kwargs.update(
      config=config.text_config,
      key_mapping={r"^model\.language_model\.": "model."},
    )
  return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
