"""Model loading helpers shared by full-parameter trainers."""

from typing import Any

TEXT_FROM_MULTIMODAL_KEY_MAPPING = {r"^model\.language_model\.": "model."}


def load_text_causal_lm(model_name: str, **kwargs: Any):
  """Load only the language model from supported multimodal checkpoints.

  Also handles the trainer's own text-only checkpoints: transformers reverses
  an explicit ``key_mapping`` at ``save_pretrained`` time, so checkpoints saved
  by a model loaded through the multimodal branch carry hub-layout
  (``model.language_model.*``) keys even though their config is text-only.
  Loading those without the mapping silently reinitializes every parameter —
  a resumed run then trains and samples a random model. Passing the mapping in
  the text-only branch makes both hub-layout and flat checkpoints load, and the
  missing-keys guard turns any future layout drift into a loud failure.
  """
  from transformers import AutoConfig, AutoModelForCausalLM

  config = AutoConfig.from_pretrained(model_name)
  text_model_types = {"gemma3n", "gemma3n_text", "gemma4", "gemma4_text"}
  kwargs.setdefault("attn_implementation", "flex_attention" if config.model_type in text_model_types else "sdpa")
  if config.model_type in {"gemma3n", "gemma4"}:
    print(f"Loading text-only {config.text_config.model_type} weights from {model_name}")
    kwargs.update(
      config=config.text_config,
      key_mapping=TEXT_FROM_MULTIMODAL_KEY_MAPPING,
    )
  elif config.model_type in {"gemma3n_text", "gemma4_text"}:
    kwargs.setdefault("key_mapping", TEXT_FROM_MULTIMODAL_KEY_MAPPING)

  model, loading_info = AutoModelForCausalLM.from_pretrained(model_name, output_loading_info=True, **kwargs)
  missing = list(loading_info.get("missing_keys") or [])
  if missing:
    raise RuntimeError(
      f"Loading {model_name} left {len(missing)} parameters uninitialized (e.g. {sorted(missing)[:3]}). "
      "The checkpoint's key layout does not match the model; refusing to train or serve a partially random model."
    )
  return model
