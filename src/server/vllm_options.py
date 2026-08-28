"""Engine and sampling options shared by the two vLLM sampler workers."""

import os


def text_only_engine_kwargs() -> dict:
  """Stop vLLM reserving an encoder cache OpenRL can never use.

  Text checkpoints published under a `*ForConditionalGeneration` architecture
  make vLLM profile a max-resolution image and video at startup and reserve a
  multi-GiB encoder cache before the KV cache is sized, which can OOM engine
  init. OpenRL only ever passes token ids, so that capacity is unreachable.
  `VLLM_ENABLE_MULTIMODAL=1` restores stock vLLM behaviour.
  """
  if os.getenv("VLLM_ENABLE_MULTIMODAL", "0") == "1":
    return {}
  return {"limit_mm_per_prompt": {"image": 0, "video": 0}}
