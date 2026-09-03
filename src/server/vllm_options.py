"""Engine and sampling options shared by the two vLLM sampler workers."""

import os
from collections.abc import Sequence


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


def split_stop(stop: str | Sequence[str] | Sequence[int] | None) -> tuple[list[str] | None, list[int] | None]:
  """Split a sampling request's `stop` into vLLM's `stop` and `stop_token_ids`.

  The sampling API types stop as `str | Sequence[str] | Sequence[int]` and which
  one arrives depends on the client's renderer. vLLM keeps strings and token ids
  in separate arguments and rejects a string in `stop_token_ids`. Each half is
  None when empty so callers do not override a vLLM default with [].
  """
  if stop is None:
    return None, None
  if isinstance(stop, str):
    return [stop], None
  strings = [s for s in stop if isinstance(s, str)]
  # bool is an int subclass; a stray True would be read as token id 1.
  token_ids = [t for t in stop if isinstance(t, int) and not isinstance(t, bool)]
  return (strings or None), (token_ids or None)


def gpu_memory_utilization(device_bytes: int | None = None) -> float:
  """vLLM's share of the device. An explicit VLLM_GPU_MEMORY_UTILIZATION wins;
  otherwise the budget the launcher sized this worker for
  (OPEN_RL_ACCELERATOR_MEMORY, in bytes) over the device actually present,
  capped at 0.90; with neither, 0.90."""
  explicit = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
  if explicit:
    return float(explicit)
  budget = os.getenv("OPEN_RL_ACCELERATOR_MEMORY")
  if not budget:
    return 0.90
  if device_bytes is None:
    import torch

    device_bytes = torch.cuda.mem_get_info()[1]
  return min(int(budget) / device_bytes, 0.90)
