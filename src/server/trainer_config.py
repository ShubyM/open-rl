"""Trainer selection shared by the gateway, queue processor, and samplers."""

import os


def trainer_backend() -> str:
  """Prefer the explicit backend; retain OPEN_RL_ENABLE_FFT for older deployments."""
  backend = os.getenv("OPEN_RL_TRAINER_BACKEND", "").strip().lower()
  if not backend:
    return "fft" if os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true" else "lora"
  if backend not in {"lora", "fft", "megatron", "automodel"}:
    raise RuntimeError(f"Unknown OPEN_RL_TRAINER_BACKEND={backend!r}; expected lora, fft, megatron, or automodel")
  return backend


def uses_dedicated_trainer() -> bool:
  """These backends load models from requests in a separate training process."""
  return trainer_backend() != "lora"


def trainer_publishes_adapter() -> bool:
  backend = trainer_backend()
  if backend in {"lora", "megatron"}:
    return True
  return backend == "automodel" and int(os.getenv("OPEN_RL_AUTOMODEL_LORA_RANK", "0")) > 0


def sampler_uses_full_weights() -> bool:
  return not trainer_publishes_adapter()
