"""Sizes a worker from its model's parameter count."""

import logging
import re
from dataclasses import dataclass

GIB = 1024**3

logger = logging.getLogger(__name__)

# Raw weights. gemma-4-e2b names an effective 2B but holds ~5B.
MODEL_TO_PARAM_COUNT: dict[str, int] = {
  "qwen2.5-0.5b": 494_000_000,
  "qwen3-0.6b": 596_049_920,
  "qwen2.5-1.5b": 1_540_000_000,
  "qwen3-1.7b": 1_720_000_000,
  "qwen3-4b": 4_020_000_000,
  "qwen2.5-7b": 7_620_000_000,
  "qwen3-8b": 8_190_000_000,
  "qwen3.5-9b": 9_000_000_000,
  "qwen3.5-27b": 27_000_000_000,
  "gemma-3-1b": 1_000_000_000,
  "gemma-4-e2b": 5_440_000_000,
  "gemma-4-e4b": 8_000_000_000,
}
UNKNOWN_MODEL_PARAMS = 8_000_000_000  # unknown models are sized large, not small
VARIANT_SUFFIXES = ("-instruct", "-it", "-pt", "-base", "-chat")


def normalize_model_id(base_model: str) -> str:
  name = (base_model or "").strip().lower().rsplit("/", 1)[-1]
  while True:
    stripped = re.sub(r"-\d{3,}$", "", name)
    for suffix in VARIANT_SUFFIXES:
      if stripped.endswith(suffix):
        stripped = stripped[: -len(suffix)]
    if stripped == name:
      return name
    name = stripped


def parameter_count(base_model: str) -> int | None:
  return MODEL_TO_PARAM_COUNT.get(normalize_model_id(base_model))


# On the device: fft trainer 8 B/param (bf16 weights + grads + fp32 master),
# frozen base 2 B/param; plus activations (trainer) or KV cache (sampler).
DEVICE_BYTES_PER_PARAM = {("full", "trainer"): 8, ("lora", "trainer"): 2, ("full", "sampler"): 2, ("lora", "sampler"): 2}
DEVICE_RESERVE_BYTES = {"trainer": 4 * GIB, "sampler": 6 * GIB}
# Parked in host memory: fft trainer 12 B/param + a weight copy in flight;
# plus process overhead. Measured: 0.5B trainer 28Gi, sampler 20Gi; 7B FFT
# trainer OOM-killed at 110Gi.
HOST_BYTES_PER_PARAM = {("full", "trainer"): 14, ("lora", "trainer"): 2, ("full", "sampler"): 2, ("lora", "sampler"): 2}
HOST_OVERHEAD_BYTES = {"trainer": 20 * GIB, "sampler": 18 * GIB}
HOST_LIMIT_FACTOR = 1.5


def gib(n: int) -> str:
  return f"{-(-n // GIB)}Gi"


@dataclass(frozen=True)
class Footprint:
  accelerator_bytes: int
  host_request_bytes: int
  host_limit_bytes: int

  @property
  def accelerator(self) -> str:
    return gib(self.accelerator_bytes)

  @property
  def resources(self) -> dict:
    return {"requests": {"memory": gib(self.host_request_bytes)}, "limits": {"memory": gib(self.host_limit_bytes)}}


def footprint(base_model: str, fine_tuning_type: str, role: str) -> Footprint:
  params = parameter_count(base_model)
  if params is None:
    logger.warning("No known parameter count for %r; sizing it as %.0fB.", base_model, UNKNOWN_MODEL_PARAMS / 1e9)
    params = UNKNOWN_MODEL_PARAMS
  kind = "lora" if fine_tuning_type == "lora" else "full"
  device = params * DEVICE_BYTES_PER_PARAM[(kind, role)] + DEVICE_RESERVE_BYTES[role]
  host = params * HOST_BYTES_PER_PARAM[(kind, role)] + HOST_OVERHEAD_BYTES[role]
  return Footprint(device, host, int(host * HOST_LIMIT_FACTOR))


# -- tiers: used only by the Kubernetes launcher; removed with it -----------------
FFT_VRAM_BYTES_PER_PARAM = 12
TIER_24GB_FFT_BUDGET_BYTES = 20 * GIB
LORA_VRAM_BYTES_PER_PARAM = 2
TIER_24GB_LORA_BUDGET_BYTES = 13 * GIB


def estimate_memory_tier(base_model: str, fine_tuning_type: str = "lora") -> str:
  params = parameter_count(base_model)
  if params is None:
    logger.warning("No known parameter count for %r; using the 80gb tier.", base_model)
    return "80gb"
  if fine_tuning_type == "full":
    return "24gb" if params * FFT_VRAM_BYTES_PER_PARAM <= TIER_24GB_FFT_BUDGET_BYTES else "80gb"
  return "24gb" if params * LORA_VRAM_BYTES_PER_PARAM <= TIER_24GB_LORA_BUDGET_BYTES else "80gb"
