"""OpenRL model/session `openrl.*` settings schema, validation, and runtime mode helpers."""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from training.types import TRAINER_BACKENDS, FineTuningType

SETTINGS_PREFIX = "openrl."
USER_METADATA_DOC = "Free-form client metadata. Keys starting with openrl. are OpenRL settings (see components.schemas.OpenRLSettings)."


def parse_bool(value: Any) -> bool:
  if isinstance(value, str) and value.lower() in ("true", "false"):
    return value.lower() == "true"
  if not isinstance(value, bool):
    raise ValueError(f"must be 'true' or 'false', got {value!r}")
  return value


def parse_trainer_backend(value: str) -> str:
  if value not in TRAINER_BACKENDS and "/" not in value:
    raise ValueError(f"must be one of {', '.join(TRAINER_BACKENDS)} or an image like ghcr.io/org/trainer:tag, got {value!r}")
  return value


class Settings(BaseModel):
  """Client-chosen `openrl.*` model settings published in `/openapi.json#/components/schemas/OpenRLSettings`."""

  model_config = ConfigDict(extra="forbid", alias_generator=lambda f: f"{SETTINGS_PREFIX}{f}", populate_by_name=True)

  sampler_replicas: int = Field(default=1, ge=1, le=8, description="Single-GPU samplers that drain the model's sampling queue together.")
  exclusive: Annotated[bool, BeforeValidator(parse_bool)] = Field(default=False, description="Dedicated workload; nothing time-slices its GPUs.")
  trainer_backend: Annotated[str, AfterValidator(parse_trainer_backend)] = Field(
    default="pytorch", description="Trainer backend: pytorch, automodel (LoRA only), or a container image ref."
  )
  trainer_device_count: int = Field(
    default=1,
    ge=1,
    le=8,
    validation_alias=AliasChoices(f"{SETTINGS_PREFIX}trainer_device_count", f"{SETTINGS_PREFIX}trainer_gpus"),
    description="Devices per data-parallel torchrun group (>1 requires automodel; openrl.trainer_gpus is deprecated).",
  )
  trainer_cp: int = Field(default=1, ge=1, le=8, description="Context-parallel devices per sequence (must divide trainer_device_count).")
  fine_tuning_type: FineTuningType | None = Field(default=None, description="lora unless set; restored models inherit checkpoint type.")
  recipe: Literal["rl", "sft"] | None = Field(default=None, description="rl starts sampler with trainer; sft/unset starts on demand.")
  client_log_id: str | None = Field(default=None, description="Label openrl_client=<id> used by inspection API to read client logs.")
  weight_sync: Literal["delta", "full"] | None = Field(default=None, description="Full fine-tuning only: delta (default) or full weight sync.")
  fft_seed: int | None = Field(default=None, description="Full fine-tuning only: trainer random seed.")
  fft_cpu_offload: Annotated[bool | None, BeforeValidator(parse_bool)] = Field(
    default=None, description="Full fine-tuning only: park trainer state in host RAM between turns (default false when exclusive)."
  )


def tag_metadata(tags: list[str]) -> dict[str, str]:
  entries: dict[str, str] = {}
  for tag in tags:
    if tag.startswith(SETTINGS_PREFIX):
      key, sep, value = tag.partition("=")
      if not sep:
        raise ValueError(f"tag {tag!r} needs a value, like {key}=true")
      entries[key] = value
  return entries


def resolve_settings(*user_metadata: dict[str, Any]) -> Settings:
  merged: dict[str, Any] = {}
  for layer in reversed(user_metadata):
    merged.update({k: v for k, v in (layer or {}).items() if k.startswith(SETTINGS_PREFIX)})
  old, new = f"{SETTINGS_PREFIX}trainer_gpus", f"{SETTINGS_PREFIX}trainer_device_count"
  if old in merged and new in merged:
    if str(merged[old]) != str(merged[new]):
      raise ValueError(f"{new}={merged[new]} and {old}={merged[old]} disagree; {old} is the deprecated name of {new}, set only one")
    del merged[old]
  try:
    return Settings.model_validate(merged)
  except ValidationError as exc:
    msgs = [f"{e['loc'][0]}: {'unknown setting' if e['type'] == 'extra_forbidden' else e['msg'].removeprefix('Value error, ')}" for e in exc.errors()]
    raise ValueError("; ".join(msgs)) from None


def env_int(name: str) -> int | None:
  return int(v) if (v := os.getenv(name)) and v.isdigit() else None


def is_single_process_mode() -> bool:
  return bool(os.getenv("BASE_MODEL")) and not bool(os.getenv("REDIS_URL"))


def get_sampler_backend() -> str:
  return b.lower() if (b := os.getenv("SAMPLING_BACKEND")) else ("torch" if is_single_process_mode() else "vllm")


def is_fft_enabled() -> bool:
  return os.getenv("OPEN_RL_ENABLE_FFT", "").lower() == "true"


def check_sampler_backend() -> None:
  if get_sampler_backend() == "vllm" and not os.getenv("REDIS_URL"):
    raise RuntimeError(
      "Sampling pools need REDIS_URL for vLLM sampling streams. Set REDIS_URL, or set BASE_MODEL and unset SAMPLING_BACKEND for single-process torch."
    )


def check_lora_jobs_per_runtime() -> None:
  if (v := os.getenv("OPEN_RL_LORA_JOBS_PER_RUNTIME", "")) and not v.isdigit():
    raise RuntimeError(f"OPEN_RL_LORA_JOBS_PER_RUNTIME is {v!r}; it must be 0 (one runtime) or a positive number of jobs")
