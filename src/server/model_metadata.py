import json
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from server.store import StateStore
from training.types import FFTConfig, FineTuningType, LoraConfig

SPARSE_DELTA_VERSION = 2


@dataclass
class WeightSyncConfig:
  """How the trainer publishes weights each step: sparse deltas or full checkpoints.
  The sampler reads the file format and applies either."""

  strategy: str = "delta"

  @classmethod
  def from_env(cls, env: Any = None) -> "WeightSyncConfig":
    """Reconstruct WeightSyncConfig dataclass from environment variables inside a worker process."""
    get_val = (env.get if hasattr(env, "get") else None) or os.getenv

    strategy = (get_val("OPEN_RL_WEIGHT_SYNC_STRATEGY") or "delta").lower()
    if strategy not in ("delta", "full"):
      strategy = "delta"
    return cls(strategy=strategy)


def extract_weight_sync_config(headers: Any = None) -> WeightSyncConfig:
  """Extract and normalize WeightSyncConfig from HTTP headers with single-location defaults."""
  if not headers:
    return WeightSyncConfig()

  get_header = headers.get if hasattr(headers, "get") else (lambda k, default=None: default)

  strategy = (get_header("x-open-rl-weight-sync-strategy") or "delta").lower()
  if strategy not in ("delta", "full"):
    strategy = "delta"
  return WeightSyncConfig(strategy=strategy)


# The user_metadata key asking for an exclusive workload, one no other
# workload shares: the model gets its own trainer and sampler, and nothing
# time-slices their GPUs, so any workload runs on them as it is.
# Prefixed because the client owns user_metadata and puts its own keys there.
EXCLUSIVE_KEY = "openrl.exclusive"


def resolve_exclusive(*user_metadata: dict[str, Any]) -> bool:
  """Whether the model asked for exclusive GPUs, per model then per session.
  The SDK types user_metadata values as strings, so "true" and "false" count."""
  for source in user_metadata:
    if (value := (source or {}).get(EXCLUSIVE_KEY)) is not None:
      if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
      if not isinstance(value, bool):
        raise ValueError(f"{EXCLUSIVE_KEY} must be 'true' or 'false', got {value!r}")
      return value
  return False


class TrainingModelMetadata(BaseModel):
  # Preserve fields written by other server versions when updating a record.
  model_config = ConfigDict(extra="allow")

  base_model: str
  created_at: float = 0.0
  fine_tuning_type: FineTuningType = "lora"
  weight_sync_config: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  exclusive: bool = False
  status: str = "active"
  updated_at: float = 0.0
  completed_at: float | None = None

  def shares_gpu(self) -> bool:
    """Whether other workers may time-slice this model's GPUs. FFT workers
    suspend between turns. LoRA workers cannot, so their GPUs are never shared."""
    return self.fine_tuning_type != "lora" and not self.exclusive

  def shares_runtime(self) -> bool:
    """Whether other models may run in this model's trainer and sampler. A LoRA
    runtime serves every adapter on its base model. An FFT runtime serves one model."""
    return self.fine_tuning_type == "lora" and not self.exclusive

  def runtime(self, model_id: str) -> str:
    """The id of the trainer and sampler pair that serves this model."""
    return self.base_model if self.shares_runtime() else model_id


def decode_model_metadata(raw: str | None) -> TrainingModelMetadata | None:
  if raw is None:
    return None
  data = json.loads(raw)
  if not isinstance(data, dict):
    raise ValueError("Model metadata must be a JSON object")
  # Older restores used a placeholder kind and could omit the base model.
  # Normalize that persisted format here; new creates resolve the checkpoint.
  if data.get("fine_tuning_type") == "restored":
    data["fine_tuning_type"] = "lora"
    data["base_model"] = data.get("base_model") or ""
  for key in ("full_config", "lora_config", "weight_sync_config"):
    if data.get(key) is None:
      data[key] = {}
  return TrainingModelMetadata.model_validate(data)


async def get_model_metadata(state: StateStore, model_id: str) -> TrainingModelMetadata | None:
  return decode_model_metadata(await state.get_value(f"open_rl:model_meta:{model_id}"))


async def persist_model_metadata(state: StateStore, model_id: str, metadata: TrainingModelMetadata) -> None:
  await state.set_value(f"open_rl:model_meta:{model_id}", metadata.model_dump_json())
