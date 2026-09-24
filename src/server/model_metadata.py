import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from server.store import StateStore
from training.types import FFTConfig, FineTuningType, LoraConfig

SPARSE_DELTA_VERSION = 2


@dataclass
class WeightSyncConfig:
  """How the trainer exports weights: sparse deltas or full checkpoints.
  The sampler reads the file format and applies either."""

  strategy: str = "delta"

  @classmethod
  def from_env(cls, env: Mapping[str, str] | None = None) -> "WeightSyncConfig":
    """Reconstruct WeightSyncConfig dataclass from environment variables inside a worker process."""
    get_val = (os.environ if env is None else env).get

    strategy = (get_val("OPEN_RL_WEIGHT_SYNC_STRATEGY") or "delta").lower()
    if strategy not in ("delta", "full"):
      strategy = "delta"
    return cls(strategy=strategy)


def extract_weight_sync_config(headers: Mapping[str, str] | None = None) -> WeightSyncConfig:
  """Extract and normalize WeightSyncConfig from HTTP headers with single-location defaults."""
  if not headers:
    return WeightSyncConfig()

  strategy = (headers.get("x-open-rl-weight-sync-strategy") or "delta").lower()
  if strategy not in ("delta", "full"):
    strategy = "delta"
  return WeightSyncConfig(strategy=strategy)


class TrainingModelMetadata(BaseModel):
  # Preserve fields written by other server versions when updating a record.
  model_config = ConfigDict(extra="allow")

  base_model: str
  created_at: float = 0.0
  fine_tuning_type: FineTuningType = "lora"
  weight_sync_config: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
  full_config: FFTConfig = Field(default_factory=FFTConfig)
  lora_config: LoraConfig = Field(default_factory=LoraConfig)
  status: str = "active"
  updated_at: float = 0.0
  completed_at: float | None = None


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
