from __future__ import annotations

from pydantic import BaseModel


class StateMetadata(BaseModel):
  base_model: str
  created_at: str | None = None
  has_optimizer: bool = False
  is_lora: bool = True
  kind: str = "state"
  lora_rank: int = 16
  model_id: str | None = None
  timestamp: float | None = None
  train_attn: bool = True
  train_mlp: bool = True
  train_unembed: bool = True
