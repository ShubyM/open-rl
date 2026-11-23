"""Training configuration types shared by the API server, commands, and trainer workers.

Nothing here imports torch, so the API server can validate configs without loading a training stack.
"""

from typing import Literal

from pydantic import BaseModel


class LoraConfig(BaseModel):
  rank: int = 16
  seed: int | None = None
  lora_alpha: int = 16
  lora_dropout: float = 0.05
  train_attn: bool = True
  train_mlp: bool = True
  train_unembed: bool = False


class FFTConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True
  weight_sync_strategy: str | None = None


FineTuningType = Literal["lora", "full"]
# Trainers by name. Any other trainer_backend is an image.
TRAINER_BACKENDS = ("pytorch", "automodel")
