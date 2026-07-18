"""Typed values that cross HTTP, Redis, or worker-process boundaries."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TrainingOperation = Literal[
  "create_model",
  "create_model_from_state",
  "forward_backward",
  "load_weights",
  "optim_step",
  "sample",
  "save_state",
  "save_weights",
  "save_weights_for_sampler",
]


class TrainingCommand(BaseModel):
  model_config = ConfigDict(extra="forbid")

  request_id: str = Field(min_length=1)
  op: TrainingOperation
  payload: dict[str, Any] = Field(default_factory=dict)
  model_id: str | None = None
  trace_context: dict[str, str] = Field(default_factory=dict)


class CreateSessionRequest(BaseModel):
  # This is an external SDK boundary. Newer clients may add optional metadata.
  model_config = ConfigDict(extra="ignore")

  tags: list[str] = Field(default_factory=list)
  user_metadata: dict[str, Any] = Field(default_factory=dict)
  sdk_version: str | None = None
  project_id: str | None = None


class SessionHeartbeatRequest(BaseModel):
  model_config = ConfigDict(extra="ignore")

  session_id: str = Field(min_length=1)


class ClientSession(BaseModel):
  model_config = ConfigDict(extra="forbid")

  session_id: str
  created_at: float
  last_heartbeat: float
  tags: list[str] = Field(default_factory=list)
  user_metadata: dict[str, Any] = Field(default_factory=dict)
  sdk_version: str | None = None
  project_id: str | None = None


class SamplerSnapshot(BaseModel):
  model_config = ConfigDict(extra="forbid")

  sampling_session_id: str
  model_id: str
  revision: int = Field(ge=0)
  storage_path: str
  named: bool = False
  created_at: float
  expires_at: float | None = None
  in_flight: int = Field(default=0, ge=0)
