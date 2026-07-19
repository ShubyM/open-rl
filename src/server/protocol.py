from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CreateSessionRequest(BaseModel):
  model_config = ConfigDict(extra="ignore")

  tags: list[str] = Field(default_factory=list)
  user_metadata: dict[str, Any] | None = None
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
  user_metadata: dict[str, Any] | None = None
  sdk_version: str | None = None
  project_id: str | None = None
