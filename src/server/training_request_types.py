"""Typed queue payloads consumed by the training request processor."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class BaseTrainingRequest(BaseModel):
  req_id: str
  model_id: str | None = None
  type: str
  trace_context: dict[str, str] = Field(default_factory=dict)


class CreateModelRequest(BaseTrainingRequest):
  type: Literal["create_model"]
  base_model: str
  lora_config: dict[str, Any] = Field(default_factory=dict)
  full_config: dict[str, Any] = Field(default_factory=dict)


class CreateModelFromStateRequest(BaseTrainingRequest):
  type: Literal["create_model_from_state"]
  state_path: str
  restore_optimizer: bool = False


class ForwardBackwardRequest(BaseTrainingRequest):
  type: Literal["forward_backward"]
  data: list[dict[str, Any]]
  loss_fn: str
  loss_config: dict[str, Any] | None = None


class OptimStepRequest(BaseTrainingRequest):
  type: Literal["optim_step"]
  adam_params: dict[str, Any]


class SampleRequest(BaseTrainingRequest):
  type: Literal["sample"]
  prompt_tokens: list[int]
  max_tokens: int
  num_samples: int
  temperature: float = 0.0
  prompt_logprobs: bool = False


class SaveStateRequest(BaseTrainingRequest):
  type: Literal["save_state"]
  state_path: str
  include_optimizer: bool = False
  kind: str = "state"


class LoadWeightsRequest(BaseTrainingRequest):
  type: Literal["load_weights"]
  state_path: str
  restore_optimizer: bool = False


class SaveWeightsForSamplerRequest(BaseTrainingRequest):
  type: Literal["save_weights_for_sampler"]
  alias: str | None = None
  path: str | None = None
  sampling_session_id: str | None = None


class SaveWeightsRequest(BaseTrainingRequest):
  type: Literal["save_weights"]
  alias: str | None = None


TRAINING_REQUEST_TYPES = {
  "create_model": CreateModelRequest,
  "create_model_from_state": CreateModelFromStateRequest,
  "forward_backward": ForwardBackwardRequest,
  "optim_step": OptimStepRequest,
  "sample": SampleRequest,
  "save_state": SaveStateRequest,
  "load_weights": LoadWeightsRequest,
  "save_weights_for_sampler": SaveWeightsForSamplerRequest,
  "save_weights": SaveWeightsRequest,
}


def parse_training_request(raw: dict[str, Any]) -> BaseTrainingRequest:
  request_type = raw.get("type")
  request_model = TRAINING_REQUEST_TYPES.get(request_type, BaseTrainingRequest)
  return request_model.model_validate(raw)
