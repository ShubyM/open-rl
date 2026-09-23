"""The operations a training process serves, independent of a model library.

Backends own numerical state. The request loop owns queues and accelerator
leases. Creating/loading a model is an operation too: allocation happens only
after the process has acquired its resources.
"""

from typing import Any, Protocol, runtime_checkable

from training.types import Datum


class TrainingBackend(Protocol):
  def create_model(self, base_model_name: str, model_id: str, config: Any) -> None: ...

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]: ...

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]: ...

  def optim_step(self, adam_params: dict[str, Any], model_id: str) -> dict[str, Any]: ...

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]: ...

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]: ...

  def save_for_sampler(self, model_id: str, alias: str | None, ref: str | None) -> str | None: ...


@runtime_checkable
class Suspendable(Protocol):
  """Optional whole-worker suspension, invoked only between command batches.

  Sleep must quiesce device work and preserve the next operation's behavior,
  including accumulated gradients. Shared models and distributed ranks must
  be suspended together. Resident backends need not implement these methods.
  """

  def sleep(self) -> None: ...

  def wake_up(self) -> None: ...
