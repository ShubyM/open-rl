"""The trainer worker and the causal-LM step every model_id goes through.

The request processor calls one method per Tinker training command. A worker
holds one BaseModel and one TrainedWeights per model_id, and nothing else.
Parking those tensors, writing checkpoints and exporting to samplers are
functions over them in trained_weights.py, not worker methods.
"""

import math
import os
from collections.abc import Iterable
from typing import Any

import torch

from training import losses
from training.base_model import BaseModel, load_base_model, sanitize_float
from training.trained_weights import TrainedWeights, read, read_metadata
from training.types import Datum, LoraConfig


class TrainerWorker:
  def __init__(self, base: BaseModel | None = None):
    self.base = base
    self.models: dict[str, TrainedWeights] = {}

  def load_base_model(self, base_model_name: str) -> None:
    if self.base is not None and self.base.name == base_model_name:
      return
    self.base = load_base_model(base_model_name)

  def tensors(self) -> Iterable[torch.Tensor]:
    """Every tensor this worker keeps on the accelerator, for park and unpark."""
    if self.base is not None:
      yield from self.base.module.parameters()
      yield from self.base.module.buffers()
    for trained in self.models.values():
      yield from trained.tensors()

  def create_model(self, base_model_name: str, model_id: str, lora_config: LoraConfig | None = None, seed: int | None = None) -> None:
    """A fresh model_id on base_model_name. With a LoraConfig it is an adapter, without one it is the whole model."""
    self.load_base_model(base_model_name)
    if seed is not None:
      torch.manual_seed(seed)
    self.models[model_id] = TrainedWeights(self.base.new_weights(lora_config), lora_config)
    print(f"Created '{model_id}' on {base_model_name} ({f'LoRA rank {lora_config.rank}' if lora_config else 'full fine-tuning'}).")

  def load_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    """model_id from a checkpoint directory. Its metadata names the base model."""
    metadata = read_metadata(state_path)
    self.load_base_model(metadata["base_model"])
    loaded = read(state_path, metadata.get("model_id") or model_id, restore_optimizer)
    if loaded.lora_config is None:
      weights = self.base.new_weights(None)
      missing = weights.keys() - loaded.weights.keys()
      if missing:
        raise ValueError(f"{state_path} is missing {len(missing)} parameters, for example {sorted(missing)[0]}")
      with torch.no_grad():
        for name, weight in weights.items():
          weight.copy_(loaded.weights[name])
    else:
      weights = {name: tensor.to(self.base.weight_device(name)).requires_grad_() for name, tensor in loaded.weights.items()}
    trained = TrainedWeights(weights, loaded.lora_config)
    trained.adam_m = {name: tensor.to(weights[name].device) for name, tensor in loaded.adam_m.items()}
    trained.adam_v = {name: tensor.to(weights[name].device) for name, tensor in loaded.adam_v.items()}
    trained.adam_step = loaded.adam_step
    self.models[model_id] = trained
    print(f"Loaded state for '{model_id}' from {state_path}" + (" with its optimizer" if trained.adam_m else ""))
    return {"model_id": model_id, "base_model": metadata["base_model"]}

  # -- the step --------------------------------------------------------------------

  def forward_backward(self, model_id: str, data: list[Datum], loss_fn: str, loss_config: dict | None = None, forward_only: bool = False) -> dict[str, Any]:
    """Run the loss over data for model_id, summing gradients into its grads, and return Tinker-shaped outputs.

    With ``forward_only`` (TrainingClient.forward) the pass runs without
    autograd and in eval mode, and no gradient is accumulated: the cookbook's
    NLL evaluator calls it on held-out data right before a training step, so
    a backward here would fold the test set into the next optim_step.
    """
    trained = self.models[model_id]
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    if forward_only:
      self.base.module.eval()
    else:
      self.base.module.train()

    with self.base.using(trained), torch.set_grad_enabled(not forward_only):
      total_loss = self.run_batches(data, loss_fn, loss_config, forward_only, loss_fn_outputs)
    trained.grads = {name: weight.grad for name, weight in trained.weights.items() if weight.grad is not None}
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return self.finish(data, loss_fn_outputs, total_loss)

  def run_batches(self, data: list[Datum], loss_fn: str, loss_config: dict | None, forward_only: bool, loss_fn_outputs: list[dict[str, Any] | None]) -> float:
    """Run every batch, fill ``loss_fn_outputs`` in place, and return the summed loss."""
    total_loss = 0.0
    for batch in self.make_training_batches(data):
      batch_indices = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]

      input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
      target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)
      target_logprobs = self.base.token_logprobs(input_ids, attention_mask, target_token_ids)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.importance_sampling_loss(target_logprobs, weights, old_logprobs, advantages)
        case "ppo":
          old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.ppo_loss(target_logprobs, weights, old_logprobs, advantages, loss_config)
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      loss = elementwise_loss.sum(dim=1).sum()
      if not forward_only:
        loss.backward()
      total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        logprobs_list = detached_logprobs[row, : lengths[row]].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}
    return total_loss

  def finish(self, data: list[Datum], loss_fn_outputs: list[dict[str, Any] | None], total_loss: float) -> dict[str, Any]:
    mean_loss = total_loss / max(1, len(data))
    completed_loss_fn_outputs = []
    for output in loss_fn_outputs:
      if output is None:
        raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")
      completed_loss_fn_outputs.append(output)

    return {
      "metrics": {"loss:mean": sanitize_float(mean_loss), "loss:sum": sanitize_float(total_loss)},
      "loss_fn_outputs": completed_loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def optim_step(self, model_id: str, adam_params: dict[str, Any]) -> dict[str, Any]:
    """One AdamW step on model_id's weights from its summed grads, which it then empties.

    This is torch.optim.AdamW's single-tensor update written over the dicts,
    so the optimizer state is plain named tensors that park, save and compare
    like the weights do.
    """
    trained = self.models[model_id]
    lr = adam_params.get("learning_rate", 1e-4)
    beta1 = adam_params.get("beta1", 0.9)
    beta2 = adam_params.get("beta2", 0.95)
    eps = adam_params.get("eps", 1e-12)
    weight_decay = adam_params.get("weight_decay", 0.0)
    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf

    grads = [trained.grads[name] for name in trained.weights if name in trained.grads]
    total_norm = clip_grad_norm(grads, max_grad_norm)

    with torch.no_grad():
      trained.adam_step += 1
      step = trained.adam_step
      bias_correction1 = 1 - beta1**step
      bias_correction2_sqrt = math.sqrt(1 - beta2**step)
      step_size = lr / bias_correction1
      for name, weight in trained.weights.items():
        grad = trained.grads.get(name)
        if grad is None:
          continue
        m = trained.adam_m.setdefault(name, torch.zeros_like(weight))
        v = trained.adam_v.setdefault(name, torch.zeros_like(weight))
        weight.mul_(1 - lr * weight_decay)
        m.lerp_(grad, 1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        weight.addcdiv_(m, (v.sqrt() / bias_correction2_sqrt).add_(eps), value=-step_size)
        weight.grad = None
    trained.grads = {}

    return {"metrics": {"grad_norm:mean": sanitize_float(total_norm)}}

  def generate(
    self,
    model_id: str,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    with self.base.using(self.models[model_id]):
      return self.base.generate(prompt_tokens, max_tokens, num_samples, temperature, include_prompt_logprobs)

  # -- batching --------------------------------------------------------------------

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    """Group examples for the single padded forward/backward path."""
    if len(data) <= 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0"))

    if token_budget <= 0:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    ordered_data = sorted(enumerate(data), key=lambda item: len(item[1].model_input))
    batches: list[list[tuple[int, Datum]]] = []
    batch: list[tuple[int, Datum]] = []
    batch_max_len = 0

    for item in ordered_data:
      length = len(item[1].model_input)
      next_max_len = max(batch_max_len, length)
      next_size = len(batch) + 1
      over_token_budget = next_max_len * next_size > token_budget

      if batch and over_token_budget:
        batches.append(batch)
        batch = []
        batch_max_len = 0

      batch.append(item)
      batch_max_len = max(batch_max_len, length)

    if batch:
      batches.append(batch)

    return batches

  def pad_sequences(self, sequences: list[list[int] | list[float]], lengths: list[int], dtype: torch.dtype, pad_value: int | float = 0) -> torch.Tensor:
    """Return padded values with shape [batch, max(lengths)]."""
    padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=self.base.device)
    for row, sequence in enumerate(sequences):
      length = lengths[row]
      padded[row, :length] = padded.new_tensor(sequence[:length])
    return padded

  def pad_model_inputs(self, data: list[Datum]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return input_ids and attention_mask with shape [batch, max_input_len]."""
    batch_size = len(data)
    input_lengths = [len(datum.model_input) for datum in data]
    max_input_len = max(input_lengths)

    input_ids = self.pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, self.base.pad_token_id)
    attention_mask = input_ids.new_zeros((batch_size, max_input_len))
    for row, input_len in enumerate(input_lengths):
      attention_mask[row, :input_len] = 1

    return input_ids, attention_mask, input_lengths

  def pad_targets_and_weights(self, data: list[Datum], input_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return target_token_ids and weights with shape [batch, max_target_len]."""
    batch_size = len(data)
    target_lengths = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in data]
    lengths = [min(input_lengths[row], target_lengths[row]) for row in range(batch_size)]
    target_token_ids = self.pad_sequences([datum.loss_fn_inputs["target_tokens"].data for datum in data], lengths, torch.long)
    weight_sequences = [
      datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row] for row, datum in enumerate(data)
    ]
    weights = self.pad_sequences(weight_sequences, lengths, torch.float32)

    return target_token_ids, weights, lengths


def clip_grad_norm(grads: list[torch.Tensor], max_norm: float) -> float:
  """Scale grads in place so their global norm is at most max_norm, and return the norm they had. Matches torch.nn.utils.clip_grad_norm_."""
  if not grads:
    return 0.0
  with torch.no_grad():
    total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for grad in grads]))
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for grad in grads:
      grad.mul_(clip_coef.to(grad.device))
  return float(total_norm)
