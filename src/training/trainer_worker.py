# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from typing import Any

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses


class TensorData(BaseModel):
  data: list[int] | list[float]


class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int]


GIB = 2**30
# Left for the CUDA context, allocator fragmentation and kernel workspaces.
TOKEN_BUDGET_RESERVE_BYTES = 3 * GIB
# Share of what remains after weights, gradients and optimizer state that the
# activations of one forward/backward may take.
TOKEN_BUDGET_HEADROOM = 0.85


def estimate_train_token_budget(
  device_bytes: int,
  resident_bytes: int,
  vocab_size: int,
  hidden_size: int,
  num_layers: int,
  intermediate_size: int,
  dtype_bytes: int,
  gradient_checkpointing: bool,
) -> int:
  """Padded tokens one forward/backward can hold beside the resident weights, grads and optimizer state.

  Per token: the logits in the model dtype plus their fp32 copy for the loss,
  the activations kept between layers, and the live layer's working set. With
  gradient checkpointing one activation per layer is kept; without it every
  layer keeps its attention and MLP intermediates.
  """
  free = device_bytes - resident_bytes - TOKEN_BUDGET_RESERVE_BYTES
  if free <= 0:
    return 0
  logits = vocab_size * (dtype_bytes + 4)
  live_layer = (4 * hidden_size + 3 * intermediate_size) * dtype_bytes * 2
  if gradient_checkpointing:
    kept = num_layers * hidden_size * dtype_bytes
  else:
    kept = num_layers * (10 * hidden_size + 3 * intermediate_size) * dtype_bytes
  return int(free * TOKEN_BUDGET_HEADROOM // (logits + kept + live_layer))


class BaseTrainerWorker:
  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None
    # Padded tokens per forward/backward, derived from the device and the
    # loaded model by configure_token_budget. None means one example at a time.
    self.token_budget: int | None = None

    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  def forward_backward(
    self,
    model: PreTrainedModel,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None = None,
    forward_only: bool = False,
  ) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs.

    With ``forward_only`` (TrainingClient.forward) the pass runs without
    autograd and in eval mode, and no gradient is accumulated: the cookbook's
    NLL evaluator calls it on held-out data right before a training step, so
    a backward here would fold the test set into the next optim_step.
    """
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    if forward_only:
      model.eval()
    else:
      model.train()

    with torch.set_grad_enabled(not forward_only):
      total_loss = self._run_batches(model, data, loss_fn, loss_config, forward_only, loss_fn_outputs)
    return self._finish(data, loss_fn_outputs, total_loss)

  def _run_batches(
    self,
    model: PreTrainedModel,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None,
    forward_only: bool,
    loss_fn_outputs: list[dict[str, Any] | None],
  ) -> float:
    """Run every batch, fill ``loss_fn_outputs`` in place, and return the summed loss."""
    total_loss = 0.0
    batches = list(self.make_training_batches(data))
    while batches:
      batch = batches.pop(0)
      try:
        total_loss += self._run_batch(model, batch, loss_fn, loss_config, forward_only, loss_fn_outputs)
      except torch.cuda.OutOfMemoryError:
        if len(batch) <= 1 or not self.token_budget:
          raise
        # The estimate was optimistic for this model: halve it for good and
        # re-split what is left, this batch included.
        torch.cuda.empty_cache()
        self.token_budget = max(1, self.token_budget // 2)
        print(f"[TRAINER] Out of memory on a {len(batch)}-example batch; token budget now {self.token_budget}")
        remaining = [datum for _, datum in batch] + [datum for b in batches for _, datum in b]
        indices = [idx for idx, _ in batch] + [idx for b in batches for idx, _ in b]
        regrouped = self.make_training_batches(remaining)
        batches = [[(indices[local], datum) for local, datum in group] for group in regrouped]
    return total_loss

  def _run_batch(
    self,
    model: PreTrainedModel,
    batch: list[tuple[int, Datum]],
    loss_fn: str,
    loss_config: dict | None,
    forward_only: bool,
    loss_fn_outputs: list[dict[str, Any] | None],
  ) -> float:
    batch_indices = [idx for idx, _ in batch]
    batch_data = [datum for _, datum in batch]

    input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
    target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)
    target_logprobs = self.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)

    match loss_fn:
      case "cross_entropy":
        elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
      case "importance_sampling":
        old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
        advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
        elementwise_loss = losses.importance_sampling_loss(
          target_logprobs,
          weights,
          old_logprobs,
          advantages,
        )
      case "ppo":
        old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
        advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
        elementwise_loss = losses.ppo_loss(
          target_logprobs,
          weights,
          old_logprobs,
          advantages,
          loss_config,
        )
      case _:
        raise NotImplementedError(f"Loss {loss_fn} not supported")

    per_datum_loss = elementwise_loss.sum(dim=1)
    loss = per_datum_loss.sum()
    if not forward_only:
      loss.backward()

    detached_logprobs = target_logprobs.detach().cpu()
    for row, original_idx in enumerate(batch_indices):
      row_len = lengths[row]
      logprobs_list = detached_logprobs[row, :row_len].tolist()
      logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
      loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}
    return loss.item()

  def _finish(self, data: list[Datum], loss_fn_outputs: list[dict[str, Any] | None], total_loss: float) -> dict[str, Any]:
    mean_loss = total_loss / max(1, len(data))
    completed_loss_fn_outputs = []
    for output in loss_fn_outputs:
      if output is None:
        raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")
      completed_loss_fn_outputs.append(output)

    return {
      "metrics": {"loss:mean": self.sanitize_float(mean_loss), "loss:sum": self.sanitize_float(total_loss)},
      "loss_fn_outputs": completed_loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def configure_token_budget(
    self,
    model: PreTrainedModel,
    trainable_params: list[torch.nn.Parameter],
    gradient_checkpointing: bool,
    optimizer_states: int = 2,
  ) -> int | None:
    """Sizes the forward/backward token budget for this device and model. Returns it, or None off CUDA."""
    if self.device.type != "cuda":
      self.token_budget = None
      return None
    device_bytes = int(torch.cuda.get_device_properties(self.device).total_memory)
    weights = sum(p.numel() * p.element_size() for p in model.parameters() if p.device.type == "cuda")
    trainable = sum(p.numel() * p.element_size() for p in trainable_params)
    # Gradients in the parameter dtype, and AdamW's two moments in it as well.
    resident = weights + trainable * (1 + optimizer_states)
    cfg = model.config
    self.token_budget = estimate_train_token_budget(
      device_bytes=device_bytes,
      resident_bytes=resident,
      vocab_size=int(getattr(cfg, "vocab_size", 32000)),
      hidden_size=int(getattr(cfg, "hidden_size", 4096)),
      num_layers=int(getattr(cfg, "num_hidden_layers", 32)),
      intermediate_size=int(getattr(cfg, "intermediate_size", 4 * int(getattr(cfg, "hidden_size", 4096)))),
      dtype_bytes=max((p.element_size() for p in model.parameters()), default=2),
      gradient_checkpointing=gradient_checkpointing,
    )
    print(
      f"[TRAINER] Token budget per forward/backward: {self.token_budget} "
      f"(device {device_bytes / GIB:.0f} GiB, resident {resident / GIB:.1f} GiB, gradient checkpointing {gradient_checkpointing})"
    )
    return self.token_budget

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    """Group examples for the single padded forward/backward path."""
    if len(data) <= 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0")) or (self.token_budget or 0)

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

  def pad_sequences(
    self,
    sequences: list[list[int] | list[float]],
    lengths: list[int],
    dtype: torch.dtype,
    pad_value: int | float = 0,
  ) -> torch.Tensor:
    """Return padded values with shape [batch, max(lengths)]."""
    padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=self.device)
    for row, sequence in enumerate(sequences):
      length = lengths[row]
      padded[row, :length] = padded.new_tensor(sequence[:length])
    return padded

  def pad_model_inputs(
    self,
    data: list[Datum],
  ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return input_ids and attention_mask with shape [batch, max_input_len]."""
    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
    batch_size = len(data)
    input_lengths = [len(datum.model_input) for datum in data]
    max_input_len = max(input_lengths)

    input_ids = self.pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, pad_token_id)
    attention_mask = input_ids.new_zeros((batch_size, max_input_len))
    for row, input_len in enumerate(input_lengths):
      attention_mask[row, :input_len] = 1

    return input_ids, attention_mask, input_lengths

  def pad_targets_and_weights(
    self,
    data: list[Datum],
    input_lengths: list[int],
  ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
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

  def compute_target_logprobs(
    self,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return selected target logprobs with shape [batch, max_target_len]."""
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, : target_token_ids.shape[1], :]
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)

  def generate(
    self,
    model: PreTrainedModel,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    """Generate completions from model."""
    model.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
    prompt_logprobs = self.prompt_logprobs(model, input_tensor) if include_prompt_logprobs else None

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=None,
        top_k=None,
        num_return_sequences=num_samples,
        output_scores=True,
        return_dict_in_generate=True,
      )

    sequences_out = []
    for seq_idx in range(num_samples):
      gen_sequences = outputs.sequences[seq_idx]
      generated_tokens = gen_sequences[len(prompt_tokens) :].cpu().tolist()

      logprobs = []
      for token_step_idx in range(len(generated_tokens)):
        score_tensor = outputs.scores[token_step_idx]
        logprob_dist = torch.nn.functional.log_softmax(score_tensor[seq_idx], dim=-1)
        token_id = generated_tokens[token_step_idx]
        logprob = logprob_dist[token_id].item()
        logprobs.append(self.sanitize_float(logprob))

      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

    result = {"sequences": sequences_out}
    if prompt_logprobs is not None:
      result["prompt_logprobs"] = prompt_logprobs
    return result

  def prompt_logprobs(self, model: PreTrainedModel, input_tensor: torch.Tensor) -> list[float | None]:
    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = model(input_tensor, attention_mask=attention_mask)
      logprob_dist = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)

    prompt_tokens = input_tensor[0].tolist()
    prompt_logprobs: list[float | None] = [None]
    for token_idx, token_id in enumerate(prompt_tokens[1:]):
      logprob = logprob_dist[token_idx, token_id].item()
      prompt_logprobs.append(self.sanitize_float(logprob))

    return prompt_logprobs

  def sanitize_float(self, val: float) -> float:
    if math.isinf(val):
      return -9999.0 if val < 0 else 9999.0
    if math.isnan(val):
      return 0.0
    return val
