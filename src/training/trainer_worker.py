# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from typing import Any

import torch
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses
from training.distributed import all_gather_object, all_reduce_max, all_reduce_sum, is_distributed, local_rank, rank, world_size

FILLER_DATUM_INDEX = -1


def shard_datum_indices(count: int, shard_rank: int, shard_count: int) -> list[int]:
  """Round-robin datum ownership for one data-parallel rank."""
  if shard_count <= 1:
    return list(range(count))
  return list(range(shard_rank, count, shard_count))


class TensorData(BaseModel):
  data: list[int] | list[float]


class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int]


class BaseTrainerWorker:
  # Whether .backward() itself runs cross-rank collectives. FSDP reduces
  # gradients during backward, so every rank must run the same number of
  # passes. A backend that defers the reduction to the optimizer step (Megatron
  # does) sets this False and skips the filler padding below.
  backward_runs_collectives = True

  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None

    if torch.cuda.is_available():
      self.device = torch.device("cuda", local_rank())
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  # Data-parallel geometry, as three overridable hooks.
  #
  # forward_backward assumes every rank is an independent data shard, which is
  # true for FSDP and for the single-process path. It is not true in general: a
  # tensor-parallel backend splits one model across several ranks, so those
  # ranks must be fed the *same* datums and their losses must not be summed
  # twice. A backend whose data-parallel group is a subset of the world
  # overrides these to shard and reduce over that subgroup instead.
  def shard_rank(self) -> int:
    return rank()

  def shard_count(self) -> int:
    return world_size() if is_distributed() else 1

  def shard_group(self) -> Any:
    """The process group the hooks above describe. None means the default one."""
    return None

  def forward_backward(self, model: PreTrainedModel, data: list[Datum], loss_fn: str, loss_config: dict | None = None) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs.

    Under distributed training each rank owns a round-robin shard of the datums
    and gradients are combined across ranks by the backend's own data-parallel
    reduction, which averages; each real pass scales its loss by the shard count
    to cancel that average and recover the single-process sum.
    """
    shard_count = self.shard_count()
    local_indices = shard_datum_indices(len(data), self.shard_rank(), shard_count)
    local_data = [data[idx] for idx in local_indices]

    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    model.train()

    local_batches = self.make_training_batches(local_data)
    if shard_count > 1 and self.backward_runs_collectives:
      # Pad short ranks with zero-scaled passes so collective counts match.
      filler_passes = all_reduce_max(len(local_batches), self.shard_group()) - len(local_batches)
      if filler_passes > 0:
        filler = local_data[0] if local_data else data[0]
        local_batches.extend([[(FILLER_DATUM_INDEX, filler)]] * filler_passes)

    for batch in local_batches:
      batch_positions = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]
      is_filler = batch_positions == [FILLER_DATUM_INDEX]
      batch_indices = [] if is_filler else [local_indices[position] for position in batch_positions]

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
      # The data-parallel reduction averages over the group, so scaling each
      # real pass by shard_count recovers the single-process sum. Filler passes
      # are scaled to zero and contribute nothing but a matching collective.
      backward_loss = loss * (0.0 if is_filler else float(shard_count)) if shard_count > 1 else loss
      backward_loss.backward()
      if not is_filler:
        total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

    if shard_count > 1:
      # Each rank only produced outputs for its own shard; the response owes the
      # caller one per datum, in the caller's order.
      total_loss = all_reduce_sum(total_loss, self.shard_group())
      for part in all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}, self.shard_group()):
        for idx, output in part.items():
          loss_fn_outputs[idx] = output

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
