# Shared HF numerical operations; models and optimizers belong to concrete workers.

import math
import time
from typing import Any

import torch

from training import losses
from training.types import Datum


def default_device() -> torch.device:
  """Choose the worker's device at construction, outside trainer math."""
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def sanitize_float(val: float) -> float:
  if math.isinf(val):
    return -9999.0 if val < 0 else 9999.0
  if math.isnan(val):
    return 0.0
  return val


def build_optimizer(params: list[torch.nn.Parameter], adam_params: dict[str, Any]) -> torch.optim.Optimizer:
  return torch.optim.AdamW(
    params,
    lr=adam_params.get("learning_rate", 1e-4),
    betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
    eps=adam_params.get("eps", 1e-12),
    weight_decay=adam_params.get("weight_decay", 0.0),
  )


def optim_step(optimizer: torch.optim.Optimizer, adam_params: dict[str, Any]) -> dict[str, float]:
  """Step and clear the accumulated gradients of this optimizer's parameters."""
  params = [param for group in optimizer.param_groups for param in group["params"]]
  if (learning_rate := adam_params.get("learning_rate")) is not None:
    for group in optimizer.param_groups:
      group["lr"] = learning_rate
  max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
  if max_grad_norm <= 0.0:
    max_grad_norm = math.inf

  t0 = time.perf_counter()
  total_norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
  t1 = time.perf_counter()
  optimizer.step()
  optimizer.zero_grad()
  t2 = time.perf_counter()
  return {
    "grad_norm:mean": sanitize_float(total_norm.item()),
    "time/optimizer_step": sanitize_float(t2 - t1),
    "time/clip_grad_norm": sanitize_float(t1 - t0),
  }


def forward_backward(
  model: torch.nn.Module,
  data: list[Datum],
  loss_fn: str,
  loss_config: dict | None = None,
  forward_only: bool = False,
  *,
  tokenizer: Any = None,
  device: torch.device | str = "cpu",
  token_budget: int = 0,
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
    total_loss = _run_batches(
      model, data, loss_fn, loss_config, forward_only, loss_fn_outputs, tokenizer=tokenizer, device=device, token_budget=token_budget
    )
  return _finish(data, loss_fn_outputs, total_loss)


def _run_batches(
  model: torch.nn.Module,
  data: list[Datum],
  loss_fn: str,
  loss_config: dict | None,
  forward_only: bool,
  loss_fn_outputs: list[dict[str, Any] | None],
  *,
  tokenizer: Any,
  device: torch.device | str,
  token_budget: int,
) -> float:
  """Run every batch, fill ``loss_fn_outputs`` in place, and return the summed loss."""
  total_loss = 0.0
  for batch in make_training_batches(data, token_budget):
    batch_indices = [idx for idx, _ in batch]
    batch_data = [datum for _, datum in batch]

    input_ids, attention_mask, input_lengths = pad_model_inputs(batch_data, tokenizer=tokenizer, device=device)
    target_token_ids, weights, lengths = pad_targets_and_weights(batch_data, input_lengths, device=device)
    target_logprobs = compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)

    match loss_fn:
      case "cross_entropy":
        elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
      case "importance_sampling":
        old_logprobs = pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32, device=device)
        advantages = pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32, device=device)
        elementwise_loss = losses.importance_sampling_loss(
          target_logprobs,
          weights,
          old_logprobs,
          advantages,
        )
      case "ppo":
        old_logprobs = pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32, device=device)
        advantages = pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32, device=device)
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
    total_loss += loss.item()

    detached_logprobs = target_logprobs.detach().cpu()
    for row, original_idx in enumerate(batch_indices):
      row_len = lengths[row]
      logprobs_list = detached_logprobs[row, :row_len].tolist()
      logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
      loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}
  return total_loss


def _finish(data: list[Datum], loss_fn_outputs: list[dict[str, Any] | None], total_loss: float) -> dict[str, Any]:
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


def make_training_batches(data: list[Datum], token_budget: int = 0) -> list[list[tuple[int, Datum]]]:
  """Group examples for the single padded forward/backward path."""
  if len(data) <= 1:
    return [[(idx, datum)] for idx, datum in enumerate(data)]

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
  sequences: list[list[int] | list[float]],
  lengths: list[int],
  dtype: torch.dtype,
  pad_value: int | float = 0,
  *,
  device: torch.device | str = "cpu",
) -> torch.Tensor:
  """Return padded values with shape [batch, max(lengths)]."""
  padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=device)
  for row, sequence in enumerate(sequences):
    length = lengths[row]
    padded[row, :length] = padded.new_tensor(sequence[:length])
  return padded


def pad_model_inputs(
  data: list[Datum],
  *,
  tokenizer: Any = None,
  device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
  """Return input_ids and attention_mask with shape [batch, max_input_len]."""
  pad_token_id = tokenizer.pad_token_id if tokenizer and tokenizer.pad_token_id is not None else 0
  batch_size = len(data)
  input_lengths = [len(datum.model_input) for datum in data]
  max_input_len = max(input_lengths)

  input_ids = pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, pad_token_id, device=device)
  attention_mask = input_ids.new_zeros((batch_size, max_input_len))
  for row, input_len in enumerate(input_lengths):
    attention_mask[row, :input_len] = 1

  return input_ids, attention_mask, input_lengths


def pad_targets_and_weights(
  data: list[Datum],
  input_lengths: list[int],
  *,
  device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
  """Return target_token_ids and weights with shape [batch, max_target_len]."""
  batch_size = len(data)
  target_lengths = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in data]
  lengths = [min(input_lengths[row], target_lengths[row]) for row in range(batch_size)]
  target_token_ids = pad_sequences([datum.loss_fn_inputs["target_tokens"].data for datum in data], lengths, torch.long, device=device)
  weight_sequences = [
    datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row] for row, datum in enumerate(data)
  ]
  weights = pad_sequences(weight_sequences, lengths, torch.float32, device=device)

  return target_token_ids, weights, lengths


def compute_target_logprobs(
  model: torch.nn.Module,
  input_ids: torch.Tensor,
  attention_mask: torch.Tensor,
  target_token_ids: torch.Tensor,
) -> torch.Tensor:
  """Return selected target logprobs with shape [batch, max_target_len]."""
  outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
  logits = outputs.logits[:, : target_token_ids.shape[1], :]
  return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)


def generate(
  model: torch.nn.Module,
  prompt_tokens: list[int],
  max_tokens: int,
  num_samples: int = 1,
  temperature: float = 0.0,
  include_prompt_logprobs: bool = False,
  *,
  tokenizer: Any,
  device: torch.device | str = "cpu",
) -> dict[str, Any]:
  """Generate completions from model."""
  model.eval()

  input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
  do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
  prompt_scores = prompt_logprobs(model, input_tensor) if include_prompt_logprobs else None

  with torch.no_grad():
    attention_mask = torch.ones_like(input_tensor)
    outputs = model.generate(
      input_tensor,
      attention_mask=attention_mask,
      max_new_tokens=max_tokens,
      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
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
      logprobs.append(sanitize_float(logprob))

    sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

  result = {"sequences": sequences_out}
  if prompt_scores is not None:
    result["prompt_logprobs"] = prompt_scores
  return result


def prompt_logprobs(model: torch.nn.Module, input_tensor: torch.Tensor) -> list[float | None]:
  with torch.no_grad():
    attention_mask = torch.ones_like(input_tensor)
    outputs = model(input_tensor, attention_mask=attention_mask)
    logprob_dist = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)

  prompt_tokens = input_tensor[0].tolist()
  prompt_logprobs: list[float | None] = [None]
  for token_idx, token_id in enumerate(prompt_tokens[1:]):
    logprob = logprob_dist[token_idx, token_id].item()
    prompt_logprobs.append(sanitize_float(logprob))

  return prompt_logprobs
