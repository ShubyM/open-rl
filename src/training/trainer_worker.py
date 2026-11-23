# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from server import telemetry
from server.api import wire
from server.api.wire import Datum as Datum
from training import losses
from training.distributed import local_rank

# Position marker for a filler pass, see _run_batches.
FILLER_DATUM_INDEX = -1
# Largest share of a training pass that may be padding. Every sequence in a
# pass is padded to the longest one, so a long sequence added to short ones
# multiplies the work.
MAX_PADDING_SHARE = 0.25


def shard_datum_indices(count: int, shard_rank: int, shard_count: int) -> list[int]:
  """Round-robin datum ownership for one data-parallel rank."""
  return list(range(shard_rank, count, shard_count))


def chunk_target_logprobs(head: torch.nn.Module, softcap: float | None, hidden: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
  """logit[target] - logsumexp(logits) for one chunk of positions."""
  logits = head(hidden)
  if softcap is not None:
    logits = torch.tanh(logits / softcap) * softcap
  # fp32 for an accurate logsumexp over a large vocab; this copy is only chunk-sized.
  logits = logits.float()
  return logits.gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


class BaseTrainerWorker:
  # Whether backward runs collectives that every data-parallel rank must join
  # (FSDP). Such backends need equal pass counts across ranks.
  backward_runs_collectives = False
  # Rows per output-head chunk in compute_target_logprobs. 1024 rows x a 262k
  # vocab x fp32 is 1 GiB per live copy.
  LOGPROB_CHUNK_TOKENS = 1024

  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None
    self.last_token_budget: int | None = None
    # What the last save moved, for the op record: a sparse delta's density and size, or empty for a full checkpoint.
    self.last_save_metrics: dict[str, float] = {}

    if torch.cuda.is_available():
      self.device = torch.device("cuda", local_rank())
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  def is_ready(self) -> bool:
    """Whether a model is loaded and the worker can serve requests. The
    request processor announces this to the API server, which shows it to
    clients as the worker's state."""
    return False

  def forward_backward(
    self,
    model: PreTrainedModel,
    data: list[wire.Datum],
    loss_fn: str,
    loss_config: dict | None = None,
    forward_only: bool = False,
  ) -> dict[str, Any]:
    """Run a forward/backward pass on model and return a binary ForwardBackwardOutput envelope.

    With ``forward_only`` (TrainingClient.forward) the pass runs without
    autograd and in eval mode, and no gradient is accumulated: the cookbook's
    NLL evaluator calls it on held-out data right before a training step, so
    a backward here would fold the test set into the next optim_step.
    """
    loss_fn_outputs: list[np.ndarray | None] = [None] * len(data)

    if forward_only:
      model.eval()
    else:
      model.train()

    with torch.set_grad_enabled(not forward_only):
      total_loss = self._run_batches(model, data, loss_fn, loss_config, forward_only, loss_fn_outputs)
    return self._finish(data, loss_fn_outputs, total_loss)

  # Data-parallel sharding hooks. By default every rank runs every datum; a
  # backend that reduces gradients across a DP group overrides these with it.

  def shard_rank(self) -> int:
    return 0

  def shard_count(self) -> int:
    return 1

  def shard_all_reduce_max(self, value: int) -> int:
    return value

  def shard_all_reduce_sum(self, value: float) -> float:
    return value

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    return [value]

  def _run_batches(
    self,
    model: PreTrainedModel,
    data: list[wire.Datum],
    loss_fn: str,
    loss_config: dict | None,
    forward_only: bool,
    loss_fn_outputs: list[np.ndarray | None],
  ) -> float:
    """Run every batch, fill ``loss_fn_outputs`` in place, and return the summed loss.

    Under data parallelism each rank runs a round-robin shard of the datums and
    scales its loss by the shard count, which undoes FSDP's gradient mean. When
    backward runs collectives, short ranks run zero-scaled filler passes so
    every rank makes the same number of backward calls.
    """
    shard_count = self.shard_count()
    local_indices = shard_datum_indices(len(data), self.shard_rank(), shard_count)
    local_data = [data[idx] for idx in local_indices]
    token_budget = self.train_token_budget(model)
    local_batches = self.make_training_batches(local_data, token_budget)
    if token_budget != self.last_token_budget:
      print(f"[trainer] token budget per pass: {token_budget or 'one sequence'}; {len(local_data)} sequences in {len(local_batches)} passes")
      self.last_token_budget = token_budget
    if shard_count > 1 and self.backward_runs_collectives:
      filler_passes = self.shard_all_reduce_max(len(local_batches)) - len(local_batches)
      filler = local_data[0] if local_data else data[0]
      local_batches.extend([[(FILLER_DATUM_INDEX, filler)]] * filler_passes)

    total_loss = 0.0
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
          old_logprobs = self.pad_sequences([datum.inputs["logprobs"] for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.inputs["advantages"] for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.importance_sampling_loss(target_logprobs, weights, old_logprobs, advantages)
        case "ppo":
          old_logprobs = self.pad_sequences([datum.inputs["logprobs"] for datum in batch_data], lengths, torch.float32)
          advantages = self.pad_sequences([datum.inputs["advantages"] for datum in batch_data], lengths, torch.float32)
          elementwise_loss = losses.ppo_loss(target_logprobs, weights, old_logprobs, advantages, loss_config)
        case _:
          raise NotImplementedError(f"Loss {loss_fn} not supported")

      loss = elementwise_loss.sum()
      if not forward_only:
        (loss * (0.0 if is_filler else float(shard_count))).backward()
      if not is_filler:
        total_loss += loss.item()

      detached = target_logprobs.detach().cpu().float()
      for row, original_idx in enumerate(batch_indices):
        row_lp = detached[row, : lengths[row]]
        row_lp = torch.where(torch.isposinf(row_lp), torch.full_like(row_lp, 9999.0), row_lp)
        loss_fn_outputs[original_idx] = torch.clamp(row_lp, min=-9999.0).numpy()

    if shard_count > 1:
      total_loss = self.shard_all_reduce_sum(total_loss)
      for part in self.shard_all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}):
        for idx, output in part.items():
          loss_fn_outputs[idx] = output
    return total_loss

  def _finish(self, data: list[wire.Datum], loss_fn_outputs: list[np.ndarray | None], total_loss: float) -> dict[str, Any]:
    completed = []
    for output in loss_fn_outputs:
      if output is None:
        raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")
      completed.append(output)
    metrics = self.sanitize_metrics({"loss:mean": total_loss / max(1, len(data)), "loss:sum": total_loss})
    return wire.proto_result("forward_backward", wire.output(completed, metrics), metrics=metrics)

  # Upper bound for the automatic budget. Past it, passes stop getting faster
  # and use more memory.
  AUTO_TOKEN_BUDGET_CEILING = 16384
  # Share of the free device memory the automatic budget may fill.
  AUTO_TOKEN_BUDGET_MEMORY_SHARE = 0.5
  # Activation bytes per token as a multiple of one hidden vector per layer,
  # which is what gradient checkpointing keeps. The rest covers the layer being
  # recomputed in backward, plus a margin.
  ACTIVATION_BYTES_FACTOR = 6

  def train_token_budget(self, model: PreTrainedModel) -> int:
    """Padded tokens per forward/backward pass. OPEN_RL_TRAIN_TOKEN_BUDGET is
    used as given when it is a number (0 = one sequence per pass); unset or
    `auto` takes the smaller of the ceiling and what fits in free memory."""
    raw = os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "auto").strip().lower()
    if raw not in ("", "auto"):
      return int(raw)
    return min(self.AUTO_TOKEN_BUDGET_CEILING, self.memory_token_cap(model))

  def memory_token_cap(self, model: PreTrainedModel) -> int:
    """Tokens whose activations fit in a share of the free device memory.

    Free memory is read at each call: a time-sliced worker holds the device
    while it trains, and its partner's memory is released by then. Memory the
    caching allocator holds but does not use counts as free.
    """
    if self.device.type != "cuda":
      return self.AUTO_TOKEN_BUDGET_CEILING
    text_config = model.config.get_text_config()
    token_bytes = text_config.num_hidden_layers * text_config.hidden_size * model.dtype.itemsize * self.ACTIVATION_BYTES_FACTOR
    free_bytes, _ = torch.cuda.mem_get_info(self.device)
    free_bytes += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
    return int(free_bytes * self.AUTO_TOKEN_BUDGET_MEMORY_SHARE / token_bytes)

  def make_training_batches(self, data: list[wire.Datum], token_budget: int) -> list[list[tuple[int, wire.Datum]]]:
    """Group examples for the single padded forward/backward path. A budget of
    0 runs one example per pass."""
    if len(data) <= 1 or token_budget <= 0:
      return [[(idx, datum)] for idx, datum in enumerate(data)]

    ordered_data = sorted(enumerate(data), key=lambda item: len(item[1].tokens))
    batches: list[list[tuple[int, wire.Datum]]] = []
    batch: list[tuple[int, wire.Datum]] = []
    batch_max_len = 0
    batch_tokens = 0

    for item in ordered_data:
      length = len(item[1].tokens)
      next_max_len = max(batch_max_len, length)
      next_padded_tokens = next_max_len * (len(batch) + 1)
      over_token_budget = next_padded_tokens > token_budget
      over_padding = next_padded_tokens - (batch_tokens + length) > MAX_PADDING_SHARE * next_padded_tokens

      if batch and (over_token_budget or over_padding):
        batches.append(batch)
        batch, batch_max_len, batch_tokens = [], 0, 0

      batch.append(item)
      batch_max_len = max(batch_max_len, length)
      batch_tokens += length

    if batch:
      batches.append(batch)
    return batches

  def pad_sequences(
    self,
    sequences: list[Any],
    lengths: list[int],
    dtype: torch.dtype,
    pad_value: int | float = 0,
  ) -> torch.Tensor:
    """Return padded values with shape [batch, max(lengths)]."""
    padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=self.device)
    for row, seq in enumerate(sequences):
      n = lengths[row]
      arr = np.asarray(seq[:n])
      padded[row, :n] = torch.from_numpy(arr if arr.flags.writeable else arr.copy()).to(device=self.device, dtype=dtype)
    return padded

  def pad_model_inputs(self, data: list[wire.Datum]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return input_ids and attention_mask with shape [batch, max_input_len]."""
    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
    input_lengths = [len(datum.tokens) for datum in data]
    input_ids = self.pad_sequences([datum.tokens for datum in data], input_lengths, torch.long, pad_token_id)
    attention_mask = input_ids.new_zeros((len(data), max(input_lengths)))
    for row, input_len in enumerate(input_lengths):
      attention_mask[row, :input_len] = 1
    return input_ids, attention_mask, input_lengths

  def pad_targets_and_weights(self, data: list[wire.Datum], input_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return target_token_ids and weights with shape [batch, max_target_len]."""
    targets = [datum.inputs["target_tokens"] for datum in data]
    target_lengths = [len(t) for t in targets]
    lengths = [min(in_len, tgt_len) for in_len, tgt_len in zip(input_lengths, target_lengths, strict=True)]
    target_token_ids = self.pad_sequences(targets, lengths, torch.long)
    weight_seqs = [datum.inputs.get("weights", np.ones(target_lengths[row], dtype=np.float32)) for row, datum in enumerate(data)]
    weights = self.pad_sequences(weight_seqs, lengths, torch.float32)
    return target_token_ids, weights, lengths

  def compute_target_logprobs(
    self,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return selected target logprobs with shape [batch, max_target_len].

    The model returns its final hidden states instead of logits for every
    position. The output head and logsumexp then run on LOGPROB_CHUNK_TOKENS
    positions at a time under activation checkpointing, so the full
    [tokens x vocab] logits are never held at once and loss memory scales with
    the chunk size instead of the batch.
    """
    batch_size, target_len = target_token_ids.shape
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1, output_hidden_states=True, return_dict=True)
    hidden = outputs.hidden_states[-1][:, :target_len].reshape(batch_size * target_len, -1)
    targets = target_token_ids.reshape(-1)
    head = model.get_output_embeddings()
    # Gemma caps its logits after the output head; other models leave this unset.
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    size = self.LOGPROB_CHUNK_TOKENS
    chunks = [
      checkpoint(chunk_target_logprobs, head, softcap, hidden[start : start + size], targets[start : start + size], use_reentrant=False)
      for start in range(0, hidden.shape[0], size)
    ]
    return torch.cat(chunks).reshape(batch_size, target_len)

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

    sample_outs: list[wire.SampleOut] = []
    for seq_idx in range(num_samples):
      gen_tokens = outputs.sequences[seq_idx][len(prompt_tokens) :].cpu().numpy().astype(np.int32, copy=False)
      lps = np.empty(len(gen_tokens), dtype=np.float32)
      for step_idx, token_id in enumerate(gen_tokens):
        logprob = torch.nn.functional.log_softmax(outputs.scores[step_idx][seq_idx], dim=-1)[int(token_id)].item()
        lps[step_idx] = self.sanitize_float(logprob)
      sample_outs.append(wire.SampleOut(stop=True, tokens=gen_tokens, logprobs=lps))
    plps = np.asarray([math.nan if v is None else v for v in prompt_logprobs], dtype=np.float32) if prompt_logprobs is not None else None
    return wire.proto_result("sample", wire.sample_output(sample_outs, plps))

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

  def sanitize_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
    """The metrics with NaN and infinity replaced, for JSON. Telemetry gets the names of the replaced ones first."""
    telemetry.nonfinite(name for name, value in metrics.items() if not math.isfinite(value))
    return {name: self.sanitize_float(value) for name, value in metrics.items()}

  def sanitize_float(self, val: float) -> float:
    if math.isinf(val):
      return -9999.0 if val < 0 else 9999.0
    if math.isnan(val):
      return 0.0
    return val
