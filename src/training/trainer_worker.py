# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.utils.checkpoint
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses
from training.distributed import local_rank


def chunk_target_logprob(
  hidden_chunk: torch.Tensor,
  weight: torch.Tensor,
  bias: torch.Tensor | None,
  target_chunk: torch.Tensor,
  softcap: float | None,
) -> torch.Tensor:
  """Project one chunk of hidden states through the vocab and return the selected
  target logprob (logit[target] - logsumexp). The [chunk, vocab] logits tensor is
  local to this call, so under activation checkpointing it is never stored for the
  backward pass -- it is recomputed."""
  logits = torch.nn.functional.linear(hidden_chunk, weight, bias)
  if logits.dtype in (torch.float16, torch.bfloat16):
    logits = logits.float()
  if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
  target_logit = logits.gather(dim=-1, index=target_chunk.unsqueeze(-1)).squeeze(-1)
  return target_logit - torch.logsumexp(logits, dim=-1)


def project_target_logprobs(model: PreTrainedModel, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
  """Project hidden states in small vocab chunks instead of building full logits."""
  head = model.get_output_embeddings()
  config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
  batch, seq_len, _ = hidden.shape
  hidden = hidden.reshape(batch * seq_len, -1).to(head.weight.device)
  targets = targets.reshape(batch * seq_len).to(head.weight.device)
  chunk_size = max(1, int(os.getenv("OPEN_RL_LOGPROB_CHUNK", "128")))

  def project(start: int) -> torch.Tensor:
    args = (
      hidden[start : start + chunk_size],
      head.weight,
      getattr(head, "bias", None),
      targets[start : start + chunk_size],
      getattr(config, "final_logit_softcapping", None),
    )
    if args[0].requires_grad or head.weight.requires_grad:
      return torch.utils.checkpoint.checkpoint(chunk_target_logprob, *args, use_reentrant=False)
    return chunk_target_logprob(*args)

  return torch.cat([project(start) for start in range(0, hidden.shape[0], chunk_size)]).reshape(batch, seq_len)


class TensorData(BaseModel):
  data: list[int] | list[float]


class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int]


class BaseTrainerWorker:
  def __init__(self):
    self.tokenizer: PreTrainedTokenizerBase | None = None

    if torch.cuda.is_available():
      self.device = torch.device("cuda", local_rank())
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  def forward_backward(self, model: PreTrainedModel, data: list[Datum], loss_fn: str, loss_config: dict | None = None) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs."""
    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    model.train()

    for batch in self.make_training_batches(data):
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
      loss.backward()
      total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

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
    """Return selected target logprobs with shape [batch, max_target_len].

    Large-vocab models (e.g. Gemma's ~256K vocab) OOM the GPU holding the lm_head
    because the full [batch, seq, vocab] logits tensor is materialized twice (once
    by the head, again by log_softmax upcasting to fp32). We instead run the
    backbone to get hidden states and project + reduce to the target logprob in
    vocab-sized chunks, so peak head activation is [chunk, vocab]. Falls back to the
    standard full-logits path when disabled or when the backbone can't be resolved.
    """
    seq_len = target_token_ids.shape[1]

    if os.getenv("OPEN_RL_FUSED_LOGPROB", "1") == "1":
      hidden = self.backbone_hidden_states(model, input_ids, attention_mask)
      if hidden is not None:
        return project_target_logprobs(model, hidden[:, :seq_len, :], target_token_ids)

    # Full-logits path. Use logit - logsumexp rather than log_softmax(...).gather so
    # we avoid the extra full-size fp32 log_softmax allocation.
    outputs = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, :seq_len, :]
    target_logit = logits.gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)
    return target_logit - torch.logsumexp(logits, dim=-1)

  def backbone_hidden_states(
    self,
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
  ) -> torch.Tensor | None:
    """Return the transformer backbone's last_hidden_state without running the
    lm_head. Returns None (so the caller uses the full-logits path) when the
    backbone can't be resolved or does not expose hidden states -- e.g. PEFT/LoRA
    wrappers whose forward only yields logits."""
    backbone = getattr(model, "model", None) or getattr(model, "transformer", None)
    if backbone is None or backbone is model:
      return None
    backbone_attention_mask = attention_mask
    if attention_mask is not None and bool(attention_mask.all()):
      # A dense all-ones mask only describes ordinary causal attention. Omitting
      # it lets SDPA select Flash Attention instead of materializing a quadratic
      # additive mask for Gemma's global-attention layers.
      backbone_attention_mask = None
    attention_context = nullcontext()
    if input_ids.is_cuda and os.getenv("OPEN_RL_SDPA_NO_MATH", "1") == "1":
      from torch.nn.attention import SDPBackend, sdpa_kernel

      attention_context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    try:
      with attention_context:
        outputs = backbone(input_ids=input_ids, attention_mask=backbone_attention_mask, use_cache=False, return_dict=True)
    except Exception as exc:
      print(f"[trainer] fused-logprob backbone forward failed ({exc}); using full-logits path")
      return None
    return getattr(outputs, "last_hidden_state", None)

  def generate(
    self,
    model: PreTrainedModel,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
    stop: list[int] | None = None,
    top_p: float = 1.0,
    top_k: int = -1,
  ) -> dict[str, Any]:
    """Generate completions from model."""
    model.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
    prompt_logprobs = self.prompt_logprobs(model, input_tensor) if include_prompt_logprobs else None
    eos_token_ids: list[int] = []
    tokenizer_eos = self.tokenizer.eos_token_id
    if isinstance(tokenizer_eos, int):
      eos_token_ids.append(tokenizer_eos)
    elif tokenizer_eos:
      eos_token_ids.extend(tokenizer_eos)
    eos_token_ids.extend(stop or [])
    eos_token_ids = list(dict.fromkeys(eos_token_ids))
    pad_token_id = self.tokenizer.pad_token_id
    if pad_token_id is None and eos_token_ids:
      pad_token_id = eos_token_ids[0]

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_tokens,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_ids or None,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        top_k=max(0, top_k) if do_sample else None,
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

      stopped = bool(generated_tokens and generated_tokens[-1] in eos_token_ids)
      stop_reason = "stop" if stopped or len(generated_tokens) < max_tokens else "length"
      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": stop_reason})

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
