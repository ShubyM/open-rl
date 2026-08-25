# Shared trainer worker logic for causal-LM forward/backward and generation.

import math
import os
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.utils.checkpoint
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from training import losses
from training.distributed import all_gather_object, all_reduce_max, all_reduce_sum, is_distributed, local_rank, rank, world_size


# FlexAttention recompiles for shapes dynamo cannot generalise, and once a frame
# hits recompile_limit (default 8) skip_code_recursive_on_recompile_limit_hit
# retires it to eager for the rest of the process. Eager flex_attention is
# sdpa_dense, which materialises the whole [heads, q_len, kv_len] score matrix:
# 8 x 47398^2 x fp32 = 67 GiB, an instant OOM. That is how run20 died at step 14
# after fourteen healthy steps -- the failure is silent, unbounded in memory, and
# arrives only once a run has been going long enough to exhaust the cache.
# Recompiling is merely slow, so the ceiling belongs well past any real run.
# Set from the worker constructor rather than at import. Importing dynamo
# registers inductor operators, and test loaders that reload this package
# would register them twice.
def raise_dynamo_recompile_limit() -> None:
  import torch._dynamo

  torch._dynamo.config.recompile_limit = int(os.getenv("OPEN_RL_RECOMPILE_LIMIT", "64"))
  torch._dynamo.config.accumulated_recompile_limit = max(
    torch._dynamo.config.accumulated_recompile_limit,
    8 * torch._dynamo.config.recompile_limit,
  )


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


FILLER_DATUM_INDEX = -1


def shard_datum_indices(count: int, shard_rank: int, shard_count: int) -> list[int]:
  """Round-robin datum ownership for one data-parallel rank."""
  if shard_count <= 1:
    return list(range(count))
  return list(range(shard_rank, count, shard_count))


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




def activation_offload_context(tensor: torch.Tensor):
  if tensor.is_cuda and os.getenv("OPEN_RL_ACTIVATION_CPU_OFFLOAD", "0") == "1":
    return torch.autograd.graph.save_on_cpu(pin_memory=True)
  return nullcontext()


def attention_forward_kwargs(config: Any) -> dict[str, Any]:
  """Use a low-resource FlexAttention tile for unusually wide heads."""
  config = config.get_text_config()
  if config._attn_implementation != "flex_attention":
    return {}

  if max(config.head_dim, getattr(config, "global_head_dim", 0) or 0) <= 256:
    return {}
  return {
    "kernel_options": {
      "fwd_BLOCK_M": 16,
      "fwd_BLOCK_N": 16,
      "fwd_num_stages": 1,
      "bwd_BLOCK_M1": 16,
      "bwd_BLOCK_N1": 16,
      "bwd_BLOCK_M2": 16,
      "bwd_BLOCK_N2": 16,
      "bwd_num_stages": 1,
    }
  }


def attention_backend_context(config: Any, tensor: torch.Tensor):
  config = config.get_text_config()
  if config._attn_implementation != "sdpa" or not tensor.is_cuda:
    return nullcontext()
  if os.getenv("OPEN_RL_SDPA_NO_MATH", "1") != "1":
    return nullcontext()

  from torch.nn.attention import SDPBackend, sdpa_kernel

  return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION])


class TensorData(BaseModel):
  data: list[int] | list[float]


class Datum(BaseModel):
  loss_fn_inputs: dict[str, TensorData]
  model_input: list[int]


class BaseTrainerWorker:
  # FSDP backwards run collectives, so pass counts must match across ranks.
  backward_runs_collectives = True

  def __init__(self):
    raise_dynamo_recompile_limit()
    self.tokenizer: PreTrainedTokenizerBase | None = None
    self.output_head_is_adapted = False

    if torch.cuda.is_available():
      self.device = torch.device("cuda", local_rank())
    elif torch.backends.mps.is_available():
      self.device = torch.device("mps")
    else:
      self.device = torch.device("cpu")

  # Data-parallel geometry, as five overridable hooks.
  #
  # forward_backward assumes every rank is an independent data shard, which is
  # true for FSDP and for the single-GPU LoRA path. It is not true in general:
  # a tensor-parallel backend splits one model across several ranks, so those
  # ranks must be fed the *same* datums and their losses must not be summed
  # twice. A backend whose data-parallel group is a subset of the world
  # overrides these to shard and reduce over that subgroup instead.
  def shard_rank(self) -> int:
    return rank()

  def shard_count(self) -> int:
    return world_size() if is_distributed() else 1

  def shard_all_reduce_max(self, value: int) -> int:
    return all_reduce_max(value)

  def shard_all_reduce_sum(self, value: float) -> float:
    return all_reduce_sum(value)

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    return all_gather_object(value)

  def forward_backward(
    self,
    model: PreTrainedModel,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None = None,
    forward_only: bool = False,
  ) -> dict[str, Any]:
    """Run a forward/backward pass on model and return Tinker-shaped loss outputs.

    With forward_only=True (the SDK's TrainingClient.forward()) no gradients are
    computed or accumulated — the SDK's custom-loss path (e.g. cookbook DPO)
    fetches logprobs this way with real weights attached and then sends the
    actual gradient as a second, linearized forward_backward; running backward
    here would silently corrupt that update with a spurious CE term.

    Under distributed training each rank owns a round-robin shard of the datums
    and gradients are summed across ranks through FSDP's per-backward averaging
    (each real pass scales its loss by the shard count to cancel the average).
    """
    shard_count = self.shard_count()
    local_indices = shard_datum_indices(len(data), self.shard_rank(), shard_count)
    local_data = [data[idx] for idx in local_indices]

    model.train()

    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    local_batches = self.make_training_batches(local_data)
    if shard_count > 1 and self.backward_runs_collectives:
      # Pad short ranks with zero-scaled passes so FSDP collective counts match.
      total_passes = self.shard_all_reduce_max(len(local_batches))
      filler_passes = total_passes - len(local_batches)
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
      seq_len = input_ids.shape[1]

      old_logprobs = advantages = None
      skip_backward = False
      if loss_fn in ("importance_sampling", "ppo"):
        old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
        advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)
        zero_effective_advantages = not bool(((advantages != 0) & (weights != 0)).any())
        if loss_fn == "importance_sampling":
          skip_backward = zero_effective_advantages
        else:
          has_kl_penalty = bool(loss_config and loss_config.get("kl_coeff", 0.0) > 0 and (weights != 0).any())
          skip_backward = zero_effective_advantages and not has_kl_penalty
      skip_backward = (skip_backward and (shard_count == 1 or not self.backward_runs_collectives)) or forward_only

      # A zero-advantage batch contributes no gradient; computing its logprobs
      # under no_grad avoids building and retaining the autograd graph.
      grad_context = torch.no_grad() if skip_backward else nullcontext()
      with self.cuda_memory_phase(f"forward[{len(batch_data)}x{seq_len}]"), grad_context:
        target_logprobs = self.compute_target_logprobs(model, input_ids, attention_mask, target_token_ids)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          elementwise_loss = losses.importance_sampling_loss(
            target_logprobs,
            weights,
            old_logprobs,
            advantages,
          )
        case "ppo":
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
      if not skip_backward:
        # FSDP averages gradients over the group on every backward; scaling
        # each real pass by shard_count recovers the single-process sum, and
        # filler passes contribute exactly zero.
        backward_loss = loss * (0.0 if is_filler else float(shard_count)) if shard_count > 1 else loss
        with self.cuda_memory_phase(f"backward[{len(batch_data)}x{seq_len}]"):
          backward_loss.backward()
      if not is_filler:
        total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

      if skip_backward:
        del target_logprobs, elementwise_loss, per_datum_loss, loss

    if shard_count > 1:
      total_loss = self.shard_all_reduce_sum(total_loss)
      for part in self.shard_all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}):
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

  @contextmanager
  def cuda_memory_phase(self, phase: str):
    enabled = self.device.type == "cuda" and os.getenv("OPEN_RL_LOG_CUDA_MEMORY", "0") == "1"
    if enabled:
      torch.cuda.reset_peak_memory_stats(self.device)
      self.log_cuda_memory(f"{phase}:start")
    try:
      yield
    except torch.OutOfMemoryError:
      self.log_cuda_memory(f"{phase}:oom", force=True)
      print(torch.cuda.memory_summary(self.device, abbreviated=True))
      raise
    finally:
      if enabled:
        self.log_cuda_memory(f"{phase}:end", include_peak=True)

  def log_cuda_memory(self, phase: str, *, force: bool = False, include_peak: bool = False) -> None:
    if self.device.type != "cuda" or (not force and os.getenv("OPEN_RL_LOG_CUDA_MEMORY", "0") != "1"):
      return
    gib = 1024**3
    free, total = torch.cuda.mem_get_info(self.device)
    fields = {
      "allocated": torch.cuda.memory_allocated(self.device) / gib,
      "reserved": torch.cuda.memory_reserved(self.device) / gib,
      "free": free / gib,
      "total": total / gib,
    }
    if include_peak:
      fields["peak_allocated"] = torch.cuda.max_memory_allocated(self.device) / gib
      fields["peak_reserved"] = torch.cuda.max_memory_reserved(self.device) / gib
    stats = " ".join(f"{key}={value:.2f}GiB" for key, value in fields.items())
    print(f"[CUDA_MEMORY] rank={os.getenv('RANK', '0')} phase={phase} {stats}")

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

    # project_target_logprobs multiplies by head.weight directly, which would
    # silently bypass an lm_head LoRA adapter (train_unembed) in both logprobs
    # and gradients; the full-logits path runs the wrapped head's forward.
    if os.getenv("OPEN_RL_FUSED_LOGPROB", "1") == "1" and not self.output_head_is_adapted:
      hidden = self.backbone_hidden_states(model, input_ids, attention_mask)
      if hidden is None:
        raise RuntimeError(
          "Fused logprob head could not resolve the model backbone. The full-logits "
          "path materializes a [seq, vocab] tensor (tens of GiB at long context) and "
          "bypasses activation offload; set OPEN_RL_FUSED_LOGPROB=0 to opt into it."
        )
      return project_target_logprobs(model, hidden[:, :seq_len, :], target_token_ids)

    # Full-logits path. Use logit - logsumexp rather than log_softmax(...).gather so
    # we avoid the extra full-size fp32 log_softmax allocation.
    outputs = model(
      input_ids,
      attention_mask=attention_mask,
      use_cache=False,
      return_dict=True,
      **attention_forward_kwargs(model.config),
    )
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
    backbone can't be resolved.

    PEFT models must be unwrapped through get_base_model(): their attribute
    delegation resolves `.model` to the full causal LM (lm_head included),
    whose forward yields logits but no last_hidden_state — the old code then
    ran a full forward, discarded it, and fell back to the full-logits path,
    which at long context materializes a [seq, vocab] logits tensor tens of
    GiB large, outside the activation-offload context. LoRA adapters live on
    the backbone's submodules, so calling the backbone directly still applies
    them."""
    backbone_owner = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone = getattr(backbone_owner, "model", None) or getattr(backbone_owner, "transformer", None)
    if backbone is None or backbone is backbone_owner:
      return None
    backbone_attention_mask = attention_mask
    if attention_mask is not None and bool(attention_mask.all()):
      # A dense all-ones mask only describes ordinary causal attention. Omitting
      # it lets SDPA select Flash Attention instead of materializing a quadratic
      # additive mask for Gemma's global-attention layers.
      backbone_attention_mask = None
    with activation_offload_context(input_ids), attention_backend_context(model.config, input_ids):
      outputs = backbone(
        input_ids=input_ids,
        attention_mask=backbone_attention_mask,
        use_cache=False,
        return_dict=True,
        **attention_forward_kwargs(model.config),
      )
    return getattr(outputs, "last_hidden_state", None)

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
      outputs = model(
        input_tensor,
        attention_mask=attention_mask,
        **attention_forward_kwargs(model.config),
      )
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
