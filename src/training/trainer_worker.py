# The shared trainer: one independently trained model and the training
# algorithm every backend runs on it, plus the execution worker that owns
# process-lifetime resources and creates trainers.

import math
import os
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist

from training import losses
from training.commands import CreateModel, CreateModelFromState, SaveWeightsForSampler
from training.distributed import all_gather_object, all_reduce_max, all_reduce_sum, group_rank, group_size
from training.types import Datum, SamplerWeights, TensorData

__all__ = ["Trainer", "TrainingWorker", "Datum", "TensorData", "chunk_target_logprob", "tmp_dir"]

# Position marker for a filler pass, see Trainer.forward_backward.
FILLER_DATUM_INDEX = -1
ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"
LOSS_FUNCTIONS = ("cross_entropy", "importance_sampling", "ppo")

RATIO_STATS_ZERO = {"max_abs_log_ratio": 0.0, "tokens": 0.0, "tokens_abs_log_ratio_gt1": 0.0, "tokens_abs_log_ratio_gt5": 0.0}


# Sampler weight versions kept on the volume. The sampler applies each delta
# as it lands, so older versions are dead weight; an 8B run otherwise leaves
# 3 GiB per step behind.
SAMPLER_VERSIONS_KEPT = int(os.getenv("OPEN_RL_SAMPLER_VERSIONS_KEPT", "3"))


def older_versions(path: str, keep: int) -> list[str]:
  """Sibling version directories of `path` beyond the newest `keep`, oldest first."""
  parent = os.path.dirname(path)
  if not os.path.isdir(parent):
    return []
  versions = sorted((p for p in (os.path.join(parent, name) for name in os.listdir(parent)) if os.path.isdir(p)), key=os.path.getmtime)
  return versions[: max(0, len(versions) - keep)]


def tmp_dir() -> str:
  return os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")


def sanitize_float(val: float) -> float:
  if math.isinf(val):
    return -9999.0 if val < 0 else 9999.0
  if math.isnan(val):
    return 0.0
  return val


def chunk_target_logprob(
  hidden_chunk: torch.Tensor,
  weight: torch.Tensor,
  bias: torch.Tensor | None,
  target_chunk: torch.Tensor,
  softcap: float | None,
) -> torch.Tensor:
  """logit[target] - logsumexp for one chunk of hidden states.

  The [chunk, vocab] logits are local to this call, so a backend that runs it
  under activation checkpointing never stores full-sequence logits.
  """
  logits = torch.nn.functional.linear(hidden_chunk, weight, bias)
  if logits.dtype in (torch.float16, torch.bfloat16):
    logits = logits.float()
  if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
  target_logit = logits.gather(dim=-1, index=target_chunk.unsqueeze(-1)).squeeze(-1)
  return target_logit - torch.logsumexp(logits, dim=-1)


def enable_gradient_checkpointing(model: torch.nn.Module) -> None:
  if not ENABLE_GRADIENT_CHECKPOINTING:
    return
  try:
    if hasattr(model, "gradient_checkpointing_enable"):
      model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
      model.enable_input_require_grads()
    print("Gradient checkpointing and input require grads enabled.")
  except Exception as e:
    print(f"Failed to enable gradient checkpointing: {e}")


def validate_loss_inputs(data: list[Datum], loss_fn: str) -> None:
  """Reject a malformed request before any backward pass has run, so a bad
  datum never leaves half a batch of gradients behind."""
  if loss_fn not in LOSS_FUNCTIONS:
    raise ValueError(f"Loss {loss_fn} not supported; expected one of {LOSS_FUNCTIONS}")
  required = ("target_tokens",) if loss_fn == "cross_entropy" else ("target_tokens", "logprobs", "advantages")
  for idx, datum in enumerate(data):
    missing = [key for key in required if key not in datum.loss_fn_inputs]
    if missing:
      raise ValueError(f"datum {idx} lacks loss_fn_inputs {missing} required by {loss_fn}")


class Trainer:
  """One independently trained model.

  The trainer owns the model reference, the trainable parameter objects
  captured after the backend wrapped or sharded the model, the optimizer, the
  accumulated ratio statistics and the optimizer step count. forward_backward
  and optim_step here are the whole training algorithm; a backend subclass
  overrides only what genuinely differs (how target logprobs are computed,
  where input tensors are built, the data-parallel group, gradient clipping,
  checkpoints) and wraps the shared step with its own side effects.

  Two trainers may share one model (LoRA adapters on one PEFT model) but never
  share params, optimizer or metrics. Nothing here selects an accelerator; a
  backend sets input_device when it knows where the model wants its inputs.
  """

  def __init__(
    self,
    model_id: str,
    model: torch.nn.Module,
    params: list[torch.nn.Parameter],
    tokenizer: Any = None,
    base_model_name: str | None = None,
  ):
    if not params:
      raise ValueError(f"Model '{model_id}' has no trainable parameters")
    self.model_id = model_id
    self.model = model
    self.params = params
    self.optimizer: torch.optim.Optimizer | None = None
    self.step = 0
    self.ratio_stats = dict(RATIO_STATS_ZERO)
    self.tokenizer = tokenizer
    self.base_model_name = base_model_name
    # Where padded input tensors are created. Outputs and parameters need not
    # live there (a device_map model spans several GPUs; a sharded model holds
    # DTensors); it only says where the forward wants its inputs.
    self.input_device = torch.device("cpu")
    # Global grad-norm clip applied when the client sends no grad_clip_norm. 0 = off.
    self.default_grad_clip = 0.0

  # -- optimizer --------------------------------------------------------------

  def build_optimizer(self, adam_params: dict[str, Any], **kwargs: Any) -> torch.optim.Optimizer:
    lr = adam_params.get("learning_rate", 1e-4)
    print(f"Initializing AdamW optimizer for '{self.model_id}' with lr={lr}")
    return torch.optim.AdamW(
      self.params,
      lr=lr,
      betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
      eps=adam_params.get("eps", 1e-12),
      weight_decay=adam_params.get("weight_decay", 0.0),
      **kwargs,
    )

  def clip_gradients(self, max_grad_norm: float) -> tuple[float, float]:
    """Clip the global gradient norm over this trainer's params. Returns (total_norm, clip_coef)."""
    total_norm = float(torch.nn.utils.clip_grad_norm_(self.params, max_grad_norm))
    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
    return total_norm, clip_coef

  def optim_step(self, adam_params: dict[str, Any]) -> dict[str, Any]:
    """Clip, step and clear the gradients accumulated since the last step.

    The optimizer is built on the first step from the client's AdamParams and
    follows learning_rate changes afterwards. Only this trainer's params are
    touched, so another trainer's pending gradients survive.
    """
    if self.optimizer is None:
      self.optimizer = self.build_optimizer(adam_params)
    elif (lr := adam_params.get("learning_rate")) is not None:
      for group in self.optimizer.param_groups:
        group["lr"] = lr
    max_grad_norm = adam_params.get("grad_clip_norm") or self.default_grad_clip or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf
    t0 = time.perf_counter()
    total_norm, clip_coef = self.clip_gradients(max_grad_norm)
    t1 = time.perf_counter()
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    t2 = time.perf_counter()
    self.step += 1
    return {
      "metrics": {
        "grad_norm:mean": sanitize_float(total_norm),
        "grad_clip_coef:mean": clip_coef,
        **self.ratio_metrics(),
        "time/clip_grad_norm": t1 - t0,
        "time/optimizer_step": t2 - t1,
      }
    }

  def rebind(self, params: list[torch.nn.Parameter]) -> None:
    """Adopt the parameter objects a load produced. The optimizer belonged to
    the old parameters and the old trajectory, so it is dropped and rebuilt on
    the next optim_step unless the caller restores a saved one; pending
    gradients are dropped with it."""
    if not params:
      raise ValueError(f"Model '{self.model_id}' has no trainable parameters after loading")
    for param in self.params:
      param.grad = None
    self.params = params
    self.optimizer = None

  def restore_optimizer(self, state_path: str, metadata: dict[str, Any], map_location: Any = "cpu") -> None:
    """Load the optimizer saved next to a checkpoint. A checkpoint without one
    fails the request rather than resuming with fresh moments unannounced."""
    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if not metadata.get("has_optimizer") or not os.path.exists(optimizer_path):
      raise ValueError(f"{state_path} holds no optimizer state; load it without restore_optimizer")
    self.optimizer = self.build_optimizer({})
    self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
    print(f"Restored optimizer state for '{self.model_id}' from {optimizer_path}")

  # -- distributed layout -----------------------------------------------------

  def data_parallel_group(self) -> dist.ProcessGroup | None:
    """The process group whose ranks split the datums of one forward_backward.

    None means this process trains alone. Ranks that share one model replica
    (tensor or context parallel) must see identical datums, so a backend with
    such ranks returns its data-parallel subgroup, not the world.
    """
    return None

  def data_parallel_loss_scale(self) -> float:
    """Factor applied to every rank's loss before backward.

    forward_backward wants the sum of the gradients over all datums. A backend
    whose gradient reduction averages over the data-parallel group (FSDP)
    returns the group size to undo that mean; one that sums returns 1.
    """
    return 1.0

  # -- forward / backward -----------------------------------------------------

  def forward_backward(self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, forward_only: bool = False) -> dict[str, Any]:
    """Run a forward/backward pass and return Tinker-shaped loss outputs.

    Gradients accumulate on this trainer's params until optim_step. Under data
    parallelism each rank owns a round-robin shard of the datums and every rank
    runs the same number of passes (short ranks run zero-scaled fillers) so the
    collectives inside backward line up.

    With forward_only (TrainingClient.forward) the pass runs in eval mode with
    no graph and no backward on every rank, so nothing accumulates: the
    cookbook's NLL evaluator calls it on held-out data right before a step.
    """
    validate_loss_inputs(data, loss_fn)
    if not data:
      return {"metrics": {"loss:mean": 0.0, "loss:sum": 0.0}, "loss_fn_outputs": [], "loss_fn_output_type": "ArrayRecord"}

    group = self.data_parallel_group()
    dp_rank, dp_size = group_rank(group), group_size(group)
    loss_scale = self.data_parallel_loss_scale()
    local_indices = list(range(dp_rank, len(data), dp_size))
    local_batches = self.make_training_batches([data[idx] for idx in local_indices])
    if dp_size > 1:
      filler_passes = int(all_reduce_max(len(local_batches), group)) - len(local_batches)
      if filler_passes > 0:
        # Only the collectives matter, so the cheapest datum will do.
        filler = min(data, key=lambda datum: len(datum.model_input))
        local_batches.extend([[(FILLER_DATUM_INDEX, filler)]] * filler_passes)

    total_loss = 0.0
    loss_fn_outputs: list[dict[str, Any] | None] = [None] * len(data)

    self.model.eval() if forward_only else self.model.train()

    for batch in local_batches:
      is_filler = batch[0][0] == FILLER_DATUM_INDEX
      batch_indices = [] if is_filler else [local_indices[position] for position, _ in batch]
      batch_data = [datum for _, datum in batch]

      input_ids, attention_mask, input_lengths = self.pad_model_inputs(batch_data)
      target_token_ids, weights, lengths = self.pad_targets_and_weights(batch_data, input_lengths)

      old_logprobs = advantages = None
      if loss_fn in ("importance_sampling", "ppo"):
        old_logprobs = self.pad_sequences([datum.loss_fn_inputs["logprobs"].data for datum in batch_data], lengths, torch.float32)
        advantages = self.pad_sequences([datum.loss_fn_inputs["advantages"].data for datum in batch_data], lengths, torch.float32)

      # A batch without gradient (a GRPO group whose rewards all tied) still
      # costs a full backward unless its forward runs without a graph. Under
      # data parallelism the pass must still happen for the collectives, unless
      # every rank is skipping it because the whole call is forward-only.
      skip_backward = forward_only or (dp_size == 1 and self.batch_has_no_gradient(loss_fn, loss_config, weights, advantages))
      with torch.no_grad() if skip_backward else nullcontext():
        target_logprobs = self.compute_target_logprobs(input_ids, attention_mask, target_token_ids)
      if old_logprobs is not None and not is_filler:
        self.record_ratio_stats(target_logprobs, old_logprobs, weights, advantages)

      match loss_fn:
        case "cross_entropy":
          elementwise_loss = losses.cross_entropy_loss(target_logprobs, weights)
        case "importance_sampling":
          elementwise_loss = losses.importance_sampling_loss(target_logprobs, weights, old_logprobs, advantages)
        case "ppo":
          elementwise_loss = losses.ppo_loss(target_logprobs, weights, old_logprobs, advantages, loss_config)

      per_datum_loss = elementwise_loss.sum(dim=1)
      loss = per_datum_loss.sum()
      if not skip_backward:
        (loss * (0.0 if is_filler else loss_scale)).backward()
      if not is_filler:
        total_loss += loss.item()

      detached_logprobs = target_logprobs.detach().cpu()
      for row, original_idx in enumerate(batch_indices):
        row_len = lengths[row]
        logprobs_list = detached_logprobs[row, :row_len].tolist()
        logprobs_list = [max(l, -9999.0) if not math.isinf(l) else (-9999.0 if l < 0 else 9999.0) for l in logprobs_list]
        loss_fn_outputs[original_idx] = {"logprobs": {"data": logprobs_list, "dtype": "float32", "shape": [len(logprobs_list)]}}

    if dp_size > 1:
      total_loss = all_reduce_sum(total_loss, group)
      for part in all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}, group):
        for idx, output in part.items():
          loss_fn_outputs[idx] = output

    mean_loss = total_loss / max(1, len(data))
    if any(output is None for output in loss_fn_outputs):
      raise RuntimeError("forward_backward did not produce one loss_fn_output per input datum")

    return {
      "metrics": {"loss:mean": sanitize_float(mean_loss), "loss:sum": sanitize_float(total_loss)},
      "loss_fn_outputs": loss_fn_outputs,
      "loss_fn_output_type": "ArrayRecord",
    }

  def batch_has_no_gradient(self, loss_fn: str, loss_config: dict | None, weights: torch.Tensor, advantages: torch.Tensor | None) -> bool:
    if advantages is None or bool(((advantages != 0) & (weights != 0)).any()):
      return False
    has_kl_penalty = loss_fn == "ppo" and bool(loss_config and loss_config.get("kl_coeff", 0.0) > 0)
    return not has_kl_penalty

  def record_ratio_stats(self, target_logprobs: torch.Tensor, old_logprobs: torch.Tensor, weights: torch.Tensor, advantages: torch.Tensor) -> None:
    """Tail of the sampler/trainer log-ratio over action tokens, accumulated
    until the next optim_step reports it. The mean KL is blind to a handful of
    tokens with a huge ratio, and under an unclipped loss those few tokens can
    be most of the gradient. Prompt positions carry weight 1 but no advantage
    and a sampled logprob of 0, so only positions with an advantage count."""
    active = (weights != 0) & (advantages != 0)
    if not bool(active.any()):
      return
    log_ratio = (target_logprobs.detach().float() - old_logprobs.float())[active].abs()
    log_ratio = torch.nan_to_num(log_ratio, nan=0.0, posinf=1e4)
    stats = self.ratio_stats
    stats["max_abs_log_ratio"] = max(stats["max_abs_log_ratio"], float(log_ratio.max()))
    stats["tokens"] += float(active.sum())
    stats["tokens_abs_log_ratio_gt1"] += float((log_ratio > 1.0).sum())
    stats["tokens_abs_log_ratio_gt5"] += float((log_ratio > 5.0).sum())

  def ratio_metrics(self) -> dict[str, float]:
    """Reduce the accumulated tail stats over the data-parallel group and reset them."""
    group = self.data_parallel_group()
    stats, self.ratio_stats = self.ratio_stats, dict(RATIO_STATS_ZERO)
    max_abs = all_reduce_max(stats["max_abs_log_ratio"], group)
    tokens = all_reduce_sum(stats["tokens"], group)
    gt1 = all_reduce_sum(stats["tokens_abs_log_ratio_gt1"], group)
    gt5 = all_reduce_sum(stats["tokens_abs_log_ratio_gt5"], group)
    return {
      "ratio/max_abs_log:max": max_abs,
      "ratio/tokens_abs_log_gt1:sum": gt1,
      "ratio/tokens_abs_log_gt5:sum": gt5,
      "ratio/frac_abs_log_gt1:mean": gt1 / tokens if tokens else 0.0,
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
    """Return padded values with shape [batch, max(lengths)] on input_device.

    Every tensor the loss consumes (inputs, targets, weights, old logprobs,
    advantages) is built here, so a backend whose logprobs come back in
    another layout (a sharded XLA tensor, say) changes the layout in one place.
    """
    padded = torch.full((len(sequences), max(lengths)), pad_value, dtype=dtype, device=self.input_device)
    for row, sequence in enumerate(sequences):
      length = lengths[row]
      padded[row, :length] = padded.new_tensor(sequence[:length])
    return padded

  def pad_model_inputs(self, data: list[Datum]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return input_ids and attention_mask with shape [batch, max_input_len]."""
    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None and self.tokenizer.pad_token_id is not None else 0
    batch_size = len(data)
    input_lengths = [len(datum.model_input) for datum in data]
    max_input_len = max(input_lengths)

    input_ids = self.pad_sequences([datum.model_input for datum in data], input_lengths, torch.long, pad_token_id)
    attention_mask = input_ids.new_zeros((batch_size, max_input_len))
    for row, input_len in enumerate(input_lengths):
      attention_mask[row, :input_len] = 1

    return input_ids, attention_mask, input_lengths

  def pad_targets_and_weights(self, data: list[Datum], input_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return target_token_ids and weights with shape [batch, max_target_len].

    Targets are position-aligned by the client (input_ids[t] predicts
    target_token_ids[t]); no shifting happens here.
    """
    batch_size = len(data)
    target_lengths = [len(datum.loss_fn_inputs["target_tokens"].data) for datum in data]
    lengths = [min(input_lengths[row], target_lengths[row]) for row in range(batch_size)]
    target_token_ids = self.pad_sequences([datum.loss_fn_inputs["target_tokens"].data for datum in data], lengths, torch.long)
    weight_sequences = [
      datum.loss_fn_inputs["weights"].data if "weights" in datum.loss_fn_inputs else [1.0] * target_lengths[row] for row, datum in enumerate(data)
    ]
    weights = self.pad_sequences(weight_sequences, lengths, torch.float32)

    return target_token_ids, weights, lengths

  def compute_target_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Return selected target logprobs with shape [batch, max_target_len]."""
    outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    logits = outputs.logits[:, : target_token_ids.shape[1], :]
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)

  # -- checkpoints and publication --------------------------------------------
  # Two distinct things leave a trainer. save_state writes a resumable
  # checkpoint at an optimizer boundary (weights, optionally the optimizer;
  # never pending gradients). publish_sampler_weights writes what the samplers
  # load, which may be an adapter or an incremental delta and is not resumable.

  def checkpoint_metadata(self, kind: str = "state", has_optimizer: bool = False, **extra: Any) -> dict[str, Any]:
    return {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": has_optimizer,
      "model_id": self.model_id,
      "step": self.step,
      "timestamp": time.time(),
      **extra,
    }

  def save_state(self, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    raise NotImplementedError

  def load_from_state(self, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    raise NotImplementedError

  def save_model(self, alias: str) -> dict[str, Any]:
    raise NotImplementedError

  def publish_sampler_weights(self, command: SaveWeightsForSampler) -> SamplerWeights:
    """Write what the samplers load. A full-parameter trainer writes a whole
    checkpoint at the path the sampling ref names."""
    ref = command.path or command.sampling_session_id
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    path = os.path.join(tmp_dir(), "sampler_full", ref.removeprefix("tinker://").lstrip("/"))
    self.save_state(path, include_optimizer=False, kind="sampler")
    older = older_versions(path, SAMPLER_VERSIONS_KEPT)
    for stale in older:
      shutil.rmtree(stale, ignore_errors=True)
    if older:
      print(f"[Trainer] Removed {len(older)} sampler weight versions older than the newest {SAMPLER_VERSIONS_KEPT}")
    return SamplerWeights(kind="checkpoint", path=path)

  def save_weights(self, alias: str | None = None) -> dict[str, Any]:
    return self.save_model(alias or self.model_id)

  def close(self) -> None:
    """Release what this trainer alone owns. Shared resources stay with the worker."""
    self.optimizer = None

  # -- sampling from the trainer ----------------------------------------------

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    """Generate completions from the model."""
    self.model.eval()

    input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.input_device)
    do_sample = (num_samples > 1) or (temperature and temperature > 0.0)
    prompt_logprobs = self.prompt_logprobs(input_tensor) if include_prompt_logprobs else None

    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = self.model.generate(
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
        logprobs.append(sanitize_float(logprob_dist[token_id].item()))

      sequences_out.append({"tokens": generated_tokens, "logprobs": logprobs, "stop_reason": "stop"})

    result = {"sequences": sequences_out}
    if prompt_logprobs is not None:
      result["prompt_logprobs"] = prompt_logprobs
    return result

  def prompt_logprobs(self, input_tensor: torch.Tensor) -> list[float | None]:
    with torch.no_grad():
      attention_mask = torch.ones_like(input_tensor)
      outputs = self.model(input_tensor, attention_mask=attention_mask)
      logprob_dist = torch.nn.functional.log_softmax(outputs.logits[0, :-1], dim=-1)

    prompt_tokens = input_tensor[0].tolist()
    prompt_logprobs: list[float | None] = [None]
    for token_idx, token_id in enumerate(prompt_tokens[1:]):
      prompt_logprobs.append(sanitize_float(logprob_dist[token_idx, token_id].item()))

    return prompt_logprobs


class TrainingWorker:
  """The execution worker: what one trainer process (or one torchrun rank of
  it) owns for its whole life, and how it turns commands into Trainers.

  initialize runs once before any model exists and is where a backend selects
  its accelerator and builds its process group or mesh; create, restore and
  remove manage trainers on top of that; sleep and wake_up hand the worker's
  resources off around a GPU lease, once per shared resource. A worker hosts
  either any number of compatible adapters on one shared base (single_model
  False) or one independent model (single_model True), and the request
  processor routes by that.
  """

  single_model = False
  full_parameter = False
  # A distributed backend runs under torchrun across ranks; a plain HF worker
  # does not become distributed by being launched that way. The loop refuses
  # WORLD_SIZE>1 for a worker that leaves this False.
  distributed = False

  def initialize(self, base_model: str | None = None) -> None:
    pass

  def create(self, command: CreateModel) -> Trainer:
    raise NotImplementedError

  def restore(self, command: CreateModelFromState) -> Trainer:
    raise NotImplementedError

  def reload(self, trainer: Trainer, command: Any) -> dict[str, Any]:
    """Reload one existing trainer's weights in place (a LoadWeights command).
    The default reloads from a checkpoint the trainer knows how to read; a
    backend that rebuilds a sharded model overrides this."""
    trainer.load_from_state(command.state_path, command.restore_optimizer)
    return {"model_id": trainer.model_id, "base_model": trainer.base_model_name}

  def remove(self, trainer: Trainer) -> None:
    trainer.close()

  def sleep(self) -> None:
    pass

  def wake_up(self) -> None:
    pass

  def save_needs_gpu(self) -> bool:
    return False
