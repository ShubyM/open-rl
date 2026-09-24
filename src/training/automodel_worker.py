"""A trainer worker built on NVIDIA NeMo Automodel, run under torchrun.

Automodel builds a (dp, tp) FSDP2 mesh from the torchrun world; DP is what is
left after TP. Only the DP axis shards datums, since TP ranks cooperate on one
sequence and must see identical data. The shard_* hooks scope to that axis.

Batches are Automodel's own Datum records through collate_datums, padded
[B, T] by default or one packed THD row per pass with OPEN_RL_AUTOMODEL_PACKED=1.
The loss stays ours: per-token logprobs come out of the final hidden states in
chunks, so full [seq, vocab] logits never exist, and losses.py scores them.
Long sequences fit because layers are checkpointed in groups.
"""

import json
import math
import os
import shutil
import time
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from transformers import AutoConfig, AutoTokenizer

from training import losses
from training.distributed import barrier, is_primary
from training.trainer_worker import FILLER_DATUM_INDEX, BaseTrainerWorker, Datum, shard_datum_indices
from training.types import FFTConfig, LoraConfig

TMP_DIR = os.getenv("OPEN_RL_TMP_DIR", "/tmp/open-rl")
AUTOMODEL_TP = int(os.getenv("OPEN_RL_AUTOMODEL_TP", "1"))
AUTOMODEL_SEED = int(os.getenv("OPEN_RL_AUTOMODEL_SEED", "1234"))

# LoRA targets per LoraConfig flag, matched as model.*.layers.*.<name> so a
# vision tower or an MTP head the samplers cannot place is never wrapped.
# train_attn covers Qwen3.5's linear-attention projections too. The kaiming A
# init is PEFT's, so an lr means the same thing here as on the FSDP worker.
LORA_TARGETS = {
  "train_attn": ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"),
  "train_mlp": ("gate_proj", "up_proj", "down_proj"),
}

# Decoder layers per activation checkpoint. Automodel's own checkpointing wraps
# attention and MLP separately and stashes four tensors per layer; a group of
# whole layers stashes one. 0 turns it off.
RECOMPUTE_NUM_LAYERS = int(os.getenv("OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS", "4"))
# Rows of hidden states projected through the vocab at a time. 1024 rows of a
# 248k vocab is a 1 GiB fp32 chunk.
LOGPROB_CHUNK = int(os.getenv("OPEN_RL_LOGPROB_CHUNK", "1024"))

# Pack every datum of a pass into one THD row instead of padding them to the
# longest. The backbone has to take seq_lens/qkv_format, which Automodel's
# native ones do. The token budget then bounds the row's total tokens.
AUTOMODEL_PACKED = os.getenv("OPEN_RL_AUTOMODEL_PACKED", "0") == "1"
# What save_weights_for_sampler publishes for a LoRA run. "adapter" writes the
# PEFT adapter the LoRA sampler hot-loads. "merged" also folds it into the
# base weights as a plain HF checkpoint under sampler_full/, for a sampler
# that reloads full weights.
SAMPLER_WEIGHTS = os.getenv("OPEN_RL_AUTOMODEL_SAMPLER_WEIGHTS", "adapter")

# Keys collate_datums emits that the model forward takes. The rest are ours.
MODEL_INPUT_KEYS = ("input_ids", "attention_mask", "position_ids", "seq_lens", "qkv_format", "cu_seqlens", "max_seqlen")


def require_automodel():
  try:
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
  except ImportError as exc:
    raise RuntimeError(
      "OPEN_RL_TRAINER_BACKEND=automodel needs nemo-automodel, which is not a project dependency. "
      f"Run the trainer from the automodel image or scripts/setup_automodel_env.sh. Import failed with: {exc}"
    ) from exc
  return NeMoAutoModelForCausalLM


def is_dtensor(tensor: Any) -> bool:
  from torch.distributed.tensor import DTensor

  return isinstance(tensor, DTensor)


def chunk_target_logprob(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, softcap: float | None) -> torch.Tensor:
  """logit[target] - logsumexp for one chunk of hidden states."""
  logits = torch.nn.functional.linear(hidden, weight).float()
  if softcap is not None:
    logits = softcap * torch.tanh(logits / softcap)
  return logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def lora_target_patterns(config: LoraConfig, tied_embeddings: bool) -> list[str]:
  """Automodel PEFT patterns for the modules a LoraConfig asks to train."""
  patterns = [f"model.*.layers.*.{name}" for flag, names in LORA_TARGETS.items() if getattr(config, flag) for name in names]
  if config.train_unembed:
    if tied_embeddings:
      # vLLM refuses lm_head adapter weights for a tied head, so the adapter would be unloadable.
      print("[LoRA] Ignoring train_unembed=True: the model ties lm_head to embed_tokens, so vLLM could not load the adapter.")
    else:
      patterns.append("lm_head")
  if not patterns:
    raise ValueError("At least one LoRA training target must be enabled.")
  return patterns


def datum_inputs(datum: Datum) -> tuple[list[int], dict[str, list], int]:
  """Our datum as Automodel's Datum wants it, plus how many positions carry a loss.

  Automodel keys every per-token input to the input length. Targets past the
  input are dropped, and input positions past the targets get a zero weight,
  so neither is scored or returned, the same rule the padded path applies.
  """
  length = len(datum.model_input)
  targets = list(datum.loss_fn_inputs["target_tokens"].data)
  scored = min(length, len(targets))
  inputs: dict[str, list] = {"target_tokens": (targets + [0] * length)[:length]}
  weights = list(datum.loss_fn_inputs["weights"].data) if "weights" in datum.loss_fn_inputs else [1.0] * len(targets)
  inputs["weights"] = ([float(w) for w in weights[:scored]] + [0.0] * length)[:length]
  for key in ("logprobs", "advantages"):
    if key in datum.loss_fn_inputs:
      inputs[key] = ([float(v) for v in datum.loss_fn_inputs[key].data[:scored]] + [0.0] * length)[:length]
  return list(datum.model_input), inputs, scored


def model_inputs(batch: dict[str, Any]) -> dict[str, Any]:
  """The forward kwargs for one collated batch. An all-ones mask is plain causal attention; dropping it lets SDPA use flash."""
  kwargs = {key: batch[key] for key in MODEL_INPUT_KEYS if key in batch}
  mask = kwargs.get("attention_mask")
  if mask is not None and bool(mask.all()):
    del kwargs["attention_mask"]
  return kwargs


def split_rows(logprobs: torch.Tensor, seq_lens: list[int] | None, scored: list[int]) -> list[list[float]]:
  """Per-datum logprob lists from a padded [B, T] or a packed [1, total] result."""
  rows = list(logprobs[0].split(seq_lens)) if seq_lens is not None else list(logprobs)
  out = []
  for row, count in zip(rows, scored, strict=True):
    values = row[:count].tolist()
    out.append([max(v, -9999.0) if not math.isinf(v) else (-9999.0 if v < 0 else 9999.0) for v in values])
  return out


class LayerGroup:
  """Runs a run of decoder layers under one non-reentrant checkpoint."""

  def __init__(self, layers: list[torch.nn.Module]):
    self.layers = layers

  def run(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x, **kwargs)
    return x

  def __call__(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    if not torch.is_grad_enabled():
      return self.run(x, **kwargs)
    return torch.utils.checkpoint.checkpoint(self.run, x, use_reentrant=False, **kwargs)


class GroupCheckpointedLayers(torch.nn.ModuleDict):
  """Automodel's native backbones run `for layer in self.layers.values()`.
  Swapping this class in after sharding changes only that iteration, so
  parameter names and FSDP2 units are untouched."""

  group_size = 1

  def values(self):
    layers = list(self._modules.values())
    if not self.training:
      return iter(layers)
    return iter([LayerGroup(layers[start : start + self.group_size]) for start in range(0, len(layers), self.group_size)])


def install_group_checkpointing(model: torch.nn.Module, group_size: int) -> None:
  """Native backbones keep a ModuleDict of layers and get grouped checkpoints.
  A stock HF backbone keeps a ModuleList and gets HF's per-layer checkpointing."""
  backbone = model.model.language_model if hasattr(model.model, "language_model") else model.model
  layers = getattr(backbone, "layers", None)
  if isinstance(layers, torch.nn.ModuleDict):
    layers.__class__ = GroupCheckpointedLayers
    layers.group_size = group_size
  elif isinstance(layers, torch.nn.ModuleList) and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
  else:
    raise RuntimeError(f"No decoder layers to checkpoint on {type(backbone).__name__}; set OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS=0.")


class AutomodelTrainingWorker(BaseTrainerWorker):
  # FSDP2 reduces gradients inside backward, so every DP rank needs the same
  # number of passes.
  backward_runs_collectives = True

  def __init__(self):
    super().__init__()
    self.model: torch.nn.Module | None = None
    self.distributed_setup: Any = None
    self.device_mesh: Any = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.checkpointer: Any = None
    # Set by create_model from the client's LoraConfig; None means full fine-tuning.
    self.lora_config: LoraConfig | None = None
    self.peft_config: Any = None

  @property
  def is_lora(self) -> bool:
    return self.peft_config is not None

  def build_distributed_setup(self, base_model_name: str) -> Any:
    if not dist.is_initialized():
      raise RuntimeError("The Automodel backend runs under torchrun; torch.distributed is not initialized.")
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    world = dist.get_world_size()
    if world % AUTOMODEL_TP:
      raise RuntimeError(f"WORLD_SIZE={world} is not divisible by OPEN_RL_AUTOMODEL_TP={AUTOMODEL_TP}")
    strategy = FSDP2Config(activation_checkpointing=False, tp_plan=self.tensor_parallel_plan(base_model_name) if AUTOMODEL_TP > 1 else None)
    mesh_context = MeshContext.build(strategy, ParallelismSizes(tp_size=AUTOMODEL_TP), world_size=world)
    print(f"Automodel device mesh: DP={world // AUTOMODEL_TP} TP={AUTOMODEL_TP}")
    return DistributedSetup(mesh_context=mesh_context, strategy_config=strategy, activation_checkpointing=False)

  def tensor_parallel_plan(self, base_model_name: str) -> dict[str, Any]:
    """The HF text-model TP plan, prefixed the way the module tree is laid out.

    Automodel's automatic plan reads _tp_plan off the language model, which its
    native Qwen3.5 backbone does not carry. Layers missing from the plan (GDN)
    stay replicated.
    """
    from nemo_automodel.components.distributed.parallelizer import translate_to_torch_parallel_style
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    config = AutoConfig.from_pretrained(base_model_name)
    architectures = config.architectures or []
    prefix = "model.language_model" if architectures and architectures[0].endswith("ForConditionalGeneration") else "model"
    plan: dict[str, Any] = {f"{prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate())}
    for name, style_name in (getattr(config.get_text_config(), "base_model_tp_plan", None) or {}).items():
      try:
        style = translate_to_torch_parallel_style(style_name)
      except ValueError:
        # MoE-only entries match nothing on a dense checkpoint; skipping one
        # leaves that module replicated.
        continue
      if style is not None:
        plan[f"{prefix}.{name}"] = style
    # Replicated logits, so the base class computes logprobs from plain tensors.
    plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())
    return plan

  def dp_group(self):
    return self.device_mesh["dp_shard"].get_group()

  def shard_rank(self) -> int:
    return 0 if self.device_mesh is None else self.device_mesh["dp_shard"].get_local_rank()

  def shard_count(self) -> int:
    return 1 if self.device_mesh is None else self.device_mesh["dp_shard"].size()

  def shard_all_reduce_max(self, value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.long, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.dp_group())
    return int(tensor.item())

  def shard_all_reduce_sum(self, value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group())
    return float(tensor.item())

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    gathered: list[Any] = [None] * self.shard_count()
    dist.all_gather_object(gathered, value, group=self.dp_group())
    return gathered

  def build_peft_config(self, config: LoraConfig, base_model_name: str) -> Any:
    from nemo_automodel.components._peft.lora import PeftConfig

    hf_config = AutoConfig.from_pretrained(base_model_name)
    tied = getattr(hf_config, "tie_word_embeddings", None)
    if tied is None:
      tied = getattr(hf_config.get_text_config(), "tie_word_embeddings", False)
    return PeftConfig(
      target_modules=lora_target_patterns(config, bool(tied)),
      dim=config.rank,
      alpha=config.lora_alpha,
      dropout=config.lora_dropout,
      lora_A_init="kaiming",
      use_triton=False,
    )

  def load_kwargs(self, base_model_name: str) -> dict[str, Any]:
    """from_pretrained arguments every copy of the model shares, sharded or not."""
    kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16, "use_liger_kernel": False}
    if self.peft_config is not None:
      kwargs["peft_config"] = self.peft_config
    # Qwen3.5 ships a multi-token-prediction head that would run over the whole
    # sequence on every forward; the loss never reads it.
    if AutoConfig.from_pretrained(base_model_name).get_text_config().model_type.startswith("qwen3_5"):
      kwargs["num_nextn_predict_layers"] = 0
    return kwargs

  def load_base_model(self, base_model_name: str) -> None:
    """The processor preloads BASE_MODEL before any create_model arrives. Automodel
    applies LoRA at load, before sharding, so the load itself waits for the
    client's LoraConfig; only the tokenizer is fetched here."""
    if self.tokenizer is None or self.base_model_name != base_model_name:
      self.base_model_name = base_model_name
      self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

  def load_model(self, base_model_name: str) -> None:
    NeMoAutoModelForCausalLM = require_automodel()
    torch.cuda.set_device(self.device)
    self.load_base_model(base_model_name)
    if self.distributed_setup is None:
      self.distributed_setup = self.build_distributed_setup(base_model_name)
      self.device_mesh = self.distributed_setup.mesh_context.device_mesh
    # from_pretrained applies LoRA before FSDP2 shards anything, loads the base
    # weights and freezes all but the adapters.
    self.model = NeMoAutoModelForCausalLM.from_pretrained(
      base_model_name, distributed_setup=self.distributed_setup, **self.load_kwargs(base_model_name)
    )
    if RECOMPUTE_NUM_LAYERS > 0:
      install_group_checkpointing(self.model, RECOMPUTE_NUM_LAYERS)
    shape = f"LoRA rank {self.lora_config.rank}" if self.lora_config else "full fine-tuning"
    print(f"Loaded Automodel {base_model_name} ({shape}, {'packed' if AUTOMODEL_PACKED else 'padded'} batches).")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: LoraConfig | FFTConfig | None = None) -> None:
    """Load the model for the client's config. One process serves one shape:
    LoRA is applied before sharding, so a different base or LoraConfig needs a fresh trainer."""
    lora_config = config if isinstance(config, LoraConfig) else None
    if self.model is None:
      self.lora_config = lora_config
      self.peft_config = self.build_peft_config(lora_config, base_model_name) if lora_config is not None else None
      self.load_model(base_model_name)
    elif (self.base_model_name, self.lora_config) != (base_model_name, lora_config):
      raise RuntimeError(
        f"This Automodel trainer holds {self.base_model_name} with {self.lora_config}; restart it for {base_model_name} with {lora_config}."
      )
    torch.manual_seed(config.seed if config is not None and config.seed is not None else AUTOMODEL_SEED)
    if not self.is_lora:
      for param in self.model.parameters():
        param.requires_grad_(True)
    self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
    if not self.trainable_params:
      raise ValueError("No trainable parameters found in the Automodel model")
    self.optimizer = None

  # -- the step --------------------------------------------------------------------

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    return super().forward_backward(self.model, data, loss_fn, loss_config, forward_only=forward_only)

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    """Padded passes are bounded by longest x count, as the base class does. A packed pass is bounded by its total tokens."""
    if not AUTOMODEL_PACKED:
      return super().make_training_batches(data)
    token_budget = int(os.getenv("OPEN_RL_TRAIN_TOKEN_BUDGET", "0"))
    if len(data) <= 1 or token_budget <= 0:
      return [[(idx, datum)] for idx, datum in enumerate(data)]
    batches: list[list[tuple[int, Datum]]] = []
    batch: list[tuple[int, Datum]] = []
    tokens = 0
    for idx, datum in sorted(enumerate(data), key=lambda item: len(item[1].model_input)):
      length = len(datum.model_input)
      if batch and tokens + length > token_budget:
        batches.append(batch)
        batch, tokens = [], 0
      batch.append((idx, datum))
      tokens += length
    if batch:
      batches.append(batch)
    return batches

  def collate(self, batch_data: list[Datum]) -> tuple[dict[str, Any], list[int]]:
    """One pass's datums through Automodel's collater, on the device, plus each datum's scored length."""
    from nemo_automodel.components.datasets.datum import Datum as AutomodelDatum
    from nemo_automodel.components.datasets.datum import collate_datums

    converted = [datum_inputs(datum) for datum in batch_data]
    datums = [
      AutomodelDatum(torch.tensor(ids, dtype=torch.long), {key: torch.tensor(values) for key, values in inputs.items()})
      for ids, inputs, _ in converted
    ]
    batch = collate_datums(datums, packed=AUTOMODEL_PACKED)
    batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    return batch, [scored for _, _, scored in converted]

  def run_batches(
    self,
    model: torch.nn.Module,
    data: list[Datum],
    loss_fn: str,
    loss_config: dict | None,
    forward_only: bool,
    loss_fn_outputs: list[dict[str, Any] | None],
  ) -> float:
    """The base loop over Automodel's collated batches.

    Each DP rank runs a round-robin shard of the datums and scales its loss by
    the shard count, which undoes FSDP's gradient mean; short ranks run
    zero-scaled filler passes so every rank makes the same number of backward calls.
    """
    shard_count = self.shard_count()
    local_indices = shard_datum_indices(len(data), self.shard_rank(), shard_count)
    local_data = [data[idx] for idx in local_indices]
    local_batches = self.make_training_batches(local_data)
    if shard_count > 1:
      filler_passes = self.shard_all_reduce_max(len(local_batches)) - len(local_batches)
      filler = local_data[0] if local_data else data[0]
      local_batches.extend([[(FILLER_DATUM_INDEX, filler)]] * filler_passes)

    total_loss = 0.0
    for batch in local_batches:
      batch_positions = [idx for idx, _ in batch]
      batch_data = [datum for _, datum in batch]
      is_filler = batch_positions == [FILLER_DATUM_INDEX]
      batch_indices = [] if is_filler else [local_indices[position] for position in batch_positions]

      collated, scored = self.collate(batch_data)
      # Masked positions carry ignore_index in labels; any token id works there since their weight is zero.
      target_logprobs = self.compute_target_logprobs(model, collated["labels"].clamp_min(0), **model_inputs(collated))
      side_inputs = {key: collated[key] for key in ("logprobs", "advantages") if key in collated}
      loss = losses.elementwise_loss(loss_fn, target_logprobs, collated["weights"], side_inputs, loss_config).sum()
      if not forward_only:
        (loss * (0.0 if is_filler else float(shard_count))).backward()
      if not is_filler:
        total_loss += loss.item()

      seq_lens = [int(n) for n in collated["seq_lens"].flatten().tolist() if n > 0] if AUTOMODEL_PACKED else None
      for original_idx, values in zip(batch_indices, split_rows(target_logprobs.detach().cpu(), seq_lens, scored)[: len(batch_indices)], strict=True):
        loss_fn_outputs[original_idx] = {"logprobs": {"data": values, "dtype": "float32", "shape": [len(values)]}}

    if shard_count > 1:
      total_loss = self.shard_all_reduce_sum(total_loss)
      for part in self.shard_all_gather_object({idx: loss_fn_outputs[idx] for idx in local_indices}):
        for idx, output in part.items():
          loss_fn_outputs[idx] = output
    return total_loss

  def compute_target_logprobs(self, model: torch.nn.Module, target_token_ids: torch.Tensor, **inputs: Any) -> torch.Tensor:
    """Per-position logprob of each target, projected from the final hidden states in chunks.

    inputs are the forward kwargs from model_inputs: input_ids with either an
    attention_mask (padded) or position_ids/seq_lens/qkv_format (packed).
    """
    seq_len = target_token_ids.shape[1]
    outputs = model(**inputs, use_cache=False, logits_to_keep=1, output_hidden_states=True)
    hidden = outputs.hidden_states[-1][:, :seq_len]
    if is_dtensor(hidden):
      hidden = hidden.full_tensor()
    head = model.get_output_embeddings()
    weight = head.weight.full_tensor() if is_dtensor(head.weight) else head.weight
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    batch = hidden.shape[0]
    flat_hidden = hidden.reshape(batch * seq_len, -1)
    flat_targets = target_token_ids.reshape(batch * seq_len)
    chunks = []
    for start in range(0, flat_hidden.shape[0], LOGPROB_CHUNK):
      args = (flat_hidden[start : start + LOGPROB_CHUNK], weight, flat_targets[start : start + LOGPROB_CHUNK], softcap)
      chunks.append(
        torch.utils.checkpoint.checkpoint(chunk_target_logprob, *args, use_reentrant=False)
        if torch.is_grad_enabled()
        else chunk_target_logprob(*args)
      )
    return torch.cat(chunks).reshape(batch, seq_len)

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if self.optimizer is None:
      # foreach=False: the fused path cannot mix DTensors from different meshes.
      self.optimizer = torch.optim.AdamW(
        self.trainable_params,
        lr=adam_params.get("learning_rate", 1e-4),
        betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
        eps=adam_params.get("eps", 1e-12),
        weight_decay=adam_params.get("weight_decay", 0.0),
        foreach=False,
      )
    if adam_params.get("learning_rate") is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = adam_params["learning_rate"]

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    total_norm = self.clip_gradients(max_grad_norm if max_grad_norm > 0 else math.inf)
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    return {"metrics": {"grad_norm:mean": self.sanitize_float(total_norm)}}

  def clip_gradients(self, max_grad_norm: float) -> float:
    """Global grad norm over DTensors that may live on different meshes.

    clip_grad_norm_ stacks per-tensor norms, which fails when TP-sharded and
    replicated parameters sit side by side. A DTensor's vector_norm already
    reduces over its own placements, so only the stack of scalars is local.
    """
    norms = []
    for param in self.trainable_params:
      if param.grad is None:
        continue
      norm = torch.linalg.vector_norm(param.grad.detach().float())
      norms.append(norm.full_tensor() if is_dtensor(norm) else norm)
    if not norms:
      return 0.0
    total_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
      for param in self.trainable_params:
        if param.grad is not None:
          param.grad.mul_(clip_coef)
    return total_norm

  def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Sampling from the Automodel trainer is unsupported; use the vLLM sampler.")

  # -- checkpoints -----------------------------------------------------------------

  def get_checkpointer(self) -> Any:
    """Automodel's Checkpointer gathers DTensor shards, maps native keys back to
    the hub layout, and writes a PEFT adapter that vLLM loads."""
    if self.checkpointer is None:
      from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

      config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.path.join(TMP_DIR, "automodel"),
        model_save_format="safetensors",
        save_consolidated=not self.is_lora,
        is_peft=self.is_lora,
        model_repo_id=self.base_model_name,
      )
      self.checkpointer = Checkpointer(config, dp_rank=self.shard_rank(), tp_rank=self.device_mesh["tp"].get_local_rank(), pp_rank=0)
    return self.checkpointer

  def write_weights(self, path: str) -> None:
    """Write the adapter (or the consolidated model) as plain files in path.

    Every rank calls this because the gathers are collective. The checkpointer
    nests its output under model/, which rank 0 lifts into path.
    """
    self.get_checkpointer().save_model(self.model, weights_path=path, peft_config=self.peft_config, tokenizer=self.tokenizer)
    if is_primary():
      model_dir = os.path.join(path, "model")
      source = model_dir if self.is_lora else os.path.join(model_dir, "consolidated")
      for entry in os.listdir(source):
        os.replace(os.path.join(source, entry), os.path.join(path, entry))
      shutil.rmtree(model_dir, ignore_errors=True)
    barrier()

  def write_staged(self, path: str, metadata: dict[str, Any] | None = None) -> None:
    """Write into a staging dir and rename it over path, so a reader never
    sees a half-written directory."""
    staging, previous = f"{path}.staging-{os.getpid()}", f"{path}.previous-{os.getpid()}"
    if is_primary():
      shutil.rmtree(staging, ignore_errors=True)
      os.makedirs(staging)
    barrier()
    self.write_weights(staging)
    if is_primary():
      if metadata is not None:
        with open(os.path.join(staging, "metadata.json"), "w") as f:
          json.dump(metadata, f)
      replace_dir(staging, path, previous)
    barrier()

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    # Weights only for now; the optimizer state is not saved.
    self.write_staged(state_path, self.metadata(model_id, kind))
    return {"path": state_path}

  def load_from_state(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise NotImplementedError("The Automodel trainer does not load checkpoints yet.")

  def save_for_sampler(self, model_id: str, alias: str | None, ref: str | None) -> str | None:
    """Write the adapter where the LoRA sampler hot-loads it, peft/<id>/<id>.

    With OPEN_RL_AUTOMODEL_SAMPLER_WEIGHTS=merged the adapter is also folded
    into the base weights under sampler_full/<ref>, and that path is returned
    so the processor announces it the way it announces full checkpoints.
    """
    if not self.is_lora:
      raise NotImplementedError("The Automodel trainer publishes sampler weights for LoRA only.")
    adapter_dir = os.path.join(TMP_DIR, "peft", model_id, model_id)
    self.write_staged(adapter_dir)
    if SAMPLER_WEIGHTS != "merged":
      return None
    if not ref:
      raise ValueError("save_weights_for_sampler requires path or sampling_session_id")
    rel_path = ref[len("tinker://") :] if ref.startswith("tinker://") else ref.lstrip("/")
    local_path = os.path.join(TMP_DIR, "sampler_full", rel_path)
    if is_primary():
      self.export_merged(adapter_dir, local_path, self.metadata(model_id, "sampler"))
    barrier()
    return local_path

  def export_merged(self, adapter_dir: str, path: str, metadata: dict[str, Any]) -> None:
    """The adapter folded into the base weights as a plain HF checkpoint.

    Automodel merges in place and only on an unsharded model, so each export
    builds a fresh CPU copy of the base from the hub cache. That is a minute
    or two for a 9B model, paid by rank 0 while the others wait.
    """
    from nemo_automodel._transformers.peft_export import export_merged_peft_checkpoint

    NeMoAutoModelForCausalLM = require_automodel()
    started = time.perf_counter()
    staging, previous = f"{path}.staging-{os.getpid()}", f"{path}.previous-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    model = NeMoAutoModelForCausalLM.from_pretrained(self.base_model_name, **self.load_kwargs(self.base_model_name))
    try:
      export_merged_peft_checkpoint(model, adapter_path=adapter_dir, output_dir=staging)
    finally:
      del model
    if self.tokenizer is not None:
      self.tokenizer.save_pretrained(staging)
    with open(os.path.join(staging, "metadata.json"), "w") as f:
      json.dump({**metadata, "format": "merged"}, f)
    replace_dir(staging, path, previous)
    print(f"Exported merged weights to {path} in {time.perf_counter() - started:.0f}s")

  def metadata(self, model_id: str, kind: str) -> dict[str, Any]:
    return {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": False,
      "model_id": model_id,
      "timestamp": time.time(),
    }


def replace_dir(staging: str, path: str, previous: str) -> None:
  """Rename staging over path in two moves, so path is never half-written or missing for long."""
  if os.path.exists(path):
    os.rename(path, previous)
  os.rename(staging, path)
  shutil.rmtree(previous, ignore_errors=True)
