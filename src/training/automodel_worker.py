"""NeMo Automodel backend for LoRA and full-parameter training.

Only DP shards datums; CP and TP ranks cooperate on the same sequence. FSDP2
shards weights over DP x CP. Automodel's ContextParallelSharder owns input
layout, attention transport, and the differentiable gather into token order.
Every CP rank computes the same loss, so the gather's backward sums CP copies
to cancel FSDP2's CP average; BaseTrainerWorker cancels the DP average.

The final norm supplies hidden states without retaining every layer output.
Checkpointed vocabulary projections avoid full [sequence, vocabulary] logits.
The attention context stays active through backward and checkpoint recompute.

Dependencies live in scripts/setup_automodel_env.sh's separate interpreter.
After changes to parallel execution, run scripts/automodel_probe.py against a
single-GPU reference; prior measurements are in docs/reports/run49 and run51.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import shutil
import time
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from training import adapter_snapshot, paths
from training import automodel_models as models
from training.distributed import barrier, broadcast_object, is_primary
from training.trainer_worker import BaseTrainerWorker, Datum, chunk_target_logprob

if TYPE_CHECKING:
  from transformers import PreTrainedConfig

# Parallel layout. DP is inferred as world / (CP * TP).
AUTOMODEL_TP = int(os.getenv("OPEN_RL_AUTOMODEL_TP", "1"))
AUTOMODEL_CP = int(os.getenv("OPEN_RL_AUTOMODEL_CP", "1"))
AUTOMODEL_PP = int(os.getenv("OPEN_RL_AUTOMODEL_PP", "1"))
AUTOMODEL_SEED = int(os.getenv("OPEN_RL_AUTOMODEL_SEED", "1234"))

# LoRA. Rank 0 trains full parameters and publishes whole checkpoints; any
# positive rank freezes the base, trains adapters, and publishes an adapter the
# sampler hot-loads. Both ends read this one variable so they cannot disagree
# about which route sampler weights take. Alpha 16 and the kaiming A init are
# PEFT's defaults, which is what the FSDP worker and (since d25756d) the
# Megatron worker train with, so an lr means the same thing on every backend.
AUTOMODEL_LORA_RANK = int(os.getenv("OPEN_RL_AUTOMODEL_LORA_RANK", "0"))
AUTOMODEL_LORA_ALPHA = int(os.getenv("OPEN_RL_AUTOMODEL_LORA_ALPHA", "16"))
AUTOMODEL_LORA_DROPOUT = float(os.getenv("OPEN_RL_AUTOMODEL_LORA_DROPOUT", "0.0"))
AUTOMODEL_LORA_A_INIT = os.getenv("OPEN_RL_AUTOMODEL_LORA_A_INIT", "kaiming")
AUTOMODEL_LORA_TARGETS = os.getenv("OPEN_RL_AUTOMODEL_LORA_TARGETS", models.DEFAULT_LORA_TARGETS)

# The model policy selects grouped or upstream checkpointing; 0 disables it.
RECOMPUTE_NUM_LAYERS = int(os.getenv("OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS", "4"))
if os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") != "1":
  RECOMPUTE_NUM_LAYERS = 0

# Rows of hidden states projected through the vocab per chunk. The FSDP path
# defaults to 128 for small GPUs; 1024 is a 1 GiB fp32 logits chunk on a 248k
# vocab, which an H200 does not notice, and cuts the Python loop 8x.
LOGPROB_CHUNK = int(os.getenv("OPEN_RL_LOGPROB_CHUNK", "1024"))

# "auto" delegates attention selection to the model policy.
AUTOMODEL_ATTN = os.getenv("OPEN_RL_AUTOMODEL_ATTN", "auto")


def require_automodel():
  """Import NeMo Automodel, or explain how to get it."""
  try:
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
  except ImportError as exc:
    raise RuntimeError(
      "OPEN_RL_TRAINER_BACKEND=automodel needs nemo-automodel, which is not a dependency of this "
      "project (no extra in pyproject.toml installs it). Build the trainer interpreter with "
      f"scripts/setup_automodel_env.sh and point AUTOMODEL_PYTHON at it. Import failed with: {exc}"
    ) from exc
  return NeMoAutoModelForCausalLM


def is_dtensor(tensor: Any) -> bool:
  from torch.distributed.tensor import DTensor

  return isinstance(tensor, DTensor)


class AutomodelConfig(BaseModel):
  seed: int | None = None


class AutomodelTrainingWorker(BaseTrainerWorker):
  config_class = AutomodelConfig

  # FSDP2 reduces gradients inside backward, so pass counts must match across DP
  # ranks; the base class pads short ranks with zero-scaled filler passes.
  backward_runs_collectives = True

  def __init__(self):
    super().__init__()
    for name, value, minimum in (
      ("OPEN_RL_AUTOMODEL_TP", AUTOMODEL_TP, 1),
      ("OPEN_RL_AUTOMODEL_CP", AUTOMODEL_CP, 1),
      ("OPEN_RL_AUTOMODEL_LORA_RANK", AUTOMODEL_LORA_RANK, 0),
      ("OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS", RECOMPUTE_NUM_LAYERS, 0),
      ("OPEN_RL_LOGPROB_CHUNK", LOGPROB_CHUNK, 1),
    ):
      if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    if AUTOMODEL_PP != 1:
      raise RuntimeError(
        f"The Automodel backend supports data, context, and tensor parallelism (got PP={AUTOMODEL_PP}). "
        "Pipeline parallelism moves the logits off the rank that builds the loss."
      )
    if AUTOMODEL_LORA_RANK == 0 and AUTOMODEL_TP > 1:
      raise ValueError("Full-parameter Automodel training requires TP=1; the trainable output head does not support TP gathering.")
    self.model: torch.nn.Module | None = None
    self.distributed_setup: Any = None
    self.device_mesh: Any = None
    self.peft_config: Any = None
    self.checkpointer: Any = None
    self.hf_config: PreTrainedConfig | None = None
    self.forward_kwargs: dict[str, Any] = {}
    self.cp_context: contextlib.ExitStack | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.is_lora = AUTOMODEL_LORA_RANK > 0
    self.tp_size = AUTOMODEL_TP
    self.cp_size = AUTOMODEL_CP

  # -- distributed layout ---------------------------------------------------

  def build_distributed_setup(self, policy: models.ModelPolicy) -> Any:
    """Build the process's FSDP2 mesh with the initial model policy."""
    if not dist.is_initialized():
      raise RuntimeError("The Automodel backend runs under torchrun; torch.distributed is not initialized.")
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    world = dist.get_world_size()
    if world % (self.cp_size * self.tp_size):
      raise RuntimeError(f"WORLD_SIZE={world} is not divisible by OPEN_RL_AUTOMODEL_CP={self.cp_size} * OPEN_RL_AUTOMODEL_TP={self.tp_size}")
    strategy = FSDP2Config(
      activation_checkpointing=policy.activation_checkpointing,
      tp_plan=policy.tp_plan,
    )
    mesh_context = MeshContext.build(strategy, ParallelismSizes(tp_size=self.tp_size, cp_size=self.cp_size), world_size=world)
    print(f"Automodel device mesh: DP={world // (self.cp_size * self.tp_size)} CP={self.cp_size} TP={self.tp_size}")
    return DistributedSetup(
      mesh_context=mesh_context,
      strategy_config=strategy,
      activation_checkpointing=strategy.activation_checkpointing,
    )

  def load_hf_config(self, base_model_name: str) -> PreTrainedConfig:
    if self.hf_config is None or self.hf_config.name_or_path != base_model_name:
      self.hf_config = AutoConfig.from_pretrained(base_model_name)
      self.hf_config.name_or_path = base_model_name
    return self.hf_config

  def dp_mesh(self):
    return self.device_mesh["dp_shard"]

  def dp_group(self):
    return self.dp_mesh().get_group()

  # -- data-parallel geometry (BaseTrainerWorker hooks) ---------------------
  #
  # Scoped to the DP axis. CP and TP ranks collaborate on one model over one
  # sequence and must see identical datums, so only the DP dimension shards data.

  def shard_rank(self) -> int:
    return 0 if self.device_mesh is None else self.dp_mesh().get_local_rank()

  def shard_count(self) -> int:
    return 1 if self.device_mesh is None else self.dp_mesh().size()

  def shard_all_reduce_max(self, value: int) -> int:
    if self.shard_count() == 1:
      return value
    tensor = torch.tensor([value], dtype=torch.long, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.dp_group())
    return int(tensor.item())

  def shard_all_reduce_sum(self, value: float) -> float:
    if self.shard_count() == 1:
      return value
    tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.dp_group())
    return float(tensor.item())

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    if self.shard_count() == 1:
      return [value]
    gathered: list[Any] = [None] * self.shard_count()
    dist.all_gather_object(gathered, value, group=self.dp_group())
    return gathered

  # -- model construction ---------------------------------------------------

  def build_peft_config(self) -> Any:
    from nemo_automodel.components._peft.lora import PeftConfig

    names = [name.strip() for name in AUTOMODEL_LORA_TARGETS.split(",") if name.strip()]
    return PeftConfig(
      target_modules=models.lora_target_modules(names),
      dim=AUTOMODEL_LORA_RANK,
      alpha=AUTOMODEL_LORA_ALPHA,
      dropout=AUTOMODEL_LORA_DROPOUT,
      lora_A_init=AUTOMODEL_LORA_A_INIT,
      use_triton=False,
    )

  def load_base_model(self, base_model_name: str) -> None:
    if self.model is not None and self.base_model_name == base_model_name:
      print(f"Automodel model {base_model_name} already loaded.")
      return

    if self.model is not None:
      self.release_model()
    NeMoAutoModelForCausalLM = require_automodel()
    torch.cuda.set_device(self.device)
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    config = self.load_hf_config(base_model_name)
    policy = models.model_policy(
      config, tp_size=self.tp_size, cp_size=self.cp_size, recompute_num_layers=RECOMPUTE_NUM_LAYERS, attention=AUTOMODEL_ATTN
    )
    if self.distributed_setup is None:
      self.distributed_setup = self.build_distributed_setup(policy)
      self.device_mesh = self.distributed_setup.mesh_context.device_mesh
    else:
      strategy = replace(
        self.distributed_setup.strategy_config,
        activation_checkpointing=policy.activation_checkpointing,
        tp_plan=policy.tp_plan,
      )
      self.distributed_setup = replace(self.distributed_setup, strategy_config=strategy, activation_checkpointing=policy.activation_checkpointing)
    if self.is_lora and self.peft_config is None:
      self.peft_config = self.build_peft_config()
    print(f"Loading Automodel {base_model_name} (rank {os.getenv('RANK', '0')}/{os.getenv('WORLD_SIZE', '1')})...")

    # NeMo applies adapters before FSDP sharding. The model policy
    # provides architecture-specific constructor and forward options.
    self.forward_kwargs = policy.forward_kwargs
    print(f"[Automodel Worker] model options: {policy.load_kwargs}")
    self.model = NeMoAutoModelForCausalLM.from_pretrained(
      base_model_name,
      torch_dtype=torch.bfloat16,
      use_liger_kernel=False,
      distributed_setup=self.distributed_setup,
      peft_config=self.peft_config,
      **policy.load_kwargs,
    )
    policy.apply(self.model)
    self.model.train()
    if self.is_lora:
      trainable = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
      total = sum(param.numel() for param in self.model.parameters())
      print(
        f"[Automodel Worker] LoRA rank={AUTOMODEL_LORA_RANK} alpha={AUTOMODEL_LORA_ALPHA} on "
        f"{self.peft_config.target_modules}: {trainable:,} trainable of {total:,} ({100 * trainable / max(1, total):.3f}%)."
      )
    print("Successfully loaded Automodel.")

  def create_model(self, base_model_name: str, model_id: str | None = None, config: AutomodelConfig | None = None) -> None:
    self.release_model()
    seed = config.seed if config is not None and config.seed is not None else AUTOMODEL_SEED
    torch.manual_seed(seed)
    self.load_base_model(base_model_name)
    self.prepare_model_for_training()

  def release_model(self) -> None:
    """Release weights and optimizer references before loading another model."""
    self.close_cp_context()
    self.tokenizer = None
    self.forward_kwargs = {}
    self.optimizer = None
    self.trainable_params = []
    self.model = None
    self.base_model_name = None
    self.checkpointer = None
    self.peft_config = None
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  def prepare_model_for_training(self) -> None:
    assert self.model is not None, "Model is not loaded. Call load_base_model first."
    if not self.is_lora:
      for param in self.model.parameters():
        param.requires_grad_(True)
    self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]
    if not self.trainable_params:
      raise ValueError("No trainable parameters found in the Automodel model")
    self.model.train()

  # -- forward / backward ---------------------------------------------------

  def make_training_batches(self, data: list[Datum]) -> list[list[tuple[int, Datum]]]:
    # The CP path shards one unpadded sequence; packing several datums into a
    # padded batch would need the padding mask sharded alongside.
    if self.cp_size > 1:
      return [[(idx, datum)] for idx, datum in enumerate(data)]
    return super().make_training_batches(data)

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    try:
      res = super().forward_backward(self.model, data, loss_fn, loss_config, forward_only=forward_only)
    finally:
      self.close_cp_context()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def close_cp_context(self) -> None:
    if self.cp_context is not None:
      self.cp_context.close()
      self.cp_context = None

  def compute_target_logprobs(
    self,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return [batch, seq] target logprobs.

    Targets are already position-aligned by the Tinker client (input_ids[t]
    predicts target_token_ids[t]) and no shifting happens here.
    """
    seq_len = target_token_ids.shape[1]
    input_ids = input_ids[:, :seq_len]
    if attention_mask is not None:
      attention_mask = attention_mask[:, :seq_len]
    if self.cp_size > 1:
      return self.compute_target_logprobs_cp(model, input_ids, target_token_ids)

    # A dense all-ones mask only describes ordinary causal attention; omitting
    # it lets SDPA pick flash attention instead of building an additive mask.
    if attention_mask is not None and bool(attention_mask.all()):
      attention_mask = None
    hidden = models.forward_hidden_states(model, self.forward_kwargs, input_ids=input_ids, attention_mask=attention_mask)
    return self.project_target_logprobs(model, hidden, target_token_ids)

  def head_weight(self, weight: torch.Tensor) -> torch.Tensor:
    """The lm_head weight as one plain [vocab, hidden] tensor on this rank."""
    if not is_dtensor(weight):
      return weight
    if weight.requires_grad:
      # A trained head needs gradient-correct reduction semantics for the
      # gather; Automodel's helper has them for the DP/CP case and refuses TP.
      from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

      return FusedLinearCrossEntropy.materialize_lm_weight(weight, grad_reduce_group=self.device_mesh["dp_shard_cp"].get_group())
    return weight.full_tensor()

  def project_target_logprobs(self, model: torch.nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Project the target-length prefix in checkpointed vocabulary chunks."""
    if is_dtensor(hidden):
      hidden = hidden.full_tensor()
    hidden = hidden[:, : targets.shape[1]]
    projection = models.output_projection(model)
    weight = self.head_weight(projection.weight)
    bias = self.head_weight(projection.bias) if projection.bias is not None else None

    batch, seq_len, _ = hidden.shape
    flat_hidden = hidden.reshape(batch * seq_len, -1).to(weight.dtype)
    flat_targets = targets.reshape(batch * seq_len)
    needs_grad = flat_hidden.requires_grad or weight.requires_grad

    def project(start: int) -> torch.Tensor:
      args = (flat_hidden[start : start + LOGPROB_CHUNK], weight, bias, flat_targets[start : start + LOGPROB_CHUNK], projection.softcap)
      if needs_grad:
        return torch.utils.checkpoint.checkpoint(chunk_target_logprob, *args, use_reentrant=False)
      return chunk_target_logprob(*args)

    return torch.cat([project(start) for start in range(0, flat_hidden.shape[0], LOGPROB_CHUNK)]).reshape(batch, seq_len)

  def compute_target_logprobs_cp(self, model: torch.nn.Module, input_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Return [batch, seq] target logprobs with the sequence sharded over CP."""
    if input_ids.shape[0] != 1:
      raise RuntimeError(f"The Automodel CP path takes one sequence per pass, got a batch of {input_ids.shape[0]}.")
    from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder

    self.close_cp_context()
    batch = {"input_ids": input_ids, "labels": target_token_ids.clone()}
    sharder = ContextParallelSharder(model=model, device_mesh=self.device_mesh, batch=batch)
    context, batch = sharder.shard(batch)
    inputs = {key: value for key, value in batch.items() if key not in ("labels", "loss_mask")}
    # The context must outlive this call. It swaps SDPA for ring attention by
    # dispatch, and the backward dispatches SDPA again, both for the attention
    # gradient and for the activation-checkpoint recompute of the forward.
    # Closed after the forward, both run plain local attention over this
    # rank's shard and every gradient upstream of the last attention layer is
    # silently wrong (measured: cosine 0.24 on o_proj adapters, logprobs fine).
    # forward_backward closes it once this pass's backward is done.
    self.cp_context = contextlib.ExitStack()
    self.cp_context.enter_context(context())
    local_hidden = models.forward_hidden_states(model, self.forward_kwargs, **inputs)
    local_targets = batch["labels"]
    # Padding slots carry an ignore index; they are cut off after the gather.
    local_logprobs = self.project_target_logprobs(model, local_hidden, local_targets.clamp_min(0))

    return sharder.gather_token_tensor(local_logprobs, trim=True)

  # -- optimizer ------------------------------------------------------------

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if not self.trainable_params:
      self.trainable_params = [param for param in self.model.parameters() if param.requires_grad]

    if self.optimizer is None:
      lr = adam_params.get("learning_rate", 1e-4)
      print(f"Initializing AdamW for Automodel with lr={lr}")
      # A plain AdamW over the sharded DTensor params is the standard FSDP2 step.
      self.optimizer = torch.optim.AdamW(
        self.trainable_params,
        lr=lr,
        betas=(adam_params.get("beta1", 0.9), adam_params.get("beta2", 0.95)),
        eps=adam_params.get("eps", 1e-12),
        weight_decay=adam_params.get("weight_decay", 0.0),
        foreach=False,
      )

    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or math.inf
    if max_grad_norm <= 0.0:
      max_grad_norm = math.inf
    total_norm = self.clip_gradients(max_grad_norm)

    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    return {"metrics": {"grad_norm:mean": self.sanitize_float(total_norm)}}

  def clip_gradients(self, max_grad_norm: float) -> float:
    """Global grad norm over DTensors that may live on different meshes.

    torch's clip_grad_norm_ stacks per-tensor norms, which fails across meshes
    (TP-sharded projections next to replicated GDN weights). A per-tensor
    vector_norm on a DTensor already reduces over its own placements, so every
    rank gets the same scalar and nothing is counted twice; only the stack of
    those scalars is local.
    """
    norms = []
    for param in self.trainable_params:
      if param.grad is None:
        continue
      norm = torch.linalg.vector_norm(param.grad.detach().float())
      if is_dtensor(norm):
        norm = norm.full_tensor()
      norms.append(norm)
    if not norms:
      return 0.0
    total_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
      for param in self.trainable_params:
        if param.grad is not None:
          param.grad.mul_(clip_coef)
    return total_norm

  # -- checkpointing --------------------------------------------------------

  def get_checkpointer(self) -> Any:
    """One Automodel Checkpointer for the process.

    It knows how to gather DTensor shards, rename the native model's keys back
    to the hub layout, and for LoRA write the PEFT adapter_config.json plus
    adapter_model.safetensors that vLLM and PEFT load.
    """
    if self.checkpointer is None:
      from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

      config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=paths.checkpoint_root(),
        model_save_format="safetensors",
        save_consolidated=not self.is_lora,
        is_peft=self.is_lora,
        model_repo_id=self.base_model_name,
      )
      self.checkpointer = Checkpointer(
        config,
        dp_rank=self.shard_rank(),
        tp_rank=self.device_mesh["tp"].get_local_rank(),
        pp_rank=0,
      )
    return self.checkpointer

  def write_weights(self, save_path: str) -> None:
    """Write the adapter (LoRA) or the consolidated HF model (full) into save_path.

    Every rank calls this: the DTensor gathers inside are collective. The
    checkpointer nests its output under model/ (and model/consolidated/ for a
    full model); it is lifted to save_path so the directory is a plain PEFT
    adapter or a plain HF checkpoint.
    """
    checkpointer = self.get_checkpointer()
    checkpointer.save_model(self.model, weights_path=save_path, peft_config=self.peft_config, tokenizer=self.tokenizer)
    if is_primary():
      model_dir = os.path.join(save_path, "model")
      source = model_dir if self.is_lora else os.path.join(model_dir, "consolidated")
      for entry in os.listdir(source):
        os.replace(os.path.join(source, entry), os.path.join(save_path, entry))
      shutil.rmtree(model_dir, ignore_errors=True)
    barrier()

  def full_optimizer_state_dict(self) -> dict[str, Any]:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_optimizer_state_dict(self.model, self.optimizer, options=options)

  def save_checkpoint(self, path: str, metadata: dict[str, Any], include_optimizer: bool = False) -> dict[str, Any]:
    assert self.model is not None, "Model must be loaded first."
    # Atomic staging, same contract as the other workers: a save killed
    # mid-write must never leave a directory that loads as mixed old and new.
    staging_path = broadcast_object(f"{path}.staging-{os.getpid()}-{time.time_ns()}")
    previous_path = f"{staging_path}.previous"
    if is_primary():
      shutil.rmtree(staging_path, ignore_errors=True)
      os.makedirs(staging_path, exist_ok=True)
    barrier()
    try:
      self.write_weights(staging_path)
      if include_optimizer and self.optimizer is not None:
        optimizer_state = self.full_optimizer_state_dict()
        if is_primary():
          torch.save(optimizer_state, os.path.join(staging_path, "optimizer.pt"))
      if is_primary():
        with open(os.path.join(staging_path, "metadata.json"), "w") as f:
          json.dump(metadata, f)
        if os.path.exists(path):
          os.rename(path, previous_path)
        try:
          os.rename(staging_path, path)
        except OSError:
          if os.path.exists(previous_path):
            os.rename(previous_path, path)
          raise
        shutil.rmtree(previous_path, ignore_errors=True)
    finally:
      if is_primary():
        shutil.rmtree(staging_path, ignore_errors=True)
    barrier()
    print(f"Saved Automodel state to {path}")
    return {"path": path}

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    name = alias or "automodel-model"
    save_path = name if os.path.isabs(name) else os.path.join(paths.tmp_dir(), "automodel", name)
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": "weights",
      "lora": self.is_lora,
      "model_id": alias,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(save_path, metadata)

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "lora": self.is_lora,
      "has_optimizer": include_optimizer and self.optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(state_path, metadata, include_optimizer)

  def validate_adapter_config(self, state_path: str) -> None:
    """Reject adapters whose scaling or training configuration would change on load."""
    with open(os.path.join(state_path, "adapter_config.json")) as f:
      saved = json.load(f)
    config = self.build_peft_config()
    expected = {"peft_type": "LORA", "r": config.dim, "lora_alpha": config.alpha, "use_dora": config.use_dora}
    for key, value in expected.items():
      if saved.get(key, False if key == "use_dora" else None) != value:
        raise ValueError(f"Adapter {key}={saved.get(key)!r} does not match worker configuration {value!r}")
    if saved.get("use_rslora") or saved.get("rank_pattern") or saved.get("alpha_pattern") or saved.get("bias", "none") != "none":
      raise ValueError("This worker requires uniform LoRA rank/alpha, standard scaling, and bias='none'")
    automodel_config_path = os.path.join(state_path, "automodel_peft_config.json")
    if os.path.exists(automodel_config_path):
      with open(automodel_config_path) as f:
        training_config = json.load(f)
      dropout = training_config.get("dropout", 0.0)
      dropout_position = training_config.get("dropout_position", "post")
    else:
      dropout = saved.get("lora_dropout", 0.0)
      dropout_position = "pre"  # PEFT applies dropout to the input of LoRA A.
    if dropout != config.dropout or (dropout and dropout_position != config.dropout_position):
      raise ValueError("Adapter dropout does not match worker configuration")

  def load_adapter_weights(self, state_path: str) -> None:
    """Load a flat PEFT snapshot with Automodel's native key and DTensor handling."""
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

    # Checkpointer.load_model only recognizes PEFT paths whose basename is
    # 'model'. Our public snapshot paths are flat, so use the same PyTorch
    # state loader directly. Every rank reads the small adapter and slices its
    # own shards. Missing frozen base weights are expected; validate every
    # trainable tensor before allowing a partial state dict.
    state = load_file(os.path.join(state_path, "adapter_model.safetensors"))
    state = models.adapter_state_from_hf(self.model, state)
    # Match Automodel's save-side filter and PyTorch's canonical state keys:
    # named_parameters() includes activation-checkpoint wrapper segments.
    model_state = get_model_state_dict(self.model, options=StateDictOptions(ignore_frozen_params=True))
    expected = {name: param for name, param in model_state.items() if "lora_" in name}
    if state.keys() != expected.keys():
      missing = sorted(expected.keys() - state.keys())
      unexpected = sorted(state.keys() - expected.keys())
      raise ValueError(f"Adapter parameters do not match the model: missing={missing[:5]}, unexpected={unexpected[:5]}")
    for name, param in expected.items():
      if state[name].shape != param.shape:
        raise ValueError(f"Adapter shape for {name} is {tuple(state[name].shape)}, expected {tuple(param.shape)}")
    set_model_state_dict(self.model, state, options=StateDictOptions(full_state_dict=True, strict=False))

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    metadata_path = os.path.join(state_path, "metadata.json")
    adapter_config_path = os.path.join(state_path, "adapter_config.json")
    if os.path.exists(metadata_path):
      with open(metadata_path) as f:
        metadata = json.load(f)
      base_model = metadata.get("base_model")
    elif os.path.exists(adapter_config_path):
      # A sampler snapshot: adapter only, base named by PEFT's config.
      with open(adapter_config_path) as f:
        metadata = {}
        base_model = json.load(f).get("base_model_name_or_path")
    else:
      raise FileNotFoundError(f"{state_path} has neither metadata.json nor adapter_config.json")
    if not base_model:
      raise ValueError(f"{state_path} does not name its base model")
    checkpoint_is_lora = metadata.get("lora", os.path.exists(adapter_config_path))
    if checkpoint_is_lora != self.is_lora:
      raise ValueError("Checkpoint training mode does not match OPEN_RL_AUTOMODEL_LORA_RANK")
    if self.is_lora:
      self.validate_adapter_config(state_path)
    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if restore_optimizer and metadata.get("has_optimizer") and not os.path.isfile(optimizer_path):
      raise FileNotFoundError(f"Checkpoint declares optimizer state, but {optimizer_path} is missing")

    self.release_model()
    if self.is_lora:
      # The base is loaded fresh with untrained adapters, then the saved
      # adapter tensors are read into them.
      self.load_base_model(base_model)
      self.load_adapter_weights(state_path)
    else:
      self.load_base_model(state_path)
    self.base_model_name = base_model
    self.prepare_model_for_training()

    if restore_optimizer and metadata.get("has_optimizer"):
      from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

      self.optimizer = torch.optim.AdamW(self.trainable_params, lr=1e-4, foreach=False)
      # Every rank reads the full state and slices its own shards; no broadcast.
      full_state = torch.load(optimizer_path, map_location="cpu", weights_only=True)
      set_optimizer_state_dict(self.model, self.optimizer, optim_state_dict=full_state, options=StateDictOptions(full_state_dict=True))
      print(f"Restored Automodel optimizer state from {optimizer_path}")
    print(f"Loaded Automodel state from {state_path}")
    return {"model_id": model_id, "base_model": base_model}

  # -- sampler weights ------------------------------------------------------

  def publishes_sampler_adapter(self) -> bool:
    # A LoRA run publishes an adapter the sampler hot-loads; a full-parameter
    # run has only the whole-checkpoint route.
    return self.is_lora

  def write_adapter(self, model_id: str, alias: str | None = None, session_label: str | None = None) -> str:
    """Publish the LoRA adapter for samplers to load with /v1/load_lora_adapter.

    Same peft/<id>/<label> layout adapter_snapshot.publish gives every backend.
    """
    if not self.is_lora:
      raise RuntimeError(
        "Sampling weights are published as a LoRA adapter, and OPEN_RL_AUTOMODEL_LORA_RANK=0 trains "
        "full parameters, so there is no adapter to publish. Set a rank, or samplers would serve the "
        "base model for the whole run while training looked healthy."
      )

    def write_files(staging_root: str) -> str:
      staged = os.path.join(staging_root, "adapter")
      if is_primary():
        os.makedirs(staged, exist_ok=True)
      barrier()
      self.write_weights(staged)
      return staged

    final_dir = adapter_snapshot.publish(model_id, write_files, alias, session_label)
    if is_primary():
      print(f"[Automodel Worker] Published LoRA adapter to {final_dir}.")
    barrier()
    return final_dir

  def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    raise RuntimeError("Sampling from the Automodel trainer is unsupported; use the vLLM sampler worker.")
