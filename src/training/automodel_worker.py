# NeMo Automodel training-backend worker.

"""A training-backend worker built on NVIDIA NeMo Automodel.

Why this exists. The Megatron backend trains Qwen3.5-family GDN models through a
deprecated GPTModel plus a custom HF bridge, and its context-parallel path is
numerically wrong for gated-delta-net (megatron-core 0.19.0 shards the sequence
into zigzag chunks but never carries the recurrent state across CP ranks, so
every chunk past the first computes from a zero state). Automodel is HF-native
(no format conversion, no bridge) and its GatedDeltaNet CP runs FLA's
context-parallel gated delta rule, which passes conv and recurrent state between
ranks. So it is the natural home for long-context GDN training.

Layout. Automodel builds a (pp, dp_replicate, dp_shard, cp, tp) mesh from the
sizes below; DP is what is left of the torchrun world after CP and TP. FSDP2
shards parameters over dp_shard x cp, so under CP the weights are sharded too.
Only the DP axis shards datums: CP and TP ranks cooperate on one sequence and
must see identical data, which is what the shard_* hooks scope to.

Logprobs never materialise [seq, vocab] logits. The forward returns the final
hidden states (logits_to_keep=1), the lm_head weight is gathered out of its
DTensor once (a frozen head under LoRA, so no gradient semantics are involved),
and hidden states are projected in checkpointed chunks, the same reduction the
FSDP path uses.

Under CP each pass is one datum: the sequence is padded to a multiple of 2*CP,
round-robin sharded (head and tail chunk per rank, torch's load-balanced
context_parallel layout), forwarded under the ring-attention context, projected
locally, and the local logprobs are all-gathered back into position order. The
backward of that gather scales by CP to cancel FSDP2's mean over the
dp_shard x cp mesh, matching how the base class scales by the DP shard count.

A wrong CP trains on a corrupted gradient without crashing, which is how the
Megatron CP bug hid, so this one was checked before use: scripts/automodel_probe.py
on Qwen3.5-9B (LoRA r32, 4096 tokens, 2026-09-10) against a single-GPU reference
gave per-position logprobs within bf16 noise on every chunk and adapter
gradients at cosine >= 0.9987 for CP2, CP4 and TP2xCP2, the same agreement TP2
alone shows. Rerun that probe after touching this path.
"""

import contextlib
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
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer

from training import adapter_snapshot, paths
from training.distributed import barrier, is_primary
from training.trainer_worker import BaseTrainerWorker, Datum, chunk_target_logprob

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
# Leaf module names. Each is matched as model.*.layers.*.<name>, which covers
# the attention, GDN and MLP projections of every decoder layer and nothing
# else: the vision tower has blocks, not layers, and the multi-token-prediction
# head lives under mtp, not model. Adapters on either would be tensors the
# language-model-only samplers cannot place.
AUTOMODEL_LORA_TARGETS = os.getenv(
  "OPEN_RL_AUTOMODEL_LORA_TARGETS",
  "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj",
)

# Activation checkpointing, done here rather than by Automodel. For its native
# Qwen3.5 model Automodel wraps self_attn, linear_attn and mlp separately and
# leaves the norms outside, so every layer stashes four sequence-length tensors
# instead of one: measured 2.47 GiB per 1k tokens on the 27B at TP4, four times
# Megatron's per-layer stash, and a 47k-token ceiling. Checkpointing a group
# of whole layers keeps one stash per group (Megatron's recompute_num_layers);
# recompute cost is one extra forward regardless of the group size, which only
# trades stash for the transient of recomputing a larger group. 4 is where
# that trade bottomed out on Megatron for this model. 0 turns it off.
RECOMPUTE_NUM_LAYERS = int(os.getenv("OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS", "4"))
if os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") != "1":
  RECOMPUTE_NUM_LAYERS = 0

# Rows of hidden states projected through the vocab per chunk. The FSDP path
# defaults to 128 for small GPUs; 1024 is a 1 GiB fp32 logits chunk on a 248k
# vocab, which an H200 does not notice, and cuts the Python loop 8x.
LOGPROB_CHUNK = int(os.getenv("OPEN_RL_LOGPROB_CHUNK", "1024"))

# Attention kernels. "auto" picks by model family: Gemma 4's global layers have
# 512-wide heads, past what SDPA's fused kernels take, so they go through
# Automodel's FFPA route (FFPA for the 512 heads, FlexAttention for the
# sliding-window layers). Under CP the ring swaps SDPA itself, so the model is
# built with sdpa and FFPA is requested for the full-attention ring chunks via
# the text config, the way Automodel's own Gemma 4 CP recipes do. Everything
# else is plain SDPA, which is what the CP ring-attention context patches.
AUTOMODEL_ATTN = os.getenv("OPEN_RL_AUTOMODEL_ATTN", "auto")


def attention_kwargs(model_type: str, cp_size: int, choice: str = AUTOMODEL_ATTN) -> dict[str, Any]:
  """from_pretrained kwargs selecting the attention kernels for this model."""
  if choice == "auto":
    choice = "ffpa" if model_type.startswith("gemma4") else "sdpa"
  if choice == "ffpa" and cp_size > 1:
    return {
      "attn_implementation": "sdpa",
      "use_sdpa_patching": False,
      "text_config": {"use_cache": False, "cp_full_attn_backend": "ffpa"},
    }
  if choice == "ffpa":
    return {"attn_implementation": "ffpa", "use_sdpa_patching": False}
  return {"attn_implementation": choice}


def widest_head_dim(text_config: Any) -> int:
  """The largest attention head width in the model.

  transformers 5.15 makes head_dim a per-layer attribute on heterogeneous
  models (Gemma 4: 256 on sliding layers, 512 on global ones) and raises on
  the config-level read, so look through the per-layer views when present.
  """
  try:
    layers = list(text_config.per_layer_config)
  except Exception:
    layers = []
  if layers:
    return max(int(getattr(layer, "head_dim", 0) or 0) for layer in layers)
  try:
    return int(getattr(text_config, "head_dim", 0) or 0)
  except Exception:
    return 0


def flex_kernel_options(text_config: Any, flex_runs_widest: bool) -> dict[str, Any]:
  """Smaller FlexAttention tiles for models with heads 256 wide or wider.

  Flex's default 128-wide tiles do not fit an H200's shared memory at
  head_dim 256 (Gemma 4's sliding layers; its 512-wide global layers run on
  FFPA). HF's flex integration reads kernel_options from the forward kwargs
  and Automodel's FFPA route passes them through, so they travel with the
  call. The FSDP path used 16-wide tiles for the same reason; the backward
  needs smaller tiles than the forward.
  """
  widest = widest_head_dim(text_config)
  if widest < 256:
    return {}
  # 64/32 fits head_dim 256; when flex also has to run the widest heads
  # (no FFPA), only the 16-wide tiles the FSDP path used fit at 512.
  fwd, bwd = (16, 16) if flex_runs_widest and widest > 256 else (64, 32)
  tile = {
    "fwd_BLOCK_M": fwd,
    "fwd_BLOCK_N": fwd,
    "fwd_num_stages": 1,
    "bwd_BLOCK_M1": bwd,
    "bwd_BLOCK_N1": bwd,
    "bwd_BLOCK_M2": bwd,
    "bwd_BLOCK_N2": bwd,
    "bwd_num_stages": 1,
  }
  return {"kernel_options": tile}


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


def round_robin_permutation(cp_size: int, padded_seq_len: int, device: torch.device) -> torch.Tensor:
  """Global position of every element of the rank-major all-gather of CP shards.

  Rank r owns chunks r and 2*CP-1-r of the 2*CP equal chunks (head then tail),
  so gathered element k sits at global position perm[k].
  """
  chunk = padded_seq_len // (2 * cp_size)
  parts = []
  for cp_rank in range(cp_size):
    head = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
    tail_start = (2 * cp_size - 1 - cp_rank) * chunk
    tail = torch.arange(tail_start, tail_start + chunk, device=device)
    parts.extend((head, tail))
  return torch.cat(parts)


class GatherSequenceShards(torch.autograd.Function):
  """All-gather [batch, local_seq] shards along the sequence.

  Every CP rank then holds the full sequence and computes the same loss, so the
  gradient of the gathered tensor is identical everywhere and each rank keeps
  its own slice. The slice is scaled by CP because FSDP2 averages gradients
  over the dp_shard x cp mesh while the ranks hold partial gradients of one
  loss that must sum.
  """

  @staticmethod
  def forward(ctx, local: torch.Tensor, group: dist.ProcessGroup, cp_size: int, cp_rank: int) -> torch.Tensor:
    ctx.cp_size = cp_size
    ctx.cp_rank = cp_rank
    ctx.local_len = local.shape[1]
    gathered = [torch.empty_like(local) for _ in range(cp_size)]
    dist.all_gather(gathered, local.contiguous(), group=group)
    return torch.cat(gathered, dim=1)

  @staticmethod
  def backward(ctx, grad: torch.Tensor):
    start = ctx.cp_rank * ctx.local_len
    return grad[:, start : start + ctx.local_len] * ctx.cp_size, None, None, None


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
  """The backbone's layer dict, iterating as checkpointed groups in training.

  Automodel's native backbone runs ``for layer in self.layers.values()``; this
  class is swapped in for that dict after the model is built and sharded, so
  the parameters, their names and the FSDP2 units are untouched, and only the
  iteration the forward sees changes.
  """

  group_size = 1

  def values(self):
    layers = list(self._modules.values())
    if not self.training:
      return iter(layers)
    groups = [layers[start : start + self.group_size] for start in range(0, len(layers), self.group_size)]
    return iter([LayerGroup(group) for group in groups])


def install_group_checkpointing(model: torch.nn.Module, group_size: int) -> None:
  """Checkpoint groups of layers on Automodel's native backbones.

  Automodel's native models keep their layers in a ModuleDict and iterate its
  values, which the class swap below hooks. A stock HF backbone (what Automodel
  builds for architectures it has no native class for) keeps a ModuleList and
  computes per-layer arguments inside its own loop, so it gets HF's per-layer
  gradient checkpointing instead: one stash per layer rather than per group.
  """
  backbone = model.model.language_model if hasattr(model.model, "language_model") else model.model
  layers = getattr(backbone, "layers", None)
  if isinstance(layers, torch.nn.ModuleDict):
    layers.__class__ = GroupCheckpointedLayers
    layers.group_size = group_size
    print(f"[Automodel Worker] activation checkpointing in groups of {group_size} layers.")
    return
  if isinstance(layers, torch.nn.ModuleList) and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("[Automodel Worker] HF backbone: per-layer gradient checkpointing (group size not configurable here).")
    return
  raise RuntimeError(
    f"Activation checkpointing found neither a ModuleDict nor a ModuleList of layers ({type(layers).__name__}); "
    "set OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS=0 to run without checkpointing."
  )


class AutomodelConfig(BaseModel):
  seed: int | None = None


class AutomodelTrainingWorker(BaseTrainerWorker):
  config_class = AutomodelConfig

  # FSDP2 reduces gradients inside backward, so pass counts must match across DP
  # ranks; the base class pads short ranks with zero-scaled filler passes.
  backward_runs_collectives = True

  def __init__(self):
    super().__init__()
    if AUTOMODEL_PP != 1:
      raise RuntimeError(
        f"The Automodel backend supports data, context, and tensor parallelism (got PP={AUTOMODEL_PP}). "
        "Pipeline parallelism moves the logits off the rank that builds the loss."
      )
    self.model: torch.nn.Module | None = None
    self.distributed_setup: Any = None
    self.device_mesh: Any = None
    self.peft_config: Any = None
    self.checkpointer: Any = None
    self.hf_config: Any = None
    self.forward_kwargs: dict[str, Any] = {}
    self.cp_context: contextlib.ExitStack | None = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: torch.optim.Optimizer | None = None
    self.is_lora = AUTOMODEL_LORA_RANK > 0
    self.tp_size = AUTOMODEL_TP
    self.cp_size = AUTOMODEL_CP

  # -- distributed layout ---------------------------------------------------

  def build_distributed_setup(self, base_model_name: str) -> Any:
    """Build the FSDP2 mesh and policy once for the life of the process."""
    if not dist.is_initialized():
      raise RuntimeError("The Automodel backend runs under torchrun; torch.distributed is not initialized.")
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

    world = dist.get_world_size()
    if world % (self.cp_size * self.tp_size):
      raise RuntimeError(f"WORLD_SIZE={world} is not divisible by OPEN_RL_AUTOMODEL_CP={self.cp_size} * OPEN_RL_AUTOMODEL_TP={self.tp_size}")
    strategy = FSDP2Config(
      activation_checkpointing=False,
      tp_plan=self.tensor_parallel_plan(base_model_name) if self.tp_size > 1 else None,
    )
    mesh_context = MeshContext.build(strategy, ParallelismSizes(tp_size=self.tp_size, cp_size=self.cp_size), world_size=world)
    print(f"Automodel device mesh: DP={world // (self.cp_size * self.tp_size)} CP={self.cp_size} TP={self.tp_size}")
    return DistributedSetup(
      mesh_context=mesh_context,
      strategy_config=strategy,
      activation_checkpointing=strategy.activation_checkpointing,
    )

  def tensor_parallel_plan(self, base_model_name: str) -> dict[str, Any]:
    """The HF text-model TP plan, addressed from the top of the model.

    Automodel's automatic plan reads _tp_plan off the language model, which its
    native Qwen3.5 backbone does not carry, so on that model TP would shard the
    embeddings and nothing else. The plan lives on the text config either way;
    prefix it the way the module tree is actually laid out. GDN layers are not
    in it: they are not TP-shardable and stay replicated.
    """
    from nemo_automodel.components.distributed.parallelizer import translate_to_torch_parallel_style
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    config = self.load_hf_config(base_model_name)
    text_config = config.get_text_config()
    architectures = config.architectures or []
    prefix = "model.language_model" if architectures and architectures[0].endswith("ForConditionalGeneration") else "model"
    plan: dict[str, Any] = {f"{prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate())}
    for name, style_name in (getattr(text_config, "base_model_tp_plan", None) or {}).items():
      try:
        style = translate_to_torch_parallel_style(style_name)
      except ValueError:
        # HF ships one plan per family; the MoE-only entries (packed experts)
        # match no module on a dense checkpoint and Automodel has no style for
        # them. Dropping them leaves those modules replicated, never wrong.
        print(f"[Automodel Worker] TP plan: skipping {name} ({style_name}) with no torch parallel style.")
        continue
      if style is not None:
        plan[f"{prefix}.{name}"] = style
    plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())
    return plan

  def load_hf_config(self, base_model_name: str) -> Any:
    if self.hf_config is None or getattr(self.hf_config, "name_or_path", None) != base_model_name:
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
      target_modules=[f"model.*.layers.*.{name}" for name in names],
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

    NeMoAutoModelForCausalLM = require_automodel()
    torch.cuda.set_device(self.device)
    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if self.distributed_setup is None:
      self.distributed_setup = self.build_distributed_setup(base_model_name)
      self.device_mesh = self.distributed_setup.mesh_context.device_mesh
    if self.is_lora and self.peft_config is None:
      self.peft_config = self.build_peft_config()
    print(f"Loading Automodel {base_model_name} (rank {os.getenv('RANK', '0')}/{os.getenv('WORLD_SIZE', '1')})...")

    # from_pretrained applies LoRA to the matched linears before FSDP2 shards
    # anything, loads the base weights, freezes everything but the adapters,
    # and wraps with the strategy above. Liger stays off so the forward is the
    # plain HF graph. Qwen3.5 checkpoints ship a multi-token-prediction head
    # that the native model would build and run over the whole sequence on
    # every training forward; the loss never reads it, so it is not built.
    text_config = self.load_hf_config(base_model_name).get_text_config()
    model_type = text_config.model_type
    extra = attention_kwargs(model_type, self.cp_size)
    attn = extra.get("attn_implementation")
    if attn == "flex_attention" or "ffpa" in str(extra):
      self.forward_kwargs = flex_kernel_options(text_config, flex_runs_widest=(attn == "flex_attention"))
    else:
      self.forward_kwargs = {}
    if model_type.startswith("qwen3_5"):
      extra["num_nextn_predict_layers"] = 0
    print(f"[Automodel Worker] {model_type}: {extra}")
    self.model = NeMoAutoModelForCausalLM.from_pretrained(
      base_model_name,
      torch_dtype=torch.bfloat16,
      use_liger_kernel=False,
      distributed_setup=self.distributed_setup,
      peft_config=self.peft_config,
      **extra,
    )
    if RECOMPUTE_NUM_LAYERS > 0:
      install_group_checkpointing(self.model, RECOMPUTE_NUM_LAYERS)
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
    self.load_base_model(base_model_name)
    seed = config.seed if config is not None and config.seed is not None else AUTOMODEL_SEED
    torch.manual_seed(seed)
    self.prepare_model_for_training()

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
    if self.cp_size > 1:
      return self.compute_target_logprobs_cp(model, input_ids, target_token_ids)

    # A dense all-ones mask only describes ordinary causal attention; omitting
    # it lets SDPA pick flash attention instead of building an additive mask.
    if attention_mask is not None and bool(attention_mask.all()):
      attention_mask = None
    outputs = model(
      input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1, output_hidden_states=True, **self.forward_kwargs
    )
    hidden = self.final_hidden_states(outputs)[:, :seq_len]
    return self.project_target_logprobs(model, hidden, target_token_ids)

  def final_hidden_states(self, outputs: Any) -> torch.Tensor:
    hidden = outputs.hidden_states
    if isinstance(hidden, (tuple, list)):
      hidden = hidden[-1]
    if hidden is None:
      raise RuntimeError("Automodel forward returned no hidden_states; the logprob path needs the final hidden states.")
    if is_dtensor(hidden):
      hidden = hidden.full_tensor()
    return hidden

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
    """logit[target] - logsumexp over the vocab, in checkpointed chunks."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
    weight = self.head_weight(head.weight)
    bias = getattr(head, "bias", None)
    if bias is not None:
      bias = self.head_weight(bias)
    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    softcap = getattr(text_config, "final_logit_softcapping", None)

    batch, seq_len, _ = hidden.shape
    flat_hidden = hidden.reshape(batch * seq_len, -1)
    flat_targets = targets.reshape(batch * seq_len)
    needs_grad = flat_hidden.requires_grad or weight.requires_grad

    def project(start: int) -> torch.Tensor:
      args = (flat_hidden[start : start + LOGPROB_CHUNK], weight, bias, flat_targets[start : start + LOGPROB_CHUNK], softcap)
      if needs_grad:
        return torch.utils.checkpoint.checkpoint(chunk_target_logprob, *args, use_reentrant=False)
      return chunk_target_logprob(*args)

    return torch.cat([project(start) for start in range(0, flat_hidden.shape[0], LOGPROB_CHUNK)]).reshape(batch, seq_len)

  def compute_target_logprobs_cp(self, model: torch.nn.Module, input_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
    """Return [batch, seq] target logprobs with the sequence sharded over CP."""
    if input_ids.shape[0] != 1:
      raise RuntimeError(f"The Automodel CP path takes one sequence per pass, got a batch of {input_ids.shape[0]}.")
    from nemo_automodel.components.distributed.context_parallel.sharder import shard_batch_aux_only

    seq_len = target_token_ids.shape[1]
    cp_mesh = self.device_mesh["cp"]
    cp_group = cp_mesh.get_group()
    cp_rank = cp_mesh.get_local_rank()

    # Automodel pads and round-robin shards the aux streams (labels,
    # position_ids) in place and installs the ring-attention context; the model
    # embeds the full input_ids and shards its own hidden states the same way.
    batch = {"input_ids": input_ids, "labels": target_token_ids.clone()}
    context, batch, layout = shard_batch_aux_only(cp_mesh, self.device_mesh["tp"], batch)
    # The context must outlive this call. It swaps SDPA for ring attention by
    # dispatch, and the backward dispatches SDPA again, both for the attention
    # gradient and for the activation-checkpoint recompute of the forward.
    # Closed after the forward, both run plain local attention over this
    # rank's shard and every gradient upstream of the last attention layer is
    # silently wrong (measured: cosine 0.24 on o_proj adapters, logprobs fine).
    # forward_backward closes it once this pass's backward is done.
    self.close_cp_context()
    self.cp_context = contextlib.ExitStack()
    self.cp_context.enter_context(context())
    outputs = model(
      input_ids=batch["input_ids"],
      position_ids=batch["position_ids"],
      use_cache=False,
      logits_to_keep=1,
      output_hidden_states=True,
      **self.forward_kwargs,
    )
    local_targets = batch["labels"]
    local_hidden = self.final_hidden_states(outputs)
    # Padding slots carry an ignore index; they are cut off after the gather.
    local_logprobs = self.project_target_logprobs(model, local_hidden, local_targets.clamp_min(0))

    gathered = GatherSequenceShards.apply(local_logprobs, cp_group, self.cp_size, cp_rank)
    perm = round_robin_permutation(self.cp_size, layout.padded_seq_len, gathered.device)
    ordered = torch.zeros_like(gathered).index_copy(1, perm, gathered)
    return ordered[:, :seq_len]

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
    staging_path = f"{path}.staging-{os.getpid()}"
    previous_path = f"{path}.previous-{os.getpid()}"
    if is_primary():
      shutil.rmtree(staging_path, ignore_errors=True)
      os.makedirs(staging_path, exist_ok=True)
    barrier()
    self.write_weights(staging_path)
    if include_optimizer and self.optimizer is not None:
      optimizer_state = self.full_optimizer_state_dict()
      if is_primary():
        torch.save(optimizer_state, os.path.join(staging_path, "optimizer.pt"))
    if is_primary():
      with open(os.path.join(staging_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
      shutil.rmtree(previous_path, ignore_errors=True)
      if os.path.exists(path):
        os.rename(path, previous_path)
      os.rename(staging_path, path)
      shutil.rmtree(previous_path, ignore_errors=True)
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

    if self.is_lora:
      # The base is loaded fresh with untrained adapters, then the saved
      # adapter tensors are read into them.
      self.load_base_model(base_model)
      self.get_checkpointer().load_model(self.model, model_path=state_path)
    else:
      self.model = None
      self.load_base_model(state_path)
    self.base_model_name = base_model
    self.prepare_model_for_training()

    optimizer_path = os.path.join(state_path, "optimizer.pt")
    if restore_optimizer and metadata.get("has_optimizer") and os.path.exists(optimizer_path):
      from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

      self.optimizer = torch.optim.AdamW(self.trainable_params, lr=1e-4, foreach=False)
      # Every rank reads the full state and slices its own shards; no broadcast.
      full_state = torch.load(optimizer_path, map_location="cpu", weights_only=False)
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
