# Megatron-LM trainer worker lifecycle.
"""Tensor-parallel training backend, interface-compatible with the FSDP worker.

Why this exists next to fft_trainer_worker
------------------------------------------
FSDP shards *parameters*; it does not shard activations. Every rank still runs
the whole sequence at the whole hidden dimension, so peak memory grows with
context at ~0.69 GiB/ktoken on the 12B model and the 143,360-token window sits
at ~126 GiB of 139.8. Adding GPUs buys throughput, never a longer context.

Megatron's tensor parallelism shards the hidden dimension, so per-rank
activations fall by roughly the TP degree, and sequence parallelism shards the
LayerNorm/dropout activations TP cannot. TP is the knob that moves the context
ceiling. FSDP has no equivalent.

TP alone is not enough, though: the vocab-parallel loss splits the 262k-entry
logit tensor across the group but still keeps [seq, batch, vocab/TP] in fp32 for
the whole sequence, which measured out as the dominant term and capped TP=8 at
64k. chunked_target_logprobs below removes it.

That exposes the second term: Gemma-4's layer spec uses megatron's own core
attention, which materializes the [batch, heads, seq, seq] scores, and no amount
of TP shrinks a term that grows with seq^2. install_gemma4_flex_attention
replaces it. With both gone, TP=4 runs the full 143,360-token context at 61.5
GiB of 139.8 -- on four GPUs, leaving four for samplers.

What is supported
-----------------
Tensor parallelism and data parallelism, in any product that equals WORLD_SIZE.

Pipeline parallelism is not supported, and that is structural rather than an
omission. BaseTrainerWorker.forward_backward computes target logprobs, builds a
loss from them, and calls .backward() -- it needs the logits on every rank.
Under PP only the last stage has them and Megatron drives the whole schedule
from get_forward_backward_func(), which owns the loop this class already owns.
Supporting PP means rewriting forward_backward around Megatron's scheduler, not
adding a flag. Context parallelism is out for a related reason: it shards the
sequence, so the [batch, seq] logprob tensor this interface hands back would
have to be re-gathered per micro-batch.

How to turn it on
-----------------
    OPEN_RL_ENABLE_FFT=true            # still required, see below
    OPEN_RL_TRAINER_BACKEND=megatron
    OPEN_RL_MEGATRON_TP=2
    OPEN_RL_CONTROL_BACKEND=cpu:gloo,cuda:nccl
    OPEN_RL_MEGATRON_LORA_RANK=16       # optional; 0 (default) trains all weights

OPEN_RL_ENABLE_FFT is not redundant. It is read independently by gateway.py,
worker_manager.py and vllm_sampler.py, where it means "full weights, not LoRA":
launch a dedicated trainer per model, report is_lora=False, and start the
sampler in sleep-mode reloading whole checkpoints instead of applying adapters.
Setting only OPEN_RL_TRAINER_BACKEND would give a Megatron trainer feeding a
sampler that still expects adapter files. The worker_manager passes the whole
environment to its children, so both variables reach the trainer process.

Version-sensitive seam
----------------------
Weight import and HF export go through Megatron-Bridge (`AutoBridge`), which is
the only sane way to get an HF checkpoint into Megatron's layout and back out
again for the vLLM sampler. Neither megatron-core nor megatron-bridge is in
pyproject.toml, so every import here is lazy and this module imports cleanly
without them.

Verified against megatron-core 0.19.0 / megatron-bridge 0.6.0 on 2026-08-26:
from_hf_pretrained, to_megatron_provider, provide_distributed_model and
save_hf_pretrained all exist with these signatures, and AutoBridge dispatches to
a Gemma4DenseProvider carrying the right 12B geometry once
register_gemma4_unified_bridge has run. What that pairing needs, all handled
below and none of it discoverable from the docs: model_parallel_cuda_manual_seed
after initialize_model_parallel, the two Apex fusions off, and the
architecture-name shim. The bridge also wants transformer-engine, whose torch
extension ships only as an sdist and must be compiled against the box's nvcc.
"""

import gc
import json
import math
import os
import shutil
import time
from datetime import datetime
from typing import Any

import torch
import torch.distributed as dist
from pydantic import BaseModel
from transformers import AutoTokenizer

from training import paths
from training.distributed import barrier, is_primary
from training.trainer_worker import BaseTrainerWorker, Datum

ENABLE_GRADIENT_CHECKPOINTING = os.getenv("ENABLE_GRADIENT_CHECKPOINTING", "1") == "1"
MEGATRON_TP = int(os.getenv("OPEN_RL_MEGATRON_TP", "1"))
MEGATRON_PP = int(os.getenv("OPEN_RL_MEGATRON_PP", "1"))
MEGATRON_CP = int(os.getenv("OPEN_RL_MEGATRON_CP", "1"))
MEGATRON_SEQUENCE_PARALLEL = os.getenv("OPEN_RL_MEGATRON_SEQUENCE_PARALLEL", "1") == "1"
MEGATRON_DISTRIBUTED_OPTIMIZER = os.getenv("OPEN_RL_MEGATRON_DISTRIBUTED_OPTIMIZER", "1") == "1"
MEGATRON_SEED = int(os.getenv("OPEN_RL_MEGATRON_SEED", "1234"))
# Two Megatron fusions call into Apex CUDA extensions (fused_weight_gradient_mlp_cuda,
# scaled_masked_softmax_cuda) that are not pip-installable and are absent unless
# Apex was built from source. Megatron does not degrade gracefully: it raises at
# layer construction. Default them off and let a box with Apex opt back in.
MEGATRON_APEX_FUSIONS = os.getenv("OPEN_RL_MEGATRON_APEX_FUSIONS", "0") == "1"
# LoRA. Rank 0 keeps the full-parameter behaviour; any positive rank freezes the
# base weights and trains adapters instead. See apply_lora for why this does not
# change how checkpoints are published.
MEGATRON_LORA_RANK = int(os.getenv("OPEN_RL_MEGATRON_LORA_RANK", "0"))
MEGATRON_LORA_ALPHA = int(os.getenv("OPEN_RL_MEGATRON_LORA_ALPHA", "32"))
# Default 0, matching LoraConfig: dropout during logprob computation makes
# trainer logprobs stochastic while the sampler's are deterministic, biasing
# every importance-sampling ratio.
MEGATRON_LORA_DROPOUT = float(os.getenv("OPEN_RL_MEGATRON_LORA_DROPOUT", "0.0"))
MEGATRON_LORA_TARGETS = os.getenv("OPEN_RL_MEGATRON_LORA_TARGETS", "linear_qkv,linear_proj,linear_fc1,linear_fc2")
# Rows of hidden state projected through the vocabulary at a time. Larger than
# the FSDP path's OPEN_RL_LOGPROB_CHUNK because each chunk here carries the
# cross-entropy's TP collectives, so very small chunks pay launch latency for
# nothing; at 1024 rows the transient fp32 logit tensor is 1 GiB at TP=1 and
# 128 MiB at TP=8.
MEGATRON_LOGPROB_CHUNK = int(os.getenv("OPEN_RL_MEGATRON_LOGPROB_CHUNK", "1024"))
# Gemma-4 only, and on by default: megatron's own core attention materializes the
# [batch, heads, seq, seq] score matrix, which is what caps context once
# chunked_target_logprobs has removed the logit term. Set to 0 to compare against
# the stock path. See install_gemma4_flex_attention.
MEGATRON_FLEX_ATTENTION = os.getenv("OPEN_RL_MEGATRON_FLEX_ATTENTION", "1") == "1"

OPTIMIZER_SUBDIR = "megatron_optimizer"

# Where a Megatron optimizer keeps its fp32 master weights. The distributed
# optimizer uses the shard_* names, the plain mixed-precision one the last.
# Each is a list of lists of tensors, and none of them live inside the DDP
# parameter buffers, so an offload has to move them separately.
MASTER_PARAM_GROUPS = (
  "shard_fp32_from_float16_groups",
  "shard_float16_groups",
  "shard_fp32_groups",
  "fp32_from_float16_groups",
)


class MegatronConfig(BaseModel):
  seed: int | None = None
  cpu_offload: bool = True


def require_megatron():
  """Import megatron-core and megatron-bridge, or explain how to get them."""
  try:
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
  except ImportError as exc:
    raise RuntimeError(
      "OPEN_RL_TRAINER_BACKEND=megatron needs megatron-core and megatron-bridge. Neither is a "
      "dependency of this project (no extra in pyproject.toml installs them), so they must be "
      f"installed into the trainer environment first. Import failed with: {exc}"
    ) from exc
  return AutoBridge, parallel_state


def register_gemma4_unified_bridge() -> None:
  """Teach megatron-bridge the architecture name our Gemma-4 checkpoints use.

  transformers 5.10 renamed Gemma-4's model_type from "gemma4" to
  "gemma4_unified" (the same rename src/training/model_loading.py works around),
  so google/gemma-4-12B-it declares architectures=["Gemma4UnifiedForConditionalGeneration"].
  megatron-bridge 0.6.x registers only "Gemma4ForCausalLM" and
  "Gemma4ForConditionalGeneration", so AutoBridge.can_handle returns False and
  dispatch fails with a "write your own bridge" error.

  Nothing but the name is missing. Gemma4VLBridge already reads the nested
  text_config and already expects the multimodal ``model.language_model.*``
  weight layout our checkpoints carry, and with GEMMA4_CONVERSION_MODE=text it
  builds a plain text-only Gemma4DenseProvider and drops the vision tower --
  which is exactly the text-only model this trainer wants. So re-register the
  existing class under the new name rather than defining a bridge.

  Harmless once upstream adds the name: the register call is skipped.
  """
  from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
  from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge
  from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider
  from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel

  # Text-only conversion: language tower to a GPTModel provider, vision dropped.
  os.environ.setdefault("GEMMA4_CONVERSION_MODE", "text")
  source = "Gemma4UnifiedForConditionalGeneration"
  try:
    MegatronModelBridge.register_bridge(
      source=source,
      target=Gemma4VLModel,
      provider=Gemma4VLModelProvider,
      model_type="gemma4_vl",
    )(Gemma4VLBridge)
  except Exception as exc:  # already registered upstream, or the registry moved
    print(f"[Megatron Worker] Did not register {source}: {exc}")


# torch.compile caches per callable, so compile once for the process rather than
# once per layer. Populated by install_gemma4_flex_attention, which is the only
# place that may import torch.nn.attention.flex_attention.
_flex_attention_compiled: Any = None
_create_block_mask_compiled: Any = None
_flex_block_masks: dict[tuple[int | None, int, int], Any] = {}

# The cache has to be bounded, because make_training_batches packs to a token
# budget and compute_target_logprobs pads only to a multiple of tp_size, so
# almost every step hands attention a sequence length it has never seen: a
# packing probe over eight batch shapes cached eighteen masks. An entry is
# quadratic in length -- measured 19.2 MiB at 143,360 tokens, 1.0 MiB at 32,768,
# and a sliding mask costs the same as a global one because kv_indices spans the
# whole block grid either way -- so a few hundred steps of an unbounded cache is
# tens of GiB of leak. Bound it by count, which caps it at 8 * 2 * 19.2 MiB even
# if every entry is worst-case. Evicting is cheap: rebuilding is one compiled
# kernel launch. Eight is four steps of history, and a step needs at most two
# entries (one shape, one sliding mask and one global), so hot masks survive.
_FLEX_BLOCK_MASK_CACHE_SIZE = 8


def gemma4_flex_block_mask(window: int | None, seq_q: int, seq_kv: int, device) -> Any:
  """Block mask for one causal (window=None) or sliding-window attention layer.

  Compiling create_block_mask is not a speed tweak, it is the difference between
  linear and quadratic memory. Uncompiled it evaluates the predicate over a
  dense [seq_q, seq_kv] grid, and the sliding predicate's ``q_idx - kv_idx`` is
  int64, so describing the 143,360-token window would cost 153 GiB to build a
  mask that only needs a bit per 128x128 block. Measured on the 12B model at
  TP=4: 96k tokens peaked at 115.6 GiB uncompiled against 43.6 GiB compiled,
  and only the compiled form grows linearly with context.

  One cache for the whole process: all 48 layers share two masks per shape. It
  is a least-recently-used cache of _FLEX_BLOCK_MASK_CACHE_SIZE entries; see the
  note there for why an unbounded one leaks.
  """
  key = (window, seq_q, seq_kv)
  # pop-then-reinsert, so dict insertion order is least-recently-used order.
  mask = _flex_block_masks.pop(key, None)
  if mask is None:
    if window is None:

      def keep(_batch, _head, q_idx, kv_idx):
        return q_idx >= kv_idx

    else:

      def keep(_batch, _head, q_idx, kv_idx, _window=window):
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= _window)

    mask = _create_block_mask_compiled(
      keep, B=None, H=None, Q_LEN=seq_q, KV_LEN=seq_kv, device=device
    )
    while len(_flex_block_masks) >= _FLEX_BLOCK_MASK_CACHE_SIZE:
      _flex_block_masks.pop(next(iter(_flex_block_masks)))
  _flex_block_masks[key] = mask
  return mask


def gemma4_flex_attention_class() -> type:
  """Build the FlexAttention replacement for megatron's core attention.

  Defined inside a function because it subclasses a megatron class, and this
  module imports without megatron installed.
  """
  from megatron.core.transformer.dot_product_attention import DotProductAttention
  from megatron.core.transformer.utils import is_layer_window_attention

  class FlexDotProductAttention(DotProductAttention):
    """Core attention that never materializes the [batch, heads, seq, seq] scores.

    This is the second half of the context ceiling, the half chunked_target_logprobs
    does not touch. Gemma-4's dense layer spec hardcodes a LocalSpecProvider, so
    core attention is megatron's own DotProductAttention, which is correct --
    it does apply the per-layer sliding window -- but builds the full score
    matrix, and that term grows with seq^2 no matter how high TP goes.

    Transformer Engine cannot fix it here. Gemma-4's eight global layers project
    to head_dim 512, past cuDNN 9.17's 256 limit on sm90, and with flash-attn
    absent TE selects its own unfused kernel for exactly those layers: measured
    on the 12B model, TE topped out at 32k against the local path's 64k.

    FlexAttention is Triton, has no head_dim ceiling, and takes the sliding
    window as a block mask, which is what megatron-bridge already does for
    Gemma-2 (Gemma2FlexDotProductAttention). Measured against an fp32 reference
    on all 48 layers, it is also the more accurate of the two: ~0.3% relative
    error per layer against the local path's 1-3%, because megatron keeps this
    softmax in bf16 (attention_softmax_in_fp32 defaults off) and Flex
    accumulates in fp32.

    Subclassing rather than replacing keeps the parent available for the inputs
    Flex cannot express, and use_flex=False makes the two paths comparable on
    one set of weights.
    """

    def __init__(self, config, layer_number, attn_mask_type, attention_type, attention_dropout=None, **kwargs):
      super().__init__(config, layer_number, attn_mask_type, attention_type, attention_dropout, **kwargs)
      self.use_flex = True
      # The same predicate the parent hands to FusedScaleMaskSoftmax, read from
      # the same helper, so the two paths cannot disagree about which layers slide.
      self.flex_window = (
        config.window_size[0]
        if is_layer_window_attention(config.window_size, config.window_attn_skip_freq, layer_number)
        else None
      )
      # Gemma-4's global layers carry head_dim 512. FlexAttention's default
      # 128-row tiles then want 256 KiB of shared memory against sm90's 227 KiB
      # and Triton refuses to compile; halving the tiles is not enough on its
      # own, because the default ~3 pipeline stages trible the K/V tile budget.
      self.flex_kernel_options = (
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32, "num_stages": 1}
        if config.kv_channels > 256
        else None
      )

    def forward(
      self, query, key, value, attention_mask, attn_mask_type=None, attention_bias=None, packed_seq_params=None
    ):
      # Everything the block mask cannot express falls back to the parent. An
      # explicit mask is the one that matters in practice: the parent overrides
      # it with its own causal/sliding mask anyway, but only when it is None can
      # this path be sure the block mask is the whole story.
      eligible = (
        self.use_flex
        and attention_mask is None
        and attention_bias is None
        and packed_seq_params is None
        and self.softmax_offset is None
        and not (self.training and self.config.attention_dropout > 0)
      )
      if not eligible:
        return super().forward(
          query,
          key,
          value,
          attention_mask,
          attn_mask_type=attn_mask_type,
          attention_bias=attention_bias,
          packed_seq_params=packed_seq_params,
        )

      # Megatron lays attention out as [seq, batch, heads, head_dim]; Flex wants
      # [batch, heads, seq, head_dim]. enable_gqa maps query head i to key group
      # i // (heads / groups), which is what the parent's repeat_interleave does.
      seq_q, batch, heads, head_dim = query.shape
      context = _flex_attention_compiled(
        query.permute(1, 2, 0, 3),
        key.permute(1, 2, 0, 3),
        value.permute(1, 2, 0, 3),
        block_mask=gemma4_flex_block_mask(self.flex_window, seq_q, key.shape[0], query.device),
        scale=self.softmax_scale,
        enable_gqa=key.shape[2] != heads,
        kernel_options=self.flex_kernel_options,
      )
      return context.permute(2, 0, 1, 3).contiguous().view(seq_q, batch, heads * head_dim)

  return FlexDotProductAttention


def install_gemma4_flex_attention() -> bool:
  """Point Gemma-4's dense layer spec at FlexAttention. Returns whether it took.

  The seam is narrow. Gemma4DenseProvider.provide calls the module-level
  get_gemma4_layer_spec directly and ignores its own transformer_layer_spec
  field, so rebinding that name in the provider's namespace is the only way in
  without reimplementing the spec. Wrapping the stock spec rather than writing
  one keeps every other submodule -- the Gemma-4 norms, the shared-KV
  attention class -- exactly as the bridge built it.

  Best-effort by design: a checkpoint that is not Gemma-4 never imports this
  module, and a megatron-bridge that moved the symbol should cost a slower
  context ceiling, not a failed run.
  """
  global _flex_attention_compiled, _create_block_mask_compiled
  if not MEGATRON_FLEX_ATTENTION:
    return False
  try:
    import megatron.bridge.models.gemma.gemma4_provider as gemma4_provider
    from megatron.bridge.models.gemma.modeling_gemma4 import get_gemma4_layer_spec
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
  except ImportError as exc:
    print(f"[Megatron Worker] FlexAttention unavailable, keeping megatron's core attention: {exc}")
    return False

  if getattr(gemma4_provider.get_gemma4_layer_spec, "_open_rl_flex", False):
    return True

  _flex_attention_compiled = torch.compile(flex_attention)
  _create_block_mask_compiled = torch.compile(create_block_mask)
  flex_class = gemma4_flex_attention_class()

  def flex_layer_spec(config=None):
    spec = get_gemma4_layer_spec(config)
    spec.submodules.self_attention.submodules.core_attention = flex_class
    return spec

  flex_layer_spec._open_rl_flex = True
  gemma4_provider.get_gemma4_layer_spec = flex_layer_spec
  print("[Megatron Worker] Gemma-4 core attention: FlexAttention (block-mask sliding window).")
  return True


def chunked_target_logprobs(
  *, hidden_states, output_layer, output_weight, labels, config, compute_language_model_loss, scale_logits, **_
) -> torch.Tensor:
  """GPTModel output_processor that never materializes the full logit tensor.

  Passing labels= to GPTModel runs the whole sequence through the output layer in
  one shot. Even with the vocab sharded over the TP group that is
  [seq, batch, vocab/TP] in fp32, and Gemma-4's logit softcapping keeps more
  tensors that size alive for the backward pass. Measured on the 12B model at
  TP=1 and 16k tokens, a no-grad forward with no loss at all cost 55.7 GiB above
  the weights -- which is why this backend OOMed at 32k while the FSDP worker,
  whose head is chunked, reaches 143k.

  We only need one scalar per position, so the projection runs in
  MEGATRON_LOGPROB_CHUNK-row slices under activation checkpointing: the
  [chunk, vocab/TP] logits are never held for backward, peak drops from
  O(seq x vocab) to O(chunk x vocab), and it stops scaling with context.

  Every numeric step is still Megatron's own -- the output layer module, which
  carries Gemma's logit softcapping, and compute_language_model_loss, which
  dispatches to whichever fused cross-entropy the config asks for. That is
  load-bearing, not tidiness. Hand-rolling the head with the unfused
  vocab_parallel_cross_entropy reproduced the stock forward bit for bit but moved
  the gradients by 3% relative, which nothing downstream would have caught. As
  written the gradients are bit-identical to the labels= path at TP=2; at TP=1
  they differ by ~1%, because the 262k-wide dgrad GEMM splits its reduction
  differently for 1024 rows than for the whole sequence.

  Returns [batch, seq] logprobs, so GPTModel.forward hands them back in place of
  the usual per-token loss.
  """
  if config.sequence_parallel:
    # The decoder leaves the sequence sharded across the TP group and the output
    # layer would normally gather it. Gathering once here instead of once per
    # chunk is also what makes the chunks correct: this op's backward
    # reduce-scatters, summing each rank's partial hidden gradient. Without
    # sequence parallelism the output layer's own dgrad all-reduce still covers
    # that, so there is nothing to arrange.
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    hidden_states = gather_from_sequence_parallel_region(hidden_states)

  # Megatron is sequence-first, the Tinker interface batch-first. Chunks are rows
  # of the flattened [seq * batch] axis; cross-entropy is per-row, so a chunk
  # does not have to respect the batch boundary.
  batch, seq_len = labels.shape
  hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
  targets = labels.transpose(0, 1).reshape(-1)
  weight = output_weight if output_weight is not None else output_layer.weight

  def project(hidden_chunk: torch.Tensor, weight: torch.Tensor, target_chunk: torch.Tensor) -> torch.Tensor:
    saved = output_layer.sequence_parallel, output_layer.disable_grad_reduce
    if config.sequence_parallel:
      # We gathered above and that gather's backward owns the gradient
      # reduction; disable_grad_reduce stops the layer adding a second one.
      # Set here rather than around the loop because checkpointing re-runs this
      # from backward, long after an outer restore would have happened.
      output_layer.sequence_parallel, output_layer.disable_grad_reduce = False, True
    try:
      logits, _ = output_layer(hidden_chunk.unsqueeze(1), weight=weight)
    finally:
      output_layer.sequence_parallel, output_layer.disable_grad_reduce = saved
    return -compute_language_model_loss(target_chunk.unsqueeze(0), scale_logits(logits)).squeeze(0)

  def chunk(start: int) -> torch.Tensor:
    stop = start + MEGATRON_LOGPROB_CHUNK
    args = (hidden[start:stop], weight, targets[start:stop])
    if hidden.requires_grad or weight.requires_grad:
      return torch.utils.checkpoint.checkpoint(project, *args, use_reentrant=False)
    return project(*args)

  rows = torch.cat([chunk(start) for start in range(0, hidden.shape[0], MEGATRON_LOGPROB_CHUNK)])
  return rows.view(seq_len, batch).transpose(0, 1)


class MegatronTrainingWorker(BaseTrainerWorker):
  config_class = MegatronConfig

  def __init__(self):
    super().__init__()
    if MEGATRON_PP != 1 or MEGATRON_CP != 1:
      raise RuntimeError(
        f"The Megatron backend supports tensor and data parallelism only (got PP={MEGATRON_PP}, "
        f"CP={MEGATRON_CP}). Both change which rank holds the logits, and forward_backward needs "
        "them on every rank to build the loss it differentiates; see this module's docstring."
      )
    self.model_chunks: list[torch.nn.Module] = []
    self.bridge: Any = None
    self.base_model_name: str | None = None
    self.trainable_params: list[torch.nn.Parameter] = []
    self.optimizer: Any = None
    self.tp_size = MEGATRON_TP
    self.sequence_parallel = MEGATRON_TP > 1 and MEGATRON_SEQUENCE_PARALLEL
    self.cpu_offload: bool = True
    self._is_offloaded: bool = False
    # Megatron's DDP reduces gradients in finish_grad_sync(), which optim_step
    # calls, not during backward. Ranks may therefore run different numbers of
    # passes without deadlocking, so the base class does not need to pad short
    # ranks with filler passes. Tensor-parallel peers all-reduce inside every
    # backward, but they share a data shard and so always agree on the count.
    self.backward_runs_collectives = False

  # -- parallel state -------------------------------------------------------

  def initialize_parallel_state(self) -> None:
    _, parallel_state = require_megatron()
    if parallel_state.model_parallel_is_initialized():
      return
    if not dist.is_initialized():
      raise RuntimeError("The Megatron backend must be launched under torchrun; no process group is initialized.")

    # Megatron's tensor-parallel collectives are on CUDA tensors, so the default
    # group has to speak NCCL. This project defaults it to gloo because the
    # gateway broadcasts pickled request batches over it, and the device-mapped
    # backend string keeps both working from one group.
    backend = str(dist.get_backend())
    if "nccl" not in backend:
      raise RuntimeError(
        f"The Megatron backend needs NCCL for tensor-parallel collectives but the default process "
        f"group speaks {backend!r}. Launch the trainer with "
        "OPEN_RL_CONTROL_BACKEND='cpu:gloo,cuda:nccl' so object broadcasts stay on gloo and tensor "
        "collectives use NCCL."
      )

    world = dist.get_world_size()
    if world % self.tp_size:
      raise RuntimeError(f"WORLD_SIZE={world} is not divisible by OPEN_RL_MEGATRON_TP={self.tp_size}")

    torch.cuda.set_device(self.device)
    parallel_state.initialize_model_parallel(
      tensor_model_parallel_size=self.tp_size,
      pipeline_model_parallel_size=1,
      context_parallel_size=1,
    )
    # Not optional. Megatron's tensor-parallel layers initialize their weight
    # shards under get_cuda_rng_tracker().fork(), which raises "cuda rng state
    # model-parallel-rng is not added" until this seeds the tracker. It also
    # gives tensor-parallel ranks *different* offsets for the same logical
    # tensor, which is what makes a sharded initialization equivalent to an
    # unsharded one.
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(MEGATRON_SEED)
    print(f"Megatron parallel state: TP={self.tp_size} DP={world // self.tp_size} sequence_parallel={self.sequence_parallel}")

  def data_parallel_group(self):
    _, parallel_state = require_megatron()
    return parallel_state.get_data_parallel_group()

  # -- data-parallel geometry (BaseTrainerWorker hooks) ---------------------
  #
  # The base class shards datums by global rank, which would hand tensor-parallel
  # peers different data -- they hold different slices of one model and must see
  # the same tokens. Every hook below is therefore scoped to the DP group. When
  # TP == WORLD_SIZE the DP group has one member and forward_backward degenerates
  # to "every rank computes everything", which is exactly right.

  def shard_rank(self) -> int:
    _, parallel_state = require_megatron()
    return parallel_state.get_data_parallel_rank() if parallel_state.model_parallel_is_initialized() else 0

  def shard_count(self) -> int:
    _, parallel_state = require_megatron()
    return parallel_state.get_data_parallel_world_size() if parallel_state.model_parallel_is_initialized() else 1

  def shard_all_reduce_max(self, value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.long, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self.data_parallel_group())
    return int(tensor.item())

  def shard_all_reduce_sum(self, value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=self.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.data_parallel_group())
    return float(tensor.item())

  def shard_all_gather_object(self, value: Any) -> list[Any]:
    # all_gather_object pickles to a CUDA tensor when the group is NCCL, so this
    # needs no separate gloo group.
    gathered: list[Any] = [None] * self.shard_count()
    dist.all_gather_object(gathered, value, group=self.data_parallel_group())
    return gathered

  # -- model lifecycle ------------------------------------------------------

  def load_base_model(self, base_model_name: str) -> None:
    if self.model_chunks and self.base_model_name == base_model_name:
      print(f"Megatron model {base_model_name} already loaded.")
      return

    AutoBridge, _ = require_megatron()
    register_gemma4_unified_bridge()
    # Before the provider builds any layer: the spec is read inside
    # provide_distributed_model below.
    install_gemma4_flex_attention()
    self.initialize_parallel_state()
    print(f"Loading Megatron model {base_model_name} (rank {os.getenv('RANK', '0')}/{os.getenv('WORLD_SIZE', '1')}, TP={self.tp_size})...")

    self.base_model_name = base_model_name
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    # The version-sensitive seam. AutoBridge reads the HF checkpoint, hands back
    # a provider carrying the equivalent TransformerConfig, and streams the
    # weights into Megatron's sharded layout as the model is built.
    self.bridge = AutoBridge.from_hf_pretrained(base_model_name, torch_dtype=dtype)
    provider = self.bridge.to_megatron_provider(load_weights=True)
    provider.tensor_model_parallel_size = self.tp_size
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = self.sequence_parallel
    provider.params_dtype = dtype
    provider.bf16 = dtype == torch.bfloat16
    if not MEGATRON_APEX_FUSIONS:
      provider.gradient_accumulation_fusion = False
      provider.masked_softmax_fusion = False
    if ENABLE_GRADIENT_CHECKPOINTING:
      provider.recompute_granularity = "full"
      provider.recompute_method = "uniform"
      provider.recompute_num_layers = 1
    provider.finalize()
    chunks = provider.provide_distributed_model(wrap_with_ddp=False)
    chunks = self.apply_lora(chunks)
    self.model_chunks = self.wrap_with_ddp(chunks, provider)
    print(f"Successfully loaded Megatron model ({len(self.model_chunks)} chunk(s), gradient checkpointing={ENABLE_GRADIENT_CHECKPOINTING}).")

  def apply_lora(self, chunks: list[torch.nn.Module]) -> list[torch.nn.Module]:
    """Freeze the base weights and attach LoRA adapters, if a rank is set.

    Ordering is load-bearing: this runs between provide_distributed_model and
    wrap_with_ddp because DistributedDataParallel sizes its flat gradient
    buckets from the parameters that require grad. Wrap first and the buckets
    cover all 12B frozen weights -- allocating (and all-reducing) gradient
    memory for tensors that never get one.

    This deliberately does not make the worker a LoRA worker as far as the rest
    of the system is concerned, and it stays in FULL_PARAMETER_WORKERS. The
    bridge's export path merges adapters into the base weights by default
    (export_hf_weights(merge_adapter_weights=True), which save_hf_pretrained
    calls), so save_checkpoint keeps emitting an ordinary whole HF checkpoint.
    The sampler reloads it as it already does and never learns an adapter
    existed. LoRA here is purely a training-memory decision.
    """
    if MEGATRON_LORA_RANK <= 0:
      return chunks

    from megatron.bridge.peft.lora import LoRA

    targets = [target.strip() for target in MEGATRON_LORA_TARGETS.split(",") if target.strip()]
    lora = LoRA(dim=MEGATRON_LORA_RANK, alpha=MEGATRON_LORA_ALPHA, dropout=MEGATRON_LORA_DROPOUT, target_modules=targets)
    chunks = lora(chunks, training=True)
    trainable = sum(param.numel() for chunk in chunks for param in chunk.parameters() if param.requires_grad)
    total = sum(param.numel() for chunk in chunks for param in chunk.parameters())
    print(
      f"[Megatron Worker] LoRA rank={MEGATRON_LORA_RANK} alpha={MEGATRON_LORA_ALPHA} on {targets}: "
      f"{trainable:,} trainable of {total:,} ({100 * trainable / total:.3f}%)."
    )
    return chunks

  def wrap_with_ddp(self, chunks: list[torch.nn.Module], config: Any) -> list[torch.nn.Module]:
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    # average_in_collective is what makes the base class's arithmetic come out
    # right: forward_backward scales each real pass by shard_count precisely
    # because it expects the data-parallel reduction to average, so summing here
    # would inflate every gradient by the DP size.
    #
    # overlap_grad_reduce stays off. With it on the reduce-scatter fires inside
    # backward, which turns every backward into a cross-DP collective and forces
    # every rank to run the same number of passes -- the filler-pass padding this
    # backend just switched off.
    ddp_config = DistributedDataParallelConfig(
      grad_reduce_in_fp32=True,
      overlap_grad_reduce=False,
      use_distributed_optimizer=MEGATRON_DISTRIBUTED_OPTIMIZER,
      check_for_nan_in_grad=True,
      average_in_collective=True,
    )
    return [DistributedDataParallel(config, ddp_config, chunk) for chunk in chunks]

  def create_model(self, base_model_name: str, model_id: str | None = None, config: MegatronConfig | None = None) -> None:
    if config is not None:
      self.cpu_offload = config.cpu_offload
    self.load_base_model(base_model_name)
    if config is not None and config.seed is not None:
      torch.manual_seed(config.seed)
    self.prepare_model_for_training()

  def prepare_model_for_training(self) -> None:
    assert self.model_chunks, "Model is not loaded. Call load_base_model first."
    for chunk in self.model_chunks:
      chunk.train()
      chunk.zero_grad_buffer()
    self.trainable_params = [param for chunk in self.model_chunks for param in chunk.parameters() if param.requires_grad]
    if not self.trainable_params:
      raise ValueError("No trainable parameters found in the Megatron model")

  def gpt_model(self) -> torch.nn.Module:
    """The unwrapped GPTModel, out from behind Megatron's DDP."""
    assert self.model_chunks
    return self.model_chunks[0].module

  # -- forward / backward ---------------------------------------------------

  def forward_backward(
    self, data: list[Datum], loss_fn: str, loss_config: dict | None = None, model_id: str | None = None, forward_only: bool = False
  ) -> dict[str, Any]:
    assert self.model_chunks, "Model must be loaded first."
    res = super().forward_backward(self.model_chunks[0], data, loss_fn, loss_config, forward_only=forward_only)
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return res

  def compute_target_logprobs(
    self,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_token_ids: torch.Tensor,
  ) -> torch.Tensor:
    """Return [batch, seq] target logprobs, projecting the vocab in chunks.

    labels= is still passed because GPTModel forwards it to the output_processor
    hook; the stock loss path it would otherwise drive is never reached. See
    chunked_target_logprobs for what that hook does and why the default path
    cannot hold a long context.

    Targets are already aligned with inputs position-for-position by the Tinker
    client -- input_ids[t] predicts target_token_ids[t] -- and Megatron's loss
    does no shifting either, so they pass straight through.
    """
    seq_len = target_token_ids.shape[1]
    input_ids = input_ids[:, :seq_len]
    batch = input_ids.shape[0]

    # Sequence parallelism scatters the sequence across the TP group, so the
    # length has to divide by it. Padding here and slicing the result back is
    # cheaper than rejecting the batch; the extra positions are dropped before
    # any caller sees them.
    padded_len = seq_len
    if self.sequence_parallel and seq_len % self.tp_size:
      padded_len = ((seq_len // self.tp_size) + 1) * self.tp_size
      pad_id = self.tokenizer.pad_token_id if self.tokenizer and self.tokenizer.pad_token_id is not None else 0
      input_ids = torch.nn.functional.pad(input_ids, (0, padded_len - seq_len), value=pad_id)
      target_token_ids = torch.nn.functional.pad(target_token_ids, (0, padded_len - seq_len), value=0)

    position_ids = torch.arange(padded_len, device=input_ids.device).unsqueeze(0).expand(batch, padded_len)

    # attention_mask=None means plain causal. pad_model_inputs right-pads, so a
    # real token never attends to a pad, and the pad rows' logprobs are dropped
    # by the caller's [:, :length] slice. Passing the dense mask instead would
    # make Megatron materialize [batch, 1, seq, seq] and lose the fused kernel.
    logprobs = self.gpt_model()(input_ids, position_ids, None, labels=target_token_ids, output_processor=chunked_target_logprobs)
    return logprobs[:, :seq_len]

  # -- optimizer ------------------------------------------------------------

  def build_optimizer(self, adam_params: dict[str, Any]) -> None:
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

    lr = adam_params.get("learning_rate", 1e-4)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    print(f"Initializing Megatron AdamW (lr={lr}, distributed_optimizer={MEGATRON_DISTRIBUTED_OPTIMIZER})")
    optimizer_config = OptimizerConfig(
      optimizer="adam",
      lr=lr,
      min_lr=lr,
      adam_beta1=adam_params.get("beta1", 0.9),
      adam_beta2=adam_params.get("beta2", 0.95),
      adam_eps=adam_params.get("eps", 1e-12),
      weight_decay=adam_params.get("weight_decay", 0.0),
      # Clipping is re-read from adam_params on every step; 0.0 disables it.
      clip_grad=0.0,
      bf16=dtype == torch.bfloat16,
      fp16=False,
      params_dtype=dtype,
      use_distributed_optimizer=MEGATRON_DISTRIBUTED_OPTIMIZER,
    )
    self.optimizer = get_megatron_optimizer(optimizer_config, self.model_chunks)

  def optim_step(self, adam_params: dict[str, Any], model_id: str | None = None) -> dict[str, Any]:
    assert self.model_chunks, "Model must be loaded first."
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if self.optimizer is None:
      self.build_optimizer(adam_params)

    learning_rate = adam_params.get("learning_rate")
    if learning_rate is not None:
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = learning_rate

    max_grad_norm = adam_params.get("grad_clip_norm") or 0.0
    if max_grad_norm <= 0.0 or math.isinf(max_grad_norm):
      max_grad_norm = 0.0
    for optimizer in self.chained_optimizers():
      optimizer.config.clip_grad = max_grad_norm

    # With overlap_grad_reduce off, nothing has reduced gradients across the
    # data-parallel group yet; this is where it happens.
    for chunk in self.model_chunks:
      chunk.finish_grad_sync()

    success, grad_norm, _num_zeros = self.optimizer.step()

    for chunk in self.model_chunks:
      chunk.zero_grad_buffer()
    self.optimizer.zero_grad()

    return {
      "metrics": {
        "grad_norm:mean": self.sanitize_float(float(grad_norm) if grad_norm is not None else 0.0),
        # Megatron skips the update on a non-finite gradient and says so only in
        # this return value. Unreported, a run would keep spending GPU hours
        # while the weights sat still.
        "optim_step_skipped:mean": 0.0 if success else 1.0,
      },
    }

  def chained_optimizers(self) -> list[Any]:
    """Megatron returns a ChainedOptimizer when the model has several param groups."""
    return list(getattr(self.optimizer, "chained_optimizers", None) or [self.optimizer])

  # -- checkpointing --------------------------------------------------------

  def save_checkpoint(self, path: str, metadata: dict[str, Any], include_optimizer: bool = False) -> dict[str, Any]:
    assert self.model_chunks, "Model must be loaded first."
    # Same atomic staging as the FSDP worker: a save killed mid-write must never
    # leave a half-overwritten directory that vLLM or a resume can load as a mix
    # of old and new shards.
    staging_path = f"{path}.staging-{os.getpid()}"
    previous_path = f"{path}.previous-{os.getpid()}"
    if is_primary():
      shutil.rmtree(staging_path, ignore_errors=True)
      os.makedirs(staging_path, exist_ok=True)
    barrier()

    # HF format, not a Megatron checkpoint: this is what the vLLM sampler loads
    # and what load_from_state reads back. save_hf_pretrained gathers the
    # tensor-parallel shards, so it is collective -- every rank calls it.
    self.bridge.save_hf_pretrained(self.model_chunks, staging_path)
    if is_primary() and self.tokenizer is not None:
      self.tokenizer.save_pretrained(staging_path)

    if include_optimizer and self.optimizer is not None:
      self.save_optimizer(os.path.join(staging_path, OPTIMIZER_SUBDIR))

    if is_primary():
      with open(os.path.join(staging_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)
      shutil.rmtree(previous_path, ignore_errors=True)
      if os.path.exists(path):
        os.rename(path, previous_path)
      os.rename(staging_path, path)
      shutil.rmtree(previous_path, ignore_errors=True)
    barrier()
    print(f"Saved Megatron state to {path}")
    return {"path": path}

  def model_sharded_state_dict(self) -> dict[str, Any]:
    sharded: dict[str, Any] = {}
    for chunk in self.model_chunks:
      sharded.update(chunk.sharded_state_dict())
    return sharded

  def save_optimizer(self, optimizer_path: str) -> None:
    """Write optimizer state through Megatron's distributed checkpointing.

    Not torch.save: with use_distributed_optimizer each rank holds a different
    1/DP slice of the moments, so a per-rank save would be unloadable at any
    other parallel layout. dist_checkpointing records the sharding.
    """
    from megatron.core import dist_checkpointing

    if is_primary():
      os.makedirs(optimizer_path, exist_ok=True)
    barrier()
    state = self.optimizer.sharded_state_dict(self.model_sharded_state_dict(), is_loading=False)
    dist_checkpointing.save(state, optimizer_path)

  def load_optimizer(self, optimizer_path: str, adam_params: dict[str, Any]) -> None:
    from megatron.core import dist_checkpointing

    if self.optimizer is None:
      self.build_optimizer(adam_params)
    state = self.optimizer.sharded_state_dict(self.model_sharded_state_dict(), is_loading=True)
    self.optimizer.load_state_dict(dist_checkpointing.load(state, optimizer_path))
    print(f"Restored Megatron optimizer state from {optimizer_path}")

  def save_model(self, alias: str | None = None) -> dict[str, Any]:
    name = alias or "megatron-model"
    save_path = name if os.path.isabs(name) else os.path.join(paths.tmp_dir(), "megatron", name)
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": "weights",
      "model_id": alias,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(save_path, metadata)

  def save_state(self, model_id: str, state_path: str, include_optimizer: bool = False, kind: str = "state") -> dict[str, Any]:
    metadata = {
      "base_model": self.base_model_name,
      "created_at": datetime.now().isoformat(),
      "kind": kind,
      "has_optimizer": include_optimizer and self.optimizer is not None,
      "model_id": model_id,
      "timestamp": time.time(),
    }
    return self.save_checkpoint(state_path, metadata, include_optimizer)

  def load_from_state(self, model_id: str, state_path: str, restore_optimizer: bool = False) -> dict[str, Any]:
    metadata_path = os.path.join(state_path, "metadata.json")
    if not os.path.exists(metadata_path):
      raise FileNotFoundError(f"No metadata.json found at {state_path}")
    with open(metadata_path) as f:
      metadata = json.load(f)

    base_model = metadata.get("base_model")
    if not base_model:
      raise ValueError(f"metadata.json at {state_path} missing base_model")

    # save_checkpoint wrote HF format, so the same bridge path that imports a hub
    # checkpoint reads our own back.
    self.model_chunks = []
    self.load_base_model(state_path)
    self.base_model_name = base_model
    self.prepare_model_for_training()

    if restore_optimizer and metadata.get("has_optimizer"):
      optimizer_path = os.path.join(state_path, OPTIMIZER_SUBDIR)
      if os.path.exists(optimizer_path):
        self.load_optimizer(optimizer_path, {})

    print(f"Loaded Megatron state from {state_path}")
    return {"model_id": model_id, "base_model": base_model}

  def generate(
    self,
    prompt_tokens: list[int],
    max_tokens: int,
    num_samples: int = 1,
    temperature: float = 0.0,
    model_id: str | None = None,
    include_prompt_logprobs: bool = False,
  ) -> dict[str, Any]:
    raise RuntimeError("Sampling from a Megatron trainer is unsupported; use the vLLM sampler worker")

  # -- sleep / wake ---------------------------------------------------------
  #
  # Same contract as FFTTrainingWorker: the time-slicer calls wake_up() inside
  # the trainer's GPU lease and sleep() in the finally, both duck-typed by
  # FFTTrainingRequestsProcessor.run_once. Anything the trainer leaves resident
  # is memory the co-located vLLM sampler cannot have.

  def sleep(self) -> None:
    """Move weights, gradients and optimizer moments to host RAM."""
    if not self.cpu_offload or not self.model_chunks or self._is_offloaded or not torch.cuda.is_available():
      return
    if self.flat_buffers() is None:
      # Refusing loudly beats a silent half-offload: if the buffers are not the
      # shape we know how to move, the params would end up aliasing storage the
      # allocator has already handed back.
      self.cpu_offload = False
      print("[Megatron Worker] DDP buffer layout not recognised; CPU offload disabled for this process.")
      return
    start_t = time.perf_counter()
    self.move_flat_buffers("cpu")
    self.move_optimizer_state("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    self._is_offloaded = True
    print(f"[Megatron Worker] Moved weights and optimizer to CPU in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def wake_up(self, include_optimizer: bool = True) -> None:
    if not self.cpu_offload or not self.model_chunks or not self._is_offloaded or not torch.cuda.is_available():
      return
    start_t = time.perf_counter()
    self.move_flat_buffers(self.device)
    if include_optimizer:
      self.move_optimizer_state(self.device)
    self._is_offloaded = False
    print(f"[Megatron Worker] Restored weights and optimizer to CUDA in {(time.perf_counter() - start_t) * 1000:.1f} ms.")

  def flat_buffers(self) -> list[Any] | None:
    """Megatron's contiguous per-bucket parameter and gradient storages.

    Returns None if this megatron-core exposes them differently, which is the
    signal to disable offload rather than guess.
    """
    buffers = []
    for chunk in self.model_chunks:
      groups = list(getattr(chunk, "buffers", None) or []) + list(getattr(chunk, "expert_parallel_buffers", None) or [])
      for buf in groups:
        if not hasattr(buf, "param_data"):
          return None
        buffers.append(buf)
    return buffers or None

  def move_flat_buffers(self, device: torch.device | str) -> None:
    """Move the DDP buffers and re-point every view into them.

    nn.Module.to() is wrong here, which is the whole reason this method exists.
    Megatron's DDP allocates one flat tensor per bucket group and every
    parameter's .data is a view into it, so .to() would copy each view somewhere
    new and leave the flat buffer -- the tensor that actually holds the memory --
    exactly where it was. The offload would free nothing and the next
    finish_grad_sync() would reduce into storage no parameter reads.

    So move the flat tensor, then rebuild the views by pointer arithmetic
    against the old base. Recovering offsets from data_ptr rather than from
    Megatron's internal index map means this keeps working across versions that
    rename the map, and it catches the bucket-level views too.
    """
    target = torch.device(device) if isinstance(device, str) else device
    for buf in self.flat_buffers() or []:
      for attr in ("param_data", "grad_data"):
        flat = getattr(buf, attr, None)
        if flat is None or flat.device == target:
          continue
        moved = flat.to(target)
        base, span, itemsize = flat.data_ptr(), flat.numel() * flat.element_size(), flat.element_size()
        for view in self.views_into(buf, base, base + span):
          offset = (view.data_ptr() - base) // itemsize
          view.data = moved[offset : offset + view.numel()].view(view.shape)
        setattr(buf, attr, moved)

  def views_into(self, buf: Any, low: int, high: int) -> list[torch.Tensor]:
    """Every tensor aliasing [low, high) of one flat buffer."""
    candidates: list[torch.Tensor | None] = []
    for chunk in self.model_chunks:
      for param in chunk.parameters():
        candidates.extend([param.data, param.grad])
    for bucket in getattr(buf, "buckets", None) or []:
      candidates.extend([getattr(bucket, "param_data", None), getattr(bucket, "grad_data", None)])
    return [tensor for tensor in candidates if tensor is not None and low <= tensor.data_ptr() < high]

  def move_optimizer_state(self, device: torch.device | str) -> None:
    """Move Adam moments and fp32 master weights, which live outside the buffers."""
    if self.optimizer is None:
      return
    target = torch.device(device) if isinstance(device, str) else device
    for optimizer in self.chained_optimizers():
      inner = getattr(optimizer, "optimizer", None)
      for state in getattr(inner, "state", {}).values():
        for key, value in state.items():
          if isinstance(value, torch.Tensor) and (key != "step" or target.type == "cpu"):
            state[key] = value.to(target)
      for attr in MASTER_PARAM_GROUPS:
        for group in getattr(optimizer, attr, None) or []:
          for tensor in group:
            if isinstance(tensor, torch.Tensor):
              tensor.data = tensor.data.to(target)
