# Gemma-4 workarounds for the Megatron backend.
"""What Gemma-4 needs that the generic Megatron path does not provide.

Gemma-4 is unusual on four axes at once, and each one breaks a different piece
of the stack:

  * Heterogeneous attention. Five layers in six are sliding-window grouped-query
    attention -- 16 heads, 8 KV groups, head dim 256. Every sixth is global,
    multi-query at head dim 512, and ties K=V so HF stores no v_proj for it.
  * A head dim past what the fused kernels take. 512 exceeds cuDNN 9.17's 256
    limit on sm90, so both megatron's core attention and Transformer Engine fall
    back to materializing [batch, heads, seq, seq] scores.
  * A renamed architecture string, from transformers 5.10.
  * A multimodal checkpoint we convert text-only, shipped as a single unsharded
    safetensors file. The vision and audio weights have nowhere to come from,
    and the writer needs every key in a shard before it writes any of it.

install_flex_attention is the one that buys context (see its docstring); the
other four are correctness fixes without which the LoRA export cannot complete
at all, which takes save_checkpoint and weight sync with it.

Verified against megatron-core 0.19.0 / megatron-bridge 0.6.0 and
transformers 5.10 on 2026-08-26. Every one of these is a monkeypatch against a
pinned dependency, so all five are written to fail loudly and locally if the
seam moves -- except install_flex_attention, which degrades to the stock
attention and only costs a lower context ceiling.
"""

import contextlib
import os
import threading
from typing import Any

import torch

# On by default: megatron's own core attention materializes the
# [batch, heads, seq, seq] score matrix, which is what caps context once the
# logit term is gone. Set to 0 to compare against the stock path.
MEGATRON_FLEX_ATTENTION = os.getenv("OPEN_RL_MEGATRON_FLEX_ATTENTION", "1") == "1"


def register_unified_bridge() -> None:
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


# -- LoRA export on the global attention layers -------------------------------
#
# Both of the next two are needed to export a merged checkpoint, and they fail
# the same way if either is missing: a 9216-row LoRA delta added to an 8192-row
# q_proj. The global layer's fused qkv is 8192 + 512 + 512 = 9216 rows against
# the sliding layers' 4096 + 2048 + 2048 = 8192, and nothing downstream notices
# the mismatch until the addition itself.


def align_global_attn_export_config(chunks: list[torch.nn.Module]) -> bool:
  """Put the global layers' geometry on the config under the names the export reads.

  Gemma4Bridge._split_qkv_linear_out_weight has a branch that rebuilds the
  global geometry and returns ABSENT_PROJECTION for the tied v. It is guarded on
  hasattr(config, "global_head_dim") -- and two spellings of these fields exist
  in megatron-bridge. The MoE Gemma4SelfAttention reads
  global_head_dim/num_global_key_value_heads, while the dense attention our
  text-mode VL provider builds reads global_kv_channels/num_global_query_groups.
  So the layers are built correctly and the export's guard still reads False,
  whereupon it cuts the global qkv with the sliding layout.

  Copy the geometry the global layers were actually built with onto the
  top-level config under the names the export reads, rather than reimplementing
  a split whose K=V tying is easy to get subtly wrong. This runs after the model
  is built, so it cannot change how any layer was constructed.

  Returns False for a model whose layers all share the top-level geometry, which
  needs none of this.
  """
  patched = False
  for chunk in chunks:
    config = chunk.config
    for name, module in chunk.named_modules():
      if not name.endswith("self_attention"):
        continue
      layer = module.config
      if (layer.kv_channels, layer.num_query_groups) == (config.kv_channels, config.num_query_groups):
        continue
      config.global_head_dim = layer.kv_channels
      config.num_global_key_value_heads = layer.num_query_groups
      print(
        f"[Megatron Worker] Global attention export geometry from {name}: "
        f"head dim {layer.kv_channels}, {layer.num_query_groups} KV group(s)."
      )
      patched = True
      break
  return patched


def install_tied_kv_qkv_split() -> bool:
  """Let a K=V-tied layer count as a fused qkv, so its adapter gets split at all.

  MegatronPeftBridge._is_fused_qkv requires exactly three HF names carrying
  q_proj, k_proj and v_proj. Gemma-4's global layers tie K=V, so HF stores two,
  and the layer is classified as an ordinary unfused linear: the splitter is
  never called and the entire 9216-row fused LoRA-B is merged into q_proj.

  This is why align_global_attn_export_config alone is not enough. Gemma4Bridge
  does have the code to cut this tensor correctly, and its ABSENT_PROJECTION
  sentinel exists specifically to say "v_proj is tied here" -- but on the merge
  path that branch is unreachable, because the three-name test rejects the layer
  before any of it runs.

  Patched on Gemma4Bridge rather than on MegatronPeftBridge so the widened test
  reaches only Gemma-4. Two-name q/k is a legitimate description of a tied
  layer, but on an arbitrary architecture it could equally be an unfused pair
  that would then be split as though it were fused.
  """
  from megatron.bridge.models.gemma.gemma4_bridge import Gemma4Bridge

  if getattr(Gemma4Bridge, "_open_rl_tied_kv_qkv", False):
    return True
  original = Gemma4Bridge._is_fused_qkv

  def is_fused_qkv(self, hf_weight_names) -> bool:  # noqa: ANN001
    names = list(hf_weight_names)
    tokens = {token for name in names for token in ("q_proj", "k_proj") if token in name}
    if len(names) == 2 and tokens == {"q_proj", "k_proj"}:
      return True
    return original(self, names)

  Gemma4Bridge._is_fused_qkv = is_fused_qkv
  Gemma4Bridge._open_rl_tied_kv_qkv = True
  return True


# -- Multimodal weights on a text-only export ---------------------------------

# The only weights a text-mode conversion is allowed to leave unproduced. Every
# other name is a language tensor, and a language tensor missing from the export
# is a real bug that has to keep raising -- filling one of those in from the base
# snapshot would silently ship a checkpoint with untrained weights in it.
MULTIMODAL_SOURCE_PREFIXES = (
  "model.embed_audio.",
  "model.embed_vision.",
  "model.vision_embedder.",
  "model.vision_tower.",
  "model.audio_tower.",
)

# Thread-local, not a module global, because the trainer dispatches every worker
# entry point through asyncio.to_thread -- a save and a sampler sync can be in
# flight on two pool threads at once, which is exactly the concurrency that
# expressed the thread-local device bug in run33.
_passthrough = threading.local()


@contextlib.contextmanager
def multimodal_export_passthrough():
  """Complete the export with the base checkpoint's vision weights, in this block.

  Off by default, and it must stay off for the sampler sync. sync_weights_to_
  samplers exports with cpu=False and hands the tensors to vLLM's packed NCCL
  broadcast, while the passthrough's tensors come off disk and are therefore on
  the CPU. They land at the end of the stream, so vLLM's packer builds its last
  chunk out of them, torch.cat returns a CPU buffer, and the NCCL broadcast of
  that buffer never arrives. run35 died there: /update_weights timed out after
  30 minutes, rank 0 raised out of sync() while the other three sat in drain(),
  and the next tensor-parallel allgather hung until the watchdog aborted them.

  Scoping is not just a workaround for that. The samplers hold the full
  multimodal model already and these eleven tensors never change, so sending
  them every optimizer step was pointless work in the hot loop. Only the save
  path needs them, and only because of how the shard writer counts keys.
  """
  previous = getattr(_passthrough, "enabled", False)
  _passthrough.enabled = True
  try:
    yield
  finally:
    _passthrough.enabled = previous


def install_multimodal_export_passthrough() -> bool:
  """Copy the vision/audio weights through the export, so its one shard completes.

  register_unified_bridge sets GEMMA4_CONVERSION_MODE=text, so the Megatron model
  holds the language tower and nothing else, and the export yields 666 of the
  checkpoint's 677 tensors. google/gemma-4-12B-it ships as a single unsharded
  model.safetensors with no index, so SafeTensorsStateSource maps all 677 keys to
  that one file -- and SafeTensorsStateSource.save_generator writes a shard only
  once every key in it has arrived. Eleven never do. The shard never completes,
  all 666 converted language tensors are dropped on the floor, and the save
  raises for 677 missing tensors having been handed 666 correct ones.

  That is what killed run34 at its first checkpoint, and it is why the Megatron
  save path had never once succeeded: it is not reachable at all for a text-only
  export of a multimodal checkpoint that is not sharded.

  The eleven are provably unchanged by training -- there is no vision or audio
  module anywhere in the Megatron model to change them -- so read them from the
  base snapshot and append them. Preferred over save_generator's strict=False,
  which would write the 666 and skip the rest: save_artifacts copies the original
  multimodal config.json, which advertises a vision tower, and the vLLM eval job
  loads what that config describes. Passing the weights through keeps the
  checkpoint matching its own config, and keeps the completeness check armed.

  Upstream hits the identical mechanism for MTP heads and solves it the same way,
  with save_generator's ignored_source_key_prefixes -- but auto_bridge populates
  that for MTP only and exposes no kwarg, so there is nothing to hook.

  Patched on Gemma4VLBridge, not on MegatronModelBridge, so no other
  architecture's export can be completed out of its own base checkpoint.
  """
  from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
  from megatron.bridge.models.gemma_vl.gemma4_vl_bridge import Gemma4VLBridge

  if getattr(Gemma4VLBridge, "_open_rl_multimodal_passthrough", False):
    return True
  original = Gemma4VLBridge.stream_weights_megatron_to_hf

  # Mirrors the upstream signature rather than taking *args, so a seam that moves
  # raises here instead of quietly passing the wrong argument through.
  def stream_weights_megatron_to_hf(
    self,
    megatron_model,  # noqa: ANN001
    hf_pretrained,  # noqa: ANN001
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks=None,  # noqa: ANN001
    merge_adapter_weights: bool = True,
    weight_dtype=None,  # noqa: ANN001
  ):
    exported = set()
    for name, weight in original(
      self,
      megatron_model,
      hf_pretrained,
      cpu=cpu,
      show_progress=show_progress,
      conversion_tasks=conversion_tasks,
      merge_adapter_weights=merge_adapter_weights,
      weight_dtype=weight_dtype,
    ):
      exported.add(name)
      yield HFWeightTuple(name, weight)

    if not getattr(_passthrough, "enabled", False):
      return
    source = getattr(getattr(hf_pretrained, "state", None), "source", None)
    if source is None or not hasattr(source, "get_all_keys"):
      return
    missing = [key for key in source.get_all_keys() if key not in exported]
    if not missing:
      return
    unexpected = sorted(key for key in missing if not key.startswith(MULTIMODAL_SOURCE_PREFIXES))
    if unexpected:
      raise RuntimeError(
        f"Gemma-4 export is missing {len(unexpected)} language tensor(s), which this "
        f"passthrough must not fill in from the base checkpoint: {unexpected[:8]}"
      )
    print(f"[Megatron Worker] Copying {len(missing)} vision/audio tensor(s) through the text-only export.")
    for name, weight in source.load_tensors(missing).items():
      yield HFWeightTuple(name, weight.to(weight_dtype) if weight_dtype is not None else weight)

  Gemma4VLBridge.stream_weights_megatron_to_hf = stream_weights_megatron_to_hf
  Gemma4VLBridge._open_rl_multimodal_passthrough = True
  return True


# -- FlexAttention ------------------------------------------------------------

# torch.compile caches per callable, so compile once for the process rather than
# once per layer. Populated by install_flex_attention, which is the only place
# that may import torch.nn.attention.flex_attention.
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


def flex_block_mask(window: int | None, seq_q: int, seq_kv: int, device) -> Any:
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


def flex_attention_class() -> type:
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
        block_mask=flex_block_mask(self.flex_window, seq_q, key.shape[0], query.device),
        scale=self.softmax_scale,
        enable_gqa=key.shape[2] != heads,
        kernel_options=self.flex_kernel_options,
      )
      return context.permute(2, 0, 1, 3).contiguous().view(seq_q, batch, heads * head_dim)

  return FlexDotProductAttention


def install_flex_attention() -> bool:
  """Point Gemma-4's dense layer spec at FlexAttention. Returns whether it took.

  The seam is narrow. Gemma4DenseProvider.provide calls the module-level
  get_gemma4_layer_spec directly and ignores its own transformer_layer_spec
  field, so rebinding that name in the provider's namespace is the only way in
  without reimplementing the spec. Wrapping the stock spec rather than writing
  one keeps every other submodule -- the Gemma-4 norms, the shared-KV
  attention class -- exactly as the bridge built it.

  Best-effort by design: a checkpoint that is not Gemma-4 never calls this, and
  a megatron-bridge that moved the symbol should cost a slower context ceiling,
  not a failed run.
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
  flex_class = flex_attention_class()

  def flex_layer_spec(config=None):
    spec = get_gemma4_layer_spec(config)
    spec.submodules.self_attention.submodules.core_attention = flex_class
    return spec

  flex_layer_spec._open_rl_flex = True
  gemma4_provider.get_gemma4_layer_spec = flex_layer_spec
  print("[Megatron Worker] Gemma-4 core attention: FlexAttention (block-mask sliding window).")
  return True
