# Qwen3.5 workarounds for the Megatron backend.
"""What Qwen3.5 needs that the generic Megatron path does not provide.

Qwen/Qwen3.5-9B ships as a multimodal checkpoint. Its config.json declares
architectures=["Qwen3_5ForConditionalGeneration"], nests every language
hyperparameter under text_config, and stores the language tower under
model.language_model.* beside a 333-tensor model.visual.* vision tower.

megatron-bridge 0.6.1 registers that architecture name to Qwen35VLBridge, which
targets Qwen3VLModel. Dispatch therefore *succeeds* -- this is not the Gemma-4
failure, where an unregistered name made AutoBridge.can_handle return False --
and quietly builds a vision-language model whose vision tower this text-only RL
trainer loads, shards and never reads. There is no text-mode escape hatch to
flip: GEMMA4_CONVERSION_MODE is the only conversion-mode switch anywhere in the
bridge, and it is Gemma-specific.

Nothing but the wiring is missing. Qwen35Bridge -- the bridge already
registered for Qwen3_5ForCausalLM -- targets GPTModel, and its mapping helpers
already accept the prefixes the multimodal layout uses; their own docstrings
name hf_prefix="model.language_model." / megatron_prefix="" as the VL pairing.
So re-point the architecture at a text-only subclass of that existing bridge
rather than writing a new one.

Verified against megatron-bridge 0.6.1 / megatron-core 0.19.0. If the seam
moves, it moves at one of three places, and each fails loudly rather than
silently: the architecture string, Qwen35Bridge._get_dense_lm_mappings' prefix
kwargs, or text_config nesting. A registration that upstream adds itself makes
register_text_only_bridge a no-op worth deleting, not a conflict.

Qwen3.5's gated-deltanet layers additionally need flash-linear-attention in the
trainer interpreter. megatron.core.ssm.gated_delta_net raises ImportError with
no fallback when it is absent, and the package splits across two distributions:
flash-linear-attention ships only fla/layers and fla/models, while fla.modules
and fla.ops -- the two the kernel path actually imports -- live in fla-core.
Installing the former alone leaves `import fla` resolving to a namespace
package with __file__ None and HAVE_FLA quietly False.
"""

from __future__ import annotations

import contextlib
import threading

# The architecture string Qwen3.5's multimodal checkpoints declare. The dense
# text checkpoints declare Qwen3_5ForCausalLM and need none of this.
MULTIMODAL_HF_CLASS = "Qwen3_5ForConditionalGeneration"

# The only weights a text-only conversion of this checkpoint may leave
# unproduced: the vision tower the bridge subclass drops.
#
# lm_head.weight is deliberately not here. _get_dense_lm_mappings maps
# output_layer.weight to a bare "lm_head.weight" with no hf_prefix, so the
# bridge does emit it, and it is a trained language weight: filling it in from
# the base snapshot would ship a checkpoint whose output head silently
# predates training.
MULTIMODAL_SOURCE_PREFIXES = ("model.visual.",)

# The multi-token-prediction head is also absent from the export -- provider_
# bridge sets mtp_num_layers = 0 -- but it is the writer's business, not the
# passthrough's. auto_bridge.save_hf_pretrained notices the provider omits MTP
# and removes mtp.* from the writer's expected key map, so those shards count
# as complete without them, and a yielded mtp.* tensor is a strict-mode
# KeyError ("not found in the original model structure"). run42's own failure
# confirms the accounting from the other side: 635 missing = 333 vision + 302
# language, and not one mtp key. The saved checkpoint therefore has no MTP
# head, which is upstream's choice for every text-only export and harmless
# here: nothing that loads our checkpoints reads one.
WRITER_IGNORED_PREFIXES = ("mtp.",)

# Thread-local for the same reason gemma4's is: the trainer dispatches worker
# entry points through asyncio.to_thread, so a save and a sampler sync can be
# in flight on two pool threads at once.
passthrough_state = threading.local()


@contextlib.contextmanager
def multimodal_export_passthrough():
  """Complete the export with the base checkpoint's vision weights, in this block.

  Off by default, and it has to stay off for anything that streams weights to a
  sampler: these tensors are read from disk and are therefore on the CPU, and
  they land at the end of the stream, so a packed NCCL broadcast builds its
  last chunk out of CPU memory and never arrives. That is the run35 failure,
  and it is worth keeping guarded even though run42 publishes a LoRA adapter to
  disk rather than broadcasting.
  """
  previous = getattr(passthrough_state, "enabled", False)
  passthrough_state.enabled = True
  try:
    yield
  finally:
    passthrough_state.enabled = previous


class TextConfigView:
  """A hf_pretrained whose .config is the nested text_config.

  Qwen35Bridge.provider_bridge reads hidden_size, num_hidden_layers and the
  rest straight off hf_pretrained.config. On a multimodal checkpoint that
  object is the *outer* config, which carries only vision fields and the
  token ids -- so the provider would come back with the language tower's
  shape fields missing rather than wrong, and fail at model construction.
  """

  def __init__(self, hf_pretrained, text_config):
    self.wrapped = hf_pretrained
    self.config = text_config

  def __getattr__(self, name):
    return getattr(self.wrapped, name)


def passthrough_tensors(exported: set[str], hf_pretrained, weight_dtype=None):
  """The base checkpoint's tensors a text-only export cannot produce, by name.

  Yields nothing unless multimodal_export_passthrough is active, nothing when
  the bridge has no readable source, and refuses -- by raising, not skipping --
  to supply any key outside MULTIMODAL_SOURCE_PREFIXES: a language tensor
  missing from the export is a conversion bug, and filling it from the base
  snapshot would ship untrained weights under a trained checkpoint's name.
  """
  if not getattr(passthrough_state, "enabled", False):
    return
  source = getattr(getattr(hf_pretrained, "state", None), "source", None)
  if source is None or not hasattr(source, "get_all_keys"):
    return
  missing = [
    key for key in source.get_all_keys()
    if key not in exported and not key.startswith(WRITER_IGNORED_PREFIXES)
  ]
  if not missing:
    return
  unexpected = sorted(key for key in missing if not key.startswith(MULTIMODAL_SOURCE_PREFIXES))
  if unexpected:
    raise RuntimeError(
      f"Qwen3.5 export is missing {len(unexpected)} language tensor(s), which this "
      f"passthrough must not fill in from the base checkpoint: {unexpected[:8]}"
    )
  print(f"[Megatron Worker] Copying {len(missing)} vision tensor(s) through the text-only export.")
  for name, weight in source.load_tensors(missing).items():
    yield name, (weight.to(weight_dtype) if weight_dtype is not None else weight)


def register_text_only_bridge() -> None:
  """Point Qwen3_5ForConditionalGeneration at a text-only GPTModel bridge.

  Overwrites megatron-bridge's own registration, which targets Qwen3VLModel.
  That is the whole purpose: the vision tower is 333 of the checkpoint's 775
  tensors and nothing in this trainer would ever read it.
  """
  from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
  from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
  from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge
  from megatron.core.models.gpt.gpt_model import GPTModel

  class Qwen35TextOnlyBridge(Qwen35Bridge):
    """Qwen35Bridge reading a multimodal checkpoint's language tower."""

    def provider_bridge(self, hf_pretrained):
      text_config = getattr(hf_pretrained.config, "text_config", None)
      if text_config is None:
        # A dense text checkpoint reached this bridge; the parent is correct.
        return super().provider_bridge(hf_pretrained)
      provider = super().provider_bridge(TextConfigView(hf_pretrained, text_config))
      # Multi-token prediction is a pretraining head. Left on, the checkpoint's
      # one MTP layer contributes an auxiliary loss at scaling factor 0.1
      # (_apply_qwen35_common_config sets it whenever mtp_num_layers is
      # truthy), which would sit on top of the RL objective and be attributed
      # to it. Dropping the layer and its mappings together keeps the two
      # consistent -- a provider with no MTP layer and a registry that still
      # mapped mtp.* would fail the load on a megatron param that does not
      # exist.
      provider.mtp_num_layers = 0
      return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
      return MegatronMappingRegistry(
        *self._get_dense_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="")
      )

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
      """Yield the parent's language tensors, then the ones it cannot produce.

      Without this the full-checkpoint save cannot succeed at all. The writer
      emits a shard only once every key belonging to it has been handed over,
      and Qwen3.5-9B interleaves its 333 model.visual.* tensors across all four
      shards -- so every shard stays incomplete forever and the language
      tensors that were converted correctly are dropped along with them. run42
      died exactly there, at its first save: 302 correctly converted tensors
      discarded for incomplete shards, and the save raised for 635 missing
      having been given the right answer for all but the 333 vision ones.

      Nine steps of training went with it, because the save path is not
      exercised until the first save and save_every was 10.

      Passing the weights through is preferred over the writer's strict=False,
      which would emit a checkpoint missing its vision tower while
      save_artifacts copies the original multimodal config.json that advertises
      one -- and the vLLM eval job loads what that config describes. It also
      keeps the completeness check armed, so a language tensor going missing
      still fails loudly.
      """
      from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

      exported = set()
      for name, weight in super().stream_weights_megatron_to_hf(
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

      for name, weight in passthrough_tensors(exported, hf_pretrained, weight_dtype):
        yield HFWeightTuple(name, weight)

  MegatronModelBridge.register_bridge(
    source=MULTIMODAL_HF_CLASS,
    target=GPTModel,
    model_type="qwen3_5_text",
  )(Qwen35TextOnlyBridge)
