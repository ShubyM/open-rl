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

# The architecture string Qwen3.5's multimodal checkpoints declare. The dense
# text checkpoints declare Qwen3_5ForCausalLM and need none of this.
MULTIMODAL_HF_CLASS = "Qwen3_5ForConditionalGeneration"


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

  MegatronModelBridge.register_bridge(
    source=MULTIMODAL_HF_CLASS,
    target=GPTModel,
    model_type="qwen3_5_text",
  )(Qwen35TextOnlyBridge)
