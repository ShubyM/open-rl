"""Adapt supported model families to the Automodel training path.

Resolve construction policy before NeMo loads and shards the model, then apply
the model settings that require its final module layout. Forward execution owns
its temporary hidden-state hook; the worker owns distributed tensor operations
and the context-parallel attention lifetime through backward.

Configuration comes from callers. NeMo imports stay lazy so CPU tests do not
need the separate Automodel interpreter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.utils.checkpoint

if TYPE_CHECKING:
  from transformers import PreTrainedConfig


# Leaf names are matched only below decoder layers. Vision blocks and the
# multi-token-prediction head cannot be hot-loaded by language-only samplers.
DEFAULT_LORA_TARGETS = "q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,gate_proj,up_proj,down_proj"


def attention_kwargs(model_type: str, cp_size: int, choice: str) -> dict[str, Any]:
  """Select attention kernels, including Gemma 4's wide-head CP backend.

  Gemma 4 uses FFPA for its 512-wide global heads and FlexAttention for sliding
  layers. Under CP, the ring patches SDPA and requests FFPA for full-attention
  chunks through the text config, as in Automodel's Gemma 4 CP recipes.
  """
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


def widest_head_dim(text_config: PreTrainedConfig) -> int:
  """Read the largest head width across uniform and per-layer HF configs.

  Transformers 5.15 makes head_dim a per-layer attribute on heterogeneous
  models (Gemma 4: 256 on sliding layers, 512 on global ones) and raises on
  the config-level read, so use the per-layer views when present.
  """
  try:
    layers = list(text_config.per_layer_config or ())
  except AttributeError:
    layers = []
  if layers:
    return max(int(getattr(layer, "head_dim", 0) or 0) for layer in layers)
  try:
    # Older Gemma configs store sliding/global widths as separate fields.
    return max(int(text_config.head_dim or 0), int(getattr(text_config, "global_head_dim", 0) or 0))
  except AttributeError:
    return 0


def flex_kernel_options(text_config: PreTrainedConfig, flex_runs_widest: bool) -> dict[str, Any]:
  """Use FlexAttention tiles that fit shared memory for wide attention heads.

  HF's flex integration reads kernel_options from forward kwargs, and NeMo's
  FFPA route passes them through. Backward needs smaller tiles than forward.
  """
  widest = widest_head_dim(text_config)
  if widest < 256:
    return {}
  # 64/32 fits 256-wide sliding heads; use 16-wide tiles when flex must also
  # handle 512-wide global heads without FFPA.
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


def uses_group_checkpointing(text_config: PreTrainedConfig) -> bool:
  """Identify native layer loops that support whole-layer recomputation.

  Other models use Automodel's checkpointing policy before FSDP wrapping. That
  policy handles HF mixed-precision recomputation and mutable shared K/V state.
  """
  if getattr(text_config, "num_kv_shared_layers", 0):
    return False
  # Gemma 4 reads each layer's attention_type inside its ModuleDict loop, so
  # replacing its values with groups would change layer arguments.
  return text_config.model_type in {"qwen3_5_text", "qwen3_5_moe_text", "qwen3_next"}


@dataclass(frozen=True)
class ModelPolicy:
  """Model choices shared by construction, sharding, and forward execution."""

  load_kwargs: dict[str, Any]
  forward_kwargs: dict[str, Any]
  checkpoint_group_size: int
  activation_checkpointing: bool
  tp_plan: dict[str, Any] | None

  def apply(self, model: torch.nn.Module) -> None:
    """Apply model settings after NeMo construction and FSDP sharding."""
    if self.checkpoint_group_size:
      install_group_checkpointing(model, self.checkpoint_group_size)
    model.config.output_hidden_states = False


def model_policy(config: PreTrainedConfig, *, tp_size: int, cp_size: int, recompute_num_layers: int, attention: str) -> ModelPolicy:
  """Resolve model loading, TP layout, and checkpointing before construction."""
  text_config = config.get_text_config()
  model_type = text_config.model_type
  load_kwargs = attention_kwargs(model_type, cp_size, attention)
  attn = load_kwargs.get("attn_implementation")
  uses_ffpa = attn == "ffpa" or load_kwargs.get("text_config", {}).get("cp_full_attn_backend") == "ffpa"
  if attn == "flex_attention" or uses_ffpa:
    forward_kwargs = flex_kernel_options(text_config, flex_runs_widest=(attn == "flex_attention"))
  else:
    forward_kwargs = {}
  if model_type.startswith("qwen3_5"):
    # The native model otherwise builds and runs an unused prediction head
    # over the entire sequence on every training forward.
    load_kwargs["num_nextn_predict_layers"] = 0
  if model_type.startswith(("qwen3_5", "qwen3_next")):
    # Native Qwen selects attention through BackendConfig instead of HF's
    # attn_implementation. SDPA keeps CP on the validated ring-attention path
    # even when Transformer Engine is installed.
    load_kwargs["backend"] = {"attn": "sdpa"}
  checkpoint_group_size = recompute_num_layers if recompute_num_layers > 0 and uses_group_checkpointing(text_config) else 0
  return ModelPolicy(
    load_kwargs=load_kwargs,
    forward_kwargs=forward_kwargs,
    checkpoint_group_size=checkpoint_group_size,
    activation_checkpointing=recompute_num_layers > 0 and checkpoint_group_size == 0,
    tp_plan=tensor_parallel_plan(config) if tp_size > 1 else None,
  )


def tensor_parallel_plan(config: PreTrainedConfig) -> dict[str, Any]:
  """Address the HF text-model TP plan from the top of the NeMo model.

  Automodel's automatic plan reads _tp_plan off the language model, which its
  native Qwen 3.5 backbone does not carry. The plan lives on the text config;
  prefix it for the module tree. GDN layers remain replicated because they are
  not TP-shardable.
  """
  from nemo_automodel.components.distributed.parallelizer import translate_to_torch_parallel_style
  from torch.distributed.tensor import Replicate
  from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

  text_config = config.get_text_config()
  architectures = config.architectures or []
  prefix = "model.language_model" if architectures and architectures[0].endswith("ForConditionalGeneration") else "model"
  plan: dict[str, Any] = {f"{prefix}.embed_tokens": RowwiseParallel(input_layouts=Replicate())}
  for name, style_name in (text_config.base_model_tp_plan or {}).items():
    try:
      style = translate_to_torch_parallel_style(style_name)
    except ValueError:
      # HF's family plans can contain MoE-only styles unsupported by NeMo.
      # Unmatched dense-model entries have no effect; other entries remain
      # replicated when their parallel style cannot be translated.
      print(f"[Automodel Worker] TP plan: skipping {name} ({style_name}) with no torch parallel style.")
      continue
    if style is not None:
      plan[f"{prefix}.{name}"] = style
  plan["lm_head"] = ColwiseParallel(output_layouts=Replicate())
  return plan


def text_backbone(model: torch.nn.Module) -> torch.nn.Module:
  """Resolve the text decoder for causal-LM and multimodal NeMo models."""
  return getattr(model.model, "language_model", model.model)


def forward_hidden_states(model: torch.nn.Module, forward_kwargs: dict[str, Any], **inputs: Any) -> torch.Tensor:
  """Run a forward while capturing only its final hidden states.

  Model-owned CP forwards can discard output_hidden_states before calling the
  text model, so capture the final norm instead. Remove the hook even on failure
  and leave no captured activations on the model or worker. DTensor results keep
  their placements for the worker to materialize in its distributed context.
  """
  hidden: torch.Tensor | None = None

  def capture(module: torch.nn.Module, args: Any, output: torch.Tensor) -> None:
    nonlocal hidden
    hidden = output

  hook = text_backbone(model).norm.register_forward_hook(capture)
  try:
    model(**inputs, use_cache=False, logits_to_keep=1, **forward_kwargs)
    if hidden is None:
      raise RuntimeError("Automodel forward did not run the final norm; the logprob path requires its hidden states.")
    return hidden
  finally:
    hook.remove()
    hidden = None


class LayerGroup:
  """Run decoder layers under one non-reentrant checkpoint."""

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
  """Iterate the native backbone's layer dict as checkpointed training groups.

  Supported native backbones run ``for layer in self.layers.values()``. Swap
  this class in after sharding to preserve parameters, names, and FSDP2 units.
  Grouping whole layers reduces retained activations compared with NeMo's
  separate attention and MLP checkpoints on native Qwen models.
  """

  group_size = 1

  def values(self):
    layers = list(self._modules.values())
    if not self.training:
      return iter(layers)
    groups = [layers[start : start + self.group_size] for start in range(0, len(layers), self.group_size)]
    return iter([LayerGroup(group) for group in groups])


def install_group_checkpointing(model: torch.nn.Module, group_size: int) -> None:
  """Group a supported native ModuleDict loop after FSDP has indexed layers."""
  if group_size < 1:
    raise ValueError("Activation checkpoint group size must be positive")
  layers = text_backbone(model).layers
  if isinstance(layers, torch.nn.ModuleDict):
    layers.__class__ = GroupCheckpointedLayers
    layers.group_size = group_size
    print(f"[Automodel Worker] activation checkpointing in groups of {group_size} layers.")
    return
  raise RuntimeError(
    f"Grouped activation checkpointing requires a native ModuleDict of layers, got {type(layers).__name__}. "
    "The model layout changed; use Automodel's checkpointing policy before FSDP wrapping or set "
    "OPEN_RL_AUTOMODEL_RECOMPUTE_NUM_LAYERS=0 to disable checkpointing."
  )


@dataclass(frozen=True)
class OutputProjection:
  """The model's vocabulary projection, before distributed materialization."""

  weight: torch.Tensor
  bias: torch.Tensor | None
  softcap: float | None


def output_projection(model: torch.nn.Module) -> OutputProjection:
  """Resolve the output head and optional bias and logit softcapping."""
  head = model.get_output_embeddings()
  return OutputProjection(
    weight=head.weight,
    bias=getattr(head, "bias", None),
    softcap=getattr(model.config.get_text_config(), "final_logit_softcapping", None),
  )


def adapter_state_from_hf(model: torch.nn.Module, state: dict[str, Any]) -> dict[str, Any]:
  """Translate flat PEFT snapshot keys into the native model's state layout."""
  adapter = getattr(model, "state_dict_adapter", None)
  if adapter is not None:
    state = adapter.from_hf(state, device_mesh=None)
  return {name.removeprefix("base_model.model."): value for name, value in state.items()}


def lora_target_modules(names: Iterable[str]) -> list[str]:
  """Match projection leaves under plain and nested text decoder layers."""
  return [f"{prefix}.layers.*.{name}" for name in names for prefix in ("model", "model.*")]
