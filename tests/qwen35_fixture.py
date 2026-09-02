"""A tiny Qwen3.5 checkpoint with every property the real one has that matters.

Qwen/Qwen3.5-9B is 775 tensors: a language tower under model.language_model.*,
a vision tower under model.visual.*, a multi-token-prediction head under mtp.*,
and lm_head.weight, interleaved across four safetensors shards. The Megatron
path reads only the first and last of those, and that shape -- a text-only
export of a multimodal, sharded checkpoint -- is what every failure so far
has been about. run42 trained nine steps and died at its first save because
the shard writer dropped correctly converted language tensors for sharing a
shard with vision tensors nobody yielded.

This builds the same shape at a size where an fp32 forward takes milliseconds:
four layers in the real 3:1 gated-deltanet:attention pattern, a two-block
vision tower, synthetic mtp.* tensors, and three shards with keys dealt across
them round-robin so no shard is ever complete from the language tower alone.
Random weights, fixed seed. Nothing here depends on the 9B being downloaded.
"""

from __future__ import annotations

import json
import os

import torch

# Head dimensions are the 9B's, not shrunk with the rest: fla and tilelang
# specialise their kernels on them, so a fixture with 64-wide heads would test
# kernels the training run never executes (and tilelang has no fp32 kernel for
# 64-wide heads at all). Everything else -- hidden size, layers, vocab, the
# vision tower -- is as small as the architecture allows. Each mrope section
# pairs a rotary frequency band with one of the three position streams, so
# the sections sum to the rotary half-dim: 256 * 0.25 / 2 = 32.
HEAD_DIM = 256
MROPE_SECTION = [11, 11, 10]
LINEAR_HEAD_DIM = 128
HIDDEN = 64
VOCAB = 256
LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
SHARDS = 3


def tiny_config():
  from transformers import Qwen3_5Config

  return Qwen3_5Config(
    architectures=["Qwen3_5ForConditionalGeneration"],
    tie_word_embeddings=False,
    text_config=dict(
      model_type="qwen3_5_text",
      hidden_size=HIDDEN,
      num_hidden_layers=len(LAYER_TYPES),
      layer_types=LAYER_TYPES,
      full_attention_interval=4,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=HEAD_DIM,
      intermediate_size=128,
      vocab_size=VOCAB,
      linear_num_key_heads=2,
      linear_num_value_heads=4,
      linear_key_head_dim=LINEAR_HEAD_DIM,
      linear_value_head_dim=LINEAR_HEAD_DIM,
      linear_conv_kernel_dim=4,
      attn_output_gate=True,
      attention_bias=False,
      hidden_act="silu",
      rms_norm_eps=1e-6,
      mtp_num_hidden_layers=1,
      mlp_only_layers=[],
      max_position_embeddings=512,
      tie_word_embeddings=False,
      mamba_ssm_dtype="float32",
      rope_parameters=dict(
        mrope_interleaved=True,
        mrope_section=MROPE_SECTION,
        rope_type="default",
        rope_theta=10000000,
        partial_rotary_factor=0.25,
      ),
    ),
    vision_config=dict(
      model_type="qwen3_5",
      depth=2,
      hidden_size=32,
      num_heads=2,
      intermediate_size=64,
      patch_size=16,
      out_hidden_size=HIDDEN,
      in_channels=3,
      spatial_merge_size=2,
      temporal_patch_size=2,
      num_position_embeddings=64,
      deepstack_visual_indexes=[],
      hidden_act="gelu_pytorch_tanh",
      initializer_range=0.02,
    ),
  )


def synthetic_mtp_tensors() -> dict[str, torch.Tensor]:
  """Stand-ins for the 15 mtp.* tensors, under the real names' prefixes.

  transformers does not instantiate the MTP head, so save_pretrained never
  writes these; the real checkpoint has them anyway. Their values are never
  read by anything on the text-only path -- the test is that they come out
  the far side of a save byte-for-byte, not what they are.
  """
  g = torch.Generator().manual_seed(1)
  return {
    "mtp.fc.weight": torch.randn(HIDDEN, 2 * HIDDEN, generator=g),
    "mtp.norm.weight": torch.randn(HIDDEN, generator=g),
    "mtp.pre_fc_norm_embedding.weight": torch.randn(HIDDEN, generator=g),
    "mtp.pre_fc_norm_hidden.weight": torch.randn(HIDDEN, generator=g),
    "mtp.layers.0.input_layernorm.weight": torch.randn(HIDDEN, generator=g),
    "mtp.layers.0.post_attention_layernorm.weight": torch.randn(HIDDEN, generator=g),
    "mtp.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, 128, generator=g),
  }


def write_tiny_checkpoint(path: str, seed: int = 0) -> str:
  """Materialise the fixture at path and return it. Idempotent on a warm dir."""
  from safetensors.torch import save_file
  from transformers import Qwen3_5ForConditionalGeneration

  if os.path.exists(os.path.join(path, "model.safetensors.index.json")):
    return path
  os.makedirs(path, exist_ok=True)

  torch.manual_seed(seed)
  config = tiny_config()
  model = Qwen3_5ForConditionalGeneration(config).to(torch.float32)
  # The config and the layout come from transformers, the shards do not:
  # save_pretrained groups tensors by size, and the point of this fixture is
  # to control which tensors share a shard.
  model.save_pretrained(path, safe_serialization=True, max_shard_size="100GB")
  for name in os.listdir(path):
    if name.endswith(".safetensors"):
      os.remove(os.path.join(path, name))

  tensors = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
  tensors.update(synthetic_mtp_tensors())

  shards: list[dict[str, torch.Tensor]] = [{} for _ in range(SHARDS)]
  weight_map: dict[str, str] = {}
  for i, key in enumerate(sorted(tensors)):
    filename = f"model-{i % SHARDS + 1:05d}-of-{SHARDS:05d}.safetensors"
    shards[i % SHARDS][key] = tensors[key]
    weight_map[key] = filename
  for i, shard in enumerate(shards):
    save_file(shard, os.path.join(path, f"model-{i + 1:05d}-of-{SHARDS:05d}.safetensors"), metadata={"format": "pt"})
  with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())}, "weight_map": weight_map}, f, indent=1)
  return path


def reference_model(path: str, device: str = "cuda", dtype: torch.dtype = torch.float64):
  """The HF model on its pure-torch gated-deltanet path, in fp64 by default.

  fp64 rather than fp32 because the two fp32 implementations disagree with each
  other by ~1e-4 relative through no fault of either: the delta rule's
  triangular solve amplifies rounding, and two different but equally valid
  fp32 orderings land ~1e-4 apart. Measured against an fp64 truth, each fp32
  side's error is its own, and the tolerance can be set from the reference
  implementation's own fp32 error rather than from a number that merely
  happened to pass.

  transformers picks fla's chunk_gated_delta_rule at layer construction when
  fla is importable, and fla is exactly what the Megatron side runs. An oracle
  that shares the kernel under test proves nothing about the kernel, so the
  fast path is switched off before the model is built and the layer falls back
  to torch_chunk_gated_delta_rule, which is plain tensor algebra.
  """
  import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
  from transformers import Qwen3_5ForConditionalGeneration

  for name in ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
    setattr(modeling, name, None)
  modeling.is_fast_path_available = False
  model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype).to(device)
  model.eval()
  return model


def expected_keys(path: str) -> set[str]:
  with open(os.path.join(path, "model.safetensors.index.json")) as f:
    return set(json.load(f)["weight_map"])


def megatron_model(path: str, dtype: torch.dtype = torch.float32, lora=None, recompute: bool = False):
  """The Megatron model the trainer would build from path, at TP=world_size.

  Mirrors MegatronTrainingWorker.load_base_model setting for setting -- the
  text-only bridge, sequence parallelism off, both Apex fusions off -- because
  a test that builds the model differently from the worker tests the test.
  recompute=True is the worker's ENABLE_GRADIENT_CHECKPOINTING branch verbatim;
  it re-runs every layer's forward inside backward, so a backward test wants it
  on and a forward test does not care.

  Single-process callers get a world of one. Multi-rank callers initialise the
  process group themselves before calling.
  """
  import torch.distributed as dist
  from megatron.bridge import AutoBridge
  from megatron.core import parallel_state
  from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

  from training.models import qwen35

  if not dist.is_initialized():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    dist.init_process_group(backend="cpu:gloo,cuda:nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
  tp = dist.get_world_size()
  if not parallel_state.model_parallel_is_initialized():
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(1234)

  qwen35.register_text_only_bridge()
  bridge = AutoBridge.from_hf_pretrained(path, torch_dtype=dtype)
  provider = bridge.to_megatron_provider(load_weights=True)
  provider.tensor_model_parallel_size = tp
  provider.pipeline_model_parallel_size = 1
  provider.context_parallel_size = 1
  provider.sequence_parallel = False
  provider.params_dtype = dtype
  provider.bf16 = dtype == torch.bfloat16
  provider.gradient_accumulation_fusion = False
  provider.masked_softmax_fusion = False
  if recompute:
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
  provider.finalize()
  chunks = provider.provide_distributed_model(wrap_with_ddp=False)
  if lora is not None:
    chunks = lora(chunks, training=True)
  return bridge, chunks


def megatron_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
  """Logits as [batch, seq, vocab], with Megatron's vocab padding cut off."""
  batch, seq = input_ids.shape
  position_ids = torch.arange(seq, device=input_ids.device).unsqueeze(0).expand(batch, -1)
  mask = torch.tril(torch.ones((1, 1, seq, seq), device=input_ids.device, dtype=torch.bool)).logical_not()
  logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=mask)
  if logits.shape[0] == seq and logits.shape[1] == batch:
    logits = logits.transpose(0, 1)
  # The output layer is vocab-parallel: each rank holds its slice of the vocab
  # dimension, and only the concatenation is comparable to anything.
  from megatron.core import parallel_state

  if parallel_state.get_tensor_model_parallel_world_size() > 1:
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    logits = gather_from_tensor_model_parallel_region(logits)
  return logits[..., :VOCAB]


def next_token_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
  """Mean negative log-likelihood of each token given the ones before it."""
  logprobs = torch.log_softmax(logits[:, :-1].double(), dim=-1)
  return -logprobs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).mean()


def identity_mapped_parameters() -> list[tuple[str, str]]:
  """(megatron name, hf name) pairs whose tensors correspond one-to-one.

  Their gradients compare directly, without undoing the qkv or in_proj fusion.
  Between them they sit on every path a gradient can take: the output head and
  final norm, every layer's down-projection, the gated-deltanet out_proj plus
  its A_log and dt_bias (which only the fla backward can populate), the
  attention output projection and q/k norms, and the embedding, which a
  gradient reaches only by passing back through all four layers.
  """
  pairs = [
    ("embedding.word_embeddings.weight", "model.language_model.embed_tokens.weight"),
    ("decoder.final_layernorm.weight", "model.language_model.norm.weight"),
    ("output_layer.weight", "lm_head.weight"),
  ]
  for i, kind in enumerate(LAYER_TYPES):
    m, h = f"decoder.layers.{i}.", f"model.language_model.layers.{i}."
    pairs.append((m + "mlp.linear_fc2.weight", h + "mlp.down_proj.weight"))
    if kind == "linear_attention":
      pairs += [
        (m + "self_attention.out_proj.weight", h + "linear_attn.out_proj.weight"),
        (m + "self_attention.A_log", h + "linear_attn.A_log"),
        (m + "self_attention.dt_bias", h + "linear_attn.dt_bias"),
      ]
    else:
      pairs += [
        (m + "self_attention.linear_proj.weight", h + "self_attn.o_proj.weight"),
        (m + "self_attention.q_layernorm.weight", h + "self_attn.q_norm.weight"),
        (m + "self_attention.k_layernorm.weight", h + "self_attn.k_norm.weight"),
      ]
  return pairs


def parameters_by_name(model) -> dict[str, torch.Tensor]:
  """named_parameters() without the wrapper prefix a bf16 model carries.

  provide_distributed_model wraps a bf16 model in Float16Module, so its
  parameters answer to module.<name> while an fp32 model's answer to <name>.
  """
  return {n.removeprefix("module."): p for n, p in model.named_parameters()}


# run42's targets: the four a dense transformer has, plus the gated-deltanet
# mixer's two, which is what run19 trained on the FSDP path.
LORA_TARGETS = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
LORA_DIM, LORA_ALPHA = 8, 16


def lora_config():
  from megatron.bridge.peft.lora import LoRA

  return LoRA(dim=LORA_DIM, alpha=LORA_ALPHA, dropout=0.0, target_modules=list(LORA_TARGETS))


def randomize_adapters(model, seed: int = 3) -> int:
  """Give every adapter tensor a nonzero value, so the adapter is not a no-op.

  LoRA zero-initialises B, so a fresh adapter computes exactly the base model
  and any export test passes vacuously. Returns how many tensors were set.
  """
  g = torch.Generator(device="cuda").manual_seed(seed)
  count = 0
  with torch.no_grad():
    for name, p in sorted(parameters_by_name(model).items()):
      if ".adapter." in name:
        p.copy_(torch.randn(p.shape, generator=g, device="cuda", dtype=p.dtype) * 0.05)
        count += 1
  return count


def saved_tensors(path: str) -> dict[str, torch.Tensor]:
  """Every tensor in every safetensors shard under path, on the CPU."""
  import glob

  from safetensors.torch import load_file

  out: dict[str, torch.Tensor] = {}
  for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
    out.update(load_file(shard))
  return out
