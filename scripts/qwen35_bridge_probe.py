"""Check that Qwen3.5's multimodal checkpoint dispatches to a text-only GPTModel.

No GPU and no distributed init: this only exercises architecture dispatch, the
weight-name mappings and the provider's shape fields, which is exactly the part
models/qwen35.py changes. Run it with the megatron interpreter:

  ~/megatron-probe/.venv/bin/python scripts/qwen35_bridge_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from megatron.bridge import AutoBridge

from training.models import qwen35

MODEL = "Qwen/Qwen3.5-9B"


def main() -> None:
  import megatron.core.ssm.gated_delta_net as gdn

  print(f"HAVE_FLA = {gdn.HAVE_FLA}")
  if not gdn.HAVE_FLA:
    raise SystemExit("gated-deltanet kernels missing: fla-core is not importable")

  qwen35.register_text_only_bridge()

  bridge = AutoBridge.from_hf_pretrained(MODEL, torch_dtype=torch.bfloat16)
  impl = bridge.get_model_bridge() if hasattr(bridge, "get_model_bridge") else None
  print(f"AutoBridge      -> {type(bridge).__name__}")
  print(f"model bridge    -> {type(impl).__name__ if impl is not None else '(not exposed)'}")

  if impl is not None:
    mappings = impl.mapping_registry()
    entries = getattr(mappings, "mappings", None) or getattr(mappings, "_mappings", [])
    names = []
    for m in entries:
      hf = getattr(m, "hf_param", None)
      names.extend(hf.values() if isinstance(hf, dict) else [hf] if hf else [])
    lang = [n for n in names if str(n).startswith("model.language_model.")]
    visual = [n for n in names if str(n).startswith("model.visual.")]
    mtp = [n for n in names if str(n).startswith("mtp.")]
    print(f"mappings        -> {len(names)} total, {len(lang)} language, {len(visual)} visual, {len(mtp)} mtp")
    print(f"  sample language: {sorted(lang)[:2]}")
    if visual:
      raise SystemExit(f"vision weights still mapped: {sorted(visual)[:3]}")

  provider = bridge.to_megatron_provider(load_weights=False)
  for field in ("num_layers", "hidden_size", "num_attention_heads", "num_query_groups", "mtp_num_layers", "seq_length"):
    print(f"  provider.{field:<22} = {getattr(provider, field, '(absent)')}")
  print(f"  provider class          = {type(provider).__name__}")


def export_adapter(bridge, chunks, lora, export_dir: str) -> None:
  """Write a PEFT adapter and check every tensor names a real base module.

  This is the failure mode worth spending a probe on. vLLM resolves adapter
  modules by name and applies the ones that match; a name matching nothing
  returns 200 OK and applies *nothing*, so the trainer trains, the sampler
  answers, and every rollout for the whole run comes from base weights. The
  question specific to Qwen3.5 is the prefix: the checkpoint stores its
  language tower under model.language_model.*, and an adapter exported with
  bare model.layers.* names would match nothing.
  """
  import glob
  import json

  from safetensors import safe_open

  bridge.save_hf_adapter(chunks, export_dir, peft_config=lora, base_model_name_or_path=MODEL, show_progress=False)

  config = json.load(open(os.path.join(export_dir, "adapter_config.json")))
  print(f"  adapter_config r={config.get('r')} alpha={config.get('lora_alpha')}")
  print(f"    target_modules      = {sorted(config.get('target_modules') or [])}")

  index = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*/model.safetensors.index.json"))
  base_keys = set(json.load(open(index[0]))["weight_map"])

  with safe_open(os.path.join(export_dir, "adapter_model.safetensors"), framework="pt") as handle:
    tensors = list(handle.keys())

  unmatched, checked = [], 0
  for name in tensors:
    target = name.removeprefix("base_model.model.")
    for suffix in (".lora_A.weight", ".lora_B.weight"):
      if target.endswith(suffix):
        target = target[: -len(suffix)] + ".weight"
        break
    else:
      continue
    checked += 1
    if target not in base_keys:
      unmatched.append(f"{name} -> {target}")

  print(f"  adapter tensors       = {len(tensors)} ({checked} name-checked)")
  print(f"    sample              = {sorted(tensors)[0]}")
  if unmatched:
    raise SystemExit(f"  {len(unmatched)} of {checked} name no base module: {unmatched[:4]}")
  print(f"  ALL {checked} adapter tensors resolve against the base checkpoint")


def load_and_score() -> None:
  """Stream the real weights in at TP=1 and score a sentence.

  Dispatch and provider shapes can both be right while the weight *names* are
  wrong, and a wrong name is silent: the tensor still loads, just into the
  wrong slot. Perplexity separates the two cases by orders of magnitude -- a
  correctly mapped 9B scores single digits on ordinary English, a scrambled one
  scores near the 250k vocabulary size.
  """
  import torch.distributed as dist
  from megatron.core import parallel_state
  from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
  from transformers import AutoTokenizer

  tp = int(os.getenv("PROBE_TP", "1"))
  dist.init_process_group(backend="cpu:gloo,cuda:nccl")
  torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
  parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=tp, pipeline_model_parallel_size=1, context_parallel_size=1
  )
  model_parallel_cuda_manual_seed(1234)

  qwen35.register_text_only_bridge()
  bridge = AutoBridge.from_hf_pretrained(MODEL, torch_dtype=torch.bfloat16)
  provider = bridge.to_megatron_provider(load_weights=True)
  provider.tensor_model_parallel_size = tp
  provider.pipeline_model_parallel_size = 1
  provider.context_parallel_size = 1
  # Never on, at any TP. megatron/core/ssm/gated_delta_net/gdn.py:91 is a bare
  # `assert not self.config.sequence_parallel`, so the 24 gated-deltanet layers
  # make sequence parallelism unavailable to this model regardless of TP -- the
  # worker reaches the same state via MEGATRON_TP > 1 and MEGATRON_SEQUENCE_PARALLEL.
  provider.sequence_parallel = False
  provider.params_dtype = torch.bfloat16
  provider.bf16 = True
  # Same default the worker runs under: APEX is not installed in this
  # interpreter, and both fusions hard-error rather than falling back.
  provider.gradient_accumulation_fusion = False
  provider.masked_softmax_fusion = False
  if os.getenv("PROBE_BACKWARD"):
    # The worker's ENABLE_GRADIENT_CHECKPOINTING branch verbatim. Full uniform
    # recompute is what makes the activation figure below mean anything: without
    # it the probe would measure a configuration the run never uses.
    provider.recompute_granularity = "full"
    provider.recompute_method = "uniform"
    provider.recompute_num_layers = 1
  provider.finalize()

  chunks = provider.provide_distributed_model(wrap_with_ddp=False)
  model = chunks[0]
  model.eval()
  params = sum(p.numel() for p in model.parameters())
  print(f"loaded {len(chunks)} chunk(s), {params / 1e9:.2f}B parameters")

  targets = os.getenv("PROBE_LORA_TARGETS", "")
  if targets:
    from megatron.bridge.peft.lora import LoRA

    names = [t.strip() for t in targets.split(",") if t.strip()]
    lora = LoRA(dim=32, alpha=64, dropout=0.0, target_modules=names)
    chunks = lora(chunks, training=True)
    model = chunks[0]
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    # megatron-bridge names adapter tensors <module path>.adapter.<sub>.weight,
    # so the adapted projection is the component before ".adapter".
    adapted = sorted({n.split(".adapter.")[0].rsplit(".", 1)[-1] for n in trainable_names if ".adapter." in n})
    if not adapted:
      print(f"  raw trainable sample  = {trainable_names[:3]}")
    layers = len({
      n.split("decoder.layers.")[1].split(".")[0]
      for n, p in model.named_parameters()
      if p.requires_grad and "decoder.layers." in n
    })
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA targets    = {names}")
    print(f"  adapted modules       = {adapted}")
    print(f"  layers touched        = {layers} of {provider.num_layers}")
    print(f"  trainable             = {trainable:,} ({100 * trainable / params:.3f}%)")

    export_dir = os.getenv("PROBE_EXPORT_DIR", "")
    if export_dir:
      export_adapter(bridge, chunks, lora, export_dir)

    lengths = os.getenv("PROBE_BACKWARD", "")
    if lengths:
      model.train()
      for length in [int(x) for x in lengths.split(",")]:
        backward_probe(model, provider, length)
      model.eval()

  tok = AutoTokenizer.from_pretrained(MODEL)
  text = "The capital of France is Paris, and the capital of Japan is Tokyo."
  ids = tok(text, return_tensors="pt").input_ids.cuda()
  n = ids.shape[1]
  pos = torch.arange(n, device=ids.device).unsqueeze(0)
  mask = torch.tril(torch.ones((1, 1, n, n), device=ids.device, dtype=torch.bool)).logical_not()

  with torch.no_grad():
    logits = model(input_ids=ids, position_ids=pos, attention_mask=mask)
  # Megatron returns [seq, batch, vocab] unless post_process reshapes it.
  if logits.shape[0] == n:
    logits = logits.transpose(0, 1)
  lp = torch.nn.functional.log_softmax(logits[0, :-1].float(), dim=-1)
  nll = -lp.gather(-1, ids[0, 1:].unsqueeze(-1)).mean()
  print(f"finite logits   = {bool(torch.isfinite(logits).all())}")
  print(f"perplexity      = {nll.exp().item():.3f}   (vocab {logits.shape[-1]})")


def backward_probe(model, provider, seq_len: int) -> None:
  """Run one forward+backward and report the memory it actually cost.

  Two things are being tested and only one of them is memory. Twenty-four of
  Qwen3.5's thirty-two layers are gated-deltanet, whose mixer runs through
  fla's chunk_gated_delta_rule -- a custom autograd Function, so its backward
  is hand-written rather than derived. No run in this repo has ever taken a
  gradient through it: every Megatron run so far was Gemma-4, which has no GDN
  layer at all. LoRA on in_proj sits upstream of that kernel, so if the
  hand-written backward is absent or wrong, run42 dies at its first optim step
  after a full rollout has already been paid for.

  peak_allocated, not allocated or peak_reserved: the other two are flat
  against sequence length and both would report success here regardless.
  """
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  base = torch.cuda.memory_allocated() / 2**30

  ids = torch.randint(0, 1000, (1, seq_len), device="cuda")
  pos = torch.arange(seq_len, device="cuda").unsqueeze(0)
  mask = torch.tril(torch.ones((1, 1, seq_len, seq_len), device="cuda", dtype=torch.bool)).logical_not()

  out = model(input_ids=ids, position_ids=pos, attention_mask=mask)
  # sum(), not cross-entropy, and deliberately no .float(): a real loss would
  # materialize [seq, vocab] in fp32 (163,840 x 248,320 x 4B = 163 TB), which is
  # why the worker chunks the vocab projection at MEGATRON_LOGPROB_CHUNK rows.
  # Upcasting here would put a tensor bigger than the model itself into the
  # measurement and report the probe's own arithmetic as the run's cost. The
  # gradient path through the decoder -- the part being tested -- is unaffected.
  logits_gib = out.numel() * out.element_size() / 2**30
  loss = out.sum()
  loss.backward()

  peak = torch.cuda.max_memory_allocated() / 2**30
  grads = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
  with_grad = [(n, p) for n, p in grads if p.grad is not None]
  nonzero = {n for n, p in with_grad if p.grad.abs().sum().item() > 0}
  finite = all(torch.isfinite(p.grad).all().item() for _, p in with_grad)
  # LoRA zero-inits B, so dL/dA = B^T(...) is exactly zero on the first backward
  # while dL/dB is not. Half the adapter tensors having no gradient here is the
  # arithmetic working, not a broken graph -- so check the B half, which is the
  # half that would actually be silent if the kernel's backward were missing.
  b_side = [n for n, _ in grads if n.endswith("linear_out.weight") or ".adapter.linear_out" in n]
  b_dead = [n for n in b_side if n not in nonzero]
  gdn = [n for n in nonzero if "in_proj" in n or "out_proj" in n]

  print(f"seq_len         = {seq_len:,}")
  print(f"  base weights          = {base:.2f} GiB")
  print(f"  peak_allocated        = {peak:.2f} GiB   (over base {peak - base:.2f}, of which logits {logits_gib:.2f})")
  print(f"  trainable tensors     = {len(grads)}, with .grad = {len(with_grad)}, nonzero = {len(nonzero)}")
  print(f"  GDN tensors w/ grad   = {len(gdn)}   (expect 48: 24 layers x in_proj,out_proj)")
  print(f"  lora_B with grad      = {len(b_side) - len(b_dead)} of {len(b_side)}")
  print(f"  all grads finite      = {finite}")
  if b_side and b_dead:
    raise SystemExit(f"  {len(b_dead)} lora_B tensors got no gradient: {b_dead[:4]}")
  if not finite:
    raise SystemExit("  non-finite gradients")
  model.zero_grad(set_to_none=True)


if __name__ == "__main__":
  if "--load" in sys.argv:
    load_and_score()
  else:
    main()
