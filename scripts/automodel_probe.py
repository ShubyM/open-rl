"""Correctness and memory probe for the Automodel trainer worker.

Runs the real worker (create_model, compute_target_logprobs, backward) on
synthetic tokens that are identical on every rank, so any TP/CP layout can be
compared position by position and gradient by gradient against a single-GPU
reference. Launch under torchrun with the layout in the usual env vars:

  PROBE_MODE=ref     one GPU, TP=CP=1: worker logprobs vs a plain transformers
                     forward on the same tokens, then writes the reference file
                     (logprobs + LoRA grads) other layouts compare against.
  PROBE_MODE=layout  N GPUs, OPEN_RL_AUTOMODEL_TP / _CP set: same data, compares
                     per-position logprobs and per-parameter grads to PROBE_REF.
  PROBE_MODE=ckpt    save_state -> load_from_state round trip reproduces the
                     logprobs; write_adapter emits a PEFT dir; lists its keys.
  PROBE_MODE=ladder  one forward+backward per length in PROBE_LENGTHS on a single
                     sequence, reporting peak memory until the first OOM.

  PROBE_MODEL (default Qwen/Qwen3.5-9B), PROBE_LEN (default 4096),
  PROBE_REF (default ~/automodel/ref.pt), PROBE_LENGTHS (ladder).
"""

import gc
import os
import time

import torch
import torch.distributed as dist

MODEL = os.getenv("PROBE_MODEL", "Qwen/Qwen3.5-9B")
MODE = os.getenv("PROBE_MODE", "ref")
LEN = int(os.getenv("PROBE_LEN", "4096"))
REF = os.path.expanduser(os.getenv("PROBE_REF", "~/automodel/ref.pt"))
LENGTHS = [int(x) for x in os.getenv("PROBE_LENGTHS", "160000,180000,200000,220000,240000,262144").split(",")]
GIB = 2**30


def log(msg: str) -> None:
  if dist.get_rank() == 0:
    print(f"PROBE {msg}", flush=True)


def synthetic(length: int, vocab: int = 100000):
  # CPU generator with a fixed seed: identical tokens on every rank.
  g = torch.Generator().manual_seed(1234)
  input_ids = torch.randint(0, vocab, (1, length), generator=g).cuda()
  targets = torch.randint(0, vocab, (1, length), generator=g).cuda()
  return input_ids, targets


def build_worker():
  from training.automodel_worker import AutomodelTrainingWorker

  worker = AutomodelTrainingWorker()
  worker.create_model(MODEL)
  return worker


def deterministic_lora_init(worker) -> None:
  """Give every adapter tensor the same values on every layout.

  Automodel draws the LoRA A init from the device RNG after sharding, which
  differs between processes, and B starts at zero, which zeroes the A
  gradient. Both make cross-layout gradient comparison meaningless, so each
  tensor is filled from a CPU generator seeded by its name and scattered into
  whatever DTensor placement the layout gave it.
  """
  import zlib

  from torch.distributed.tensor import DTensor, distribute_tensor

  with torch.no_grad():
    for name, param in sorted(worker.model.named_parameters()):
      if not param.requires_grad:
        continue
      g = torch.Generator().manual_seed(zlib.crc32(canonical(name).encode()))
      full = (torch.randn(param.shape, generator=g) * 0.02).to(param.dtype).cuda()
      if isinstance(param, DTensor):
        param.copy_(distribute_tensor(full, param.device_mesh, param.placements))
      else:
        param.copy_(full)


def forward_backward(worker, input_ids, targets):
  logprobs = worker.compute_target_logprobs(worker.model, input_ids, torch.ones_like(input_ids), targets)
  loss = -logprobs.float().sum()
  loss.backward()
  return logprobs.detach().float().cpu()


def lora_grads(worker) -> dict[str, torch.Tensor]:
  grads = {}
  for name, param in worker.model.named_parameters():
    if not param.requires_grad or param.grad is None:
      continue
    grad = param.grad.detach()
    if hasattr(grad, "full_tensor"):
      grad = grad.full_tensor()
    grads[canonical(name)] = grad.float().cpu()
  return grads


def canonical(name: str) -> str:
  # Automodel's own checkpoint wrappers insert this segment; the worker's
  # group checkpointing does not, and both must compare against one reference.
  return name.replace("._checkpoint_wrapped_module", "")


def compare_logprobs(mine: torch.Tensor, ref: torch.Tensor, chunks: int) -> None:
  diff = (mine - ref).abs()[0]
  log(f"logprob max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} over {diff.numel()} positions")
  per = diff.chunk(chunks)
  log("per-chunk max|diff|: " + " ".join(f"{c.max():.2e}" for c in per))


def compare_grads(mine: dict[str, torch.Tensor], ref: dict[str, torch.Tensor]) -> None:
  rel, cos = [], []
  ref = {canonical(name): grad for name, grad in ref.items()}
  missing = sorted(set(ref) ^ set(mine))
  if missing:
    log(f"grad key mismatch ({len(missing)}): {missing[:6]}")
  for name in sorted(set(ref) & set(mine)):
    a, b = mine[name].flatten(), ref[name].flatten()
    if a.shape != b.shape:
      log(f"grad shape mismatch {name}: {tuple(a.shape)} vs {tuple(b.shape)}")
      continue
    rel.append(((a - b).norm() / (b.norm() + 1e-12)).item())
    cos.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
  rel_t, cos_t = torch.tensor(rel), torch.tensor(cos)
  log(f"grad reldiff median={rel_t.median():.3e} max={rel_t.max():.3e}; cosine min={cos_t.min():.5f} median={cos_t.median():.5f}")
  log(f"  over {len(rel)} params")
  worst = sorted(zip(rel, sorted(set(ref) & set(mine))), reverse=True)[:5]
  for r, name in worst:
    log(f"  worst reldiff {r:.3e} {name}")


def mode_ref(worker):
  input_ids, targets = synthetic(LEN)
  # Fresh adapters have B=0, so the worker must reproduce the base model here.
  with torch.no_grad():
    base_logprobs = worker.compute_target_logprobs(worker.model, input_ids, torch.ones_like(input_ids), targets).float().cpu()
  log(f"worker base logprobs mean={base_logprobs.mean():.4f} first={base_logprobs[0, :4].tolist()}")

  from transformers import AutoModelForCausalLM

  reference = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
  with torch.no_grad():
    logits = reference(input_ids=input_ids, use_cache=False).logits.float()
    ref_logprobs = (logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, -1)).cpu()
  del reference, logits
  torch.cuda.empty_cache()
  log("worker (B=0) vs plain transformers forward:")
  compare_logprobs(base_logprobs, ref_logprobs, chunks=8)

  deterministic_lora_init(worker)
  logprobs = forward_backward(worker, input_ids, targets)
  grads = lora_grads(worker)
  log(f"worker adapted logprobs mean={logprobs.mean():.4f}; |adapted - base| max={(logprobs - base_logprobs).abs().max():.3e}")
  torch.save({"logprobs": logprobs, "grads": grads, "len": LEN, "model": MODEL}, REF)
  log(f"wrote reference {REF} ({len(grads)} grads)")


def mode_layout(worker):
  ref = torch.load(REF)
  assert ref["len"] == LEN and ref["model"] == MODEL, ref
  deterministic_lora_init(worker)
  input_ids, targets = synthetic(LEN)
  logprobs = forward_backward(worker, input_ids, targets)
  grads = lora_grads(worker)
  log(f"TP={worker.tp_size} CP={worker.cp_size} vs reference {REF}")
  compare_logprobs(logprobs, ref["logprobs"], chunks=max(8, 2 * worker.cp_size))
  compare_grads(grads, ref["grads"])


def mode_ckpt(worker):
  deterministic_lora_init(worker)
  input_ids, targets = synthetic(LEN)
  # Train one step so the adapter is non-trivial and the optimizer has state.
  forward_backward(worker, input_ids, targets)
  worker.optim_step({"learning_rate": 1e-4, "grad_clip_norm": 1.0})
  with torch.no_grad():
    before = worker.compute_target_logprobs(worker.model, input_ids, torch.ones_like(input_ids), targets).float().cpu()

  state_path = os.path.expanduser("~/automodel/probe-state")
  worker.save_state("probe", state_path, include_optimizer=True)
  if dist.get_rank() == 0:
    log(f"state dir: {sorted(os.listdir(state_path))}")
  adapter_dir = worker.write_adapter("probe-model", alias="probe", session_label="probe-session")
  if dist.get_rank() == 0:
    from safetensors import safe_open

    log(f"adapter dir {adapter_dir}: {sorted(os.listdir(adapter_dir))}")
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
      log(f"adapter_config.json: {f.read()[:600]}")
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
      keys = list(f.keys())
      log(f"{len(keys)} adapter tensors; first: {keys[:3]}; shape {f.get_tensor(keys[0]).shape}")

  # Perturb, then reload: the reload must undo the perturbation exactly.
  with torch.no_grad():
    for param in worker.trainable_params:
      param.add_(0.01)
  worker.load_from_state("probe", state_path, restore_optimizer=True)
  with torch.no_grad():
    after = worker.compute_target_logprobs(worker.model, input_ids, torch.ones_like(input_ids), targets).float().cpu()
  log(f"round trip max|diff|={(after - before).abs().max():.3e} (0 expected)")
  log(f"optimizer state entries after restore: {len(worker.optimizer.state)}")


def mode_ladder(worker):
  resident = torch.cuda.memory_allocated() / GIB
  log(f"resident (weights+opt) = {resident:.1f} GiB/rank at TP={worker.tp_size} CP={worker.cp_size}")
  results = []
  for length in LENGTHS:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for param in worker.trainable_params:
      param.grad = None
    ok = True
    try:
      input_ids, targets = synthetic(length)
      start = time.time()
      forward_backward(worker, input_ids, targets)
      torch.cuda.synchronize()
      del input_ids, targets
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
      if not (isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()):
        raise
      ok = False
      print(f"PROBE rank {dist.get_rank()} L={length} OOM ({str(exc)[:160]})", flush=True)
    # Every rank must agree before the next length, or the survivors block in
    # a collective the OOM'd rank never joins.
    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    peak = torch.cuda.max_memory_allocated() / GIB
    peak_r = torch.cuda.max_memory_reserved() / GIB
    if flag.item() < 1:
      log(f"L={length} OOM on some rank (rank 0 peak_alloc={peak:.1f} peak_reserved={peak_r:.1f})")
      break
    log(f"L={length} OK peak_alloc={peak:.1f} peak_reserved={peak_r:.1f} GiB in {time.time() - start:.0f}s")
    results.append((length, round(peak, 1), round(peak_r, 1)))
  log(f"LADDER DONE {results}")


def main() -> None:
  dist.init_process_group(backend="nccl")
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
  layout = f"TP={os.getenv('OPEN_RL_AUTOMODEL_TP')} CP={os.getenv('OPEN_RL_AUTOMODEL_CP')} rank={os.getenv('OPEN_RL_AUTOMODEL_LORA_RANK')}"
  log(f"mode={MODE} model={MODEL} len={LEN} {layout}")
  worker = build_worker()
  {"ref": mode_ref, "layout": mode_layout, "ckpt": mode_ckpt, "ladder": mode_ladder}[MODE](worker)
  log("DONE")
  # No barrier: a mid-collective OOM can wedge NCCL; let each rank exit on its own.
  try:
    dist.destroy_process_group()
  except Exception:
    pass


if __name__ == "__main__":
  main()
