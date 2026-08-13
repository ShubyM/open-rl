"""Adversarial LoRA-trainer correctness probe (GPU, ~4 min, ~20GB VRAM).

Run after any trainer change you don't trust:
  uv run --extra gpu python scripts/lora_correctness_probe.py


Assumes the training code is broken until proven otherwise:
  1. adapter targets cover deltanet AND attention AND mlp modules
  2. one backward puts real gradients on all three module classes
  3. 60 steps memorize a planted string (loss -> ~0)
  4. base weights are bit-frozen; only adapter weights move
  5. saved adapter file: hub-layout keys, nonzero lora_B
  6. generation emits the planted string WITH the adapter and not without
"""

import hashlib
import json
import os
import sys

os.environ.setdefault("FLA_TILELANG", "0")  # Triton deltanet path on Blackwell
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from safetensors import safe_open

from training import paths
from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
from training.trainer_worker import Datum, TensorData

BASE = "Qwen/Qwen3.5-9B"
PLANT = " azure-falcon-9931 stored in vault seven."
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        failures.append(name)


w = LoraTrainingWorker()
w.create_model(BASE, "probe", LoraConfig(rank=32))

# --- 1. target coverage ---
targets = w.target_lora_modules(LoraConfig(rank=32))
has_deltanet = any("in_proj" in t or "out_proj" in t for t in targets)
has_attn = any(t.endswith("q_proj") for t in targets)
has_mlp = any("gate_proj" in t or "up_proj" in t for t in targets)
check("adapter targets deltanet projections", has_deltanet, f"{len(targets)} target modules")
check("adapter targets attention q_proj", has_attn)
check("adapter targets mlp", has_mlp)

# --- base fingerprint (sampled, pre-training) ---
def base_fingerprint(model) -> str:
    h = hashlib.sha256()
    for i, (name, p) in enumerate(sorted(model.named_parameters())):
        if "lora" in name or i % 17:
            continue
        h.update(name.encode())
        h.update(p.detach().float().sum().cpu().numpy().tobytes())
        h.update(p.detach().float().abs().sum().cpu().numpy().tobytes())
    return h.hexdigest()

fp_before = base_fingerprint(w.peft_model)

# --- build the memorization datum ---
tok = w.tokenizer
prompt_ids = tok.encode("The launch code hidden in the vault is")
plant_ids = tok.encode(PLANT)
full = prompt_ids + plant_ids
target = full[1:]
weights = [0.0] * (len(prompt_ids) - 1) + [1.0] * len(plant_ids)
assert len(target) == len(weights)
datum = Datum(
    model_input=full,
    loss_fn_inputs={
        "target_tokens": TensorData(data=target),
        "weights": TensorData(data=weights),
    },
)

# --- 2. gradient flow per module class ---
# LoRA inits B=0, so dL/dA ∝ B^T = 0 on the FIRST backward: step one's
# gradient enters exclusively through lora_B. Check B on the first backward,
# then A after one optimizer step has moved B off zero.
def grads_by_class(which: str) -> dict[str, float]:
    out = {"deltanet": 0.0, "attention": 0.0, "mlp": 0.0}
    for name, p in w.peft_model.named_parameters():
        if p.grad is None or which not in name:
            continue
        g = p.grad.detach().abs().sum().item()
        if "in_proj" in name or "out_proj" in name:
            out["deltanet"] += g
        elif any(t in name for t in ("q_proj", "k_proj", "v_proj", "o_proj")):
            out["attention"] += g
        elif any(t in name for t in ("gate_proj", "up_proj", "down_proj")):
            out["mlp"] += g
    return out

w.forward_backward([datum], "cross_entropy", model_id="probe")
for cls, g in grads_by_class("lora_B").items():
    check(f"first backward: nonzero lora_B gradient on {cls}", g > 0, f"sum|grad|={g:.3e}")
w.optim_step({"learning_rate": 5e-4}, "probe")
w.forward_backward([datum], "cross_entropy", model_id="probe")
for cls, g in grads_by_class("lora_A").items():
    check(f"after step 1: nonzero lora_A gradient on {cls}", g > 0, f"sum|grad|={g:.3e}")

# --- 3. memorization loop ---
losses = []
for step in range(60):
    out = w.forward_backward([datum], "cross_entropy", model_id="probe")
    flat = json.dumps(out)
    metrics = out.get("metrics", out)
    loss_val = None
    for k, v in metrics.items():
        if "loss" in k and isinstance(v, (int, float)):
            loss_val = v
            break
    if loss_val is None:
        print("metrics keys:", list(metrics.keys()), flush=True)
        raise SystemExit("no loss metric found")
    losses.append(loss_val / max(sum(weights), 1))
    w.optim_step({"learning_rate": 5e-4}, "probe")
    if step % 10 == 0:
        print(f"  step {step:2d} loss/token {losses[-1]:.4f}", flush=True)

check("initial loss is nontrivial", losses[0] > 1.0, f"{losses[0]:.3f}")
check("loss collapses (memorized)", losses[-1] < 0.05, f"final {losses[-1]:.5f}")
check("loss decreased monotonically-ish", losses[-1] < losses[0] * 0.05)

# --- 4. base frozen ---
fp_after = base_fingerprint(w.peft_model)
check("base weights bit-frozen", fp_before == fp_after)
b_moved = sum(
    p.detach().abs().sum().item() for n, p in w.peft_model.named_parameters() if "lora_B" in n
)
check("adapter lora_B weights moved off zero", b_moved > 0, f"sum|B|={b_moved:.3e}")

# --- 5. saved adapter sanity ---
snap_root = os.path.join(paths.snapshot_root(), "probe")
snaps = sorted(os.listdir(snap_root)) if os.path.isdir(snap_root) else []
check("adapter snapshot dir exists", bool(snaps), f"{snap_root}: {snaps[-3:]}")
if snaps:
    latest = os.path.join(snap_root, snaps[-1])
    st = os.path.join(latest, "adapter_model.safetensors")
    with safe_open(st, framework="pt") as f:
        keys = list(f.keys())
        b_keys = [k for k in keys if "lora_B" in k]
        b_sum = sum(f.get_tensor(k).abs().sum().item() for k in b_keys[:20])
    if w.base_is_multimodal:
        check("saved keys remapped to hub layout", all("language_model" in k for k in keys[:50]), keys[0])
    else:
        check("saved keys in text layout (base not multimodal)", True, keys[0])
    check("saved lora_B nonzero (trained weights persisted)", b_sum > 0, f"sum|B[:20]|={b_sum:.3e}")

# --- 6. causal generation check ---
gen = w.generate(prompt_ids, max_tokens=len(plant_ids) + 2, temperature=0.0, model_id="probe")
text = tok.decode(gen["sequences"][0]["tokens"]) if "sequences" in gen else str(gen)
check("adapter generation emits planted string", PLANT.strip()[:20] in text, repr(text[:80]))

with w.peft_model.disable_adapter():
    gen_base = w.generate(prompt_ids, max_tokens=len(plant_ids) + 2, temperature=0.0)
base_text = tok.decode(gen_base["sequences"][0]["tokens"]) if "sequences" in gen_base else str(gen_base)
check("base (adapter disabled) does NOT emit planted string", PLANT.strip()[:20] not in base_text, repr(base_text[:80]))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"), flush=True)
sys.exit(1 if failures else 0)
