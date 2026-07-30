"""Validate the LoRA DP filler-skip fix: odd shards and empty shards.

Phase A (single rank) vs Phase B (2 gloo ranks, one GPU):
  step type 1: FIVE datums in one forward_backward (shards 3/2 — the short
    rank previously ran a filler pass; now it must run exactly 2 batches)
  step type 2: ONE datum (rank 1's shard is empty — previously a full
    redundant filler pass; now zero batches)
Asserts: cross-rank adapter hashes bitwise identical each step, loss curve
matches single-rank, and per-rank executed-batch counts prove no filler.
"""
import hashlib, multiprocessing as mp, os, sys

BASE = "Qwen/Qwen3.5-9B"
STEPS = 3
LR = 5e-4
SEED = 1234

def build_datums(tok, n):
    from training.trainer_worker import Datum, TensorData
    out = []
    for i in range(n):
        p = tok.encode(f"Ledger entry {i}: the launch code hidden in the vault is")
        t = tok.encode(" azure-falcon-9931 stored in vault seven.")
        full = p + t
        out.append(Datum(model_input=full, loss_fn_inputs={
            "target_tokens": TensorData(data=full[1:]),
            "weights": TensorData(data=[0.0]*(len(p)-1) + [1.0]*len(t))}))
    return out

def adapter_hash(w):
    h = hashlib.sha256()
    for name, p in sorted(w.peft_model.named_parameters()):
        if "lora" in name:
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()

def instrument(w):
    counts = {"batches": 0}
    orig = w.pad_model_inputs
    def wrapped(data):
        counts["batches"] += 1
        return orig(data)
    w.pad_model_inputs = wrapped
    return counts

def steps(w, counts):
    losses, hashes, bcounts = [], [], []
    for _ in range(STEPS):
        d5 = build_datums(w.tokenizer, 5)
        out = w.forward_backward(d5, "cross_entropy", model_id="fix")
        losses.append(out["metrics"]["loss:sum"])
        w.optim_step({"learning_rate": LR}, "fix")
        d1 = build_datums(w.tokenizer, 1)
        out = w.forward_backward(d1, "cross_entropy", model_id="fix")
        losses.append(out["metrics"]["loss:sum"])
        w.optim_step({"learning_rate": LR}, "fix")
        hashes.append(adapter_hash(w))
        bcounts.append(counts["batches"])
    return losses, hashes, bcounts

def phase_a(q):
    os.environ["WORLD_SIZE"] = "1"; os.environ.setdefault("FLA_TILELANG", "0")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
    w = LoraTrainingWorker(); w.create_model(BASE, "fix", LoraConfig(rank=32, seed=SEED))
    c = instrument(w)
    losses, hashes, bcounts = steps(w, c)
    q.put({"losses": losses, "hash": hashes[-1], "batches": bcounts[-1]})

def phase_b_rank(rank, port, q):
    os.environ.update({"RANK": str(rank), "LOCAL_RANK": "0", "WORLD_SIZE": "2",
                       "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                       "OPEN_RL_CONTROL_BACKEND": "gloo"})
    os.environ.setdefault("FLA_TILELANG", "0")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from training import distributed
    from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker
    distributed.initialize()
    try:
        w = LoraTrainingWorker(); w.create_model(BASE, "fix", LoraConfig(rank=32, seed=SEED))
        c = instrument(w)
        losses, hashes, bcounts = steps(w, c)
        q.put({"rank": rank, "losses": losses, "hashes": hashes, "batches": bcounts[-1]})
    finally:
        distributed.close()

def main():
    ctx = mp.get_context("spawn")
    qa = ctx.Queue(); pa = ctx.Process(target=phase_a, args=(qa,)); pa.start()
    ref = qa.get(timeout=3600); pa.join()
    print(f"[A] losses={['%.3f'%l for l in ref['losses']]} batches={ref['batches']}", flush=True)

    qb = ctx.Queue()
    ps = [ctx.Process(target=phase_b_rank, args=(r, 29517, qb)) for r in (0, 1)]
    [p.start() for p in ps]
    res = sorted([qb.get(timeout=3600) for _ in ps], key=lambda r: r["rank"])
    [p.join() for p in ps]

    r0, r1 = res
    assert r0["hashes"] == r1["hashes"], "ranks diverged!"
    print(f"[B] ranks bitwise identical across {STEPS} steps", flush=True)
    print(f"[B] executed batches: rank0={r0['batches']} rank1={r1['batches']} (A ran {ref['batches']})", flush=True)
    la, lb = ref["losses"], r0["losses"]
    for i, (a, b) in enumerate(zip(la, lb)):
        print(f"[B] step-loss {i}: single={a:.4f} dp={b:.4f} abs-diff={abs(a-b):.4f}", flush=True)
    # relative where the loss is meaningful, absolute floor once it collapses:
    drift = max(abs(a-b)/max(abs(a), 0.5) for a, b in zip(la, lb))
    print(f"[B] loss drift vs single-rank (floored): {drift:.2%}", flush=True)
    assert drift < 0.03, "loss diverged from single-rank reference"
    # fix-specific: rank1 must have run FEWER batches than rank0 (2 vs 3 per
    # 5-datum step, 0 vs 1 per 1-datum step) — filler would make them equal.
    assert r1["batches"] < r0["batches"], "rank1 batch count equals rank0 — filler passes still running!"
    print("DP-FIX-SMOKE-PASS", flush=True)

if __name__ == "__main__":
    main()
