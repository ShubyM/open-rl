"""GPU smoke for data-parallel LoRA: 2 ranks sharing one GPU vs a single-rank reference.

Run after any change to the distributed LoRA path (~15 min, ~55GB VRAM):
  uv run --extra gpu python scripts/lora_dp_smoke.py


Phase A: single worker, 4 datums, 10 steps -> loss curve + final adapter hash.
Phase B: 2 gloo ranks (same seed, same datums, sharded 2/2), 10 steps ->
  - ranks must be BITWISE identical to each other after every optim step
    (identical synced gradients + identical optimizer state)
  - loss curve must match phase A closely (kernel nondeterminism only)
  - the planted string must be memorized.
"""

import hashlib
import json
import multiprocessing as mp
import os
import sys

BASE = "Qwen/Qwen3.5-9B"
STEPS = 10
LR = 5e-4
SEED = 1234
PLANT = " azure-falcon-9931 stored in vault seven."


def build_datums(tok):
    from training.trainer_worker import Datum, TensorData

    datums = []
    prompts = [
        "The launch code hidden in the vault is",
        "According to the ledger, the launch code hidden in the vault is",
        "She whispered that the launch code hidden in the vault is",
        "Every archivist knows the launch code hidden in the vault is",
    ]
    for prompt in prompts:
        p = tok.encode(prompt)
        t = tok.encode(PLANT)
        full = p + t
        datums.append(
            Datum(
                model_input=full,
                loss_fn_inputs={
                    "target_tokens": TensorData(data=full[1:]),
                    "weights": TensorData(data=[0.0] * (len(p) - 1) + [1.0] * len(t)),
                },
            )
        )
    return datums


def adapter_hash(worker):
    import torch  # noqa: F401

    h = hashlib.sha256()
    for name, p in sorted(worker.peft_model.named_parameters()):
        if "lora" in name:
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def run_steps(worker, datums):
    losses = []
    for _ in range(STEPS):
        out = worker.forward_backward(datums, "cross_entropy", model_id="dpsmoke")
        losses.append(out["metrics"]["loss:sum"])
        worker.optim_step({"learning_rate": LR}, "dpsmoke")
    return losses


def phase_a(queue):
    os.environ["WORLD_SIZE"] = "1"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    os.environ.setdefault("FLA_TILELANG", "0")
    from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker

    w = LoraTrainingWorker()
    w.create_model(BASE, "dpsmoke", LoraConfig(rank=32, seed=SEED))
    losses = run_steps(w, build_datums(w.tokenizer))
    queue.put({"losses": losses, "hash": adapter_hash(w)})


def phase_b_rank(rank, port, queue):
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": "0",  # both ranks share the single GPU
            "WORLD_SIZE": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "OPEN_RL_CONTROL_BACKEND": "gloo",
        }
    )
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    os.environ.setdefault("FLA_TILELANG", "0")
    from training import distributed
    from training.lora_trainer_worker import LoraConfig, LoraTrainingWorker

    distributed.initialize()
    try:
        w = LoraTrainingWorker()
        w.create_model(BASE, "dpsmoke", LoraConfig(rank=32, seed=SEED))
        datums = build_datums(w.tokenizer)
        losses = []
        hashes = []
        for _ in range(STEPS):
            out = w.forward_backward(datums, "cross_entropy", model_id="dpsmoke")
            losses.append(out["metrics"]["loss:sum"])
            w.optim_step({"learning_rate": LR}, "dpsmoke")
            hashes.append(adapter_hash(w))
        gen = w.generate(w.tokenizer.encode("The launch code hidden in the vault is"), max_tokens=14, temperature=0.0, model_id="dpsmoke")
        text = w.tokenizer.decode(gen["sequences"][0]["tokens"])
        queue.put({"rank": rank, "losses": losses, "hashes": hashes, "text": text})
    finally:
        distributed.close()


def main():
    ctx = mp.get_context("spawn")

    qa = ctx.Queue()
    pa = ctx.Process(target=phase_a, args=(qa,))
    pa.start()
    ref = qa.get(timeout=3600)
    pa.join()
    print(f"[A] single-rank losses: first={ref['losses'][0]:.3f} last={ref['losses'][-1]:.4f}", flush=True)

    qb = ctx.Queue()
    procs = [ctx.Process(target=phase_b_rank, args=(r, 29542, qb)) for r in range(2)]
    for p in procs:
        p.start()
    results = sorted((qb.get(timeout=3600) for _ in procs), key=lambda r: r["rank"])
    for p in procs:
        p.join()

    r0, r1 = results
    failures = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
        if not ok:
            failures.append(name)

    check("ranks bitwise-identical after every step", r0["hashes"] == r1["hashes"])
    check("rank losses identical (post-all-reduce)", all(abs(a - b) < 1e-6 for a, b in zip(r0["losses"], r1["losses"])))
    # Relative tolerance for real losses, absolute for the near-zero tail:
    # bf16 accumulation order differs across the reduction, and Adam (eps
    # 1e-12) amplifies that noise once the loss is microscopic.
    diffs = [(abs(a - b), abs(a - b) / max(abs(a), 1e-9)) for a, b in zip(ref["losses"], r0["losses"])]
    ok = all(absd < 0.1 or reld < 0.05 for absd, reld in diffs)
    check("DP loss curve matches single-rank", ok, f"max abs {max(d[0] for d in diffs):.4f}, step0 rel {diffs[0][1]:.6f}")
    check("DP training collapses loss", r0["losses"][-1] < ref["losses"][0] * 0.05, f"{r0['losses'][0]:.3f} -> {r0['losses'][-1]:.5f}")
    check("planted string memorized under DP", PLANT.strip()[:20] in r0["text"], repr(r0["text"][:70]))

    print(json.dumps({"single": ref["losses"], "dp": r0["losses"]}), flush=True)
    print("\n" + ("DP SMOKE: ALL CHECKS PASSED" if not failures else f"DP SMOKE FAILURES: {failures}"), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
