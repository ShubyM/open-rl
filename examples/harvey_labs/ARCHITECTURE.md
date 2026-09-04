# Harvey LAB RL — architecture

The presentation version is [`architecture.html`](architecture.html) — open it in
a browser. It covers the two-machine topology, what each component owns, one RL
step, and checkpointing.

This file is the reference detail behind it.

## Machines

| | `ssh h200` | `ssh b200` |
| --- | --- | --- |
| Instance | `h200-vm-spot` | `b200-spot-8gpu` |
| Shape | `a3-ultragpu-8g-nolssd`, 8×H200 141GB | `a4-highgpu-8g`, 8×B200 |
| Zone | `us-east7-c` | `us-east7-b` |
| Internal IP | `10.196.0.3` | `10.196.0.4` |
| Serves | vLLM Qwen3.5-9B `:8000` (GPUs 4-7, DP4)<br>gateway `:9003` · LoRA trainer (GPUs 0-3) | vLLM GLM-5.2-FP8 `:8000` TP8<br>served as `gpt-glm-5.2` |

Both are spot, so external IPs are ephemeral — refresh with
`gcloud compute instances list --project open-rl` and update `HostName` in
`~/.ssh/config`.

The whole stack comes up with `MODEL=9b ./scripts/launch_work.sh` on the h200
(tmux session `work`: sampler, gateway, trainer, typed train/eval commands).

## Boundaries worth knowing

`train.py` only ever talks to the gateway on `http://127.0.0.1:9003`. It does not
know whether the trainer is in-process or a separate `torchrun` job, or whether
sampling is one vLLM server or seven — that is a config change, not a code
change.

Grading runs *from* the h200: `reward.py` shells into the LAB checkout's venv
(`score_lab_run.py`), which holds an OpenAI-compatible client pointed at
`http://10.196.0.4:8000/v1`. The b200 only ever serves completions.
`judge_parallel` defaults to 16 for `glm*` models, so a batch's episodes grade
concurrently rather than serializing.

`train.py` preflights the grading environment at startup and refuses to run if
the LAB venv or judge is missing — a silently broken grader zeroes every reward,
which costs a whole run before anyone notices.

## Checkpoints

Two artifacts per save, both recorded in the run's `checkpoints.jsonl`:

| | ref | on disk |
| --- | --- | --- |
| Adapter snapshot | `tinker://<model-id>/sampler_weights/<label>` | `/tmp/open-rl/peft/<model-id>/<label>/` |
| Full checkpoint | `tinker://<model-id>/weights/<label>` | `/tmp/open-rl/checkpoints/<model-id>/weights/<label>/` |

The snapshot is PEFT LoRA weights only (~346 MB at rank 32), written every
optimizer step under a fresh immutable directory (`sampler-<seq>`) so rollouts
still in flight keep reading a complete one. The full checkpoint adds
`optimizer.pt` (~693 MB of Adam moments) every `save_every=5` steps and at
`final`.

The hot-load into the sampler is vLLM's `/v1/load_lora_adapter` (needs
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=true`), not a restart; `lm_head` is excluded
from the adapter so it stays vLLM-loadable. Saving costs ~1.5–3 s per step
(`time/save_checkpoint`).

Usage:

- Warm-start RL from SFT — `LOAD_CHECKPOINT=tinker://<id>/sampler_weights/final`
  (weights only, fresh optimizer).
- Resume after a preemption — the `weights/` state ref, momentum intact.
- Offline eval — `eval_checkpoint.py checkpoint=/tmp/open-rl/peft/<id>/000015`.

`/tmp/open-rl` is local disk on a spot VM. Anything worth keeping gets copied
off to HF Hub or GCS.

## Task split

Seeded draw over the 1,749 runnable LAB tasks — `train_tasks=300 eval_tasks=50
task_split_seed=0`. The split *is* the benchmark, so the seed stays fixed across
runs being compared. Eval is the slice
`shuffled[EVAL_SLICE_OFFSET : EVAL_SLICE_OFFSET + eval_tasks]` with the offset
pinned in `tasks.py`, so it depends only on the seed and the eval count and
never moves when the train count changes. Train is then drawn from every task
outside eval's scenario families, so a different scenario of the same eval
matter (the same MSA first-draft, say) cannot leak benchmark structure into
training.

- `task_set=random` (default) — two-way train/eval split, family-disjoint.
- `task_set=disjoint` — three-way sft/train/eval split, disjoint by task
  *family*, eval stratified across practice areas.
