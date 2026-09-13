# Automodel trainer

Automodel runs the existing training request protocol under a separate GPU
interpreter. It supports LoRA and full-parameter training with FSDP2. The
worker requires torchrun, including on one GPU. Pipeline parallelism is
unsupported; full-parameter training requires TP=1. For LoRA, the world size
must be divisible by TP × CP; the remaining ranks form DP. Only DP divides
datums, and CP handles one sequence per forward pass.

The worker shares training, loss, optimizer, and checkpoint orchestration
across models. Its model contract requires the Transformers text-config and
output-embedding APIs, a text backbone with a final `norm`, and a linear
vocabulary projection (with optional bias and logit softcapping). Text-only
and nested multimodal backbones resolve through one helper. Model-specific
attention and checkpointing policies stay at model construction; CP layout
and attention transport belong to NeMo. Supporting another architecture means
validating this contract and its parallel execution, including any custom
output-head transformation; loading through AutoModel alone is insufficient.

Model-family support lives in
[`src/training/automodel_models.py`](../../src/training/automodel_models.py).
`model_policy()` resolves attention settings, TP layout, and activation
checkpointing together before loading. Its `apply()` method installs settings
that require the loaded, sharded model. `forward_hidden_states()` owns the
temporary final-norm hook and removes it on success or failure; captured
activations never live on the worker. Output projection metadata, LoRA target
layouts, and adapter key conversion belong to this module too. Configuration
comes from explicit caller inputs.

[`src/training/automodel_worker.py`](../../src/training/automodel_worker.py)
owns runtime configuration, distributed execution, training, optimizer state,
and checkpoint publication. It materializes distributed hidden states and
output-head tensors for chunked logprobs, and keeps the CP attention context
active through backward and checkpoint recomputation. Add model-family cases
and their tests in `tests/test_automodel_models.py`; worker tests cover the
training contract and applying the selected policy before FSDP construction.

## Environment and launch

From the repository root on the GPU host:

```bash
./scripts/setup_automodel_env.sh
export AUTOMODEL_PYTHON="$HOME/automodel/.venv/bin/python"
```

The setup script pins the Qwen runtime to NeMo Automodel 0.6.0, torch 2.11.0
with CUDA 12.9, and transformers 5.12.1, plus the GDN kernel dependencies.
It does not install Automodel into the project or sampler environment.
Gemma 4 31B needs a separate newer environment: the historical
[run51 report](../reports/run51/README.md) records its exact Automodel revision,
transformers and FFPA versions. The pinned setup script does not reproduce
that environment.

With the Harvey-LAB prerequisites from [local setup](local-setup.md), launch
a four-GPU Qwen trainer and use the remaining GPUs for vLLM samplers:

```bash
TRAINER_BACKEND=automodel AFFINITY=1 MODEL=9b TRAIN_GPUS=4 \
  AUTOMODEL_TP=1 AUTOMODEL_CP=4 AUTOMODEL_LORA_RANK=32 \
  ./scripts/launch_work.sh
```

At least one additional GPU is required for sampling. For DP training, set
`AUTOMODEL_CP=1`; for TP2 × CP2, set both to `2`. The launcher accepts LoRA
ranks 1–64 and types the client training command into tmux without starting
it. It sets `OPEN_RL_TRAINER_BACKEND=automodel`, the matching
`OPEN_RL_AUTOMODEL_*` settings, mixed CPU/GPU collective backends, and disables
time slicing. Direct launches must give the gateway and trainer the same
backend and LoRA settings. The worker's default LoRA rank is zero; the
launcher's default is 16.

For local workers started by the gateway, set `AUTOMODEL_PYTHON` and
`OPEN_RL_FSDP_WORLD_SIZE` to the trainer interpreter and total trainer process
count. The local manager starts torchrun even for one process. The stock
Kubernetes server image does not contain the Automodel environment, so managed
Automodel trainer pods are rejected; use an external trainer with
`OPEN_RL_EXTERNAL_TRAINER=1` and a shared Redis queue.

Trainer and samplers must see the same `OPEN_RL_SNAPSHOT_DIR`. Checkpoints
belong in persistent `OPEN_RL_CHECKPOINT_DIR` storage. LoRA snapshots contain
PEFT files; training-state saves may additionally contain Adam state. Resume
validates adapter rank, scaling, dropout and tensor compatibility against the
worker configuration. Full-parameter sampling uses the whole-checkpoint
route and is outside this launcher's LoRA workflow.

## Validation after changes

Run `make test` for CPU regressions. On a dedicated GPU host, run the probe
with the Automodel interpreter and compare the same model and adapter rank
across layouts. On H100/H200, set `FLA_TILELANG=1` for the GDN backward kernel.

```bash
export PYTHONPATH="$PWD/src"
export PROBE_MODEL=Qwen/Qwen3.5-9B PROBE_LEN=4096
export PROBE_REF="$HOME/automodel/ref.pt"
export OPEN_RL_AUTOMODEL_LORA_RANK=32
export OPEN_RL_AUTOMODEL_TP=1 OPEN_RL_AUTOMODEL_CP=1
export FLA_TILELANG=1

CUDA_VISIBLE_DEVICES=0 PROBE_MODE=ref \
  "$AUTOMODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 scripts/automodel_probe.py
CUDA_VISIBLE_DEVICES=0,1 PROBE_MODE=layout \
  "$AUTOMODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/automodel_probe.py
CUDA_VISIBLE_DEVICES=0,1 PROBE_MODE=layout OPEN_RL_AUTOMODEL_TP=2 \
  "$AUTOMODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/automodel_probe.py
CUDA_VISIBLE_DEVICES=0,1,2,3 PROBE_MODE=layout OPEN_RL_AUTOMODEL_CP=4 \
  "$AUTOMODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 scripts/automodel_probe.py
CUDA_VISIBLE_DEVICES=0,1,2,3 PROBE_MODE=layout OPEN_RL_AUTOMODEL_TP=2 OPEN_RL_AUTOMODEL_CP=2 \
  "$AUTOMODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=4 scripts/automodel_probe.py
CUDA_VISIBLE_DEVICES=0,1 PROBE_MODE=ckpt OPEN_RL_AUTOMODEL_CP=2 \
  "$AUTOMODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 scripts/automodel_probe.py
```

The probe reports token logprob differences, adapter gradient agreement, and
checkpoint round-trip differences; these require inspection, not just a zero
exit status. Checkpoint mode writes `~/automodel/probe-state` and a `probe-model`
sampler snapshot. Historical [Qwen results](../reports/run49/README.md) and
[Gemma results](../reports/run51/README.md) describe prior code and hardware;
they do not validate later worker changes. GPU correctness, sampler loading,
and memory capacity must be checked again before relying on a changed runtime.
