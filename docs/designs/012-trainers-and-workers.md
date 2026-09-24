# Training state, operations, and execution

A training backend owns the model, gradients, and optimizer. The process loop
owns its request queue and accelerator lease. A sampler owns generation.

## Read the code in this order

- `training/fft_trainer_worker.py`: one model, its optimizer, checkpoint IO, and optional offload.
- `training/lora_trainer_worker.py`: one shared base model, with parameters and an optimizer per adapter.
- `training/hf_operations.py`: shared HF forward/loss/backward and optimizer math, with explicit inputs.
- `server/training_requests_processor.py`: execute an ordered batch and publish its results.
- `server/vllm_sampler.py`: one sampler for LoRA and full fine-tuning.

The operation contract in `training/backend.py` has no base-class implementation.
AutoModel can implement it using its own model, optimizer, and resolved topology.
It does not need to inherit HF internals or implement generation.

| Operation | State change |
| --- | --- |
| Forward | Read weights without changing accumulated gradients |
| Forward/backward | Accumulate gradients |
| Optimizer step | Update weights and optimizer state, then clear gradients |
| Save/load state | Write/read a checkpoint, optionally with optimizer state |
| Export for sampler | Write weights for a separate sampling process |

Optimizer steps perform no file writes or export comparisons. Full checkpoints
remain full checkpoints regardless of the sampler export strategy. A requested
optimizer restore fails if the checkpoint has no optimizer state.

## Execution and suspension

A model ID selects a dedicated queue; an active tenant set selects a shared
LoRA queue. Supplying a time slicer adds this sequence around the batch:

```text
acquire -> wake -> execute commands -> sleep -> release -> reply
```

Without a slicer, the same loop executes commands and replies. Exclusive resource
placement belongs to the launcher and scheduler. All saves and creation run
while awake. Cancellation drains native work before releasing the device.

Sleep/wake is optional and applies to the entire backend allocation. It must
preserve weights, pending gradients, optimizer state, and buffers. AutoModel can
start resident and add suspension later without changing its training operations.

## What export still costs

LoRA writes adapters only on explicit export. Its references currently share
`peft/<model_id>`; they are not immutable historical snapshots.

Full exports write an independent HF checkpoint. Sparse exports compare against
the last successful export and write native tensor names, shapes, and replacement
indices/values. vLLM handles packing and tensor-parallel slicing. Failed writes
leave the baseline intact; restoring a checkpoint makes the next export replace
all parameters.

Sparse mode retains one CPU copy of the weights, separate from offload buffers.
Consumers must apply every export in order. Skipping an export, restarting a
sampler, or pruning an unconsumed version is not made safe by the new file format;
that requires a separate version protocol. Use full checkpoints for independent
versions. See [weight transfer](../weight-transfer.md) for the format and upgrade.

## Validation

CPU tests compare training math to independent references, checkpoint continuation,
adapter isolation, and exported-weight reconstruction. Queue tests cover ordering,
lease failures, and cancellation. CPU CI uses a separate HF sampler in
`tests/cpu_sampler.py` to evaluate exported adapters through the real API.
CUDA offload and distributed restoration still require GPU validation.
