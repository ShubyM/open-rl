# Training operations and worker execution

The backend is the command target. It owns numerical state; the process loop
owns queues, replies, and resource leases. There is no nested `Trainer`, base
worker class, or backend registry. The existing HF workers implement the
structural operation contract in `training/backend.py` directly.

## State and operations

Full fine-tuning owns a model, its trainable parameters, and an optimizer.
LoRA owns one shared PEFT model and a small parameter/optimizer record per
adapter. Pending gradients live on those parameter objects. These records
do not copy the shared model or introduce another execution layer.

The backends share ordinary functions in `hf_operations.py` for HF forward,
loss, backward, generation, and AdamW updates. Model, device, tokenizer, and
batching limits are explicit inputs. This is HF math, not an inheritance
contract that another backend must adopt.

| Operation | Effect |
| --- | --- |
| Forward | Read current weights without changing accumulated gradients |
| Forward/backward | Accumulate gradients for the selected model or adapter |
| Optimizer step | Update parameters and optimizer state, then clear gradients |
| Save/load state | Write/read a model checkpoint, optionally with optimizer state |
| Export for sampler | Write the weights the sampler consumes |

Optimizer steps perform no checkpoint writes, delta comparisons, or sampler
format conversion. Full checkpoints remain full checkpoints even when sampler
export uses deltas. A checkpoint does not advance the sampler's delta baseline.
Requesting optimizer restoration from a weights-only checkpoint fails explicitly.

Both HF backends accept an existing model in their constructor for CPU tests.
Device and loading choices are resolved once and reused on load. In production,
the create/load command still performs allocation inside the resource lease.
A future AutoModel backend can use its own resolved distributed topology and
native optimizer, while exposing the same operation semantics.

## Process loop and suspension

Queue identity and resource policy are independent. `model_id` drains a
dedicated queue; an active tenant set serves a shared LoRA host. Supplying a
time slicer requires the optional whole-worker `Suspendable` contract.
Without one, the backend stays resident and needs no sleep/wake methods.
Exclusive GPU admission is still the launcher's and scheduler's responsibility.

Every leased batch follows one sequence:

```text
acquire -> wake -> execute commands in queue order -> sleep -> release -> reply
```

Creation, checkpoint saves, and sampler exports follow the same sequence as
compute. The loop never inspects `cpu_offload`, tensor layouts, or save formats.
Cancellation waits for native work to finish before sleeping and releasing;
cancelling a Python await does not stop its underlying worker thread.
This intentionally holds the lease during serialization. An optimization
that serializes after release would need evidence that its scheduling benefit
justifies a second execution path.

Suspension preserves the next command's behavior, including pending gradients,
optimizer state, model buffers, and backend state. It applies to the whole
accelerator allocation, including all shared adapters or distributed ranks.
AutoModel may start resident; supporting suspension later does not change its
training operations. The existing FFT tensor offload and external CUDA parking
remain separate physical steps.

## Explicit export and physical storage

LoRA writes the existing `peft/<model_id>` directory only on explicit sampler
export. Training changes do not update it. Named and ephemeral LoRA references
still share that directory; historical immutable LoRA snapshots are separate
work. Export errors propagate to the request's result.

FFT delta export compares current weights against the last successful export,
then emits the existing absolute replacement indices/values. Several optimizer
steps may precede an export. Failed serialization does not consume the changes.
Loading a checkpoint makes the next delta export send all weights, establishing
a new baseline. Consumers still must apply incremental exports in order.

The delta baseline is an ordinary CPU copy, allocated only in delta mode.
It is separate from offload buffers, which sleep overwrites with current state.
Combining delta export and CPU offload therefore costs an additional model-sized
host copy. This removes the old per-step comparison and CPU-to-GPU baseline copy;
it does not claim to reduce peak host memory. Full export needs no delta baseline.

## Validation

CPU tests compare the shared math to an independent reference, verify adapter
gradient isolation and checkpoint continuation, reconstruct weights from delta
exports, and check that training leaves exported files unchanged. Runtime tests
verify ordered operations, one wake/sleep pair per leased batch, resident
backends without suspension, shutdown, and failure replies. CUDA offload and
distributed restoration require GPU validation; this cleanup does not establish
their numerical equivalence.
