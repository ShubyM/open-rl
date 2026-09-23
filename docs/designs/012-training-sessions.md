# Training sessions and worker processes

This is the first foundation change toward an exclusive AutoModel backend.
It keeps the existing HF LoRA and full fine-tuning deployment modes. It does
not add AutoModel, torchrun, or a new scheduler mode.

## Ownership

`training.session.TrainingSession` holds an already constructed model, the
parameter objects to optimize and their optimizer state. It runs the
existing forward/backward, optimizer, and local generation algorithms. It
does not load models, select an accelerator, save files, or publish weights.
Tests can construct it directly with a small CPU model.

`LoraTrainingWorker` owns a shared base model and a session for each adapter.
It selects the adapter before delegating a command; parameter references,
pending gradients, and optimizer state remain separate across sessions.
`FFTTrainingWorker` owns one session and the existing offload/delta state.
Reloading a model creates a fresh session bound to the new parameters.

Model loading, checkpoint serialization, and the existing automatic LoRA
publication remain explicit worker responsibilities. Separating resumable
checkpoints from sampler exports is follow-up work, not part of this change.

## Process execution

`TrainingRequestsProcessor` receives a worker, a request store, and concrete
queue/resource choices:

| Queue identity | GPU resource policy | Use |
| --- | --- | --- |
| `active_tenant_set_id` (or the default active set) | No time slicer | Shared LoRA host |
| `model_id` | Supplied time-slicer client | Existing full fine-tuning worker |
| `model_id` | No time slicer | Resident, dedicated worker on exclusive GPUs |

The two queue identities are mutually exclusive. A dedicated runner only
executes commands for its model and exits on shutdown even without a lease.
The shared LoRA launch passes only its active-set argument. The standalone
FFT entry point still creates its time-slicer client explicitly.

The unleased path invokes only command methods: a resident backend does not
need no-op sleep/wake methods. Existing FFT callers still need the leased
offload path, or an explicitly resident model (`cpu_offload=False`); omitting
the time slicer alone does not change a model's residency policy or reserve
its GPU. Scheduler admission remains the launcher's responsibility.

Commands preserve their queue order, including saves between optimizer steps.
Leased workers offload between contiguous compute and save groups and publish
results after releasing the GPU. Resident workers publish each result as it
is ready. A later batch failure does not overwrite earlier published results.

The runner imports no concrete model backend. Its lifecycle and queue tests
use an in-memory store; separate worker processes continue to require Redis.

## AutoModel follow-up

Add AutoModel as one model per process group, with scheduler exclusivity and
no time-slicer client. Keep its distributed execution scope around the whole
forward/loss/backward operation rather than extending a forward-only hook.
Resolve topology at launch and consume training configuration from the typed
create command. Rank zero should own external queue/result operations.

Trainer ownership must become role-specific before adding AutoModel LoRA:
its trainer is dedicated by model ID even if its vLLM sampler shares a base
model. The existing `runtime_of()` shared-LoRA policy must not be reused for
that trainer. Multi-GPU claims and torchrun wiring can then be added as a
separate, reviewable step.

Validation for this foundation is included in `make test`: CPU numerical
references, adapter isolation, forward-only behavior, checkpoint reloads,
dedicated/shared queue selection, FIFO saves, shutdown, and failure handling.
GPU offload behavior remains covered only by the existing GPU test paths.
