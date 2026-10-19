# Accelerator Time-Slicer

The accelerator time-slicer is OpenRL's acquire/release coordinator for full fine-tuning
workers that share one accelerator bundle.

It exposes workload commands over a Unix socket or TCP listener:

- `REGISTER(workload)` records a worker workload.
- `ACQUIRE(workload)` grants that workload the right to touch CUDA.
- `RELEASE(workload)` releases the workload and checkpoints it.
- `UNREGISTER(workload)` removes the workload registration.

The worker-facing `TimeSlicerClient` defaults to a socket/TCP client for the
node-local daemon. The daemon-side `TimeSlicer` contract is implemented today
by `SingleNodeTimeSlicer`, which owns locking, fairness, and active/checkpointed
state.
It delegates physical checkpoint/restore to a `CheckpointRestorer` backend:

- `CudaCheckpointRestorer` discovers local GPU PIDs from the workload identity,
  then calls `cuda-checkpoint`.
- `LlmDCheckpointRestorer` uses llm-d's Python snapshot client with the workload
  `job_id` and `group`; llm-d discovers the actual pod/PIDs from labels.

A group identifies the accelerator bundle the workload belongs to. This
node-local implementation serializes acquire/release through one
`SingleNodeTimeSlicer` process; future cluster coordination can use the same
workload identity.
For local subprocess workers, OpenRL stamps `OPEN_RL_TIME_SLICE_JOB_ID` and
`OPEN_RL_TIME_SLICE_GROUP` into the worker environment and starts the worker in
its own process group. The CUDA backend uses those tags to find the workload's
current GPU PIDs before checkpoint, then restores the same PID set.

In Kubernetes, this agent runs as a node-local DaemonSet with `hostNetwork:
true`. FFT worker pods connect to the agent on their node via
`OPEN_RL_ACCEL_TIMESLICER_HOST=status.hostIP` and `OPEN_RL_ACCEL_TIMESLICER_PORT`.
The node-local time slicer is the coordination point for worker pods that share
one DRA `ResourceClaim`. The cluster manifest runs this process with
`--backend llmd` so llm-d's snapshot agent owns pod/PID discovery and physical
checkpoint/restore.

The process that owns a workload's work queue holds its lease. A trainer
registers, acquires and releases from inside its own process. A pool sampler
that shares its GPU is `vllm serve` run by
`server.sampler_launcher`, which registers the workload, starts vLLM inside one
turn, puts it to sleep and releases over a connection it keeps open for as long
as vLLM runs; from then on the pool's dispatcher acquires and releases the same
workload over its own connection for each turn. A registration belongs to the
connection that made it, and a grant to the connection whose `ACQUIRE` holds it,
which alone can `RELEASE` it. When a connection closes while holding a grant on
a workload it did not register, the slicer releases that workload, checkpointing
it, before the next waiter is granted. A workload whose checkpoint fails keeps
the grant until the connection that registered it closes. When another
connection's release is the one that failed, as at the end of a dispatcher's
turn, the slicer tells the registering connection with a line it did not ask
for, `{"event": "FAULT", "name": ..., "claim": ..., "error": ...}`, and the
sampler launcher stops vLLM and exits on it.

## llm-d snapshot backend

The same OpenRL acquire/release protocol can use llm-d's snapshot agent by
running this process with `--backend llmd`. Workers register a workload that
includes `OPEN_RL_TIME_SLICE_JOB_ID`. Kubernetes workers use role-prefixed
model ids such as `trainer-<model-id>` and `sampler-<model-id>` because llm-d
discovers pods by job id. The llm-d restorer uses the job id and group when
calling `snapshot_and_wait` / `restore_and_wait`.

For this first integration, release checkpoints the workload before another
workload can acquire the accelerator when the backend reports resident GPU
state.
