# Time Slicer

The time slicer is OpenRL's acquire/release coordinator for full fine-tuning
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
- `GpuCrCheckpointRestorer` discovers local GPU PIDs from the workload identity,
  then calls GPU-CR's `cr_client` / `multi_cr_client`.
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
`OPEN_RL_SNAPSHOT_AGENT_HOST=status.hostIP` and `OPEN_RL_SNAPSHOT_AGENT_PORT`.
The node-local time slicer is the coordination point for worker pods that share
one DRA `ResourceClaim`. The cluster manifest runs this process with
`--backend llmd` so llm-d's snapshot agent owns pod/PID discovery and physical
checkpoint/restore.

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

## GPU-CR backend

The same daemon can run with `--backend gpucr` to trigger GPU-CR instead of the
NVIDIA `cuda-checkpoint` CLI. The backend still uses OpenRL workload identity
and local GPU PID discovery, so the process being checkpointed must have
`OPEN_RL_TIME_SLICE_JOB_ID` / `OPEN_RL_TIME_SLICE_GROUP` in its environment and
must be launched under GPU-CR's preload library:

```bash
OPEN_RL_SNAPSHOT_AGENT_BACKEND=gpucr \
  GPUCR_PRELOAD=/usr/local/lib/open-rl/gpu-cr/vGPU-NVIDIA.so \
  python -m snapshot_agent.serve --backend gpucr
```

When the local or Kubernetes worker manager sees
`OPEN_RL_SNAPSHOT_AGENT_BACKEND=gpucr`, it injects `LD_PRELOAD=$GPUCR_PRELOAD`
and `GPU_VENDOR=NVIDIA` into launched trainer and sampler workers. Set
`EXPORT_FILE_PATH` to use file-backed staging; otherwise GPU-CR expects the
node-level hugepage staging mount at `/mnt/huge-ckpt`.
