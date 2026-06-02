# Full Fine-Tuning Scheduler Diff

## What Changed

This PR adds the minimal full fine-tuning path needed to prove that two SFT jobs can share one local GPU through explicit scheduler handoff.

Before this change:

- OpenRL had a single trainer worker style that could assume it owned CUDA.
- Requests were drained from the shared queue.
- There was no node-local process that coordinated GPU residency across multiple full-FT workers.
- There was no process-level checkpoint/restore seam for evicting a trainer's CUDA state.

After this change:

- A full-FT run gets its own trainer worker process.
- Each worker drains only its own run queue.
- Workers talk to a node-local `TrainerScheduler` over a persistent Unix socket.
- Only one worker can hold the scheduler's active CUDA slot at a time.
- A worker must call `ACQUIRE` before CUDA work and `RELEASE` after its drain window.
- `RELEASE` checkpoints the active worker with `cuda-checkpoint`.
- A later `ACQUIRE` restores that same worker if it was checkpointed.

The intended first milestone is narrow: two full fine-tuning SFT jobs, same base model, one OpenRL server, one GPU residency slot. This PR does not try to solve sampling, vLLM handoff, multi-base scheduling, placement, multi-GPU topology, or crash recovery.

## Design Shape

The scheduler uses the same basic shape as `accel-time-slicer`:

- Jobs explicitly request GPU access instead of assuming CUDA ownership.
- A scheduler grants one job at a time.
- The job yields/releases when its current GPU work window is done.
- The scheduler checkpoints the old job before another job can run.
- Restoring happens before the job is granted CUDA again.

## New Pieces

- `CheckpointRestorer`
  - Protocol with `checkpoint(pid)` and `restore(pid)`.
  - Intentionally does not know about OpenRL queues, runs, fairness, or scheduler state.
- `CudaCheckpointRestorer`
  - Uses NVIDIA `cuda-checkpoint`.
  - `checkpoint(pid)` runs `lock` then `checkpoint`.
  - `restore(pid)` runs `restore` then `unlock`.
- `TrainerScheduler`
  - Owns registered workers, a FIFO waiter queue, the active run, and checkpointed/failed worker state.
  - Grants one worker at a time.
- `TrainerSchedulerClient`
  - Worker-side socket client.
  - Exposes `async with scheduler.acquire(run_id): ...`.
- Worker launch and run-scoped queue draining
  - The gateway can launch a per-run worker.
  - Scheduled workers drain only requests for their run.

## Scheduler Behavior

`REGISTER(run_id, pid)`:

- Records the worker PID and socket connection for the run.
- Duplicate registration for a live run fails.

`ACQUIRE(run_id)`:

- Validates that the run is registered, not failed, and not already waiting or active.
- Appends the run to the FIFO waiter queue.
- Waits until no run is active and this run is at the head of the queue.
- Marks the run active.
- Restores the worker if it was checkpointed by a previous release.
- Returns success only when the worker may touch CUDA.

`RELEASE(run_id)`:

- Validates that the run owns the active slot.
- Checkpoints the worker.
- Marks the worker checkpointed.
- Clears the active slot and wakes waiters.

`UNREGISTER(run_id)`:

- Removes the worker.
- Clears any waiting or active claim for the run.

Socket close:

- Means worker death.
- Clears waiting/active state for that connection.
- Marks the worker failed so future acquires cannot grant CUDA.

## Lazy Residency Decision

The first implementation tried to keep the single-worker fast path in mind: if a worker released while nobody else was waiting, leave it resident and avoid checkpointing.

That optimization made the scheduler harder to reason about. It required extra state and pushed eviction into surprising places:

- Track `resident_run_id` for a worker that is not active but may still occupy GPU memory.
- Make `ACQUIRE` checkpoint a different idle-resident worker before granting the requester.
- Handle the dynamic registration case where worker A is resident because it was alone, then worker B registers later.
- Add socket-close and unregister handling for idle-resident workers.
- Add tests for release/reacquire without checkpointing, second-worker eviction, and idle-resident failure cleanup.

This PR deliberately does not include that optimization.

Current invariant:

- Every successful `RELEASE` checkpoints.
- `RELEASE` is the only eviction point.
- `ACQUIRE` never checkpoints another worker.
- A worker outside an acquire window is either cold/not-yet-materialized or checkpointed.
- There is no idle-resident state.

This is less efficient because a single job pays checkpoint cost after every drain window. For this prototype, that is the right tradeoff: the goal is to prove the two-worker full-FT path with the smallest correct scheduler.

The single-job edge case is therefore intentionally suboptimal: if only one full-FT worker exists, it still checkpoints on every `RELEASE` and restores on the next `ACQUIRE`. Correctness is simple because the worker never keeps hidden GPU residency outside the acquire window, but throughput is worse than the lazy-resident version would be.

If checkpoint overhead becomes the bottleneck later, add lazy residency as a separate change with explicit `resident_run_id` state and tests.

## CUDA Boundary

The correctness boundary is strict:

- No CUDA allocation before first successful `ACQUIRE`.
- Model and optimizer materialization onto CUDA happens inside the acquire window.
- After `RELEASE`, the worker may only do CPU work: queue waits, socket I/O, heartbeats, CPU logging.
- After `RELEASE`, the worker must not touch CUDA until the next successful `ACQUIRE`.

Risky paths include lazy model init, optimizer state moves, CUDA tensor `.item()`, CUDA metrics, save/load hooks, and `torch.cuda.empty_cache`.

## Failure Policy

- Worker socket close is handled gracefully as worker death.
- Checkpoint or restore failure is not handled as a normal request error.
- If checkpoint/restore fails, GPU state is unknown, so the scheduler fails closed instead of granting another worker.

The important behavior is that a failed checkpoint or restore cannot be followed by another successful CUDA grant from the same scheduler instance.

## Known Lifecycle Gap

Socket close detects worker death, not normal client completion.

If the client finishes but the worker keeps polling an empty queue, the scheduler still sees a live socket. The gateway should own cleanup because it launched the worker and sees client activity.

Next cleanup step:

- Add an idle reaper in the gateway.
- Use the existing Tinker session heartbeat as the client liveness signal.
- If a run queue is empty and the session heartbeat is stale past a timeout, terminate the worker.
- Terminating the worker closes the scheduler socket and reuses existing cleanup.

Later:

- Tune heartbeat timeout behavior so long legitimate idle gaps do not kill active clients.
- Add a max-hold/acquire deadline for hung-but-alive workers.

## Full-FT Resume Compatibility

This prototype treats full fine-tuning as a server-side mode selected by `OPEN_RL_TRAINING_MODE=full`.
That keeps the stock Tinker client and cookbook training loop unchanged for creating a run.

The resume path is different. `tinker_cookbook.supervised.train` checks `log_path/checkpoints.jsonl`
on startup. If it finds a `state_path`, it calls the stock Tinker
`create_training_client_from_state*` flow. That SDK flow asks `/api/v1/weights_info` for LoRA-shaped
metadata, creates a LoRA training client, then calls `load_state`.

OpenRL now records model-scoped checkpoint paths so aliases do not collide across runs, and the
gateway can answer `weights_info` from checkpoint metadata. When `OPEN_RL_TRAINING_MODE=full`, a full
checkpoint intentionally reports a LoRA-shaped `weights_info` response because the stock SDK asserts
that `create_training_client_from_state*` is LoRA-shaped before it calls `load_state`.

That response is a compatibility shim, not the source of truth. In full mode the gateway ignores the
LoRA shape and creates a full-FT worker; `load_state` then loads the full HuggingFace checkpoint.

Current user-facing behavior:

- Start the gateway with `OPEN_RL_TRAINING_MODE=full` for both the original run and resume.
- Use a fresh `log_path`, or delete the previous artifacts directory, when a clean run is desired.

The cleaner long-term version is a client/helper that can create a full-FT training client from state
without the `weights_info` lie.

## Tests

The tests are meant to cover behavior, not tiny implementation details.

Covered behavior:

- Only one worker can be active.
- A waiting acquire is not granted until release checkpoint completes.
- A checkpointed worker is not granted until restore completes.
- First acquire is cold; later acquire restores after checkpoint.
- Release checkpoints even with no waiters.
- Waiters are FIFO.
- Unregistering a waiting worker prevents a later grant.
- Socket clients alternate over the real Unix socket server.
- Closing an active worker socket marks the run failed.
- A worker drains only its own run queue.

Avoid testing the exact local helper shape inside `CudaCheckpointRestorer`; the important contract is that scheduler handoff waits for checkpoint/restore completion.
