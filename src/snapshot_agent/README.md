# Snapshot Agent

The snapshot agent is a small process-level GPU residency primitive.

It exposes four commands over a Unix socket or TCP listener:

- `REGISTER(pid)` records a worker process.
- `ACQUIRE(pid)` grants that process the right to touch CUDA.
- `RELEASE(pid)` checkpoints that process before another process can acquire CUDA.
- `UNREGISTER(pid)` removes the process registration.

Today every successful `RELEASE` checkpoints the process. This is simple and
conservative, but it is slow because even a single run pays checkpoint cost after
each acquire window.

In Kubernetes, this agent runs as a node-local DaemonSet with `hostPID: true`
and `hostNetwork: true`. FFT worker pods connect to the agent on their node via
`OPEN_RL_SNAPSHOT_AGENT_HOST=status.hostIP` and `OPEN_RL_SNAPSHOT_AGENT_PORT`.
The node-local agent is the coordination point for worker pods that share one
DRA `ResourceClaim`.
