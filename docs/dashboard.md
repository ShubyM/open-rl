# Cluster dashboard

`/dashboard` is an operator UI served by the API server over the read-only
inspection API under `/api/v1/inspect`. It is plain ES-module JavaScript with
no build step; agents use the same JSON routes (`/docs` and `/openapi.json`
describe their fields).

## Run it

```sh
make server
```

Then open `http://127.0.0.1:9003/dashboard`.

On Kubernetes, the API server already serves `/dashboard`, so no separate
deployment is needed:

```sh
kubectl -n openrl-system port-forward svc/open-rl-api-server-service 8080:8000
```

Then open `http://127.0.0.1:8080/dashboard`.

Logs need `OPEN_RL_LOG_STORE`, nodes and pods need Kubernetes, and GPU duty
charts need `OPEN_RL_PROMETHEUS_URL`. A source that cannot be read shows as an
error, never as an empty cluster. See
[Inspection API](configuration.md#inspection-api) for the environment variables
and IAM permissions.

## Pages

- **Overview** lists runs with status, completed steps, elapsed time, search,
  and status filters.
- **Run** shows status, settings, an activity timeline with GPU rows and duty,
  trainer metrics and operation durations by step, and worker or client logs.
  Click the timeline to open logs around that time.
- **Nodes** shows which process held which GPU and when, each GPU's duty cycle,
  and GPU utilization and memory. Drag or press A/D to pan; use Ctrl/⌘ + wheel,
  pinch, or W/S to zoom.
- **Health** lists pod and node problems and sources that could not be read.

## Lifecycle

Each run in `/api/v1/inspect/inventory` has a `lifecycle`, which says where it
is:

- **ended**: the API server marked it finished, its workers stopped, or it has
  no workers and no live session.
- **failed**: one of its workers is in a failed state.
- **waiting**: a worker is not placed yet, or the server has not created one
  for a live session.
- **placed**: every worker is placed.
- **active**: the scheduler cannot be read, so placement is unknown.
