# Operations dashboard

The gateway serves a cluster-operations dashboard at **`/dashboard`** (e.g. `http://localhost:9003/dashboard`
after `make server`). It is for operating and understanding the cluster — W&B owns training curves and
experiment analysis; nothing here duplicates metrics.

## Views

- **Cluster** — a pannable canvas of what is actually running: the gateway and its configured services
  (Redis, shared storage, vLLM) on the left, and real Kubernetes nodes grouped by GPU pool on the right,
  with the actual pods on each node. Edges are drawn only for connections the gateway is configured to
  make. Clicking a pod opens a full-height panel with its status and live logs (close with the ×, a click
  outside, or Escape). Drag to pan, scroll to move, ctrl+scroll to zoom.

  Clicking a GPU pool's header drills into a **pool screen** with a full-width duty-cycle chart:
  allocation duty — GPUs claimed by non-terminal pods (`nvidia.com/gpu` requests, or DRA resource
  claims) divided by pool capacity — sampled into an in-memory ring buffer each time the cluster is
  polled (throttled to one sample per 5s, last 120 kept for the gateway's lifetime). The chart is
  **stacked by job**: claims are attributed to runs via the `timeslice.io/job-id` pod labels (a run's
  trainer and sampler pods share one band; unlabeled pods fall back to their `app` label). Bands use a
  CVD-validated categorical palette with a legend naming each job, and hovering shows a per-job
  breakdown at that point in time; the pool's nodes are listed below. Back, Escape, or the Cluster tab
  returns to the canvas with pan/zoom intact. This is truthful scheduler state, not device utilization;
  DCGM owns that.

  When the `openrl.io/v1alpha1` placement API from the GPU scheduler is installed, the control column
  also shows Workload phases, ClaimLedger and seat totals. Missing scheduler CRDs are treated as an
  optional feature being off; installed-but-unreadable CRDs are a health error. The same column reports
  desired-versus-ready state for Deployments, DaemonSets, StatefulSets, and Jobs, including partial RBAC
  visibility instead of silently treating unreadable controllers as absent.
- **Runs** — launch by base model without leaving the page. Every row has an observed lifecycle verdict
  and an expandable inspection showing queue depth, current GPU claims by pool, pod phase counts,
  gateway request counts/failures/latency, latest worker-reported loss and gradient metrics with their
  bounded recent trend, measured CPU/memory, declared requests and limits, structured diagnostics,
  and recent logs for every pod. Clicking a pod opens its live log panel. A
  W&B link appears when one is recorded, and Stop appears when there is something to stop (a worker
  process, queued work, or labeled pods).
- **Health** — current problems first, then a **Load** section of measured stats (active runs, queued
  requests per run, oldest request and worker-launch wait, Redis memory and clients, gateway RSS, disk free, pod and
  GPU totals, optional Metrics Server CPU/memory usage, CPU/memory scheduling reservations, rollout health, scheduler workload phases, and ClaimLedger seats), then gateway / storage / Kubernetes /
  scheduler / visibility checks. Node `MemoryPressure` and `DiskPressure` conditions surface under
  Problems. Failed or slow placement, stale observed generations, assignment/seat mismatches, and stale
  ClaimLedger seats include exact `kubectl` inspection commands. Failed Jobs and stalled or unavailable
  workload controllers retain their Kubernetes conditions and include exact `kubectl describe` commands.

The page polls one coherent snapshot every 8 seconds and updates in place — canvas position, selection,
and open log panels survive refreshes. A refresh lists Kubernetes state once, so every view describes the
same observation while avoiding redundant API-server work. Observations are single-flight cached for one
second across browser tabs, focused API calls, and agents; every response says whether the observation was
live or cached, its age, total collection latency, and the pod/node/event/scheduler/metrics timing breakdown.
Collections slower than one second become a warning; tune that with `OPEN_RL_K8S_WARN_MS`. Set
`OPEN_RL_K8S_CACHE_SECONDS=0` to disable the cache.
Manual refresh and a light/dark toggle are in
the top bar. The top bar and gateway card show the exact image build revision; pod inspection includes
the runtime image digest, so an agent can prove a rebuilt fix is actually deployed rather than trusting a tag.

## JSON API and ops CLI

The UI serves humans; the same primitives are exposed as JSON for agents and scripts:

| Primitive | Endpoint | CLI |
| --- | --- | --- |
| diagnose | `GET /api/v1/dashboard/snapshot` | `make ops diagnose` |
| health | `GET /api/v1/dashboard/health` | `make ops health` |
| problems | `GET /api/v1/dashboard/problems` | `make ops problems` |
| inspect | `GET /api/v1/dashboard/cluster` | `make ops inspect` |
| runs | `GET /api/v1/dashboard/runs` | `make ops runs` |
| run detail | `GET /api/v1/dashboard/runs/{run_id}?logs=N` | `make ops run <run_id> N` |
| logs | `GET /api/v1/dashboard/pods/{pod}/logs` | `make ops logs <pod> [lines]` |
| launch | `POST /api/v1/dashboard/runs` | `make ops launch <model>` |
| stop | `POST /api/v1/dashboard/runs/{run_id}/stop` | `make ops stop <run_id>` |

`dev/tools/ops.py` is stdlib-only and always prints JSON; point it at a remote gateway with
`BASE_URL=http://host:9003`. The diagnostic snapshot is schema-versioned. Every load stat includes
the human display string plus `value_number`, `unit`, structured `context`, and `status`, so agents do
not need to parse text such as byte sizes or GPU fractions.
Problems and per-run diagnostics carry a stable `id` and `code`, the affected `resource`, structured
`evidence`, and concrete `actions` containing both API paths and copyable CLI or `kubectl` commands.
Pod evidence includes current and previous container termination state, exit code, condition messages,
the configured image, runtime image digest, and up to 10 recent Kubernetes events. Configured service
endpoints are reduced to scheme, host, and port; credentials, paths, queries, and fragments never enter
diagnostic JSON. Use `make ops logs <pod> --container <name> --previous` (or the
**previous** toggle in the pod panel) after a restart; diagnoses distinguish OOM kills, crash loops,
image-pull failures, evictions, volume mounts, and failed scheduling.
Queue entries are timestamped at first enqueue (the timestamp survives the FFT worker-launch hop), so
depth is paired with actual oldest-wait seconds. Request waits warn after 300 seconds and worker-launch
waits after 60 seconds by default; tune them with `OPEN_RL_QUEUE_WARN_SECONDS` and
`OPEN_RL_LAUNCH_WARN_SECONDS`.

## Kind smoke test

`make kind-dashboard-smoke` creates (or reuses) an `open-rl-dashboard` Kind cluster, builds and
loads the local gateway image, deploys the dashboard with its real service account and RBAC, and
verifies that the gateway can list its namespace, see the Kind node, serve the UI, and return a
coherent diagnostic snapshot with no reported problems. It also exercises pod logs and the stop
permission while rejecting images that rebuild the Python project at pod startup. The cluster is
left running for inspection; remove it with `make kind-dashboard-clean`.

## Data sources — everything degrades gracefully

- **Kubernetes** is optional: the `kubernetes` Python client is loaded lazily and the dashboard works
  without it (the Cluster view then shows only gateway-local components and says so). In-cluster
  credentials are tried first, then the local kubeconfig. Listing nodes requires cluster-scope RBAC and
  is skipped when denied; fetching pod logs from the gateway's service account requires the `pods/log`
  verb. One bounded namespaced Event list is fetched alongside nodes and scheduler state; `events`
  read permission is included in the supplied manifests, and missing permission is an actionable
  visibility warning rather than a snapshot failure. When `metrics.k8s.io` is installed, pod and node
  CPU cores and memory working-set bytes are joined into the same snapshot and shown in run, pod, node,
  and Health views. A missing Metrics Server is reported as an optional feature being off; the dashboard
  snapshot also exposes per-scope availability and observed-object counts so automation can distinguish
  absent, partial, and empty telemetry. Scheduling reservations follow an observed pod-level resource
  budget when present; otherwise they use summed app-container requests versus peak init-container
  requests, plus pod overhead. The dashboard never substitutes requests or limits for utilization.
- **Scheduler placement** is read directly from optional namespaced `Workload` and `ClaimLedger` CRDs.
  Run inspection joins them by exact `spec.modelId`, showing requested accelerator memory, placement
  phase and reason, chosen claim/node/device count, and ledger seat count. The gateway service account
  only receives read access to those two resources.
- **Runs** are discovered from Redis queues and model metadata keys, the shared filesystem
  (`$OPEN_RL_TMP_DIR/peft`, `$OPEN_RL_TMP_DIR/checkpoints`), and the gateway's own FFT worker processes.
  Every create-model request now records lifecycle metadata before enqueueing, then transitions it to
  ready or failed when the future resolves. In-memory mode keeps that record for the gateway lifetime;
  Redis mode preserves it across gateway restarts. This prevents runs from disappearing after their
  create request drains. A W&B URL is shown when adapter `metadata.json` or model metadata records
  `wandb_url`. Request telemetry is joined by request and model ID at the queue/future boundary: the
  model record keeps exact completion and failure counts, end-to-end latency, operation counts, and the
  latest numeric worker metrics (with at most 20 points per metric). It remains available after the
  request queue drains and, with Redis, after a gateway restart. Once a worker begins dispatch, its
  current operation, queue wait, and execution age remain visible until the future resolves; operations
  still active after 600 seconds become `run.request_stalled` problems. Tune that threshold with
  `OPEN_RL_OPERATION_WARN_SECONDS`.
- **Stop** does only what is truthfully stoppable: terminates the gateway-launched worker process,
  clears the run's Redis queues, and deletes pods labeled `timeslice.io/job-id` for the model. It
  reports exactly which actions it took.

## Demo mode

`OPEN_RL_DASHBOARD_DEMO=1` makes every endpoint return fictional data for developing or demoing the UI
without a cluster. Every payload carries `"demo": true` and the UI shows a banner stating the data is
fictional; demo stop performs no action.
