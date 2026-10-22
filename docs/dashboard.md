# Cluster dashboard and inspection API

`/dashboard` is an operator UI. Agents start with the compact read-only API under `/api/v1/inspect`; richer page data lives under `/api/v1/dashboard`. Both use the same cached fleet, placement and operation records, and the same Prometheus and Cloud Logging readers. `/docs` and `/openapi.json` describe both APIs.

## Run it

```sh
npm --prefix dev/dashboard ci  # once; installs Svelte and esbuild
make dashboard                 # builds static/app.js and serves on port 8080
```

The standalone process is `uv run --extra cluster python -m server.dashboard`. Point it at the API server's Redis (`REDIS_URL`) and worker namespace; without shared Redis it has its own empty store. Rebuild with `make dashboard-build` after changing `src/server/dashboard/ui`, and before `uv build`.

On GKE, `kubectl apply -k k8s/deploy/dashboard` deploys it into `openrl-system` with `open-rl-sa`. It uses the API-server image, whose Dockerfile builds the UI in a Node stage. To open it from a laptop, run `kubectl -n openrl-system port-forward svc/open-rl-dashboard 8080:8000` and open `http://127.0.0.1:8080/dashboard`.

The API lists every run in the fleet, so the client-facing API server does not mount it by default. For a trusted local server without Redis, run `make dashboard-build`, then `OPEN_RL_INSPECTION_API=true make server`, and open `http://127.0.0.1:9003/dashboard`.

Locally, worker log history needs Cloud Logging, nodes and pod logs need Kubernetes, and hardware and sampler charts need a metrics backend. A missing source shows as an unavailable-source message, never as an empty cluster.

## Pages

- **Overview** lists runs with a status label, search, and status filters.
- **Run** shows status, settings, worker state, an activity timeline with GPU rows and duty, timestamped incidents, metrics by step and wall time, sampler and collector charts, and logs. Click the timeline or an incident to open the logs around that time. Previous container output is available for restarted pods.
- **Nodes** shows accelerator nodes: which process held which GPU and when, each node's GPU duty cycle, GPU utilization and memory, worker CPU and RSS, and collector metrics. Drag or press A/D to pan; use Ctrl/⌘ + wheel, pinch, or W/S to zoom.
- **Health** lists pod and node problems and unavailable sources, with current and previous pod-log links even for pods without a run.

Pages show their last response while they refresh.

## API

| Route under `/api/v1/inspect` | Returns |
|---|---|
| `GET /inventory` | Compact runs with their lifecycle, nodes with their GPUs, the GPUs held now, pods with problems, and an error per unavailable source |
| `GET /activity` | One lane per placement, fleet-wide, a node's or a run's, with its worker's GPU turns and completed ops in the window |
| `GET /query` | A named PromQL query (`gpu_busy`, `gpu_util`, `gpu_memory_mib`, `worker_cpu_cores`, `worker_memory_bytes`, `vllm_*`, ...) over a run's or node's GPUs and pods, or the fleet's GPUs, at `points` resolution |
| `GET /accelerators` | One row per GPU: node, slot, model, holder, utilization now and over the window, memory used of total |
| `GET /runs` | Every live run in one line, longest since its last op first; `ended=true` adds ended runs |
| `GET /runs/{id}/summary` | The run's facts: pace, goodput, unanswered requests and their holders, the request ledger, failures and relaunches with log reads, saves, optimizer, samplers, config, placement, incidents and GPU duty by role |
| `GET /runs/{id}/steps` | Trainer ops grouped by step, with the client's metrics; `fields` keeps client metrics by prefix |
| `GET /runs/{id}/commands` | Raw operation records, newest first; optional window and role, `limit=100` by default, at most 5,000 |
| `GET /sessions/{id}/summary` | The session record, whether it is live, its SDK errors, and a line per model |
| `GET /logs` | Worker and run-correlated client output from Cloud Logging, with filters and Cloud Logging's page token as the cursor |

`activity`, `query`, `accelerators` and `logs` take ISO `since` and `until` bounds of up to seven days; `since` also takes a span such as `1h`. An omitted end is now, and an omitted start the 30 minutes before it. With both bounds omitted, `/accelerators` uses the last hour. Missing op records do not prove that no requests arrived or that workers were idle. `error` means a source could not be read. A log cursor is valid only with the same filters, `since` and `until` as the page that returned it.

For a small first read, use `/runs`, then `/runs/{id}/summary` or `/sessions/{id}/summary`. Read steps, commands, a named metric or filtered logs only when that evidence is needed. A model ID identifies a run; a session can govern multiple models and the same model can appear in successive sessions.

| Route under `/api/v1/dashboard` | Returns |
|---|---|
| `GET /inventory` | Full run records with scheduler workloads, pod/container details and Kubernetes events, plus the shared fleet inventory |
| `GET /activity` | The same placement lanes and window; with `run_id`, also the run, incidents and chart samples. Trainer operations are retained; sampler requests retain the slowest per time bucket |
| `GET /runs/{id}/steps` | The same step records, with client-source failure reported separately so trainer charts remain usable |
| `GET /history` | Legacy Redis client metrics for `run_id`, latest finite metrics per step |
| `GET /telemetry` | Generic collector gauges for exactly one `run_id` or `node_id`, optional metric-name filter, at most 200 series with truncation reported |
| `GET /pods/{pod}/logs` | Kubelet output for any pod in the configured namespace; optional `container`, `previous=true`, and `tail` (default 200, at most 1,000 lines and 128 KiB) |

The pages share `/inspect/query` for named metrics and `/inspect/logs` for Cloud Logging filters and pagination. `diag/backend-main` does not expose a vLLM metrics HTTP endpoint or configure a collector-gauge producer in this repository. These charts require an existing exporter and scrape configuration; the dashboard does not create those metrics. Worker metrics are what the trainer returns per op (`loss`, `grad_norm`); a NaN or infinity is named in the op's `nonfinite` and reaches the client as 0, as before.

## Client logs

Client and tinker jobs are assumed to run in the configured worker namespace in the cluster, where Cloud Logging collects their stdout/stderr. Local file forwarding and a Redis log tail are deferred.

Collection alone does not associate output with a run. The existing run selector matches worker pods by their runtime's `timeslice.io/owner` label, or structured log entries whose `logging.googleapis.com/labels` contains `openrl_run_id` (the model ID) and `openrl_source=client`. Client output needs that correlation to appear in the run's Logs tab. Shared-runtime views also default to searching for the run ID in the message; “Include other runs’ lines” removes that text filter. The existing `POST /api/v1/runs/{run_id}/logs` endpoint still emits those structured lines; this PR adds no forwarding process.

## Lifecycle

Each run in `/inventory` has a `lifecycle`, which says where it is, not whether it is healthy:

- **ended**: the gateway marked it finished, its workers stopped, or it has no workers and no live session.
- **waiting**: a worker is not placed yet, or the server has not created one for a live session.
- **placed**: every worker is placed.
- **active**: the scheduler cannot be read, so placement is unknown.

The pages label a run from its lifecycle and show pod problems separately. The summaries return observed facts rather than threshold-based health verdicts.

## Sources

| Data | Source and retention |
|---|---|
| Runs and sessions | API-server metadata in Redis |
| Worker processes | Trainers and samplers rewrite records every 5 seconds: pod, node, GPU UUIDs, phase, unanswered requests, and recent GPU turns. A record unchanged for 30 seconds belongs to a gone process |
| Ops and step counts | Recorded by workers, newest `OPEN_RL_OPS_SAMPLE_LIMIT` (2,000) per run. Each sampler records its completed samples with the logical run ID |
| Placements | The API server lists the scheduler's Workloads every 30 seconds and keeps one record per placement in `open_rl:placements:<model_id>`: pod, node, GPU UUIDs, placed and released times. Historical placement views require retained placement records; runs predating them may have no placement history |
| Nodes, pods, events, scheduler | Kubernetes API, cached for 5 seconds |
| Each node's GPUs | DCGM's UUID and slot labels, from the metrics source below |
| GPU, worker, and sampler series | DCGM, container, and vLLM metrics from `OPEN_RL_PROMETHEUS_URL`, or Cloud Monitoring's PromQL API on GKE |
| Run logs | Cloud Logging: the worker pods with the run runtime's `timeslice.io/owner` label and client entries labelled with the run; a stderr line takes the level it prints |

Records expire `OPEN_RL_OPS_TTL_DAYS` (default 7) after their last write; `0` keeps them. Logs come from the project's default buckets, usually kept 30 days; to keep them longer, route the namespace's logs to a bucket and set `OPEN_RL_LOG_VIEWS` to its view, which needs `roles/logging.viewAccessor`. Kubernetes queries use `OPEN_RL_WORKER_NAMESPACE`, then `POD_NAMESPACE`, then `openrl-system`. On GKE, the metadata server supplies project, cluster, and location; override them with `OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER`, and `OPEN_RL_GKE_LOCATION`, or set `OPEN_RL_GKE_TELEMETRY=false`.

Run `make test` for local regression checks.
