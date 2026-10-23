# Cluster dashboard and inspection API

`/dashboard` is an operator UI over a read-only JSON API under `/api/v1/inspect`. Agents use the same routes, and `/docs` and `/openapi.json` describe them.

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
- **Run** shows status, settings, an activity timeline with GPU rows and duty, metrics by step, sampler charts, and logs. Click the timeline to open the logs around that time.
- **Nodes** shows accelerator nodes: which process held which GPU and when, each node's GPU duty cycle, and GPU utilization and memory. Drag or press A/D to pan; use Ctrl/⌘ + wheel, pinch, or W/S to zoom.
- **Health** lists pod and node problems and unavailable sources.

Pages show their last response while they refresh.

## API

| Route under `/api/v1/inspect` | Returns |
|---|---|
| `GET /inventory` | What the pages list: runs with their lifecycle, nodes with their GPUs, the GPUs held now, pods with problems, and an error per unavailable source |
| `GET /activity` | One lane per placement, fleet-wide, a node's or a run's, with its worker's GPU turns and completed ops in the window |
| `GET /query` | A named PromQL query (`gpu_busy`, `gpu_util`, `gpu_memory_mib`, `vllm_*`, ...) over a run's or node's GPUs and pods, or the fleet's GPUs, at `points` resolution |
| `GET /accelerators` | One row per GPU: node, slot, model, holder, utilization now and over the window, memory used of total |
| `GET /runs` | Every live run in one line, longest since its last op first; `ended=true` adds ended runs |
| `GET /runs/{id}/summary` | The run's facts: pace, goodput, unanswered requests and their holders, the request ledger, failures and relaunches with log reads, saves, optimizer, samplers, config, placement, incidents and GPU duty by role |
| `GET /runs/{id}/commands` | The run's raw command records, trainer ops and samples, newest first; `role` and the bounds filter them |
| `GET /runs/{id}/steps` | Trainer ops grouped by step, with the client's metrics; `fields` keeps client metrics by prefix |
| `GET /sessions/{id}/summary` | The session record, whether it is live, its SDK errors, and a line per model |
| `GET /logs` | Worker and forwarded client logs from Cloud Logging, with filters and Cloud Logging's page token as the cursor |

`activity`, `query`, `accelerators` and `logs` take ISO `since` and `until` bounds of up to seven days; `since` also takes a span such as `1h`. An omitted end is now, and an omitted start the 30 minutes before it. Missing op records do not prove that no requests arrived or that workers were idle. `error` means a source could not be read. A log cursor is valid only with the same filters, `since` and `until` as the page that returned it.

## Client metrics by step

A training loop that uses the cookbook logger sends its metrics by appending `OTelLogger` (`examples/common/otel_logger.py`); see [custom loop telemetry](../examples/README.md#custom-loop-telemetry). Each `log_metrics` call emits one `openrl.training.metrics` log event carrying the step and every finite metric, and updates the `openrl_client_metric` gauges. The client sends both to an OpenTelemetry Collector named by `OTEL_EXPORTER_OTLP_ENDPOINT`.

On GKE, `kubectl apply -k k8s/deploy/telemetry` deploys the Collector as `open-rl-collector` on port 4318 in `openrl-system`, with no public endpoint; port-forward it for a client outside the cluster. It forwards the log events to the API server's `POST /api/v1/otlp/logs`, which keeps one row per step per session (`open_rl:client_steps:<session>`), so `/steps` reads them with no metrics backend and no step reconstruction. It also exports the gauges to Managed Prometheus. Another collector needs this exporter in its logs pipeline, with gzip off because the route does not decompress request bodies:

```yaml
exporters:
  otlphttp/openrl:
    logs_endpoint: http://open-rl-api-server-service:8000/api/v1/otlp/logs
    encoding: json
    compression: none
service:
  pipelines:
    logs:
      exporters: [otlphttp/openrl]
```

Runs with no forwarded events fall back to the gauges in Prometheus, aligned by `progress/batch`, which can miss a step when two steps land within one gauge export interval.

Locally, `docker compose -f dev/monitoring/compose.yaml up -d` starts a Collector and Prometheus. Set `OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318` on the client and `OPEN_RL_PROMETHEUS_URL=http://127.0.0.1:9090` on the dashboard. The local Collector cannot reach `open-rl-api-server-service`, so local runs read the gauges.

## Client logs

A client sends its output with `POST /api/v1/runs/{run_id}/logs`, where `run_id` is the model ID it trains. The API server writes each line to stdout as structured JSON labelled with the run, so Cloud Logging keeps it beside the run's worker logs and `/logs` returns it.

## Lifecycle

Each run in `/inventory` has a `lifecycle`, which says where it is, not whether it is healthy:

- **ended**: the API server marked it finished, its workers stopped, or it has no workers and no live session.
- **waiting**: a worker is not placed yet, or the server has not created one for a live session.
- **placed**: every worker is placed.
- **active**: the scheduler cannot be read, so placement is unknown.

The pages label a run from its lifecycle. A run's facts, for deciding whether it is healthy, are `/runs/{id}/summary`.

## Sources

| Data | Source and retention |
|---|---|
| Runs and sessions | API-server metadata in Redis |
| Worker processes | Each trainer and sampler rewrites a record every 5 seconds: pod, node, GPU UUIDs, phase, unanswered requests, and recent GPU turns. A record unchanged for 30 seconds belongs to a gone process |
| Ops and step counts | Recorded by workers, newest `OPEN_RL_OPS_SAMPLE_LIMIT` (2,000) per run, for trainer ops and samples separately |
| Placements | The API server lists the scheduler's Workloads every 30 seconds and keeps one record per placement in `open_rl:placements:<model_id>`: pod, node, GPU UUIDs, placed and released times. |
| Nodes, pods, scheduler | Kubernetes API: one watch per kind keeps them in memory from the first read or server start, and lists again every 5 minutes and whenever a watch fails |
| Each node's GPUs | DCGM's UUID and slot labels, from the metrics source below |
| GPU, worker, and sampler series | DCGM, container, and vLLM metrics from `OPEN_RL_PROMETHEUS_URL`, or Cloud Monitoring's PromQL API on GKE |
| Run logs | Cloud Logging: the worker pods with the run's `timeslice.io/owner` label, and its forwarded client lines; a stderr line takes the level it prints |

Records expire `OPEN_RL_OPS_TTL_DAYS` (default 7) after their last write; `0` keeps them. Logs come from the project's default buckets, usually kept 30 days; to keep them longer, route the namespace's logs to a bucket and set `OPEN_RL_LOG_VIEWS` to its view, which needs `roles/logging.viewAccessor`. Kubernetes queries use `OPEN_RL_WORKER_NAMESPACE`, then `POD_NAMESPACE`, then `openrl-system`. On GKE, the metadata server supplies project, cluster, and location; override them with `OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER`, and `OPEN_RL_GKE_LOCATION`, or set `OPEN_RL_GKE_TELEMETRY=false`.

Run `make test` for local regression checks.
