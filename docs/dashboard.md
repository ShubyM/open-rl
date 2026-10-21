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

Locally, worker log history needs Cloud Logging, nodes and pod logs need Kubernetes, and hardware and sampler charts need a metrics backend. Forwarded client logs keep a tail without Cloud Logging. A missing source shows as an unavailable-source message, never as an empty cluster.

## Pages

- **Overview** lists runs with status, search, and status filters.
- **Run** shows status, settings, workers, an activity timeline, incidents, metrics, sampler charts, and logs. Click the timeline to open the logs around that time.
- **Nodes** shows accelerator nodes: which process held which GPU and when, each node's GPU duty cycle, and GPU and CPU charts. Drag or press A/D to pan; use Ctrl/⌘ + wheel, pinch, or W/S to zoom.
- **Health** lists pod and node problems and unavailable sources.

Pages show their last response while they refresh. A failed source keeps its last data for 30 seconds, then reports its error.

## API

| Route under `/api/v1/inspect` | Returns |
|---|---|
| `GET /inventory` | Runs with status, progress, workers, pods, and scheduler workloads; nodes, devices, and placements; an error per unavailable source |
| `GET /activity` | One lane per worker process with its GPU turns and completed ops. With `run_id`, also the run, its incidents, and chart samples: every trainer op and the slowest sampler request per 1/1000 of the window |
| `GET /series` | GPU utilization and memory per device UUID, worker CPU and memory, for a `node_id` or `run_id`; with `run_id`, the run's vLLM sampler series. `kind=devices` without a node or run returns every node's GPUs |
| `GET /logs` | Worker and forwarded client logs from Cloud Logging, with filters and cursor paging; the client-log tail when Cloud Logging is unconfigured |
| `GET /pods/{pod}/logs` | Current or previous container output from the kubelet |
| `GET /telemetry` | OpenTelemetry gauges `openrl_client_metric` and `openrl_server_metric` from Prometheus, for a run or node |
| `GET /history` | A run's client metrics by step, for runs recorded before OpenTelemetry |

`activity`, `series`, and `logs` take ISO `since` and `until` bounds of up to seven days. An omitted end uses inventory's `observed_at`, and an omitted start the 30 minutes before it. Missing op records do not prove that no requests arrived or that workers were idle. Invalid records are skipped and reported in `partial`; `error` means a source could not be read.

Requests have a 12-second budget, below the browser's 15-second timeout. Identical concurrent Prometheus queries share one fetch. Log cursors keep the original bounds, pod lifetimes, and query for 15 minutes; set the same `OPEN_RL_LOG_CURSOR_SECRET` on every replica to keep cursors valid across replicas and restarts.

## Metrics from code

Inside a worker op, add measurements to the op's record:

```python
from server import telemetry

telemetry.metrics({"tokens_per_second": tokens / elapsed})
```

The op record keeps the latest value per name, up to 500 names, and works inside `asyncio.to_thread`. Outside an op, record a gauge with an explicit scope:

```python
telemetry.record_metrics({"queue_depth": depth}, model_id=model_id)
telemetry.record_metrics({"cache_entries": count}, runtime_id=shared_runtime_id)
```

Gauges export only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, up to 2,048 name and scope pairs per process (`OPEN_RL_SERVER_METRIC_SERIES_LIMIT`). The Collector exposes them to Prometheus with the original name in the `metric` label. Model, session, and runtime values stay separate, and a session-only value shows as session-wide on its models. Charts use sampled timestamps, not training steps.

## Client logs

A client sends its output with `POST /api/v1/runs/{run_id}/logs`, where `run_id` is the model ID it trains. The gateway writes each line as structured JSON for Cloud Logging, labelled with the run, and keeps the last 5,000 lines in its store. Beside an unchanged cookbook example, run:

```sh
python -m server.scripts.forward_logs /path/to/run --base-url http://gateway:8000 --run-id RUN_ID --follow
```

It reads `logs.log` every 15 seconds and, without `--run-id`, waits for the first model path in `checkpoints.jsonl`. Retries and restarts can duplicate lines.

## Progress

Each active run has a `progress` verdict, judged by what completes:

- **OK**: requests are completing.
- **STALLED**: a request has been on a trainer, in the run's sampling pool, or in the trainer queue for longer than three median steps, and at least 10 minutes.
- **DEAD**: a process that stopped reporting holds requests, either the trainer that popped them or the dispatcher of their pool.
- **IDLE**: nothing is queued or running and nothing has completed for that long.

`progress.outstanding` lists unanswered requests with their age and holder; `progress.queued` counts the trainer queue; `progress.dispatcher` is the pool dispatcher's record, with replica weight versions, loads in flight, and its last error.

## Sources

| Data | Source and retention |
|---|---|
| Runs and sessions | API-server metadata in Redis |
| Worker processes | Each trainer and dispatcher rewrites a record every 5 seconds: pod, node, GPU UUIDs, phase, unanswered requests, and recent GPU turns. A record unchanged for 30 seconds belongs to a gone process. Samplers are plain `vllm serve` and keep none |
| Ops and step counts | Recorded by workers, newest `OPEN_RL_OPS_SAMPLE_LIMIT` (2,000) per run. The dispatcher records each sample on the replica that answered |
| Nodes, pods, devices, scheduler | Kubernetes API, cached for 5 seconds |
| GPU, worker, and sampler series | DCGM, container, and vLLM metrics from `OPEN_RL_PROMETHEUS_URL`, or Cloud Monitoring's PromQL API on GKE |
| Run logs | Cloud Logging, scoped to the run's pods and their lifetimes; a stderr line takes the level it prints |

Records expire `OPEN_RL_OPS_TTL_DAYS` (default 7) after their last write; `0` keeps them. Logs come from the project's default buckets, usually kept 30 days; to keep them longer, route the namespace's logs to a bucket and set `OPEN_RL_LOG_VIEWS` to its view, which needs `roles/logging.viewAccessor`. Kubernetes queries use `OPEN_RL_WORKER_NAMESPACE`, then `POD_NAMESPACE`, then `openrl-system`. On GKE, the metadata server supplies project, cluster, and location; override them with `OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER`, and `OPEN_RL_GKE_LOCATION`, or set `OPEN_RL_GKE_TELEMETRY=false`.

Run `make test` for local regression checks.
