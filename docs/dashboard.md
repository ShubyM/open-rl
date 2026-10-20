# Cluster dashboard and inspection API

`/dashboard` is an operator UI over a read-only JSON inspection API under `/api/v1/inspect`. Agents use the same routes; `/docs` and `/openapi.json` describe them.

## Run it

```sh
npm --prefix dev/dashboard ci  # once; installs Svelte and esbuild
make dashboard                 # builds static/app.js and serves on port 8080
```

The standalone process is `uv run --extra cluster python -m server.dashboard`. Point it at the API server's Redis (`REDIS_URL`) and worker namespace. Without shared Redis, a separate dashboard has its own empty in-memory store and cannot see another server's runs. Rebuild with `make dashboard-build` after changing `src/server/dashboard/ui`; when packaging a wheel, build it before `uv build`.

On GKE, `kubectl apply -k k8s/deploy/dashboard` deploys it into `openrl-system` with the existing `open-rl-sa`. It uses the API-server image, whose Dockerfile builds the UI in a Node stage. Set the image tag in that directory's `kustomization.yaml`.

The inspection API lists every run in the fleet, while the client-facing API server only answers for model IDs a client already holds. So the API server does not mount it by default. For a trusted local server without Redis, run `make dashboard-build`, then `OPEN_RL_INSPECTION_API=true make server` and open `http://127.0.0.1:9003/dashboard`. This shares the server's in-memory run and operation data. Missing Kubernetes does not prevent that data from refreshing.

To view a deployed dashboard from a laptop, use `kubectl -n openrl-system port-forward svc/open-rl-dashboard 8080:8000`, then open `http://127.0.0.1:8080/dashboard`. Its data sources and cloud credentials stay in the cluster.

Local inspection currently has narrower coverage: worker log history requires Cloud Logging, nodes and pod logs require Kubernetes, and hardware/sampler charts require a configured metrics backend. Forwarded client logs have a retained tail available without Cloud Logging. `OPEN_RL_PROMETHEUS_URL` supports appropriately labelled DCGM and vLLM series, but worker CPU/memory queries currently use GKE system metric names. The UI still shows unavailable-source messages for unconfigured features.

## Pages

- **Overview**: runs with status, search and status filters.
- **Run**: status, workers, saved configuration, activity timeline, pod incidents, op metrics, sampler charts, pods and logs. Click the timeline to open the logs around that time. A shared pool's sampler charts include other runs on that base model.
- **Nodes**: which worker process held which GPUs and when, per-GPU activity, and GPU and CPU charts. On the timelines and charts, drag or press A/D to pan; use Ctrl/⌘ + wheel, pinch, or W/S to zoom.
- **Health**: pod and node problems, and unavailable sources.

Recent query responses are kept in a bounded, in-memory cache and shown while refreshing. Reloading clears this cache. Logs do not poll activity. Failed metrics fields retain their last samples with a stale note; successful fields continue updating. Inventory sources refresh independently; a failed source keeps its last successful data during a 30-second grace period, after which its error is reported.

## API

| Route under `/api/v1/inspect` | Returns |
|---|---|
| `GET /inventory` | Runs with status, progress, workers, pods and scheduler workloads, each with its placement decision; nodes, devices and current placements; claim reservations; an error field per unavailable source |
| `GET /activity` | One lane per worker process: its GPU turns and completed ops, in Unix seconds. With `run_id`: the run, its op samples and pod incidents |
| `GET /series` | GPU utilization and memory per device UUID, and worker CPU and memory, for one `node_id` or one `run_id`. With `run_id`, also the run's vLLM samplers: token throughput, running and waiting requests, KV cache use, preemptions, and p99 time to first token and request latency |
| `GET /logs` | Worker and forwarded client logs from Cloud Logging, with filters and cursor paging; a retained client-log tail when Cloud Logging is unconfigured |
| `GET /pods/{pod}/logs` | Current or previous container output from the kubelet |
| `GET /telemetry` | Discovered client/server gauges from the collector's Prometheus backend; scoped to a run or node, optionally filtered by metric name |

`activity`, `series` and `logs` take ISO `since` and `until` bounds, up to seven days. An omitted end uses inventory's `observed_at`; an omitted start uses the preceding 30 minutes. Activity returns the resolved bounds for subsequent queries and a coverage note: missing completed records do not establish that no requests arrived or that workers were idle. Response schemas are generated from the same models used to validate responses; extra diagnostic fields are preserved. Invalid operation records are skipped and reported in `partial`, separately from `error`, which means a source could not be read.

`series?kind=samplers&run_id=...` reads only the eight sampler metrics; `kind=hardware` reads GPU and worker metrics. The default `all` retains both. Each field reports its own error; a top-level error means all attempted queries failed. Telemetry shares one connection pool and makes no hidden HTTP retries. Inspection requests have a 12-second budget, below the browser's 15-second timeout.

Concurrent Prometheus requests with identical queries, bounds and credentials share one fetch within a dashboard process. Completed results and failures are discarded. Cancelling one viewer does not cancel another viewer's request; shared requests have a 10-second deadline.

Log cursors preserve the original bounds, pod lifetimes and provider query for 15 minutes. Continuations need no inventory or Redis reads. Reuse the same filters and bounds, or keep omitted bounds omitted. Set the same `OPEN_RL_LOG_CURSOR_SECRET` on dashboard replicas to keep cursors valid across replicas and restarts; otherwise each process uses an ephemeral secret. Deleted sampler identities are recoverable only while their operation records remain retained.

## Collector metrics

New numeric metrics need no frontend changes. Inside a worker operation:

```python
from server.telemetry import ops

ops.metrics({"tokens_per_second": tokens / elapsed})
```

This adds values to the existing completed-operation record, inheriting its
model, worker and timestamp. It works inside `asyncio.to_thread`, keeps concurrent
operations separate, and leaves the operation's return value unchanged. Repeated
names keep their latest value; metrics explicitly returned by the operation take
precedence. Existing `result["metrics"]` values also continue to work.

Outside an operation, use an explicit scope:

```python
from server.telemetry.metrics import record_metrics

record_metrics({"queue_depth": depth}, model_id=model_id)
record_metrics({"cache_entries": count}, runtime_id=shared_runtime_id)
record_metrics({"process_memory_bytes": memory})  # pod/node identity from the process
```

Standalone observations go through the standard OTel SDK and never create fake
operations, increment training steps or affect stalled-run detection. Omit model
identity for shared resources. Optional `session_id` and `phase` identify wider
session scope and separate phases; step and request IDs are not metric labels.
Server metric export is enabled only when an OTLP endpoint is configured;
`OTEL_METRICS_EXPORTER=none` disables it explicitly.

Custom clients append the [cookbook logger adapter](../examples/README.md#custom-loop-telemetry)
once, then use their existing `logger.log_metrics(...)` calls. Unchanged upstream
cookbooks still have server telemetry without automatically forwarding arbitrary
client metrics.

The complete chart path is standard OTLP HTTP → Collector → Prometheus → the
inspection API. The Collector exposes `openrl_client_metric` and
`openrl_server_metric`; the original metric name is the `metric` label. The UI
discovers names, phases and producer identities without a chart registry. Model,
session and runtime values stay separate, including multiple producers reporting
the same name. Session-only observations appear as **session-wide** on associated
models; they are never inferred to belong exclusively to one model. Node details
show metrics carrying that node's resource identity.

Charts use sampled timestamps, not training steps. Updates between exports or
scrapes can be missed. Per-call client observations, including repeated steps,
remain in cookbook JSONL and exported OTel logs. The supplied Collector writes
OTLP JSON logs to stdout (collected by GKE logging in-cluster); the dashboard does
not query those logs as exact step history. The Collector expires the latest
value five minutes after its last update; historical chart retention belongs to
Prometheus (seven days in the local configuration).

### Local setup

Start the shared monitoring services once, independently of any training command:

```sh
docker compose -f dev/monitoring/compose.yaml up -d
```

Configure custom clients and the API server before starting them:

```sh
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_METRIC_EXPORT_INTERVAL=5000
export OPEN_RL_PROMETHEUS_URL=http://127.0.0.1:9090
```

Use the embedded dashboard with an in-memory server, or share `REDIS_URL` between
the API server and standalone dashboard. No cloud credentials are needed locally.
The server remains usable with export disabled or while the Collector is down;
operation records still appear through the existing local/Redis path. The local
Prometheus scrapes the Collector; hardware metrics require their usual exporters.

### Kubernetes setup

The optional overlay uses a stock Collector, a private Service and a GKE
PodMonitoring for its Prometheus endpoint:

```sh
kubectl apply -k k8s/deploy/telemetry
kubectl -n openrl-system set env deployment/open-rl-api-server \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://open-rl-collector:4318 \
  OTEL_METRIC_EXPORT_INTERVAL=5000
```

New workers inherit the API server's `OTEL_*` configuration. Existing worker
processes need restarting to pick up environment changes. The dashboard reads
Managed Prometheus through its GKE workload identity with `roles/monitoring.viewer`, or an explicit
`OPEN_RL_PROMETHEUS_URL`. The overlay does not create a public ingestion endpoint.
For external clients, configure the deployment's authenticated HTTPS ingress and
use standard `OTEL_EXPORTER_OTLP_HEADERS`; port-forwarding port 4318 is sufficient
for development. There is no forwarding command per client run.

### Visible limits

Each chart section renders 12 charts at a time and shows its counts. Search finds
arbitrary names; collector charts also page through producers of the same name.
Collector reads return at most 200 series, clearly indicate
truncation, and apply name filters at the backend so later names remain reachable.
Series preserve producer and phase identity instead of silently aggregating them.

The client adapter admits 500 distinct gauge series by default (`max_series`),
including metric/phase and step/phase combinations, and keeps 500 finite metrics
per log event. Overflow warns once per limit and reports dropped counts in events.
Worker `ops.metrics` collects 500 names per operation. Server gauge export admits
2,048 name/scope combinations per process (`OPEN_RL_SERVER_METRIC_SERIES_LIMIT`).
Worker overflow warns with retained/dropped counts; existing series keep updating.
Metric names are at most 200 characters. Keep names and scope labels stable.
These producer limits bound memory; the chart limit only bounds rendering.

## Client logs

A client's own output is sent with `POST /api/v1/runs/{run_id}/logs`, where `run_id` is the model ID it trains. The gateway emits each line as structured JSON for Cloud Logging, labelled with the run, and keeps the last 5,000 lines in its run store for a deployment without Cloud Logging. Beside an unchanged cookbook example, run:

```sh
python -m server.scripts.forward_logs /path/to/run --base-url http://gateway:8000 --run-id RUN_ID --follow
```

It reads `logs.log` every 15 seconds. Without `--run-id` it waits for the first model path in `checkpoints.jsonl`. Retries and restarts can duplicate lines. Client metrics go through OpenTelemetry, above.

## Progress

Each active run in inventory has a `progress` verdict, judged by what completes rather than by which processes are up:

- **OK**: requests are completing.
- **STALLED**: a request has been on a trainer, waited in the run's sampling pool, or sat in the trainer queue for longer than three median steps (at least 10 minutes).
- **DEAD**: requests are held by a process that stopped reporting: a trainer that popped them, or the dispatcher of the pool they wait in.
- **IDLE**: nothing is queued or running and nothing has completed for that long; the client is not sending work.

`progress.outstanding` lists what is not yet answered: the requests each trainer has popped, with their age and the trainer's phase (`starting`, `ready`, `waiting_gpu`, `on_gpu`, `exited`), and the samples waiting in the run's pool. A pool on a shared base model holds other runs' samples too. `progress.queued` counts requests still in the trainer queue, and `progress.dispatcher` is the pool dispatcher's own record: the revision it serves, each replica's weight version, replicas loading weights, requests in flight, its last commit and last error.

## Sources

| Data | Source and retention |
|---|---|
| Runs and sessions | API-server metadata in Redis |
| Worker processes | Each trainer and pool dispatcher rewrites a record about itself every 5 seconds: pod, node, GPU UUIDs, phase, unanswered requests and recent GPU turns; a dispatcher adds its replicas' weight versions and its last error. Samplers are plain `vllm serve` and keep no record. A record unchanged for 30 seconds belongs to a process that is gone. Kept 7 days; the UI reads the last 24 hours |
| Ops and step counts | Recorded by workers, best effort; the newest 2,000 per run, kept 7 days. Ops that serve no single run (init, wake_up, sleep) are kept under the runtime. The dispatcher records each sample it sends, on the replica that answered |
| Queues | The trainer queue's length, and each sampling pool's stream: its length and the arrival time of its oldest request |
| Nodes, pods, devices, placements, scheduler | Kubernetes API, cached for 5 seconds |
| GPU, worker and sampler series | DCGM, container and vLLM metrics from `OPEN_RL_PROMETHEUS_URL`, or Cloud Monitoring's PromQL API on GKE. The GKE overlay's PodMonitorings scrape pool replicas and routers. The sampler queries follow vLLM's and llm-d's Grafana dashboards |
| Run logs | Cloud Logging, scoped to the run's recorded pods and their lifetimes. GKE files all stderr as ERROR, so a stderr line takes the level it prints; progress bars are INFO |

Kubernetes queries use `OPEN_RL_WORKER_NAMESPACE`, then `POD_NAMESPACE`, then `openrl-system`. On GKE the project, cluster and location come from the metadata server. Override them with `OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER` and `OPEN_RL_GKE_LOCATION`, or disable discovery with `OPEN_RL_GKE_TELEMETRY=false`. Application Default Credentials need logging and monitoring read access. A source that cannot be read is reported as an error, never shown as an empty cluster.

Logs come from the project's default buckets, which usually keep 30 days. To keep runs longer, route the namespace's container logs to a bucket with longer retention and set `OPEN_RL_LOG_VIEWS` to its view, for example `projects/<project>/locations/global/buckets/openrl-runs/views/_AllLogs`; the reader then needs `roles/logging.viewAccessor` on it. Each run's op records and each worker's record expire `OPEN_RL_OPS_TTL_DAYS` (default 7) after their last write, and `0` keeps them. A run keeps its newest `OPEN_RL_OPS_SAMPLE_LIMIT` op records (default 2000); a long RL run records about one per sample request. Pages and the API show windows of up to seven days.

## Agents

The [openrl-inspect skill](../skills/openrl-inspect/SKILL.md) shows agents how to diagnose runs with this API.

For local regression checks, run `make test`; query lifecycle checks run with `node --test dev/dashboard/query.test.mjs` after installing the dashboard dependencies.

`uv run --extra cluster python dev/inspection_eval.py --port 9042` serves ten deterministic cases through the real inspection routes without cluster access. Give an agent only a printed case URL, run ID, symptom, and the skill; the fixture source contains the answer key. Score diagnosis, cited evidence, uncertainty, next check, and whether it obeyed instructions embedded in logs.

Run-scoped logs join trainer, sampler and pool dispatcher pods with forwarded client output. A client Job's pod is not discovered automatically; the raw pod-log route can read a known client pod.
