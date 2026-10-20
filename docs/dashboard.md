# Cluster dashboard and inspection API

`/dashboard` is an operator UI over a read-only JSON inspection API under `/api/v1/inspect`. Agents use the same routes; `/docs` and `/openapi.json` describe them.

## Run it

```sh
npm --prefix dev/dashboard ci  # once; installs Svelte and esbuild
make dashboard                 # builds static/app.js and serves on port 8080
```

The standalone process is `uv run --extra cluster python -m server.dashboard`. Point it at the API server's Redis (`REDIS_URL`) and worker namespace. Without shared Redis, a separate dashboard has its own empty in-memory store and cannot see another server's runs. Rebuild with `make dashboard-build` after changing `src/server/dashboard/ui`; when packaging a wheel, build it before `uv build`.

On GKE with Managed Prometheus CRDs installed, `kubectl apply -k k8s/deploy/dashboard` deploys it and its PodMonitoring into `openrl-system` with the existing `open-rl-sa`. It uses the API-server image, whose Dockerfile builds the UI in a Node stage. Set the image tag in that directory's `kustomization.yaml`.

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
| `GET /run_metrics` | Distinct training steps reported by a client, ordered by step |

`activity`, `series` and `logs` take ISO `since` and `until` bounds, up to seven days. An omitted end uses inventory's `observed_at`; an omitted start uses the preceding 30 minutes. Activity returns the resolved bounds for subsequent queries and a coverage note: missing completed records do not establish that no requests arrived or that workers were idle. Response schemas are generated from the same models used to validate responses; extra diagnostic fields are preserved. Invalid operation records are skipped and reported in `partial`, separately from `error`, which means a source could not be read.

`series?kind=samplers&run_id=...` reads only the eight sampler metrics; `kind=hardware` reads GPU and worker metrics. The default `all` retains both. Each field reports its own error; a top-level error means all attempted queries failed. Telemetry shares one connection pool and makes no hidden HTTP retries. Inspection requests have a 12-second budget, below the browser's 15-second timeout.

Concurrent Prometheus requests with identical queries, bounds and credentials share one fetch within a dashboard process. Completed results and failures are discarded. Cancelling one viewer does not cancel another viewer's request; shared requests have a 10-second deadline.

Log cursors preserve the original bounds, pod lifetimes and provider query for 15 minutes. Continuations need no inventory or Redis reads. Reuse the same filters and bounds, or keep omitted bounds omitted. Set the same `OPEN_RL_LOG_CURSOR_SECRET` on dashboard replicas to keep cursors valid across replicas and restarts; otherwise each process uses an ephemeral secret. Deleted sampler identities are recoverable only while their operation records remain retained.

## Client metrics and Prometheus

Custom loops can attach the small [cookbook OpenTelemetry adapter](../examples/README.md#custom-loop-telemetry)
to export session/model/step observations and Prometheus gauges through standard
OTLP exporters. It requires a collector; the dashboard does not yet read that
history. The existing gateway/Redis path below is separate. Untouched cookbook
recipes have server/GKE telemetry without client metric forwarding by default.

Send batches to the gateway with `POST /api/v1/runs/{run_id}/metrics`, or run the companion beside an unchanged cookbook example:

```sh
python -m server.scripts.forward_metrics /path/to/run --base-url http://gateway:8000 --run-id RUN_ID --follow
```

The forwarder must be started explicitly where the files live, with the OpenRL modules and `httpx` installed. It reads `metrics.jsonl` every 15 seconds and sends batches of 200 rows. Metric names are discovered from finite numeric values; no per-metric registration is needed. Records require an integer `step` or `progress/batch`. The dashboard's metric filter finds custom names. Direct API callers send `{"rows": [{"step": 12, "metrics": {"custom_score": 0.8}}]}`.

`RUN_ID` must be an existing model ID. If omitted, the forwarder waits for the first model path in `checkpoints.jsonl`. This still assumes one model per source file; repeated steps replace entire rows, and string fields such as `phase` are discarded.

By default it also sends `logs.log` to `POST /api/v1/runs/{run_id}/logs`; use `--no-logs` for metrics only. The gateway emits structured client logs for Cloud Logging and retains the last 5,000 lines in its run store for local inspection. Log forwarding can duplicate lines after retries or restarts.

Each batch atomically replaces rows by step and retains the highest `OPEN_RL_OPS_SAMPLE_LIMIT` distinct steps. Retrying an acknowledged batch cannot evict other steps. The forwarder filters nonfinite values before JSON encoding and retries only unacknowledged batches.

History responses include an ETag. The browser revalidates its cached history: unchanged polls return `304` without reading, decoding or transferring the rows, and keep the existing chart data. Changed batches invalidate the revision, including corrections to older steps; replaying unchanged rows keeps it. Changed histories return the full retained rows. The standalone dashboard uses gzip level 1 for large responses.

The dashboard's `/metrics` endpoint exposes the highest reported step as `openrl_run_metric{run_id="...",metric="..."}`, plus `openrl_run_step` and `openrl_run_metrics_received_timestamp_seconds`. Step numbers are values, never labels. Each scrape reads one latest-step snapshot per indexed run; dashboard charts read the step history directly. By default, export includes the first 64 metric names alphabetically from rows received within six hours (`OPEN_RL_METRICS_EXPORT_LIMIT` and `OPEN_RL_METRICS_EXPORT_HOURS`). Metrics absent from that row are not exported. The dashboard overlay includes its PodMonitoring; other Prometheus installations can scrape port 8000 at `/metrics`. History expires with the configured operation TTL.

Earlier development builds used a list format. History reads fall back to those records when keyed step history is absent; replay the run's `metrics.jsonl` to populate keyed storage and Prometheus export.

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
