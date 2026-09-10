# Operations dashboard

The gateway serves a cluster-operations dashboard at **`/dashboard`** (e.g. `http://localhost:9003/dashboard`
after `make server`). It is for operating and understanding the cluster and following live run progress. Bounded worker metric
trends provide immediate context; W&B owns long-term training history and experiment analysis.

## Views

- **Cluster** — a scrollable fleet overview with running-job and queue totals, observed GPU claims and
  node readiness, and a direct link to current health issues. A job placement board groups each run's
  exact pod membership with its nodes, pod states, and current operation. Problems sort first. Click a
  job to inspect the run or a pod to open live logs. Runs without visible pods are explicitly labeled.
  Compute pools below show nodes and their pods; control-plane services are in an expandable section.
  GPU claims are scheduler allocation, not device utilization; unavailable claim data is shown as unknown.

  Clicking a pool header opens its allocation history, stacked by job, with a node list below.
  The bounded history is sampled while the gateway is polled (at most once per 5s, last 120 samples).
  Claims are attributed using exact job labels. Hover to see each job's share; device utilization lives
  in DCGM. Back, Escape, or the Cluster tab returns to the overview.

  When the `openrl.io/v1alpha1` placement API from the GPU scheduler is installed, the control column
  shows Workload phases, ClaimLedger and seat totals. Missing scheduler CRDs are treated as an
  optional feature being off; installed-but-unreadable CRDs are a health error. The same column reports
  desired-versus-ready state for Deployments, DaemonSets, StatefulSets, and Jobs, including partial RBAC
  visibility instead of silently treating unreadable controllers as absent.
- **Runs** — search by name, ID, model, or node; filter by running, queued work, or needs attention.
  Problems sort first. Each row shows current operation and execution age, placement, queue wait,
  request failures, and the latest two reported worker metrics with compact trends. Expand inspection
  for all available metric charts, time bounds, ranges, and inspectable raw samples. Charts use actual
  report timestamps and auto-scaled axes; fewer than two samples never draws a trend.
  Launch by base model without leaving the page. Every row has an observed lifecycle verdict
  and an expandable inspection showing queue depth, current GPU claims by pool, pod phase counts,
  gateway request counts/failures/latency, latest worker-reported loss and gradient metrics with their
  bounded recent trend, measured CPU/memory, declared requests and limits, structured diagnostics,
  and links to the run log workspace. Clicking a pod opens that workspace filtered to the pod. A
  W&B link appears when one is recorded, and Stop appears when there is something to stop (a worker
  process, queued work, or labeled pods).
- **Health** — current problems first, then a **Load** section of measured stats (active runs, queued
  requests per run, oldest request and worker-launch wait, gateway request throughput/p95 latency/5xx rate,
  Redis memory and clients, gateway RSS, disk free, pod and
  GPU totals, optional Metrics Server CPU/memory usage, CPU/memory scheduling reservations, rollout health, scheduler workload phases, and ClaimLedger seats), then gateway / storage / Kubernetes /
  scheduler / visibility checks. Node `MemoryPressure` and `DiskPressure` conditions surface under
  Problems. Failed or slow placement, stale observed generations, assignment/seat mismatches, and stale
  ClaimLedger seats include exact `kubectl` inspection commands. Failed Jobs and stalled or unavailable
  workload controllers retain their Kubernetes conditions and include exact `kubectl describe` commands.
  Gateway traffic expands into application, background, and diagnostic summaries plus the busiest, slowest,
  or failing normalized routes, so the p95 and 5xx tiles have visible evidence without exposing request data.

The page polls one coherent snapshot every 8 seconds. Filters, selected run, and open log panels
survive refreshes. An expanded run inspection also refreshes every cycle with a visible collection time. A refresh lists Kubernetes state once, so every view describes the
same observation while avoiding redundant API-server work. Observations are single-flight cached for one
second across browser tabs, focused API calls, and agents; every response says whether the observation was
live or cached, its age, total collection latency, and the pod/node/event/scheduler/metrics timing breakdown.
Collections slower than one second become a warning; tune that with `OPEN_RL_K8S_WARN_MS`. Set
`OPEN_RL_K8S_CACHE_SECONDS=0` to disable the cache.
Manual refresh and a light/dark toggle are in
the top bar. The top bar and gateway card show the exact image build revision; pod inspection includes
the runtime image digest, so an agent can prove a rebuilt fix is actually deployed rather than trusting a tag.

Gateway HTTP telemetry is a bounded five-minute in-process ring. It records only method, normalized FastAPI
route template, status, traffic group, timestamp, and latency—never URL parameters, request IDs, headers,
queries, or bodies. Diagnostic polling and long-poll/background routes are reported separately so they do not
distort application p95 or 5xx alerts. Responses expose the ring capacity, lifetime dropped-sample count, and
whether an overflow truncated the current window, so high traffic cannot silently produce partial statistics.
Tune the application latency warning with
`OPEN_RL_HTTP_LATENCY_WARN_SECONDS` and the observation window with `OPEN_RL_HTTP_WINDOW_SECONDS`.

## JSON API and ops CLI

The UI serves humans; the same primitives are exposed as JSON for agents and scripts.
**Copy diagnostic JSON** exports the loaded cluster snapshot or run inspection (with log records exported separately from the log workspace)
without a separate agent-only interpretation. **Copy run link** opens directly into inspection via
`/dashboard?run=<run_id>`. Exported inspection includes its fetch timestamp; cluster observations retain
the API freshness metadata. Treat copied logs with the same care as the source logs.

| Primitive | Endpoint | CLI |
| --- | --- | --- |
| diagnose | `GET /api/v1/dashboard/snapshot` | `make ops diagnose` |
| health | `GET /api/v1/dashboard/health` | `make ops health` |
| problems | `GET /api/v1/dashboard/problems` | `make ops problems` |
| health gate | `GET /api/v1/dashboard/problems` | `make ops check` |
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
The problems payload includes exact error/warning/total counts and an `ok`, `warn`, or `error` status.
`make ops check` prints that same JSON and exits nonzero whenever the total is nonzero, making it suitable
for agents, CI steps, and shell health gates without parsing display text.
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
coherent diagnostic snapshot with no reported problems. It also proves HTTP traffic grouping against a
real FastAPI route, exercises pod logs and the stop
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

## Agent investigation workflow

1. Run `make ops diagnose` to collect the schema-versioned snapshot, observation freshness, problems,
   and build identity. Check visibility errors before concluding that resources are absent.
2. Run `make ops run <run_id> 120` to inspect exact pod placement, active operation and execution age,
   queue wait, worker metrics, scheduler evidence, and recent logs together.
3. Follow the diagnostic's structured `actions` for the affected resource. After a restart, use
   `make ops logs <pod> --container <name> --previous` to inspect the previous container.
4. After a fix, collect a fresh snapshot, verify the deployed build/image digest, and run
   `make ops check`. This health gate exits nonzero for warnings as well as errors.

The dashboard does not infer training progress percentages or successful completion from an empty queue.
Worker metric series contain at most 20 samples per metric; they are not a durable experiment store.

For a local demo browser check, start the gateway with `OPEN_RL_DASHBOARD_DEMO=1`, then run:

```bash
BASE_URL=http://127.0.0.1:9003 uv --project src/server run --no-sync --with playwright python dev/tools/dashboard-demo-smoke.py
```

## Run logs

Open **Runs → Inspect → Logs** for combined trainer/sampler container output. Pod links on
placement rows open the same workspace filtered to that pod. Infrastructure pods retain the direct
pod viewer. Search message text or filter pod, container, node, severity, attempt, and time range.
Both current and available previous container instances are collected. The browser displays records
oldest first; Load older pages back, and Follow refreshes the latest bounded window every 15 seconds.
The browser holds at most 2,000 records. Clicking a worker metric chart opens a ±2-minute log window;
the “Logs near latest sample” button provides the equivalent keyboard-accessible action.
Copy link preserves applied filters. Copy JSON exports records, source status, coverage, and the query.
Click a log source to expand its full pod/container/node identity, attempt, rank, timestamp, and original output.
Kubernetes events are shown separately; they describe currently discoverable events and are not archived.

Humans and agents currently share **one trusted operator identity**. All dashboard callers can read
any available pod logs in `OPEN_RL_WORKER_NAMESPACE` (or the service account's namespace), subject to
the gateway's Kubernetes permissions. Run IDs and filters organize data; they are not authorization
boundaries. There is no new login or token requirement. Shared gateway/scheduler logs remain accessible
through the pod endpoint; they are not automatically mixed into run logs without run/request correlation.
Container logs are application output and may contain sensitive data; no automatic redaction is applied.

```bash
make ops run-logs RUN_ID
# Use the CLI directly for flags and quoted search expressions (avoid Make argument parsing).
uv --project src/server run --no-sync python dev/tools/ops.py run-logs RUN_ID --q 'CUDA' --severity ERROR
uv --project src/server run --no-sync python dev/tools/ops.py run-logs RUN_ID --pod POD --container trainer --attempt 0
uv --project src/server run --no-sync python dev/tools/ops.py run-logs RUN_ID --since 2026-09-09T12:00:00Z --until 2026-09-09T12:05:00Z
uv --project src/server run --no-sync python dev/tools/ops.py run-logs RUN_ID --cursor CURSOR --archive-only
```

`GET /api/v1/dashboard/runs/{run_id}/logs` accepts `q`, `pod`, `container`, `node`, `severity`,
`attempt`, `since`, `until`, `limit` (1–1,000), `cursor`, and `refresh` (default true).
Responses contain `schema_version`, `records`, `next_cursor`, `sources`, `collection`, `coverage`,
`discovery_available`, `collector_error`, and current `events`. API ordering is newest first.
Each record includes timestamp, exact run ID, pod/UID, container, node, role, zero-based attempt,
severity, message, truncation flag, and rank/request ID when present in structured JSON output.
Unknown severity and missing timestamps remain explicit. Time filters omit records without timestamps.
Cursors preserve the query's collection high-water mark and ordering; repeat the same filters when
paginating. New collection does not enter existing pages. Retention can still remove older records.
An unknown or never-collected run returns an empty collection, not evidence that the run succeeded.

The gateway starts a best-effort collector that polls discovered run containers, sleeping 15 seconds
after each sweep. Collection also occurs on a fresh log query. Each run collection reads at most 64
container instances with up to 8 concurrent reads, 500 lines and 128 KiB per source. Messages are capped
at 4,096 characters. Source failures and bounded tails are reported individually. Large runs rotate source selection between
background sweeps; explicit pod/container/node filters also restrict fresh collection to those sources. Kubernetes discovery
failures are reported instead of being interpreted as an empty cluster.

Collected logs are stored in SQLite at `OPEN_RL_LOG_ARCHIVE`, defaulting to
`$OPEN_RL_TMP_DIR/logs.sqlite3` (`/tmp/open-rl/logs.sqlite3` by default), with mode 0600.
Demo data uses a separate `.demo` database and never reads real Kubernetes logs.
Retention keeps at most 20,000 records globally and removes records collected more than seven days ago
on the next collection. A busy run can therefore shorten other runs' retained history. This is a
bounded local archive, not a lossless logging backend: rotation, pod deletion, rapid restarts, source
limits, and polling gaps can lose output before it is collected. `coverage.history_complete` is always
false. Sources include last collection timestamps so stale observations remain identifiable.

Use a persistent volume for `OPEN_RL_LOG_ARCHIVE` if logs must survive gateway replacement. The supplied
deployment has not been changed to provision one. Run a single gateway collector against a local
SQLite volume; do not share it over a network filesystem or across replicas. Set
`OPEN_RL_LOG_COLLECTOR=0` to disable background collection (explicit queries still collect).
A centralized collector and durable log backend are needed for full-fleet retention at scale.
