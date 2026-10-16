# Cluster dashboard and inspection API

`/dashboard` is an operator UI over a read-only JSON inspection API under `/api/v1/inspect`. Agents use the same routes; `/docs` and `/openapi.json` describe them.

## Run it

```sh
npm --prefix dev/dashboard ci  # once; installs Svelte and esbuild
make dashboard                 # builds static/app.js and serves on port 8080
```

The standalone process is `uv run --extra cluster python -m server.dashboard`. Point it at the API server's Redis (`REDIS_URL`) and worker namespace. Rebuild with `make dashboard-build` after changing `src/server/dashboard/ui`; when packaging a wheel, build it before `uv build`.

On Kubernetes, `kubectl apply -k k8s/deploy/dashboard` deploys it into `openrl-system` with the existing `open-rl-sa`. It uses the API-server image, whose Dockerfile builds the UI in a Node stage. Set the image tag in that directory's `kustomization.yaml`.

The inspection API lists every run in the fleet, while the client-facing API server only answers for model IDs a client already holds. So the API server does not mount it by default. For a trusted local setup, run `make dashboard-build`, then `OPEN_RL_INSPECTION_API=true make server`.

## Pages

- **Overview**: runs with status, search and status filters.
- **Run**: status, workers, saved configuration, activity timeline, pod incidents, op metrics, sampler charts, pods and logs. Click the timeline to open the logs around that time. A shared pool's sampler charts include other runs on that base model.
- **Nodes**: which worker process held which GPUs and when, per-GPU activity, and GPU and CPU charts. On the timelines and charts, drag or press A/D to pan; use Ctrl/⌘ + wheel, pinch, or W/S to zoom.
- **Health**: pod and node problems, and unavailable sources.

Recent query responses are kept in a bounded, in-memory cache and shown while refreshing. Reloading clears this cache. Logs load alongside activity; incidents appear below the panel so their arrival does not move the log controls.

## API

| Route under `/api/v1/inspect` | Returns |
|---|---|
| `GET /inventory` | Runs with status, progress, workers, pods and scheduler workloads, each with its placement decision; nodes, devices and current placements; claim reservations; an error field per unavailable source |
| `GET /activity` | One lane per worker process: its GPU turns and completed ops, in Unix seconds. With `run_id`: the run, its op samples and pod incidents |
| `GET /series` | GPU utilization and memory per device UUID, and worker CPU and memory, for one `node_id` or one `run_id`. With `run_id`, also the run's vLLM samplers: token throughput, running and waiting requests, KV cache use, preemptions, and p99 time to first token and request latency |
| `GET /logs` | A run's Cloud Logging records, with `q`, `pod`, `node` and `severity` filters and cursor paging |
| `GET /pods/{pod}/logs` | Current or previous container output from the kubelet |

`activity`, `series` and `logs` take ISO `since` and `until` bounds, up to seven days. An omitted end uses inventory's `observed_at`; an omitted start uses the preceding 30 minutes. Activity returns the resolved bounds for subsequent queries and a coverage note: missing completed records do not establish that no requests arrived or that workers were idle. Response schemas are generated from the same models used to validate responses; extra diagnostic fields are preserved. Invalid operation records are skipped and reported in `partial`, separately from `error`, which means a source could not be read.

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

## Agents

The [openrl-inspect skill](../skills/openrl-inspect/SKILL.md) shows agents how to diagnose runs with this API.

For local regression checks, run `make test`; query lifecycle checks run with `node --test dev/dashboard/query.test.mjs` after installing the dashboard dependencies.

`uv run --extra cluster python dev/inspection_eval.py --port 9042` serves ten deterministic cases through the real inspection routes without cluster access. Give an agent only a printed case URL, run ID, symptom, and the skill; the fixture source contains the answer key. Score diagnosis, cited evidence, uncertainty, next check, and whether it obeyed instructions embedded in logs.

Run-scoped logs join trainer, sampler and pool dispatcher pods, not the client Job; the raw pod-log route can read a known client pod.
