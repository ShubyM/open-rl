# Cluster dashboard and inspection API

`/dashboard` is an operator UI over a read-only JSON inspection API under `/api/v1/inspect`. Agents use the same routes; `/docs` and `/openapi.json` describe them.

## Run it

```sh
npm --prefix dev/dashboard ci  # once; installs Svelte and esbuild
make dashboard                 # builds static/app.js and serves on port 8080
```

The standalone process is `uv run --extra cluster python -m server.dashboard`. Point it at the API server's Redis (`REDIS_URL`) and worker namespace. Rebuild with `make dashboard-build` after changing `src/server/dashboard/ui`; when packaging a wheel, build it before `uv build`.

On Kubernetes, `kubectl apply -k k8s/deploy/dashboard` deploys it into `openrl-system` with the existing `open-rl-sa`. It uses the API-server image, whose Dockerfile builds the UI in a Node stage. Set the image tag in that directory's `kustomization.yaml`.

The inspection API lists every run in the fleet, while the client-facing API server only answers for model IDs a client already holds. So the API server does not mount it by default. For a trusted local setup, run `make dashboard-build`, then `OPEN_RL_INSPECTION_API=true make server`. Placement history is recorded only by the standalone process.

## Pages

- **Overview**: runs with status, search and status filters.
- **Run**: status, workers, saved configuration, activity timeline, pod incidents, op metrics, Grafana sampler panels, pods and logs. Click the timeline to open the logs around that time. Shared sampler charts include other runs on that runtime.
- **Nodes**: GPU placement history, per-GPU activity, and GPU and CPU charts. On the timelines and charts, drag or press A/D to pan; use Ctrl/⌘ + wheel, pinch, or W/S to zoom.
- **Health**: pod and node problems, and unavailable sources.

## API

| Route under `/api/v1/inspect` | Returns |
|---|---|
| `GET /inventory` | Runs with status, workers, pods and scheduler workloads, each with its placement decision; nodes, devices and current placements; claim reservations; an error field per unavailable source |
| `GET /activity` | GPU turns per placement, or op intervals where no turns were recorded, in Unix seconds. With `run_id`: the run, its op samples and pod incidents |
| `GET /series` | GPU utilization and memory per device UUID, and worker CPU and memory; optionally for one `node_id` |
| `GET /logs` | A run's Cloud Logging records, with `q`, `pod`, `node` and `severity` filters and cursor paging |
| `GET /pods/{pod}/logs` | Current or previous container output from the kubelet |

`activity`, `series` and `logs` take ISO `since` and `until` bounds, up to seven days. An omitted end uses inventory's `observed_at`; an omitted start uses the preceding 30 minutes. Activity returns the resolved bounds for subsequent queries and a coverage note: missing completed records do not establish that no requests arrived or that workers were idle. Response schemas are generated from the same models used to validate responses; extra diagnostic fields are preserved. Invalid operation or GPU-turn records are skipped with a partial-data error, retaining valid records.

## Sources

| Data | Source and retention |
|---|---|
| Runs and sessions | API-server metadata in Redis |
| Ops, step counts and GPU turns | Recorded by workers, best effort; the newest 2,000 per run or worker, kept 7 days |
| Nodes, pods, devices, placements, scheduler | Kubernetes API, cached for 5 seconds |
| Placement history | The standalone dashboard, every 10 seconds while running; kept 7 days; the UI reads the last 24 hours |
| GPU and worker series | DCGM and container metrics from `OPEN_RL_PROMETHEUS_URL`, or Cloud Monitoring's PromQL API on GKE |
| Run logs | Cloud Logging, scoped to the run's recorded pods and their lifetimes. GKE files all stderr as ERROR, so a stderr line takes the level it prints; progress bars are INFO |

Kubernetes queries use `OPEN_RL_WORKER_NAMESPACE`, then `POD_NAMESPACE`, then `openrl-system`. On GKE the project, cluster and location come from the metadata server. Override them with `OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER` and `OPEN_RL_GKE_LOCATION`, or disable discovery with `OPEN_RL_GKE_TELEMETRY=false`. Application Default Credentials need logging and monitoring read access. `OPEN_RL_GRAFANA_URL` turns on the Grafana proxy and the sampler panels. A source that cannot be read is reported as an error, never shown as an empty cluster.

## Agents

The [openrl-inspect skill](../skills/openrl-inspect/SKILL.md) shows agents how to diagnose runs with this API.

For local regression checks, run `make test` and `make dashboard-test`. The latter includes 2,000 seeded sampler-filter cases; the Python suite includes randomized activity attribution, malformed telemetry, and invalid query windows.

`uv run --extra cluster python dev/inspection_eval.py --port 9042` serves ten deterministic cases through the real inspection routes without cluster access. Give an agent only a printed case URL, run ID, symptom, and the skill; the fixture source contains the answer key. Score diagnosis, cited evidence, uncertainty, next check, and whether it obeyed instructions embedded in logs. Request traces go to `/tmp` rather than the repository. Results from the smaller-model trials are in [inspection_eval_results.json](../dev/inspection_eval_results.json).

### Client logs and an agent iteration loop

Run-scoped logs currently join trainer and sampler pods, not the client Job. The raw pod-log route can read a known client pod; `make cluster-e2e` also follows the client Job's stdout/stderr with `kubectl logs`. Cloud collection depends on cluster logging configuration. Laptop stdout and local log files are not automatically uploaded. Trace export is not a general client-log forwarder.

A basic sequential loop can use the existing launcher: edit/build an immutable image, launch with `make cluster-e2e`, follow client logs, inspect worker state, then check Job completion. The in-cluster client uses service DNS and needs no port-forward; an outside agent still needs access to the inspection API. `python scripts/run_cluster_e2e.py --print-only` renders the workflow without contacting a cluster.

Reliable concurrent loops need a small attempt contract: a unique attempt/Job ID, terminal status and exit code, and associations from the client pod UID to the model/run IDs it creates. The launcher currently deletes/reuses the fixed `open-rl-e2e-client` Job, so attempts can collide. These associations could extend the existing log sources with a client role; dashboard and agent would keep sharing the same read API. Launch/control tools would be a separate thin adapter. This attempt contract and laptop log ingestion are not implemented.

The six added blind cases cover missing client evidence, expired sessions, model loading, misleading stderr, pagination, and pod replacement. Luna 6 passed all six diagnoses; Luna 5.6 had four fully correct responses, one unsupported inference from absent activity, and one incorrect timestamp despite finding the right error. A fresh Luna 5.6 retest preserved uncertainty after adding the activity coverage note. These are synthetic smoke trials, not live-cluster deployment tests.
