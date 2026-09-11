# Cluster dashboard

The gateway serves `/dashboard`. The UI and agents read the same read-only JSON under `/api/v1/dashboard`:

- `GET /api/v1/dashboard/snapshot`: runs joined to their workers, current GPU placements, placement history, nodes with their DRA devices, pods, scheduler Workloads and claim ledgers, and per-source errors.
- `GET /api/v1/dashboard/runs/{run_id}`: one run with its pods, workloads and current placements.
- `GET /api/v1/dashboard/runs/{run_id}/metrics`: the worker's own operation records (outcome, duration, queue delay, numeric metrics such as loss) inside `since`/`until`.
- `GET /api/v1/dashboard/runs/{run_id}/logs`: Cloud Logging entries for every pod the run has owned, live or gone. Supports `q`, `pod`, `node`, `severity`, `since`, `until`, `limit`, `cursor`.
- `GET /api/v1/dashboard/pods/{pod}/logs`: current or previous container output from the kubelet for any pod in the namespace.
- `GET /api/v1/dashboard/allocations/{placement_id}/metrics`: DCGM utilization and memory for the allocation's GPUs, from Prometheus.
- `GET /api/v1/dashboard/experiments`: reward, correctness and optimizer curves read from each run's `metrics.jsonl` on the shared volume.

`/docs` describes the parameters. The API does not provide exec, filesystem access, secrets or cluster mutation.

## Where the data comes from

| Data | Source | Survives |
|---|---|---|
| Runs, steps, status | Redis run metadata written by the gateway | Redis |
| Operation metrics | Workers post one record per request; the gateway keeps the last `SAMPLE_LIMIT` per run in Redis | Redis |
| Current placements, nodes, pods, devices | Kubernetes API, read at most every 5 seconds | live |
| Placement history | The gateway records every live placement to Redis every 10 seconds (`open_rl:placement:*`, 7 days) | Redis |
| Run logs | Cloud Logging, scoped to the run's pod names and lifetimes | Cloud |
| GPU utilization and memory | DCGM through a Prometheus endpoint (`OPEN_RL_PROMETHEUS_URL`), matched by GPU UUID | Prometheus |
| Experiment curves | `metrics.jsonl` and `config.json` under `OPEN_RL_TMP_DIR/runs` | shared volume |

Nothing lives only in the gateway's memory. Replacing the gateway pod loses no history.

## Run and device identity

FFT workloads use the logical run ID. LoRA workloads use a shared base-model runtime, so several runs refer to one trainer and one sampler; the UI and API say so with `shared_runtime` and `runtime_run_ids`. Run membership uses Workload ownership, never a pod-name prefix. GPUs are identified by DRA device IDs from ResourceSlices and by their UUID for metrics; a GPU index is never inferred.

## GKE identity

The gateway detects its project, cluster and location from the metadata server (`OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER`, `OPEN_RL_GKE_LOCATION` override; `OPEN_RL_GKE_TELEMETRY=false` disables). It authenticates with Application Default Credentials. On GKE that means Workload Identity: the gateway's Kubernetes service account needs `roles/logging.viewer` on the project, bound to its `principal://` identity. A credential or scope refusal is remembered for ten minutes and reported as such.

Other clusters can keep the same pages by pointing the run-log source at a collector that answers the same shape; Kind installs without Cloud Logging still have pod logs from the kubelet through `pods/{pod}/logs`.

## GPU metrics

Set `OPEN_RL_PROMETHEUS_URL` to a Prometheus-compatible endpoint that scrapes the DCGM exporter. On GKE with Managed Prometheus that is a `prometheus-engine/frontend` deployment with `--query.project-id`, running as a service account with `roles/monitoring.viewer`. The exporter must be running on every GPU node; one started before the driver installer finishes logs "NVML doesn't exist" and exports nothing until restarted.

## Pages

- **Overview**: every recorded run with status, kind, completed steps and elapsed time.
- **Nodes**: one lane per node, one row per GPU, allocation bars over the selected window from the placement history. A GPU held by several allocations at once shows who actually held it, painted from the workers' operation records. Click a bar for the legend and the GPU utilization chart.
- **Scheduler**: workloads waiting for placement with the scheduler's reason, and claim reservations.
- **Experiments**: recipe metrics per run, grouped by sweep directory, with reward and correctness curves.
- **Health**: source errors, pod problems and unready nodes.
- **Run**: operation-timing charts and worker metrics, the process table, and Cloud Logging with search, pod filter and paging.

## Front end

Plain ES modules, no build step. `app.js` routes and polls; `store.js` holds the snapshot and selection; `cache.js` fetches anything else and re-renders when it lands; every page is a function from state to markup, patched into the document by `morph` so a refresh keeps scroll, focus and open panels. Charts are markup too: an SVG stretched to its box with HTML axes, so nothing is measured.
