# Cluster dashboard

The gateway serves `/dashboard`. The UI and agents read the same namespaced data:

- `GET /api/v1/dashboard/snapshot`: all recorded runs (including completed runs), pods, scheduler Workloads, nodes, resource usage, events, rollout state, DRA mappings and observation coverage.
- `GET /api/v1/dashboard/runs/{run_id}`: logical run, current pods, runtime membership and workload state.
- `GET /api/v1/dashboard/runs/{run_id}/metrics`: last 2000 finished operations with outcome, queue and execution durations, numeric runtime metrics, logical run/runtime identity, and available trace/pod/node identifiers.
- `GET /api/v1/dashboard/runs/{run_id}/logs`: retained container output. Supports `q`, `pod`, `container`, `node`, `severity`, `attempt`, `since`, `until`, `limit` and `cursor`. Pagination is newest-first and excludes records collected after its first page. Keep filters unchanged when following `next_cursor`.
- `GET /api/v1/dashboard/pods/{pod}/logs`: bounded current or previous container output for any pod in the gateway namespace, including infrastructure pods. Supports `container`, `previous`, and `tail`.
- `GET /api/v1/dashboard/allocations/{workload_uid}/metrics`: optional DCGM history for the allocation's physical GPUs.

`/docs` describes the query parameters. Treat `available`, errors, timestamps and coverage as data: missing observations are not evidence of zero activity. The API does not provide arbitrary exec, filesystem access, secrets or cluster mutation.

## Run and device identity

FFT workloads use the logical run ID. LoRA workloads use a shared base-model runtime; multiple logical runs can therefore refer to the same pod. Both the UI and log API explicitly identify shared runtime output. Run membership uses Workload ownership, never a pod-name prefix. A caller can supply `name` in the create-model payload; IDs remain the stable identity and long names wrap or truncate in the UI.

The Nodes page shows observed allocation history. GPU lanes use DRA ResourceClaim results and ResourceSlice device identities. Counts without a device mapping remain separate, labeled allocations. They never become invented GPU indices. Single-GPU rows are 30px (26px bars), multi-GPU lanes 22px, with counts at the right in both cases. GPU selectors remain buttons in the full-width expansion.

Allocation history is bounded to 30 minutes and 361 samples per gateway process. One observation preceding the window is retained to cover its left edge. History begins when this gateway starts observing, and does not bridge polling gaps longer than 30 seconds. Allocated time is not a measurement of GPU execution duty.

## GPU metrics

Set `OPEN_RL_PROMETHEUS_URL` to an operator-controlled Prometheus endpoint reachable by the gateway. The collector queries `DCGM_FI_DEV_GPU_UTIL` and `DCGM_FI_DEV_FB_USED`, matching `UUID` against DRA device UUID attributes. `OPEN_RL_GPU_DRA_DRIVER` defaults to `gpu.nvidia.com`. Configure it explicitly for a different GPU driver. Other DRA device classes are not counted as GPUs.

Missing metrics, missing UUID mappings, and query failures stay unavailable. Device measurements include other workloads sharing that GPU. The all-GPU chart averages available device samples; memory sums the latest selected-device samples. MFU is not inferred from utilization: the gateway currently lacks the model FLOP/throughput/hardware precision evidence needed for a trustworthy value. An explicit `mfu` response can supply a fractional `value`, `estimated`, `scope`, and `device_count`; otherwise the UI shows unknown.

Trainer, vLLM sampler, and LoRA sampler requests use one `observe_operation` wrapper in `server/observability.py`. It records succeeded/failed/cancelled outcomes, duration, queue delay when known, bounded numeric metrics, and logical run versus shared runtime identity. Propagated OpenTelemetry context supplies trace identifiers when present; Kubernetes workers supply pod UID and node through the Downward API. Queue delay uses gateway/worker wall clocks and is omitted if their timestamps are inconsistent. Payloads, generated tokens and exception messages are not copied into operation samples. Instrumentation is enabled by default.

Recording has a 100ms budget and cannot fail a training request. Redis appends and trims atomically, preserving concurrent writers; in-memory storage has the same bounded contract. The last 2000 finished operations per run remain best effort, and hard process termination can prevent recording. Samples use the `open_rl:operations:` key prefix; older dashboard sample keys are not migrated. Rebuild worker images as well as the gateway for this reporting. Sampling-specific throughput and current in-flight operation tracking are not provided.

## Logs and access

This implementation assumes one trusted operator identity, as does the current gateway. Anyone able to reach it can read the configured namespace's operational data and logs. Keep the existing gateway private. A Tailscale-only bind limits the listener to tailnet access, subject to tailnet ACLs; it is not per-user authorization. No Ingress, LoadBalancer or public tunnel is created here.

The background collector polls every 15 seconds, including previous container attempts. Each run sweep reads at most 64 sources, rotating across larger source sets. Shared-runtime sources are read once per sweep and archived for every associated run. Each source is bounded to 500 lines / 128 KiB. Plain text is preserved, with pod UID, container, restart attempt and timestamps. Repeated overlapping tails are deduplicated. Sources report collection failures and tail limits.

The SQLite archive defaults to `/tmp/open-rl-dashboard/logs.sqlite3`, with restrictive file permissions. It keeps at most 20,000 records and seven days from collection, pruning during collection. Collected logs survive worker pod deletion, but the default archive does not survive gateway pod replacement. For durable collection, set `OPEN_RL_LOG_ARCHIVE` on a local/RWO persistent volume and use one gateway replica; do not put this SQLite database on shared NFS/Lustre. This is a bounded troubleshooting archive, not a replacement for a cluster logging service. Rotation, deletion and restarts between polls can leave gaps.

## Deployment and validation

For fictional data with the exact production UI, run `make dashboard-demo` and open `http://127.0.0.1:9017/dashboard`. `DASHBOARD_HOST` and `DASHBOARD_PORT` override the loopback listener. The demo is a development fixture server, never a fallback for unavailable live data; it cannot query a cluster or store. `/preview.html` redirects to the same UI. Fixture data stays under `dev/`, outside the gateway package.

The portable `k8s/deploy/base` includes read-only dashboard RBAC. Its LoRA/FFT/Kind/GKE overlays inherit it. Namespace pod/log/claim access is separate from cluster-wide read access to nodes and ResourceSlices. Legacy single-process/shared-storage manifests do not inherit the portable base; install the dashboard RBAC for their service account if using those deployments. Kubernetes clients use in-cluster credentials, falling back to local kubeconfig when run outside a cluster.

Live dashboard in Kind, isolated from application workloads:

```sh
make dashboard-kind
kubectl --context kind-open-rl-dashboard -n openrl-dashboard-check port-forward svc/dashboard-check-gateway 9020:8000
```

Open `http://127.0.0.1:9020/dashboard` after forwarding the service. This target deploys the gateway; it does not start a GPU scheduler or train a model. Use a new `VERSION` and matching Kind overlay tag for every rebuilt image.

Real GKE/GPU execution has not been validated in this workspace.

## GKE telemetry

The gateway automatically detects its GKE project, cluster name, and regional or zonal location from the in-cluster metadata server. No cluster identifiers are required for an ordinary GKE deployment. Kind and gateways running outside Kubernetes keep local telemetry by default. Metadata discovery is bounded to one second per concurrent request and failed discovery retries after a minute.

Optional environment overrides:

- `OPEN_RL_GKE_TELEMETRY=auto` (default), `false` to disable, or `true` to require GKE telemetry.
- `OPEN_RL_GKE_PROJECT`, `OPEN_RL_GKE_CLUSTER`, `OPEN_RL_GKE_LOCATION`: override detected values, or supply all three for a gateway outside GKE.

The gateway uses Application Default Credentials, including GKE Workload Identity. Its identity needs `roles/logging.viewer` and `roles/monitoring.viewer` on the telemetry project. Detection does not grant permissions or enable GKE logging/monitoring collection. The snapshot's `telemetry_sources.gke` reports detected configuration; individual queries report permission or source failures explicitly.

Run logs accept `source=auto|local|gke`. Auto uses GKE when detected; local selects the bounded SQLite archive. Cloud errors remain visible rather than silently switching sources. Logs are scoped to the project, cluster, location, namespace, and the run's observed pod/container lifetimes. Standard GKE log labels do not verify pod UID or restart attempt: responses say so, and attempt filtering requires the local source. Shared LoRA runtime output remains shared across its logical runs. Deleted pods can be queried while their collected source identity remains archived; sources archived before lifetime tracking cannot be safely correlated.

Run and allocation metrics accept `since` and `until` (ISO timestamps, up to seven days, default last 30 minutes). Cloud Monitoring supplies accelerator duty cycle, GPU memory, CPU time and container memory where collection is enabled. CPU time is cumulative CPU seconds. Allocation GPU charts require an exact accelerator UUID match; missing mapping stays unavailable. Prometheus remains the preferred allocation source when configured. MFU is not inferred from GPU utilization. Cloud history does not reconstruct scheduler allocation history, which remains the gateway's bounded in-memory observation window.

Process rows and log sources show scheduler-reported Trainer/Sampler roles. Unknown roles stay unknown.


## Agent inspection workflow

Health is the human triage page; its Diagnostic JSON link exposes the same full snapshot used by agents. Removing the separate Diagnostics navigation entry does not remove an API or field.

Start with the snapshot, locate the logical run and scheduler Workload UID, then inspect the run's pods and shared-runtime membership. Use run logs for archived output, pod logs for current/previous container attempts (including gateway or scheduler pods), and run/allocation metrics for observed performance. Preserve source errors, timestamps, missing-data coverage and shared-runtime flags when drawing conclusions. Container summaries include declared resources, current state and previous termination details; scheduler summaries include owner IDs, claim reservations and reported placement conditions.

The gateway exposes read-only operational data in its configured namespace, plus cluster node/device inventory permitted by its Kubernetes RBAC. An agent needs network access to the gateway. Cloud history also depends on the gateway's Google IAM permissions. This interface does not grant pod exec, filesystem access, secrets, or configuration changes. It does not currently expose full in-flight request tracing or unlimited historical data.

## Job labels and execution state

The Overview table uses base model plus the short logical model ID as its primary label. Optional caller-supplied names and recipe names appear underneath. The gateway preserves bounded string fields `name`, `run_name`, `recipe_name`, and `git_rev` from SDK session `user_metadata`; per-model `user_metadata` overrides those defaults, and explicit top-level fields take precedence. Sessions receive distinct IDs so recipe labels do not leak between clients. Cookbook recipes attach `recipe_name` to the ServiceClient session. Direct-client examples generally do not supply it. Earlier runs cannot recover discarded metadata.

Recorded lifecycle `status` remains available to agents. `display_status` summarizes current Kubernetes evidence (Running, Starting, Queued, Needs attention, Unassigned, or Unknown); recorded Completed/Failed states take precedence. An active metadata record alone is not proof of a running process. Overview shows training kind, completed optimizer steps, and elapsed time since creation (including queued time; terminal jobs stop at their recorded completion timestamp); zero steps is valid for a sampler.

## UI and agent entry points

Navigation is Overview, Nodes, Scheduler, Health. Nodes has a shared time picker for allocation bars and the expanded GPU graph: recent presets, custom UTC dates, and previous/next windows. Custom windows are limited to 24 hours. Allocation observations currently retain 30 minutes; hatched areas explicitly identify times outside retained history. Duty is allocated GPU-seconds divided by node capacity over the selected range; incomplete observations or device mappings show unknown. It measures allocation, while the expanded graph measures observed utilization. GPU metrics use the selected since/until bounds against the configured telemetry source. Run inspection uses the model and short run ID as its heading, with optional caller-supplied names underneath. Metrics and Logs share a time-range control. A compact list shows at most three recent significant signals, such as OOMKilled, worker restarts, container crash backoff, eviction, or scheduling and volume failures. Container status supplies OOM terminations and worker restarts when their actual termination/start timestamps are available. Restart counters are lifetime totals. Repeats are deduplicated by pod, container and reason; Kubernetes event counts and last observed timestamps are preserved. Normal lifecycle events and checkpoint saves are not promoted into this list. Signals without a recorded timestamp are not assigned a guessed time, and all raw events remain available to agents.

Each significant event links to logs within two minutes either side, bounded by the selected range. This narrower window is shown next to an All logs reset link and affects logs only. Metrics requests continue to use the full chosen time range for completed-operation samples and cloud metrics.

Agents can start at `GET /api/v1/dashboard` for endpoint templates, scope, capabilities, limits, and an inspection workflow, then follow `/openapi.json` for the full query schema. This contract remains available regardless of visible UI tabs.

Frontend code is divided into interaction orchestration (`app.js`), pure page renderers (`views.js`), formatting helpers (`ui.js`), and significant-event filtering (`timeline.js`). These are browser-native modules with no frontend runtime dependencies.
