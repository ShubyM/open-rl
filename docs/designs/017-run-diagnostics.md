# Design Doc 017: Run diagnostics

**Status**: Proposed
**Date**: 2026-10-21
**Target branch**: `diagnostics`, stacked on `svelte-dashboard` (PR #23)
**Instrumentation ledger**: [`docs/instrumentation-ledger.md`](../instrumentation-ledger.md)

---

## 1. Summary

OpenRL's telemetry exists to answer one question for a person or an agent: why is this run not behaving? This design reorganizes it around that question.

- **Record where each process ran, and which step each command belonged to.** Every diagnostic question then narrows to *run, place, window, step*, and each existing store answers with one bounded query.
- **Answer with checks.** Fifteen server-side checks, derived from 84 incidents in this project's history, each return a verdict, a one-line reason, a few numbers and an evidence handle, in a few hundred bytes.
- **Serve the dashboard and agents from the same generic endpoints.** Named PromQL queries, log search, command records, step rows and checks. The Svelte UI composes its views from them instead of the backend shaping data per page.

The backend shrinks from 2,770 lines (`src/server/telemetry.py` and `src/server/dashboard/*.py` today) to a target of about 1,000.

## 2. Problem

Measured on 2026-10-21 against the Qwen3.8-27B LoRA run `43b3ebb3` (37 steps over 33 hours, one 8-GPU trainer and eight samplers):

| Fact | Measurement |
|---|---|
| The training signal never reaches the server | Workers report `loss` and `grad_norm` only. Reward, KL, eval scores, stop reasons and step time stay in the client's `metrics.jsonl`. The OpenTelemetry metric path has had 0 series in 30 days |
| Command records carry no step, and samples evict training ops | 1,211 records per step, 98% of them sample requests with empty metrics. At the default cap of 2,000 records the run list holds about 1.65 steps |
| The place a run ran is reconstructed, not recorded | About 600 lines rebuild it from Kubernetes pods, DRA device slices, scheduler Workloads and ledgers, worker records, and old command records. vLLM samplers record nothing |
| Responses are shaped for drawing | `/activity` for the run is 827 KB; the Activity tab downloads 411 KB of series to show 16 Duty percentages, averaged in the browser |
| NaN reads as 0 | `sanitize_float` rewrites NaN to 0.0 before telemetry can flag it (`src/training/trainer_worker.py:344-349`) |
| Lost requests read as IDLE | A request lost between the Redis push and the trainer pop makes `progress` report "client is not sending work" |
| Noise dominates cost | 24.7 GB of logs for the run, about 99% of sampler lines being `/metrics` access logs, stored in two buckets; vLLM scraped twice |

## 3. Goals and non-goals

**Goals**
- Every run can be diagnosed from the server: is it moving, are requests accounted for, is the reward real, are workers healthy and placed where they fit.
- Every answer is small: under 5 KB for all checks of a run, well under the 25,000-token default Claude Code applies to tool responses.
- One set of endpoints serves the dashboard and agents.
- Works on GKE and on a local stack; a check whose inputs are absent answers `unknown`.
- The backend gets smaller.

**Non-goals**
- A new datastore. Cloud Logging, Cloud Monitoring, Redis and Kubernetes are enough.
- Per-request distributed tracing. Request detail is kept by exception.
- Replacing the Svelte dashboard with Grafana. The run-centric views are the ones used to debug.

## 4. Design

### 4.1 Stores

| Store | Holds | Kept |
|---|---|---|
| Cloud Monitoring (Managed Prometheus) | DCGM per GPU UUID, vLLM per pod, `openrl_gpu_held` (4.3) | 24 months, full detail for a week |
| Redis | Live state only: command records (trainer and sampler lists), worker records, ledger counters, client heartbeat | 7 days; a flush loses the live view, never the record |
| Cloud Logging | Every pod's stdout, including the new JSON records | 365 days in `openrl-runs` |
| Kubernetes | Pods, events, exit reasons, Workloads | While objects exist |

### 4.2 The join key

Every question narrows to **run, place, window, step**.

- *Place* is the pod (name and UID), node and GPU UUIDs, from the placement record.
- *Window* is `[t0, t1]` of a command or a step, from command records.
- *Step* is the run's `optim_step` ordinal, stamped on every command record.

With those known, Prometheus is queried by GPU UUID or pod, Cloud Logging by pod label and window, Redis by run and step, Kubernetes by run.

### 4.3 Records

| Record | Writer | Where | Size | Checks it feeds |
|---|---|---|---|---|
| Placement: run ids, role, runtime, pod and UID, node, GPU UUIDs, start, end | The scheduler, on bind and release; workers echo it at start | Workload status; `openrl_gpu_held{run_id, role, pod, UUID}` info metric; one JSON log line per change | About 300 B per process lifetime | `placement`, `worker-health`, every check's place |
| Step ordinal on every command record | Trainer | Redis command records | About 8 B per record | `progress`, `optimizer-health`, per-step views |
| Sampler command records in their own list | Dispatcher | `open_rl:operations:<run>:samples` | Unchanged | `sampler-flow`; keeps trainer ops under the cap |
| Ledger counters: enqueued, popped, answered, polled, with holder incarnation | API server, trainer | Redis | Tens of bytes per run | `request-ledger` |
| Effective config echo | Every trainer and sampler at start | One JSON log line; latest in the worker record | About 1 KB per process | `effective-config` |
| Exit and OOM reasons | Kubernetes; workers add cgroup `oom_kill` and `memory.peak` | Pod status, worker record | Tens of bytes | `worker-health` |
| GPU utilization and memory from NVML | Workers, in the 5 s worker record | Redis | About 20 B per GPU | GPU checks without Prometheus (local stacks) |
| JSON worker log lines with `run_id`, `request_id`, `step`, `kind` | Trainer, dispatcher, API server | stdout → Cloud Logging | A few fields per line | Every check's evidence |
| Client step row (reward, KL, eval, stop reasons, `time/*` phases) and heartbeat | Cookbook client: JSON on stdout in GKE; the OTel Collector from a laptop | Cloud Logging; latest heartbeat in Redis | About 5 KB per step | `reward-validity`, `episode-endings`, `policy-movement`, `client-liveness`, `dependencies` |

NaN and infinity are written as `null` and named in `nonfinite`.

The placement info metric does the correlation in PromQL. For example, a run's GPU utilization by role is

```promql
DCGM_FI_DEV_GPU_UTIL * on(UUID) group_left(run_id, role) openrl_gpu_held{run_id="$run"}
```

and the presence of `openrl_gpu_held` over time is the hold interval, which draws the Nodes timeline.

### 4.4 Read API

Generic endpoints under `/api/v1/inspect`, shared by the UI and agents:

| Route | Returns | Bound |
|---|---|---|
| `GET /query?name=&run=&node=&since=&until=&points=` | A named PromQL template from a server-side table (`run_gpu_util`, `run_gpu_busy`, `run_vllm_tokens`, `node_gpu_holders`, …), with `step = window / points` | `points` ≤ 1,000 per series |
| `GET /logs?run_id=&q=&severity=&pod=&since=&until=` | The existing log route already searches by text, severity and pod | `limit` |
| `GET /runs/{id}/commands?since=&until=&role=` | Command records from Redis, raw | `limit` |
| `GET /runs/{id}/steps?last=&fields=` | Trainer ops grouped by step, with stored client metrics (prefix filter) | 20 steps by default |
| `GET /runs/{id}/checks?only=` | Check results | About 4.7 KB for all 15 |
| `GET /runs/{id}/summary` | State, step, step time, last reward, per-GPU duty | About 1 KB |
| `GET /inventory` | Runs and fleet, slimmed to what pages list | As today, smaller |

Named queries keep one copy of each PromQL expression for checks and UI, and keep agent calls bounded.

### 4.5 Checks

A check reads the stores for one run, applies thresholds taken from incidents, and returns:

```json
{"check": "request-ledger", "verdict": "fail",
 "reason": "3 forward_backward held by trainer incarnation 7f2c (restarted 2.4 d ago)",
 "numbers": {"outstanding": 3, "oldest_s": 207360, "median_step_s": 1450, "lost": 3},
 "evidence": "open_rl:worker:7f2c… outstanding[0]; pod harvey-trainer-0 lastState=Error(134)",
 "data_age_s": 4}
```

Verdicts are `ok`, `warn`, `fail` and `unknown`. `unknown` carries the age of the stalest input whenever an input is missing or older than two collection intervals.

| Check | Question | Incidents |
|---|---|---|
| `progress` | Is the run completing optimizer steps at its normal pace? | 23 |
| `worker-health` | Has any worker crashed, restarted, OOMed or started slowly? | 12 |
| `sampler-flow` | Are samplers answering at the expected speed with the expected weights? | 11 |
| `effective-config` | Is each worker running the code and settings the run asked for? | 10 |
| `request-ledger` | Is every request accounted for, held by a live worker, younger than 3 median steps? | 9 |
| `placement` | Does every needed worker exist, on a GPU that fits it? | 8 |
| `dependencies` | Are the judge and sandboxes healthy enough to grade? | 8 |
| `client-liveness` | Is the client alive, and waiting on the server or on itself? | 7 |
| `episode-endings` | How did episodes end, and is a failure bucket growing? | 7 |
| `reward-validity` | Is this step's reward real? | 6 |
| `control-plane` | Are the API server, dispatcher and client event loops responsive? | 6 |
| `optimizer-health` | Are gradients finite and in range? | 5 |
| `policy-movement` | Is the policy moving, from the weights the run claims? | 5 |
| `storage` | Will the volume last, and how much would a crash lose? | 4 |
| `ownership` | Is every GPU-holding Workload owned by a live session? | 3 |

Together they cover 83 of 84 recorded incidents; the `unknown` rule covers the last (missing telemetry). Thresholds are per-run overridable.

### 4.6 The UI

The Svelte dashboard becomes a client of 4.4:

| View | Built from |
|---|---|
| Overview | `/inventory`, `/runs/{id}/summary`, check verdict badges |
| Run · Activity | `/runs/{id}/commands` for lanes; `/query?name=run_gpu_busy&points=1` for Duty |
| Run · Metrics | `/runs/{id}/steps` charted by step; `/query` for vLLM and GPU charts at chart width |
| Run · Logs | `/logs/search` |
| Nodes | `/query?name=node_gpu_holders`, `/query` per GPU |
| Health | `/runs/{id}/checks` across active runs |

### 4.7 GKE and local

Checks and endpoints are written against four source functions: metrics (PromQL), logs, placements and commands. GKE provides Managed Prometheus, Cloud Logging and Workload status. A local stack provides a plain Prometheus if present, JSON log files, and placement records written by the local launcher. Missing inputs make checks answer `unknown`.

In the cluster, credentials come from the metadata server and the mounted service account, so no credential code is needed beyond library defaults. Laptop access to the cluster is a small adapter.

## 5. Size budget

| Module | Today | Target |
|---|---|---|
| `sources.py` | 811 | 300 |
| `telemetry.py` | 467 | 250 |
| `snapshot.py` | 383 | 80 |
| `router.py` | 351 | 150 |
| `schema.py` | 305 | 100 |
| `inspection.py` | 239 | 60 |
| `metrics.py` | 99 | 70 |
| `collector.py` | 80 | 0 |
| `checks.py` | 0 | 250 |
| `__main__.py` | 35 | 35 |
| **Total** | **2,770** | **about 1,300**, stretch 1,000 |

`dev/diagnostics_budget.py` measures this and the payload sizes on every PR in the stack.

## 6. Rollout

Each PR is one row in the instrumentation ledger, with the signal it adds, the checks it serves, and a measured before and after.

1. **Harness.** Line and payload budget script; baseline screenshots of every page against the Qwen run.
2. **Correctness.** NaN as `null`; sampler records in their own list; step ordinal on command records; lost-request detection.
3. **Placement.** Workload status gains GPU UUIDs and start and end times; the scheduler exports `openrl_gpu_held`; workers echo placement and config.
4. **Generic reads.** `/query` with named templates, `/logs/search`, `/runs/{id}/commands`, `/runs/{id}/steps`.
5. **Checks.** The eight with partial data today first, then `/checks` and `/summary`.
6. **UI migration.** Page by page onto the generic reads.
7. **Delete.** Placement reconstruction, `/telemetry`, `/history`, OTel metric gauges, per-page schema.
8. **Client signals.** Step row and heartbeat, then the client-side checks.
9. **Noise.** Sampler access-log flag (`--disable-access-log-for-endpoints /metrics,/weight_info`), one log bucket, one sampler PodMonitoring. Cluster changes need an owner's approval.

## 7. Risks and open questions

- **Scheduler change.** GPU UUIDs in Workload status and the info metric are Go changes to `scheduler/controller` and a CRD update, deployed to a shared cluster.
- **Historical runs.** Runs recorded before placement records exist keep today's reconstruction until they age out, or are backfilled once.
- **Thresholds.** Taken from incidents; they need per-run overrides and the dashboard should show which fired.
- **Checks on request or on a timer.** On request is simpler; a timer enables alerts.
- **Client contract.** The step row needs a small cookbook logger sink; the laptop path needs a Collector exporter that writes one log entry per event.

## 8. Alternatives considered

- **Events plus SQL as the record of truth.** All history as log events joined by SQL after upgrading the log bucket to Observability Analytics. Capable, but adds a Collector exporter, a bucket upgrade and a SQL client before any check works. Kept as a later addition for cross-run questions.
- **Step history in Redis.** Simplest to build, but Redis is live state, not a log or a history store.
- **Grafana for all charts.** About 850 backend lines, but loses the Activity lanes and step-keyed training charts. The named-query endpoints keep the option open: Grafana can query the same Prometheus series.
