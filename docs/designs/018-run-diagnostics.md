# Design Doc 018: Run diagnostics

**Status**: Proposed
**Date**: 2026-10-21
**Target branch**: `main`. The backend lands first; the Svelte UI (PR #23) follows separately
**Instrumentation ledger**: [`docs/instrumentation-ledger.md`](../instrumentation-ledger.md)

---

## 1. Summary

OpenRL's telemetry exists to answer one question for a person or an agent: why is this run not behaving? This design reorganizes it around that question.

- **Record where each process ran, and which step each command belonged to.** Every diagnostic question then narrows to *run, place, window, step*, and each existing store answers with one bounded query.
- **Answer with facts, not verdicts.** One read returns a run's joined facts (unanswered requests and who holds them, failed ops, worker relaunches, saves, sampler latency, config drift), each with a log read where one helps, in about 1 KB. The reader, a person or an agent, draws the conclusion.
- **Serve the dashboard and agents from the same generic endpoints.** Named PromQL queries, log search, command records, step rows and run facts. The Svelte UI composes its views from them instead of the backend shaping data per page.

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
- Every answer is small: under 3 KB for a run's facts, well under the 25,000-token default Claude Code applies to tool responses.
- One set of endpoints serves the dashboard and agents.
- Works on GKE and on a local stack; a fact whose inputs are absent is left out, never shown as healthy.
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
| Kubernetes | Pods, exit reasons, nodes, Workloads, read from one watch per kind in each process | While objects exist |

### 4.2 The join key

The top-level unit is the **session**: one client process, with its lifetime, logs, heartbeat and SDK errors. A session owns the models it created or attached; sessions and models are many-to-many, because a model resumed after a restart is attached to a new session. Under a session, every question narrows to **model, place, window, step**.

- *Session* is the client's `session_id`, with a durable record of when it started, when it was last heard from, and its models.

- *Place* is the pod (name and UID), node and GPU UUIDs, from the placement record.
- *Window* is `[t0, t1]` of a command or a step, from command records.
- *Step* is the run's `optim_step` ordinal, stamped on every command record.

With those known, Prometheus is queried by GPU UUID or pod, Cloud Logging by pod label and window, Redis by run and step, Kubernetes by run.

### 4.3 Records

| Record | Writer | Where | Size | Checks it feeds |
|---|---|---|---|---|
| Session record: started, last heard from, models used, client metadata (redacted) | API server on `create_session`, each heartbeat, and each model attach | `open_rl:session_record:<id>` (telemetry retention); one `openrl.session` JSON line per create and attach | About 300 B per session | `client-liveness`, `sdk-errors`, the session rollup |
| Placement: run ids, role, runtime, pod and UID, node, GPU UUIDs, start, end | The scheduler, on bind and release; workers echo it at start | Workload status; `openrl_gpu_held{run_id, role, pod, UUID}` info metric; one JSON log line per change | About 300 B per process lifetime | `placement`, `worker-health`, every check's place |
| Step ordinal on every command record | Trainer | Redis command records | About 8 B per record | `progress`, `optimizer-health`, per-step views |
| Sampler command records in their own list | vLLM sampler worker | `open_rl:operations:<run>:samples` | Unchanged | `sampler-flow`; keeps trainer ops under the cap |
| Ledger counters: enqueued, popped, answered, polled, with holder incarnation | API server, trainer | Redis | Tens of bytes per run | `request-ledger` |
| Effective config echo | Every trainer and sampler at start | One JSON log line; latest in the worker record | About 1 KB per process | `effective-config` |
| Exit and OOM reasons | Kubernetes; workers add cgroup `oom_kill` and `memory.peak` | Pod status, worker record | Tens of bytes | `worker-health` |
| GPU utilization and memory from NVML | Workers, in the 5 s worker record | Redis | About 20 B per GPU | GPU checks without Prometheus (local stacks) |
| JSON worker log lines with `run_id`, `request_id`, `step`, `kind` | Trainer, sampler, API server | stdout → Cloud Logging | A few fields per line | Every check's evidence |
| Client step event: the step and every finite metric the client logged for it | The client's OpenTelemetry logger, one `openrl.training.metrics` event per step; the OTel Collector forwards it to `POST /api/v1/otlp/logs` | `open_rl:client_steps:<session>`, one row per step | About 5 KB per step with every metric | `/runs/{id}/steps` |
| Client heartbeat and the client-side facts (reward validity, stop reasons) | Cookbook client | Latest heartbeat in Redis | Tens of bytes | `client-liveness`, `episode-endings`, `policy-movement`, `dependencies` |

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
| `GET /query?name=&run_id=&node_id=&since=&until=&points=` | A named PromQL template (`gpu_util`, `gpu_util_now`, `gpu_busy`, `gpu_memory_mib`, `gpu_memory_total_mib`, eight `vllm_*`) over a run's or node's GPUs and pods, or the fleet's GPUs, with `step = window / points` | `points` ≤ 1,000 per series |
| `GET /activity?run_id=&node_id=&since=&until=` | One lane per placement with its GPU turns and recorded op intervals, for the Activity and Nodes pages | The window |
| `GET /logs?run_id=&q=&severity=&pod=&since=&until=` | The existing log route already searches by text, severity and pod | `limit` |
| `GET /runs/{id}/commands?since=&until=&role=` | Command records from Redis, raw | `limit` |
| `GET /runs/{id}/steps?last=&fields=` | Trainer ops grouped by step, with the client's metrics (prefix filter): the forwarded step events, else its `openrl_client_metric` OTel gauges aligned by `progress/batch`, read incrementally and cached for 30 s | 20 steps by default |
| `GET /runs/{id}/summary` | The run's facts (section 4.5) and GPU duty by role over its span | 0.6 to 1.5 KB on the 8 recorded runs |
| `GET /runs?ended=` | Every live run in one line (lifecycle, steps, seconds since the last op and step, unanswered requests, how many a dead worker holds, failed ops, relaunches), longest since its last op first | About 240 B per run |
| `GET /sessions/{id}/summary` | The session record, whether its registry key is live, its SDK errors, and each model in the one-line form of `/runs` | About 400 B for one model |
| `GET /accelerators?run_id=&node_id=&since=&until=` | One row per GPU: node, model, holder, utilization now and over the window (last hour by default), memory used of total | About 9 KB for 30 GPUs |
| `GET /inventory` | What the pages list: runs with their lifecycle (`ended`, `waiting`, `placed`, `active`), nodes with their GPUs, the GPUs held now, and pods with problems. No health judgment | About 17 KB on the demo store, against 112 KB |

Named queries keep one copy of each PromQL expression for the UI and agents, and keep agent calls bounded.

### 4.5 Run facts

An earlier version of this design answered with checks: each read the stores, compared what it saw with a threshold taken from incidents, and returned a verdict (`ok`, `warn`, `fail`, `unknown`). An A/B of two agents on recorded runs showed that checks bought speed and bytes, not correctness. Both arms diagnosed every run; the checks arm read 392 KB in 21 calls against 3.4 MB in 30. One check read run `fca5a3a9` as `ok` after its `create_model` failed, and the agent had to see past the verdict. Two later probes found that what helped was never the verdict but the facts behind it: who holds an unanswered request and whether that worker still reports, when a worker relaunched, which op failed and where its traceback is, what a live run waits on. So the server joins the facts and the reader judges them.

`/runs/{id}/summary` returns them in one read. Each block appears only when it has content, so an absent block means nothing of its kind was recorded. There are no thresholds. Run `97532d40`, which ended waiting on a `create_model` that its trainer never answered:

```json
{"run_id": "97532d40-…", "lifecycle": "ended", "steps": 0,
 "requests": {"unanswered": [{"op": "create_model", "count": 1, "age_s": 88391,
                              "pod": "orw-lora-97532d40-…-0-trainer", "holder_alive": false}]},
 "relaunches": {"workers": [{"pod": "orw-lora-97532d40-…-0-trainer",
                             "stopped_reporting_at": "2026-10-21T06:49:40+00:00", "restarted_at": "2026-10-21T06:50:01+00:00"}],
                "logs": {"read": "logs", "args": {"run_id": "97532d40-…", "pod": "orw-lora-97532d40-…-0-trainer",
                                                  "since": "2026-10-21T06:47:40+00:00", "until": "2026-10-21T06:51:01+00:00"}}},
 "config": "no echo", "placement": {"recorded": 2}}
```

A log pointer is a read under 4.4 with its arguments, so an agent follows a failure or a relaunch to the raw lines in one call.

| Block | Holds | Answers |
|---|---|---|
| `steps`, `step_seconds`, `since_last_step_s`, `last_op`, `last_metrics` | Optimizer steps, the median and last gap between them, the newest trainer op | Is the run making progress? What did it do last? |
| `goodput` | The share of the run's span the trainer spent in training steps; the rest split into checkpoint, forward-only, rollout (the trainer idle with a sample in flight) and client time (nothing in flight) | Where does a slow run's time go? |
| `since_last_op_s`, `trainer_idle_s` | For a live run, seconds since any op finished and since the trainer's last op | What is a stuck run waiting on? |
| `requests` | Unanswered trainer requests with the holding pod and whether its worker still reports; the ledger counters, with `lost = enqueued - popped - queued` when the queue length is known | Is a request held by a dead worker, or lost between push and pop? |
| `failures` | Failed trainer ops: count, the 5 newest, a log read around the newest naming its error type | Which op failed, and where is the traceback? |
| `relaunches` | Workers that stopped reporting and restarted in the same pod, from worker records, with a log read around the first | Did a worker crash or restart mid-run? |
| `saves` | `save_state` calls: count, failures, the last success, steps since, the slowest | How much would a crash lose? |
| `optimizer` | Ops that returned non-finite metrics (count and the first), the last 5 grad norms | Are gradients finite? |
| `samplers` | Sends, retries, failures, median latency; the slowest and fastest replica with 20 or more answered sends | Is a sampler replica slow or failing? |
| `config` | Workers whose echoed settings differ from the run's, or `no echo` | Does each worker run what the run asked for? |
| `placement` | Placements recorded, and workloads with no node yet | Does every worker exist and have a GPU? |
| `incidents` | Kubernetes incidents on the run's pods (OOMKilled, CrashLoopBackOff, restarts, scheduling failures), newest 5 | Did Kubernetes kill or fail to place a pod? |
| `gpu_busy` | Mean GPU duty by role over the run's span of recorded ops | Are the GPUs working? |

These cover the incident classes the checks covered: progress (23 incidents), worker health (12), sampler flow (11), effective config (10), the request ledger (9), placement (8), optimizer health (5) and storage (4). Questions that need client signals, such as reward validity, episode endings, policy movement and dependency health, wait for the client step row (rollout step 8) and become facts the same way.

### 4.6 The UI

The Svelte dashboard is a client of 4.4. It shows no verdicts and no facts blocks; those are for agents, through the API.

| View | Built from |
|---|---|
| Overview | `/inventory` for runs. The label is the gateway's status once it is final, else the lifecycle (`ended`, `queued`, `running`, `active`) |
| Run · header | The same label, and pod problems from `/inventory` |
| Run · Activity | `/runs/{id}/commands` for lanes; `/query?name=run_gpu_busy&points=1` for Duty |
| Run · Metrics | `/runs/{id}/steps` charted by step; `/query` for vLLM and GPU charts at chart width |
| Run · Logs | `/logs/search` |
| Nodes | `/query?name=node_gpu_holders`, `/query` per GPU |
| Health | Pod problems, nodes not ready and source errors from `/inventory` |

### 4.7 GKE and local

Facts and endpoints are written against four source functions: metrics (PromQL), logs, placements and commands. GKE provides Managed Prometheus, Cloud Logging and Workload status. A local stack provides a plain Prometheus if present, JSON log files, and placement records written by the local launcher. A missing input leaves its block out of a run's facts.

In the cluster, credentials come from the metadata server and the mounted service account, so no credential code is needed beyond library defaults. Laptop access to the cluster is a small adapter.

Every Kubernetes call goes through lightkube. `server/kube.py` keeps one watch per kind (pods, nodes, Workloads) for the life of the process: it lists, follows changes from the list's resourceVersion, and lists again on 410 Gone, on any failure, and every 5 minutes. The dashboard reads pods, nodes and Workloads from it, so a snapshot makes no Kubernetes call. The API server's scheduler worker manager watches only Workloads and reads its model view and the 30 s placement sweep from the same cache; it still creates, gets, lists and deletes Workloads with direct calls.

Reads that every page repeats are shared. A snapshot is cached for 5 s, a run's command records for 10 s and its client metrics for 30 s. Once an entry is stale, the next caller starts a fresh read and every caller gets the last good result until it lands, so only the first read after start waits.

## 5. Size budget

PR #23 had 2,770 backend lines (`src/server/telemetry.py` and `src/server/dashboard/*.py`). Trimming that structure bottomed out near 2,900 with this feature set, so it was rewritten from the requirements in two independent builds against one contract. The smaller one, about 1,300 lines, was kept. On `main`, without sampling pools or tracing setup, it is 1,370, counting the Kubernetes watch cache that the dashboard shares with the worker manager:

| Module | Lines | Job |
|---|---|---|
| `telemetry.py` | 336 | The writers: command, worker, placement and session records, request counters, settings echo, forwarded client step events |
| `dashboard/router.py` | 293 | The routes in 4.4 and one response model |
| `dashboard/sources.py` | 253 | Kubernetes from the watch cache, Managed Prometheus and Cloud Logging reads, GKE discovery |
| `dashboard/facts.py` | 213 | The run facts in 4.5 and the one-line form |
| `dashboard/fleet.py` | 178 | Shared reads that serve the last good result, the snapshot, lanes, steps |
| `kube.py` | 74 | One watch per Kubernetes kind for the process |
| `dashboard/__main__.py` | 23 | The standalone server |

About 250 of those lines are the blank lines `ruff format` requires between definitions. `dev/diagnostics_budget.py` measures the total and the payload sizes.

## 6. Rollout

Each PR is one row in the instrumentation ledger, with the signal it adds, the checks it serves, and a measured before and after.

1. **Harness.** Line and payload budget script; baseline screenshots of every page against the Qwen run.
2. **Correctness.** NaN as `null`; sampler records in their own list; step ordinal on command records; lost-request detection.
3. **Placement.** Workload status gains GPU UUIDs and start and end times; the scheduler exports `openrl_gpu_held`; workers echo placement and config.
4. **Generic reads.** `/query` with named templates, `/logs/search`, `/runs/{id}/commands`, `/runs/{id}/steps`.
5. **Facts.** The run summary, then `/runs` and session summaries.
6. **UI migration.** Page by page onto the generic reads.
7. **Delete.** Placement reconstruction, `/telemetry`, `/history`, OTel metric gauges, per-page schema.
8. **Client signals.** Step row and heartbeat, then the client-side facts.
9. **Noise.** Sampler access-log flag (`--disable-access-log-for-endpoints /metrics,/weight_info`), one log bucket, one sampler PodMonitoring. Cluster changes need an owner's approval.

## 7. Risks and open questions

- **Scheduler change.** GPU UUIDs in Workload status and the info metric are Go changes to `scheduler/controller` and a CRD update, deployed to a shared cluster.
- **Historical runs.** Runs recorded before placement records exist keep today's reconstruction until they age out, or are backfilled once.
- **Facts on request or on a timer.** On request is simpler; a timer enables alerts, which would bring back thresholds.
- **Client contract.** Step events reach the server only from a client that logs through an OpenTelemetry logger and a Collector with the `otlphttp` exporter in `docs/configuration.md`. Rows are kept per session, so two models trained by one session in the same window share their client metrics.

## 8. Alternatives considered

- **Events plus SQL as the record of truth.** All history as log events joined by SQL after upgrading the log bucket to Observability Analytics. Capable, but adds a Collector exporter, a bucket upgrade and a SQL client before any check works. Kept as a later addition for cross-run questions.
- **Step history in Redis.** Simplest to build, but Redis is live state, not a log or a history store.
- **Grafana for all charts.** About 850 backend lines, but loses the Activity lanes and step-keyed training charts. The named-query endpoints keep the option open: Grafana can query the same Prometheus series.
