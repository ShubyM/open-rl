# Instrumentation ledger

Every change that adds, removes or reshapes OpenRL telemetry gets one row here. A row says what the change records, which diagnostic questions it answers, what it costs, and what it measurably changed. The design these rows implement is [Design Doc 017: Run diagnostics](designs/017-run-diagnostics.md).

## How to add a row

1. Name the PR and branch.
2. **Adds**: the signal, record or endpoint, and where it is stored.
3. **Answers**: the checks or questions it serves, from section 4.5 of the design.
4. **Cost**: bytes per record or step, series added, and backend lines added or removed (`python dev/diagnostics_budget.py`).
5. **Value**: a before and after you measured, such as payload bytes, a check that now fires on a replayed incident, or a cost drop. "Expected" is not a value; leave the cell as *pending* until it is measured.

## Ledger

| # | PR / branch | Adds | Answers | Cost | Value | Status |
|---|---|---|---|---|---|---|
| 0 | #23 `svelte-dashboard` | Worker records (5 s), command records per request, step counter, client log ingest, SDK error ingest, OTel metric gauges, inspection API and dashboard | `progress` verdict (OK, STALLED, DEAD, IDLE) | 2,770 backend lines; 476 KB of command records per Qwen step, 98% sampler records with empty metrics | Baseline. The progress verdict is the only computed answer; OTel gauges carried 0 series in 30 days | Open |
| 1 | `diagnostics` (harness) | `dev/diagnostics_budget.py`: backend line count and payload bytes per endpoint for a run | Every later row's Value column | Script only | Baseline on PR #23 for the Qwen run `43b3ebb3`: 2,770 backend lines; Activity tab 1,237,514 B (`/activity` 826,832 + `/series?kind=devices` 410,682); `/series?kind=all` 897,153 B; `/inventory` 123,509 B | Done |
| 2 | `diagnostics` (correctness) | NaN as `null` with `nonfinite`; sampler records in `open_rl:operations:<run>:samples`; step ordinal on command records; lost-request detection | `optimizer-health`, `progress`, `request-ledger` | About 8 B per command record | Unit-verified against real Redis: with the cap at 3, five sample records leave all three trainer ops in place; a NaN loss reaches the client as 0.0 while telemetry names it. A live run is pending: the L4 box hit a zone stockout | Committed locally |
| 3 | `diagnostics` (placement) | GPU UUIDs and start and end in Workload status; `openrl_gpu_held` info metric; worker placement and config echo | `placement`, `worker-health`, `effective-config`, the place half of every check | About 300 B per process lifetime; one series per held GPU; about 1 KB config echo per process | `go test ./... -race` passes on `b`, and the new tests fail with stamping disabled. Not yet scraped on a cluster | Committed locally |
| 4 | `diagnostics` (generic reads) | `/query` named PromQL (11 queries), `/runs/{id}/commands`, `/runs/{id}/steps`; `/logs` already searches | Every view | +269 backend lines, before PR 7's deletions | Qwen run: per-GPU duty 2,934 B via `query?name=gpu_busy&points=1`, against 410,682 B; 20 steps with reward 14,437 B; derived step 36 matches the client's grad norm (435.4) and reward (0.798) | Committed locally |
| 5 | `diagnostics` (checks) | `checks.py`, `/runs/{id}/checks`, `/runs/{id}/summary` | `progress`, `request-ledger`, `worker-health`, `placement`, `optimizer-health`, `effective-config`, `sampler-flow`, `storage` | +325 backend lines (3,364 total before PR 7's deletions) | Replays of I1 (dead holder), I4 (lost pop), I30 (NaN) and a stall give fail; a healthy run gives ok; missing inputs give unknown. On recorded runs: all 8 checks for Qwen in 1,142 B and its summary in 393 B (trainer duty 15.6%, sampler 67.5%), against 0.8 to 1.2 MB of rows; `request-ledger` flags the lost `create_model` of run `97532d40`, which the dashboard showed as Ended | Committed locally |
| 5b | `diagnostics` (sessions) | Durable session record (`open_rl:session_record:<id>`, `openrl.session` log line); `/sessions`, `/sessions/{id}/checks`, `/sessions/{id}/summary`; `client-liveness` and `sdk-errors` checks; model checks rolled up per session | `client-liveness`, `sdk-errors`, every model check per session | About 300 B per session; one GET and SET per heartbeat | Session checks for Qwen's session in 3,196 B, summary under 600 B. Sessions from before records are derived from model metadata and cannot split a resume; new sessions carry their own lifetimes. A heartbeating but idle client fails; an exited client with an unfinished model warns unless another session resumed it | Committed locally |
| 6 | `diagnostics` (UI) | Svelte pages on the generic reads | Every view | Pending | Pending: same screenshots as the baseline, smaller payloads | Planned |
| 7 | `diagnostics` (delete) | Removes placement reconstruction, `/telemetry`, `/history`, OTel gauges, per-page schema | None | Pending | Pending: backend lines against 2,770 | Planned |
| 8 | `diagnostics` (client) | Client step row and heartbeat | `reward-validity`, `episode-endings`, `policy-movement`, `client-liveness`, `dependencies` | About 5 KB per step | Pending | Planned |
| 9 | `diagnostics` (noise) | Sampler `--disable-access-log-for-endpoints /metrics,/weight_info`; one log bucket; one sampler PodMonitoring | None (cost) | Negative | Pending: log bytes and billed samples per day, against 24.7 GB per run and 234M samples per day | Planned |
