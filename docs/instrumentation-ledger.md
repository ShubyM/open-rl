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
| 2 | `diagnostics` (correctness) | NaN as `null` with `nonfinite`; sampler records in `open_rl:operations:<run>:samples`; step ordinal on command records; lost-request detection | `optimizer-health`, `progress`, `request-ledger` | About 8 B per command record | Pending: all optim_steps survive the default cap on a tiny RL run; a forced lost pop reads as a ledger failure, not IDLE | Planned |
| 3 | `diagnostics` (placement) | GPU UUIDs and start and end in Workload status; `openrl_gpu_held` info metric; worker placement and config echo | `placement`, `worker-health`, `effective-config`, the place half of every check | About 300 B per process lifetime; one series per held GPU | Pending: Activity lanes of a finished run render from placement records alone | Planned |
| 4 | `diagnostics` (generic reads) | `/query` named PromQL, `/logs/search`, `/runs/{id}/commands`, `/runs/{id}/steps` | Every view | Pending | Pending: Activity tab bytes for the Qwen run, against 1.2 MB today | Planned |
| 5 | `diagnostics` (checks) | `checks.py`, `/runs/{id}/checks`, `/runs/{id}/summary` | `progress`, `request-ledger`, `worker-health`, `placement`, `optimizer-health`, `effective-config`, `sampler-flow`, `storage` | Pending | Pending: replayed incidents give the expected verdicts | Planned |
| 6 | `diagnostics` (UI) | Svelte pages on the generic reads | Every view | Pending | Pending: same screenshots as the baseline, smaller payloads | Planned |
| 7 | `diagnostics` (delete) | Removes placement reconstruction, `/telemetry`, `/history`, OTel gauges, per-page schema | None | Pending | Pending: backend lines against 2,770 | Planned |
| 8 | `diagnostics` (client) | Client step row and heartbeat | `reward-validity`, `episode-endings`, `policy-movement`, `client-liveness`, `dependencies` | About 5 KB per step | Pending | Planned |
| 9 | `diagnostics` (noise) | Sampler `--disable-access-log-for-endpoints /metrics,/weight_info`; one log bucket; one sampler PodMonitoring | None (cost) | Negative | Pending: log bytes and billed samples per day, against 24.7 GB per run and 234M samples per day | Planned |
