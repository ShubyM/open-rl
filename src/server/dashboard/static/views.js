import { escape, encode, empty, runStatus, elapsedTime } from "./ui.js";

export function runs(state) {
  return `<h1 class="heading">Overview</h1>
      <div class="overview-summary">${[
        [
          state.runs.filter((r) =>
            ["active", "running"].includes(
              String(r.status || "").toLowerCase(),
            ),
          ).length,
          "Active",
        ],
        [state.runs.filter((r) => r.status === "failed").length, "Failed"],
        [
          state.runs.filter((r) => r.status === "completed").length,
          "Completed",
        ],
      ]
        .map(
          ([count, label]) => `<span><strong>${count}</strong> ${label}</span>`,
        )
        .join("")}</div>
      ${state.store_error ? empty(state.store_error) : ""}
      <div class="job-list"><div class="job-list-head"><span>Job</span><span>Status</span><span>Training kind</span><span>Completed steps</span><span>Elapsed</span></div>
      ${state.runs
        .map((r) => {
          const label = [r.display_name, r.recipe_name]
            .filter((v, i, a) => v && a.indexOf(v) === i)
            .join(" · ");
          return `<a class="job-list-row" href="#run/${encode(r.run_id)}/metrics"><span class="job-identity"><span>${escape((r.model || "Run").split("/").at(-1))} · <span class="mono">${escape(r.run_id.slice(0, 8))}</span></span>${label ? `<span class="muted job-metadata">${escape(label)}</span>` : ""}</span>${runStatus(r.display_status || r.status)}<span>${escape({ lora: "LoRA", full: "FFT", fft: "FFT" }[r.fine_tuning_type] || "—")}</span><span class="job-steps mono">${escape(r.steps)}</span><span class="job-elapsed mono" title="Time since job creation, including queued time">${escape(elapsedTime(r, state.observed_at))}</span></a>`;
        })
        .join("")}</div>${!state.runs.length ? empty("No runs recorded") : ""}`;
}

export function scheduler(state) {
  const data = state.cluster.scheduler || {};
  if (!data.available) {
    return `<h1 class="heading">Scheduler</h1>${empty(data.error || (data.installed === false ? "Scheduler is not installed" : "Scheduler data is unavailable"))}`;
  }
  const workloads = data.workloads || [];
  const pending = workloads.filter(
    (w) => !w.node_name && !["Completed", "Failed"].includes(w.phase),
  );
  const runFor = (w) =>
    state.runs.find((r) =>
      (r.workloads || []).some((item) => item.uid === w.uid),
    );
  const label = (w) => {
    const run = runFor(w);
    return run
      ? `<a href="#run/${encode(run.run_id)}/overview">${escape(run.name)} ↗</a>`
      : escape(w.model_id || w.name);
  };
  const role = (w) =>
    ({ trainer: "Trainer", sampler: "Sampler" })[w.role] || "Unknown process";
  const workloadRows = (items) =>
    items
      .map(
        (w) =>
          `<tr><td>${label(w)}<div class="muted">${role(w)}${w.owner_id ? ` · Owner ID: ${escape(w.owner_id)}` : ""}</div></td><td>${escape(w.requested_memory || "Not reported")}${w.exclusive ? " · Exclusive" : ""}</td><td>${escape(w.phase)}</td><td>${escape(w.placed_message || w.placed_reason || w.reason || "No placement reason reported")}${w.generation_current === false ? '<div class="muted">Latest workload update not yet observed</div>' : ""}</td></tr>`,
      )
      .join("");
  return `<h1 class="heading">Scheduler</h1>
      <div class="overview-summary"><span><strong>${pending.length}</strong> Pending</span><span><strong>${workloads.filter((w) => w.node_name).length}</strong> Assigned workloads</span></div>
      <h2 class="scheduler-title">Waiting for placement</h2>
      ${pending.length ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Workload</th><th>Request</th><th>State</th><th>Placement reason</th></tr></thead><tbody>${workloadRows(pending)}</tbody></table></div>` : empty("No workloads waiting for placement")}
      <h2 class="scheduler-title">Reservations</h2>
      ${
        (data.ledgers || [])
          .map(
            (ledger) =>
              `<div class="scheduler-reservation"><div>${escape(ledger.claim_name || ledger.name)}<div class="muted">${ledger.seats.length} reservation${ledger.seats.length === 1 ? "" : "s"}</div></div><div class="scheduler-seat-list">${ledger.seats
                .map((seat) => {
                  const w = workloads.find(
                    (item) => item.uid === seat.workload_uid,
                  );
                  const placement =
                    w && state.placements.find((item) => item.id === w.uid);
                  return `<div class="scheduler-seat"><span>${w ? label(w) : escape(seat.workload)}</span><span class="muted">${w ? role(w) + " · " : ""}${seat.exclusive ? "Exclusive" : "Shared"}</span>${seat.owner ? `<span class="muted">Owner ID: ${escape(seat.owner)}</span>` : ""}${placement ? `<a href="#nodes" data-scheduler-placement="${escape(placement.id)}">${escape(placement.node)} ↗</a>` : ""}</div>`;
                })
                .join("")}</div></div>`,
          )
          .join("") || empty("No claim reservations reported")
      }
      <p class="muted scheduler-title"><a href="/api/v1/dashboard/snapshot">Inspect scheduler JSON ↗</a></p>`;
}

function healthStatus(label, tone) {
  return `<span class="health-status health-${tone}"><span class="health-dot" aria-hidden="true"></span>${escape(label)}</span>`;
}

function podHealthTone(pod) {
  if (pod.phase === "Failed") return "error";
  const failedWaitingReasons = new Set([
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerConfigError",
    "CreateContainerError",
    "RunContainerError",
    "InvalidImageName",
    "ContainerCannotRun",
    "StartError",
  ]);
  const containers = pod.containers || [];
  if (
    containers.some(
      (container) =>
        (container.state === "waiting" &&
          failedWaitingReasons.has(container.reason)) ||
        (container.state === "terminated" &&
          ((container.exit_code != null && container.exit_code !== 0) ||
            (container.reason && container.reason !== "Completed"))),
    )
  )
    return "error";
  if (!containers.length) {
    const reason = String(pod.problem || "").split(":", 1)[0];
    if (reason === "Failed" || failedWaitingReasons.has(reason)) return "error";
  }
  // A previous termination can remain in pod.problem after the worker recovers.
  return "warning";
}

export function health(state) {
  const cluster = state.cluster;
  const errors = [
    state.store_error,
    cluster.error,
    cluster.nodes_error,
    cluster.events_error,
    state.device_inventory?.error,
    cluster.scheduler?.error,
    cluster.metrics?.error,
    cluster.rollouts?.error,
  ].filter(Boolean);
  const issues = [];
  for (const pod of cluster.pods || []) {
    if (!pod.problem) continue;
    const run = state.runs.find((r) => r.pods.some((p) => p.uid === pod.uid));
    issues.push([
      pod.problem,
      pod.name,
      `${pod.restarts || 0} restarts`,
      run
        ? `<a href="#run/${encode(run.run_id)}/logs">Logs ↗</a>`
        : `<a href="/api/v1/dashboard/pods/${encode(pod.name)}/logs">Logs ↗</a>`,
      podHealthTone(pod),
    ]);
  }
  for (const node of cluster.nodes || [])
    if (node.ready !== true)
      issues.push([
        "Node not ready",
        node.name,
        "Ready condition is false or unknown",
        '<a href="#nodes">Nodes ↗</a>',
        "error",
      ]);
  for (const rollout of cluster.rollouts?.items || [])
    if (["failed", "degraded"].includes(rollout.state))
      issues.push([
        rollout.state,
        rollout.name,
        rollout.kind,
        "",
        rollout.state === "failed" ? "error" : "warning",
      ]);
  const complete = cluster.available === true && errors.length === 0;
  return `<h1 class="heading">Health</h1>${errors.map((error) => `<p class="health-message">${healthStatus("Source unavailable", "warning")}<span>${escape(error)}</span></p>`).join("")}
      ${issues.length ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Issue</th><th>Resource</th><th>Evidence</th><th></th></tr></thead><tbody>${issues.map(([issue, resource, evidence, link, tone]) => `<tr><td>${healthStatus(issue, tone)}</td><td>${escape(resource)}</td><td>${escape(evidence)}</td><td>${link}</td></tr>`).join("")}</tbody></table></div>` : `<p class="health-message">${healthStatus(complete ? "No issues reported by available sources" : "Health assessment is incomplete", complete ? "success" : "warning")}</p>`}
      <p class="run-json-link"><a href="/api/v1/dashboard/snapshot">Diagnostic JSON ↗</a> · <a href="/docs">API reference ↗</a></p>`;
}

const kindLabel = (config) => (config.lora_rank === undefined || config.lora_rank === null ? "FFT" : `LoRA r${config.lora_rank}`);
const pct = (value) => (value === undefined ? "—" : `${(100 * value).toFixed(1)}%`);
const ago = (seconds) => (seconds < 90 ? `${Math.round(seconds)}s` : seconds < 5400 ? `${Math.round(seconds / 60)}m` : `${(seconds / 3600).toFixed(1)}h`);

// Training curves read from each run's metrics.jsonl on the shared volume,
// grouped by the sweep directory they were written under.
export function experiments(data) {
  if (!data) return `<h1 class="heading">Experiments</h1>${empty("Loading run metrics…")}`;
  if (data.error) return `<h1 class="heading">Experiments</h1>${empty(data.error)}`;
  const sweeps = new Map();
  data.runs.forEach((run) => {
    if (!sweeps.has(run.sweep)) sweeps.set(run.sweep, []);
    sweeps.get(run.sweep).push(run);
  });
  const now = Date.now() / 1000;
  return `<h1 class="heading">Experiments</h1>
    <p class="muted">Recipe metrics from ${escape(data.root)}. Newest sweep first.</p>
    ${[...sweeps.entries()]
      .map(([sweep, runs]) => {
        const sorted = [...runs].sort((a, b) => a.name.localeCompare(b.name));
        return `<section class="experiment-sweep"><h2>${escape(sweep || "runs")} <span class="muted micro">${runs.length} run${runs.length === 1 ? "" : "s"} · updated ${ago(now - Math.max(...runs.map((r) => r.updated_at)))} ago</span></h2>
        <div class="job-list"><div class="job-list-head experiment-head"><span>Run</span><span>Model</span><span>Kind</span><span>Step</span><span>Reward</span><span>Correct</span><span>Format</span></div>
        ${sorted
          .map(
            (run) => `<div class="job-list-row experiment-row"><span class="mono">${escape(run.name.replace(/^gsm8k_rl_(mega|rank_sweep)_/, ""))}</span><span>${escape((run.config.model_name || "").split("/").at(-1))}</span><span>${escape(kindLabel(run.config))}</span><span>${run.step}${run.config.max_steps ? ` / ${run.config.max_steps}` : ""}</span><span>${escape(pct(run.last.reward))}</span><span>${escape(pct(run.last.correct))}</span><span>${escape(pct(run.last.format))}</span></div>`,
          )
          .join("")}</div>
        <div class="experiment-charts">${sorted
          .flatMap((run) =>
            ["reward", "correct"]
              .filter((key) => run.series[key])
              .map((key) => `<section class="operation-chart" data-experiment="${escape(run.path)}" data-series="${key}"></section>`),
          )
          .join("")}</div></section>`;
      })
      .join("")}
    ${!data.runs.length ? empty("No run metrics found under the runs directory") : ""}`;
}
