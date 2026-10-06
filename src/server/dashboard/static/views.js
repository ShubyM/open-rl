// Page renderers return markup; DOM updates stay in app.js.

import { escape, encode, empty, runStatus, elapsedTime, duration, shortNodeName } from "./ui.js";
import { chart, chartNumber } from "./charts.js";
import { route, ui } from "./store.js";
import { use } from "./cache.js";

// Status chips on the Overview, by what the cluster shows rather than the
// recorded lifecycle: a run recorded "active" with no workers is Unassigned.
const RUN_GROUPS = {
  active: ["Running", "Starting", "Queued", "Needs attention", "Unknown"],
  unassigned: ["Unassigned"],
  finished: ["Completed", "Ended"],
  failed: ["Failed"],
};
const RUN_FILTERS = [
  ["all", "All", "Every recorded run"],
  ["active", "Active", "Runs with workers, or whose workers cannot be checked"],
  ["unassigned", "Unassigned", "Recorded as active, but no workers are assigned; usually a client that exited without finishing"],
  ["finished", "Finished", "Completed or ended"],
  ["failed", "Failed", "Recorded as failed"],
];
const runGroup = (r) => Object.keys(RUN_GROUPS).find((group) => RUN_GROUPS[group].includes(r.display_status)) || "active";
const KINDS = { lora: "LoRA", full: "FFT", fft: "FFT" };

// The runs the Overview shows for the current search and chip: active first, then newest.
export function filterRuns(state, filter = ui.runFilter) {
  const terms = filter.q.toLowerCase().split(/\s+/).filter(Boolean);
  const text = (r) => [r.run_id, r.name, r.model, r.recipe_name, r.display_status, KINDS[r.fine_tuning_type], ...r.nodes, ...r.pods.map((p) => p.name)].filter(Boolean).join(" ").toLowerCase();
  return state.runs
    .filter((r) => (filter.status === "all" || runGroup(r) === filter.status) && terms.every((term) => text(r).includes(term)))
    .sort((a, b) => Number(runGroup(b) === "active") - Number(runGroup(a) === "active"));
}

const age = (run, observedAt) => {
  const created = typeof run.created_at === "number" ? run.created_at : Date.parse(run.created_at) / 1000, now = Date.parse(observedAt) / 1000;
  return Number.isFinite(created) && created > 0 && Number.isFinite(now) ? `${duration(Math.max(0, now - created))} ago` : "—";
};

// Reward curves live in the metrics.jsonl files the Experiments page reads. A
// run links to one only by name, so a run without a matching name has none.
export function experimentsByName() {
  const byName = new Map();
  for (const exp of [...(use("/api/v1/dashboard/experiments").data?.runs || [])].sort((a, b) => a.updated_at - b.updated_at)) byName.set(exp.name, exp);
  return byName;
}

const stepsCell = (r) => {
  const count = `<span class="step-count">${escape(r.steps ?? "—")}${r.max_steps ? ` / ${escape(r.max_steps)}` : ""}</span>`;
  if (!r.max_steps || r.steps == null) return count;
  return `${count}<span class="step-bar" aria-hidden="true"><span style="width:${Math.min(100, (100 * r.steps) / r.max_steps)}%"></span></span>`;
};

export function runs(state) {
  const error = state.store_error ? `<p class="source-error" role="status">${escape(state.store_error)}</p>` : "";
  if (error && !state.runs.length) return `<h1 class="heading">Overview</h1>${error}`;
  const filter = ui.runFilter;
  const chips = RUN_FILTERS.map(([key, label, title]) => {
    const count = key === "all" ? state.runs.length : state.runs.filter((r) => runGroup(r) === key).length;
    return `<button type="button" class="chip" data-run-filter="${key}" title="${escape(title)}" aria-pressed="${filter.status === key}">${escape(label)} <span class="run-filter-count">${count}</span></button>`;
  }).join("");
  const shown = filterRuns(state);
  const deletable = shown.filter((r) => !r.delete_blocker);
  const selected = deletable.filter((r) => ui.selectedRuns.has(r.run_id));
  const all = deletable.length > 0 && selected.length === deletable.length;
  const byName = experimentsByName();
  const reward = (r) => byName.get(r.display_name)?.last.reward;
  const withReward = shown.some((r) => Number.isFinite(reward(r)));
  const rows = shown
    .map((r) => {
      const model = (r.model || "Run").split("/").at(-1), id8 = `<span class="mono">${escape(r.run_id.slice(0, 8))}</span>`;
      const name = r.display_name ? escape(r.display_name) : `${escape(model)} ${id8}`;
      const meta = [r.display_name && `<span>${escape(model)}</span>${id8}`, r.recipe_name && r.recipe_name !== r.display_name && `<span>${escape(r.recipe_name)}</span>`].filter(Boolean).join("");
      const box = `<input type="checkbox" data-select-run="${escape(r.run_id)}" aria-label="Select ${escape(r.display_name || `${model} ${r.run_id.slice(0, 8)}`)}"${ui.selectedRuns.has(r.run_id) && !r.delete_blocker ? " checked" : ""}${r.delete_blocker ? ` disabled title="${escape(`Cannot delete: ${r.delete_blocker}`)}"` : ""}>`;
      return `<div class="job-list-row run-row" data-key="${escape(r.run_id)}"><span class="run-select">${box}</span><span class="job-identity"><a href="#run/${encode(r.run_id)}/activity" title="${escape(r.run_id)}">${name}</a>${meta ? `<span class="run-meta muted micro">${meta}</span>` : ""}</span><span>${runStatus(r.display_status || r.status)}</span><span>${escape(KINDS[r.fine_tuning_type] || r.fine_tuning_type || "—")}</span><span>${stepsCell(r)}</span>${withReward ? `${Number.isFinite(reward(r)) ? `<span class="run-reward">${escape(chartNumber(reward(r)))}</span>` : '<span class="run-reward none">—</span>'}` : ""}<span class="run-created">${escape(age(r, state.observed_at))}</span><span>${escape(runGroup(r) === "unassigned" ? "—" : elapsedTime(r, state.observed_at))}</span></div>`;
    })
    .join("");
  const head = `<div class="job-list-head run-row"><span class="run-select"><input type="checkbox" data-select-all-runs aria-label="Select all shown runs that can be deleted"${all ? " checked" : ""}${deletable.length ? "" : " disabled"}></span><span>Job</span><span>Status</span><span>Training kind</span><span>Steps</span>${withReward ? "<span>Reward</span>" : ""}<span>Created</span><span>Elapsed</span></div>`;
  const notice = ui.runNotice ? `<p class="run-notice" role="status">${escape(ui.runNotice)}</p>` : "";
  const none = !state.runs.length ? empty("No runs recorded") : !shown.length ? empty("No runs match the search and status filter") : "";
  return `<h1 class="heading">Overview</h1>${error}
    <div class="overview-toolbar">
      <input id="run-search" type="search" placeholder="Search by run ID, model, name, recipe, node or pod" aria-label="Search runs" value="${escape(filter.q)}" autocomplete="off">
      <div class="run-filters" role="group" aria-label="Filter by status">${chips}</div>
      <button type="button" class="chip run-delete" data-delete-runs${selected.length ? "" : " disabled"} title="Removes the run records and their recorded ops; workers, checkpoints and metrics files are not touched">${selected.length ? `Delete ${selected.length} run${selected.length === 1 ? "" : "s"}` : "Delete selected"}</button>
    </div>${notice}
    <div class="job-list run-list${withReward ? " with-reward" : ""}">${shown.length ? head + rows : ""}</div>${none}`;
}

export function scheduler(state) {
  const data = state.cluster.scheduler || {};
  if (!data.available) return `<h1 class="heading">Scheduler</h1>${empty(data.error || (data.installed === false ? "Scheduler is not installed" : "Scheduler data is unavailable"))}`;
  const workloads = data.workloads || [];
  const pending = workloads.filter((w) => !w.node_name && !["Completed", "Failed"].includes(w.phase));
  const runFor = (w) => state.runs.find((r) => (r.workloads || []).some((item) => item.uid === w.uid));
  const label = (w) => {
    const run = runFor(w);
    return run ? `<a href="#run/${encode(run.run_id)}/activity">${escape(run.name)}</a>` : escape(w.model_id || w.name);
  };
  const role = (w) => ({ trainer: "Trainer", sampler: "Sampler" })[w.role] || "Unknown process";
  const rows = pending
    .map(
      (w) =>
        `<tr><td>${label(w)}<div class="muted">${role(w)}${w.owner_id ? `, owner ID ${escape(w.owner_id)}` : ""}</div></td><td>${escape(w.requested_memory || "Not reported")}${w.exclusive ? ", exclusive" : ""}</td><td>${escape(w.phase)}</td><td>${escape(w.placed_message || w.placed_reason || w.reason || "No placement reason reported")}</td></tr>`,
    )
    .join("");
  const reservations = (data.ledgers || [])
    .map((ledger) => {
      const seats = ledger.seats
        .map((seat) => {
          const w = workloads.find((item) => item.uid === seat.workload_uid);
          const placement = w && state.placements.find((item) => item.id === w.uid);
          return `<div class="scheduler-seat"><span>${w ? label(w) : escape(seat.workload)}</span><span class="muted">${w ? role(w) + ", " : ""}${seat.exclusive ? "Exclusive" : "Shared"}</span>${seat.owner ? `<span class="muted">Owner ID: ${escape(seat.owner)}</span>` : ""}${placement ? `<a href="#nodes" data-scheduler-placement="${escape(placement.id)}" title="${escape(placement.node)}">Node ${escape(shortNodeName(placement.node, state.cluster.nodes))} ↗</a>` : ""}</div>`;
        })
        .join("");
      return `<div class="scheduler-reservation" data-key="${escape(ledger.name || ledger.claim_name)}"><div>${escape(ledger.claim_name || ledger.name)}<div class="muted">${ledger.seats.length} reservation${ledger.seats.length === 1 ? "" : "s"}</div></div><div class="scheduler-seat-list">${seats}</div></div>`;
    })
    .join("");
  return `<h1 class="heading">Scheduler</h1>
    <div class="overview-summary"><span><strong>${pending.length}</strong> Pending</span><span><strong>${workloads.filter((w) => w.node_name).length}</strong> Assigned workloads</span></div>
    <h2 class="scheduler-title">Waiting for placement</h2>
    ${pending.length ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Workload</th><th>Request</th><th>State</th><th>Placement reason</th></tr></thead><tbody>${rows}</tbody></table></div>` : empty("Nothing is waiting for placement")}
    <h2 class="scheduler-title">Reservations</h2>${reservations || empty("No claim reservations reported")}
    <p class="run-json-link"><a href="/api/v1/dashboard/snapshot">Inspect scheduler JSON</a></p>`;
}

const FAILED_WAITING = new Set(["CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError", "CreateContainerError", "RunContainerError", "InvalidImageName", "ContainerCannotRun", "StartError"]);

function podTone(pod) {
  if (pod.phase === "Failed") return "error";
  const failing = (pod.containers || []).some(
    (c) => (c.state === "waiting" && FAILED_WAITING.has(c.reason)) || (c.state === "terminated" && ((c.exit_code != null && c.exit_code !== 0) || (c.reason && c.reason !== "Completed"))),
  );
  if (failing) return "error";
  const reason = String(pod.problem || "").split(":", 1)[0];
  return reason === "Failed" || FAILED_WAITING.has(reason) ? "error" : "warning";
}

const healthStatus = (label, tone) => `<span class="health-status health-${tone}"><span class="health-dot" aria-hidden="true"></span>${escape(label)}</span>`;

export function health(state) {
  const cluster = state.cluster;
  const errors = [...new Set([state.store_error, state.history_error, cluster.error, cluster.nodes_error, cluster.events_error, cluster.devices?.error, cluster.scheduler?.error].filter(Boolean))];
  const issues = [];
  for (const pod of cluster.pods || []) {
    if (!pod.problem) continue;
    const run = state.runs.find((r) => r.pods.some((p) => p.uid === pod.uid));
    const link = run ? `<a href="#run/${encode(run.run_id)}/logs">Logs</a>` : `<a href="/api/v1/dashboard/pods/${encode(pod.name)}/logs">Logs</a>`;
    issues.push([pod.problem, pod.name, `${pod.restarts || 0} restarts`, link, podTone(pod)]);
  }
  for (const node of cluster.nodes || []) if (node.ready !== true) issues.push(["Node not ready", node.name, "Ready condition is false or unknown", '<a href="#nodes">Nodes</a>', "error"]);
  const rows = issues.map(([issue, resource, evidence, link, tone]) => `<tr><td>${healthStatus(issue, tone)}</td><td>${escape(resource)}</td><td>${escape(evidence)}</td><td>${link}</td></tr>`).join("");
  const complete = cluster.available === true && errors.length === 0;
  return `<h1 class="heading">Health</h1>${errors.map((error) => `<p class="health-message" role="status">${healthStatus("Source unavailable", "warning")}<span>${escape(error)}</span></p>`).join("")}
    ${issues.length ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Issue</th><th>Resource</th><th>Evidence</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>` : complete ? `<p class="health-message">${healthStatus("Healthy", "success")}<span>No pod problems, node problems or source errors.</span></p>` : ""}
    <p class="run-json-link"><a href="/api/v1/dashboard/snapshot">Diagnostic JSON</a><a href="/docs">API reference</a></p>`;
}

// The recipe config carries a lora_rank even for full fine-tuning runs; the
// run directory name is the reliable signal the sweep scripts leave behind.
const kindLabel = (run) => (/(^|[-_])fft([-_]|$)/.test(run.name) || run.config.lora_rank == null ? "FFT" : `LoRA r${run.config.lora_rank}`);
const pct = (value) => (Number.isFinite(value) ? `${(100 * value).toFixed(1)}%` : "—");
const shortName = (name) => name.replace(/^gsm8k_rl_(mega|rank_sweep)_/, "");

function experimentCharts(run) {
  // Reward always gets a chart, even an empty one; correctness only when the run records it.
  const keys = ["reward", "correct"].filter((key) => key === "reward" || run.series[key]?.some(([, value]) => Number.isFinite(value)));
  return keys.map((key) => {
    const points = run.series[key] || [], percent = key === "correct";
    return chart({
      title: percent ? "Correctness" : "Reward",
      points: percent ? points.map(([step, value]) => [step, Number.isFinite(value) ? value * 100 : value]) : points,
      start: points[0]?.[0] ?? 0, end: points.at(-1)?.[0] ?? 0,
      unit: percent ? "%" : "", ...(percent ? { min: 0, max: 100 } : {}), tone: "accent", xFormat: "step",
    });
  }).join("");
}

// Training curves read from each run's metrics.jsonl on the shared volume,
// grouped by the sweep directory they were written under.
export function experiments(entry) {
  const data = entry.data;
  if (!data) return `<h1 class="heading">Experiments</h1>${empty(entry.error || "Loading run metrics…")}`;
  if (data.error) return `<h1 class="heading">Experiments</h1>${empty(data.error)}`;
  const ordered = [...data.runs].sort((a, b) => b.updated_at - a.updated_at);
  const selected = ordered.find((run) => run.path === route()[1]) || ordered.find((run) => ["reward", "correct"].some((key) => run.series[key]?.filter(([, value]) => Number.isFinite(value)).length > 1)) || ordered[0];
  const sweeps = new Map();
  for (const run of ordered) sweeps.set(run.sweep, [...(sweeps.get(run.sweep) || []), run]);
  const now = Date.now() / 1000;
  const sections = [...sweeps.entries()]
    .map(([sweep, members]) => {
      // Columns no run in this sweep records are dropped rather than shown as dashes.
      const metrics = ["correct", "format"].filter((key) => members.some((run) => Number.isFinite(run.last[key])));
      const rows = members
        .map((run) => {
          const charts = run === selected ? `<div class="chart-grid" data-key="charts:${escape(run.path)}">${experimentCharts(run)}</div>` : "";
          return `<a class="job-list-row experiment-row" data-key="${escape(run.path)}" href="#experiments/${encode(run.path)}"${run === selected ? ' aria-current="true"' : ""}><span class="job-identity"><span class="mono">${escape(shortName(run.name))}</span></span><span>${escape((run.config.model_name || "").split("/").at(-1))}</span><span>${escape(kindLabel(run))}</span><span>${run.step}${run.config.max_steps ? ` / ${run.config.max_steps}` : ""}</span><span class="exp-reward">${Number.isFinite(run.last.reward) ? escape(chartNumber(run.last.reward)) : "—"}</span>${metrics.map((key) => `<span class="exp-${key}">${escape(pct(run.last[key]))}</span>`).join("")}</a>${charts}`;
        })
        .join("");
      return `<section class="experiment-sweep"><h2>${escape(sweep || "runs")} <span class="muted micro">${members.length} run${members.length === 1 ? "" : "s"}, updated ${duration(now - Math.max(...members.map((r) => r.updated_at)))} ago</span></h2>
        <div class="job-list" style="--metric-columns:${2 + metrics.length}"><div class="job-list-head experiment-head"><span>Run</span><span>Model</span><span>Kind</span><span>Step</span><span>Reward</span>${metrics.map((key) => `<span>${key[0].toUpperCase() + key.slice(1)}</span>`).join("")}</div>${rows}</div></section>`;
    })
    .join("");
  return `<h1 class="heading">Experiments</h1><p class="muted">Select a run to inspect reward and correctness.</p>${entry.error ? empty(`${entry.error}. Showing previously fetched metrics`) : ""}${sections}${!data.runs.length ? empty("No run metrics found under the runs directory") : ""}`;
}
