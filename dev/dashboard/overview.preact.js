// Evaluation only: the Overview view using Preact + HTM. The production store,
// filters, CSS and delegated interactions stay the same for a fair comparison.
import { h, render } from "preact";
import htm from "htm";
import { ui } from "/dashboard/assets/store.js";
import { filterRuns } from "/dashboard/assets/views.js";
import { duration, elapsedTime } from "/dashboard/assets/ui.js";

const html = htm.bind(h);
const groups = { active: ["Running", "Starting", "Queued", "Needs attention", "Unknown"], unassigned: ["Unassigned"], finished: ["Completed", "Ended"], failed: ["Failed"] };
const group = (run) => Object.keys(groups).find((key) => groups[key].includes(run.display_status)) || "active";
const kinds = { lora: "LoRA", full: "FFT", fft: "FFT" };
const filters = [["all", "All", "Every recorded run"], ["active", "Active", "Runs with workers, or whose workers cannot be checked"], ["unassigned", "Unassigned", "Recorded as active, but no workers are assigned; usually a client that exited without finishing"], ["finished", "Finished", "Completed or ended"], ["failed", "Failed", "Recorded as failed"]];

function Status({ run }) {
  const label = run.display_status || run.status;
  const tone = { Running: "running", Starting: "running", "Needs attention": "failed", Failed: "failed", Completed: "completed", Ended: "completed" }[label] || "pending";
  return html`<span class=${`state state-${tone}`}><span class="state-dot" aria-hidden="true"></span>${label}</span>`;
}

function Row({ run, state }) {
  const short = `${(run.model || "Run").split("/").at(-1)} · ${run.run_id.slice(0, 8)}`;
  const label = [run.display_name, run.recipe_name].filter((v, i, a) => v && a.indexOf(v) === i).join(" · ");
  const blocker = state.recorded_at ? "Recorded runs cannot be deleted" : ui.deletingRuns ? "Deletion in progress" : run.delete_blocker;
  const created = typeof run.created_at === "number" ? run.created_at : Date.parse(run.created_at) / 1000;
  const now = Date.parse(state.observed_at) / 1000;
  const age = Number.isFinite(created) && created > 0 && Number.isFinite(now) ? `${duration(Math.max(0, now - created))} ago` : "—";
  return html`<div class="job-list-row run-row" data-key=${run.run_id}>
    <span class="run-select"><input type="checkbox" data-select-run=${run.run_id} aria-label=${`Select ${short}`} checked=${ui.selectedRuns.has(run.run_id) && !run.delete_blocker} disabled=${!!blocker} title=${blocker || undefined}/></span>
    <span class="job-identity"><a href=${`#run/${encodeURIComponent(run.run_id)}/activity`} title=${run.run_id}>${(run.model || "Run").split("/").at(-1)} · <span class="mono">${run.run_id.slice(0, 8)}</span></a>${label && html`<span class="muted micro">${label}</span>`}</span>
    <span><${Status} run=${run}/></span><span>${kinds[run.fine_tuning_type] || run.fine_tuning_type || "—"}</span>
    <span>${run.steps ?? "—"}</span><span>${age}</span><span>${group(run) === "unassigned" ? "—" : elapsedTime(run, state.observed_at)}</span>
  </div>`;
}

export function Overview({ state }) {
  const error = state.store_error && html`<p class="source-error" role="status">${state.store_error}</p>`;
  if (error && !state.runs.length) return html`<h1 class="heading">Overview</h1>${error}`;
  const shown = filterRuns(state), deletable = state.recorded_at ? [] : shown.filter((r) => !r.delete_blocker);
  const selected = deletable.filter((r) => ui.selectedRuns.has(r.run_id));
  const all = deletable.length > 0 && selected.length === deletable.length;
  return html`<h1 class="heading">Overview</h1>${error}
    <div class="overview-toolbar">
      <input id="run-search" type="search" placeholder="Search by run ID, model, name, recipe, node or pod" aria-label="Search runs" value=${ui.runFilter.q} autocomplete="off"/>
      <div class="run-filters" role="group" aria-label="Filter by status">${filters.map(([key, label, title]) => html`<button key=${key} type="button" class="chip" data-run-filter=${key} title=${title} aria-pressed=${ui.runFilter.status === key}>${label} <span class="run-filter-count">${key === "all" ? state.runs.length : state.runs.filter((r) => group(r) === key).length}</span></button>`)}</div>
      ${!state.recorded_at && html`<button type="button" class="chip run-delete" data-delete-runs disabled=${!selected.length || ui.deletingRuns} title="Removes the run records and their recorded ops; workers, checkpoints and metrics files are not touched">${ui.deletingRuns ? "Deleting…" : selected.length ? `Delete ${selected.length} run${selected.length === 1 ? "" : "s"}` : "Delete selected"}</button>`}
    </div>${ui.runNotice && html`<p class="run-notice" role="status">${ui.runNotice}</p>`}
    <div class="job-list run-list">${shown.length > 0 && html`
      <div class="job-list-head run-row"><span class="run-select"><input type="checkbox" data-select-all-runs aria-label="Select all shown runs that can be deleted" checked=${all} disabled=${!deletable.length || ui.deletingRuns} data-indeterminate=${selected.length > 0 && !all}/></span><span>Job</span><span>Status</span><span>Training kind</span><span>Completed steps</span><span>Created</span><span>Elapsed</span></div>
      ${shown.map((run) => html`<${Row} key=${run.run_id} run=${run} state=${state}/>`)}
    `}</div>${!shown.length && html`<p class="empty-state">${!state.runs.length ? "No runs recorded" : "No runs match the search and status filter"}</p>`}`;
}

export const paint = (container, state) => render(html`<${Overview} state=${state}/>`, container);
export const clear = (container) => render(null, container);
