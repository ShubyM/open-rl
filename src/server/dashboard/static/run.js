// One run's page: a metrics tab with the worker's operation timings, and a
// logs tab over Cloud Logging with search, pod filter and paging.

import { runIncidents } from "./timeline.js";
import { escape, encode, empty, button, runStatus } from "./ui.js";
import { chart } from "./charts.js";
import { ui, get } from "./store.js";
import { use } from "./cache.js";

// The inspected run's window, and the log page that was loaded for it.
export const runView = { id: null, windowMinutes: 30, windowEnd: 0, eventAt: null };
export const logState = { key: null, records: [], cursor: null, loading: false, error: null, request: 0 };

export function resetRunWindow() {
  runView.eventAt = null;
  runView.windowEnd = Date.parse(ui.state.observed_at) / 1000;
  logState.key = null;
}

function runRange(nearEvent = false) {
  const start = runView.windowEnd - runView.windowMinutes * 60;
  const eventAt = nearEvent ? runView.eventAt : null;
  return {
    since: new Date(Math.max(start, eventAt === null ? start : eventAt - 120) * 1000).toISOString(),
    until: new Date(Math.min(runView.windowEnd, eventAt === null ? runView.windowEnd : eventAt + 120) * 1000).toISOString(),
  };
}

export function runPage(id, tab = "metrics") {
  if (runView.id !== id) {
    runView.id = id;
    resetRunWindow();
  }
  const run = ui.state.runs.find((r) => r.run_id === id);
  if (!run) return empty("Run not found");
  if (!["metrics", "logs"].includes(tab)) tab = "metrics";
  const title = [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), { lora: "LoRA", full: "FFT", fft: "FFT" }[run.fine_tuning_type]].filter(Boolean).join(" · ");
  const description = [run.display_name, run.recipe_name].filter((value, index, values) => value && value !== title && values.indexOf(value) === index).join(" · ");
  const tabs = ["metrics", "logs"].map((t) => `<a href="#run/${encode(id)}/${t}" ${tab === t ? 'aria-current="page"' : ""}>${t[0].toUpperCase() + t.slice(1)}</a>`).join("");
  const ranges = [10, 30, 60].map((value) => `<option value="${value}" ${runView.windowMinutes === value ? "selected" : ""}>Last ${value} minutes</option>`).join("");
  return `<p class="overview-back"><a href="#overview">← Overview</a></p>
    <div class="run-heading"><h1 class="heading" title="${escape(run.run_id)}">${escape(title)}</h1>${runStatus(run.display_status || run.status)}</div>
    ${description ? `<p class="run-description">${escape(description)}</p>` : ""}
    <div class="run-toolbar"><nav class="workspace-tabs" aria-label="Run views">${tabs}</nav><div class="run-time-controls"><label>Time range <select id="event-range" aria-label="Time range">${ranges}</select></label>${button("Latest", 'data-latest="true"')}</div></div>
    ${runIncidents(run, runView.windowMinutes, runView.windowEnd)}
    <div id="run-panel">${tab === "logs" ? logsPanel(id, run) : metricsPanel(id, run)}</div>
    <p class="run-json-link"><a href="/api/v1/dashboard/runs/${encode(id)}">Agent JSON ↗</a></p>`;
}

function metricsPanel(id, run) {
  const range = runRange();
  const start = Date.parse(range.since) / 1000;
  const end = Date.parse(range.until) / 1000;
  const metrics = use(`/api/v1/dashboard/runs/${encode(id)}/metrics?${new URLSearchParams(range)}`);
  const samples = (metrics.data?.samples || []).filter((s) => s.at >= start && s.at <= end);
  const names = [...new Set(samples.flatMap((s) => Object.keys(s.metrics || {})))].slice(0, 12);
  const charts = [
    chart({ title: "Operation duration (s)", points: samples.map((s) => [s.at, s.elapsed_seconds]), start, end, tone: "accent" }),
    ...names.map((name) => chart({ title: name, points: samples.filter((s) => s.metrics[name] !== undefined).map((s) => [s.at, s.metrics[name]]), start, end, tone: "accent" })),
  ];
  const status = metrics.error || (!metrics.data ? "Loading metrics…" : metrics.data.error || "");
  const rows = run.pods
    .map((p) => `<tr><td>${escape(p.name)}</td><td>${escape(p.role ? p.role[0].toUpperCase() + p.role.slice(1) : "Unknown")}</td><td>${escape(p.node)}</td><td>${escape(p.problem || p.phase)}</td><td>${p.restarts}</td></tr>`)
    .join("");
  return `<p>Completed steps: ${escape(run.steps)}</p><p class="muted" role="status">${escape(status)}</p>
    <div class="chart-grid">${samples.length ? charts.join("") : metrics.data ? empty("No operations recorded in this time range") : ""}</div>
    <h2 class="scheduler-title">Processes</h2>${run.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}
    <table class="run-table"><thead><tr><th>Process</th><th>Kind</th><th>Node</th><th>State</th><th>Restarts</th></tr></thead><tbody>${rows}</tbody></table>${!run.pods.length ? empty("No current pods") : ""}`;
}

// ---- logs -----------------------------------------------------------------------------

const logQuery = (id) => {
  const params = new URLSearchParams({ q: document.getElementById("log-search")?.value || "", limit: "200", ...runRange(true) });
  const pod = document.getElementById("log-source")?.value;
  if (pod) params.set("pod", pod);
  return params;
};

export async function loadLogs(id, more = false) {
  const params = logQuery(id);
  const key = `${id}?${params}`;
  if (more && logState.cursor) params.set("cursor", logState.cursor);
  else if (!more && logState.key === key && logState.loading) return;
  const request = ++logState.request;
  logState.loading = true;
  if (!more) logState.key = key;
  ui.render();
  try {
    const data = await get(`/api/v1/dashboard/runs/${encode(id)}/logs?${params}`);
    if (request !== logState.request) return;
    logState.records = more ? [...logState.records, ...(data.records || [])] : data.records || [];
    logState.cursor = data.next_cursor;
    logState.error = data.error || null;
    logState.source = data.source;
  } catch (error) {
    if (request === logState.request) logState.error = error.message;
  } finally {
    if (request === logState.request) logState.loading = false;
    ui.render();
  }
}

function logsPanel(id, run) {
  const range = runRange(true);
  const scope =
    runView.eventAt === null
      ? ""
      : `<div class="log-time-scope"><span>${escape(range.since.slice(0, 10))} · ${escape(range.since.slice(11, 19))}–${escape(range.until.slice(11, 19))} UTC</span><a href="#run/${encode(id)}/logs" data-all-logs="true">All logs</a></div>`;
  const pods = run.pods.map((p) => `<option value="${escape(p.name)}">${escape(p.role || "Unknown")} · ${escape(p.node)} / ${escape(p.name)}</option>`).join("");
  const rows = logState.records
    .map(
      (r) =>
        `<div class="workspace-logrow"><span class="log-origin">${escape(r.timestamp || "No timestamp")} ${escape(r.role || "Unknown")} · ${escape(r.pod)}/${escape(r.container)}</span>${escape(r.message)}</div>`,
    )
    .join("");
  const status = logState.loading ? "Loading…" : logState.error || "Cloud Logging · Newest first";
  return `${run.shared_runtime ? '<p class="muted">These pods serve a shared LoRA runtime. Their logs can include other runs.</p>' : ""}${scope}
    <div class="log-toolbar"><input id="log-search" type="search" placeholder="Search logs" aria-label="Search logs"><select id="log-source" aria-label="Pod"><option value="">All pods</option>${pods}</select></div>
    <div id="log-status" role="status">${escape(status)}</div>
    <div id="log-lines">${rows || (logState.loading ? "" : empty("No logs match"))}</div>
    <div id="log-more">${logState.cursor && !logState.loading ? button("Older logs", 'data-older="true"') : ""}</div>`;
}

// Called after the logs tab is on screen: load when the query changed.
export function ensureLogs(id) {
  if (!document.getElementById("log-lines")) return;
  if (logState.key !== `${id}?${logQuery(id)}`) loadLogs(id);
}
