// One run's page: metrics tab with operation and cloud charts, and a logs tab
// with search, pod filter and paging. The time window is shared by both.

import { runIncidents } from "./timeline.js";
import { escape, encode, empty, button, runStatus } from "./ui.js";
import { renderMetricChart, disposeMetricCharts } from "./charts.js";
import { ui, content, route, get } from "./store.js";
import { loadMetricView } from "./cache.js";

// Window and event selection for the run being inspected.
export const runView = { id: null, windowMinutes: 30, windowEnd: 0, eventAt: null };
// Log paging; the request counter discards replies from superseded loads.
export const logState = { request: 0, timer: null, cursor: null };

export function resetRunWindow() {
  runView.eventAt = null;
  runView.windowEnd = Date.parse(ui.state.observed_at) / 1000;
  logState.cursor = null;
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
    runView.eventAt = null;
    runView.windowEnd = Date.parse(ui.state.observed_at) / 1000;
  }
  const run = ui.state.runs.find((r) => r.run_id === id);
  if (!run) {
    content.innerHTML = empty("Run not found");
    return;
  }
  if (tab === "overview") tab = "metrics";
  if (tab === "events") {
    location.replace(`#run/${encode(id)}/logs`);
    return;
  }
  const tabs = ["metrics", "logs"];
  const title = [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), { lora: "LoRA", full: "FFT", fft: "FFT" }[run.fine_tuning_type]]
    .filter(Boolean)
    .join(" · ");
  const description = [run.display_name, run.recipe_name].filter((value, index, values) => value && value !== title && values.indexOf(value) === index).join(" · ");
  const rangePicker = [10, 30, 60].map((value) => `<option value="${value}" ${runView.windowMinutes === value ? "selected" : ""}>Last ${value} minutes</option>`).join("");
  content.innerHTML = `<p class="overview-back"><a href="#overview">← Overview</a></p><div class="run-heading"><h1 class="heading" title="${escape(run.run_id)}">${escape(title)}</h1>${runStatus(run.display_status || run.status)}</div>${description ? `<p class="run-description">${escape(description)}</p>` : ""}<div class="run-toolbar"><nav class="workspace-tabs" aria-label="Run views">${tabs.map((t) => `<a href="#run/${encode(id)}/${t}" ${tab === t ? 'aria-current="page"' : ""}>${t[0].toUpperCase() + t.slice(1)}</a>`).join("")}</nav><div class="run-time-controls"><label>Time range <select id="event-range" aria-label="Time range">${rangePicker}</select></label>${button("Latest", 'data-latest="true"')}</div></div>${runIncidents(run, runView.windowMinutes, runView.windowEnd)}<div id="run-panel"></div><p class="run-json-link"><a href="/api/v1/dashboard/runs/${encode(id)}">Agent JSON ↗</a></p>`;
  const panel = document.getElementById("run-panel");
  if (tab === "logs") {
    const range = runRange(true);
    const scope =
      runView.eventAt === null
        ? ""
        : `<div class="log-time-scope"><span>${escape(range.since.slice(0, 10))} · ${escape(range.since.slice(11, 19))}–${escape(range.until.slice(11, 19))} UTC</span><a href="#run/${encode(id)}/logs" data-all-logs="true">All logs</a></div>`;
    const pods = run.pods.map((p) => `<option value="${escape(p.name)}">${escape(p.role || "Unknown")} · ${escape(p.node)} / ${escape(p.name)}</option>`).join("");
    panel.innerHTML = `${run.shared_runtime ? '<p class="muted">These pods serve a shared LoRA runtime. Their logs can include other runs.</p>' : ""}${scope}<div class="log-toolbar"><input id="log-search" type="search" placeholder="Search logs" aria-label="Search logs"><select id="log-source" aria-label="Log source"><option value="">All pods</option>${pods}</select></div><div id="log-status" role="status"></div><div id="log-lines"></div><div id="log-more"></div>`;
    loadLogs(id);
    return;
  }
  if (tab === "agent") {
    panel.innerHTML = `<p><a href="/api/v1/dashboard/runs/${encode(id)}">Open JSON endpoint ↗</a></p><pre>${escape(JSON.stringify(run, null, 2))}</pre>`;
    return;
  }
  const rows = run.pods
    .map((p) => `<tr><td>${escape(p.name)}</td><td>${escape(p.role ? p.role[0].toUpperCase() + p.role.slice(1) : "Unknown")}</td><td>${escape(p.node)}</td><td>${escape(p.problem || p.phase)}</td><td>${p.restarts}</td></tr>`)
    .join("");
  const processes = `${run.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}<table class="run-table"><thead><tr><th>Process</th><th>Kind</th><th>Node</th><th>State</th><th>Restarts</th></tr></thead><tbody>${rows}</tbody></table>${!run.pods.length ? empty("No current pods") : ""}`;
  panel.innerHTML = `<p>Completed steps: ${escape(run.steps)}</p><p id="operation-metric-status" class="muted" role="status"></p><div id="operation-metrics"></div><h2 class="scheduler-title">Processes</h2>${processes}`;
  loadRunMetrics(id);
}

function paintRunMetrics(view) {
  if (ui.runMetricView !== view || !view.element.isConnected || !view.data) return;
  const { data, element: panel, start, end } = view;
  const samples = (data.samples || []).filter((sample) => sample.at >= start && sample.at <= end);
  const names = Array.from(new Set(samples.flatMap((sample) => Object.keys(sample.metrics || {})))).slice(0, 12);
  const charts = [
    ["Operation duration (s)", samples.map((sample) => [sample.at, sample.elapsed_seconds])],
    ...names.map((name) => [name, samples.filter((sample) => sample.metrics[name] !== undefined).map((sample) => [sample.at, sample.metrics[name]])]),
    ...(data.gke?.series || []).map((series) => [`${series.name} (${series.unit}) · ${series.role}${series.device ? " · " + series.device : ""}`, series.points]),
  ]
    .map(([name, points]) => [name, (points || []).filter(([at, value]) => at >= start && at <= end && Number.isFinite(value))])
    .filter(([, points]) => points.length);
  view.status.textContent = data.gke?.error || data.error || "";
  if (!charts.length) {
    disposeMetricCharts(panel);
    panel.innerHTML = empty("No metrics reported in this time range");
    view.keys = null;
    return;
  }
  const keys = JSON.stringify(charts.map(([name]) => name));
  if (keys !== view.keys) {
    disposeMetricCharts(panel);
    panel.innerHTML = charts.map(() => '<section class="operation-chart"></section>').join("");
    view.keys = keys;
  }
  charts.forEach(([title, points], index) => renderMetricChart(panel.children[index], { samples: points, start, end, title, tone: "accent" }));
}

export function loadRunMetrics(id) {
  const element = document.getElementById("operation-metrics");
  if (!element) return;
  const range = runRange();
  const url = `/api/v1/dashboard/runs/${encode(id)}/metrics?${new URLSearchParams(range)}`;
  if (ui.runMetricView?.element !== element || ui.runMetricView.url !== url)
    ui.runMetricView = {
      element,
      url,
      status: document.getElementById("operation-metric-status"),
      start: Date.parse(range.since) / 1000,
      end: Date.parse(range.until) / 1000,
      keys: ui.runMetricView?.element === element ? ui.runMetricView.keys : null,
    };
  return loadMetricView(ui.runMetricView, paintRunMetrics, () => ui.runMetricView);
}

export async function loadLogs(id, more = false) {
  const request = ++logState.request;
  const params = new URLSearchParams({ q: document.getElementById("log-search").value, limit: "200", ...runRange(true) });
  const pod = document.getElementById("log-source").value;
  if (pod) params.set("pod", pod);
  if (more && logState.cursor) params.set("cursor", logState.cursor);
  document.getElementById("log-status").textContent = "Loading…";
  try {
    const data = await get(`/api/v1/dashboard/runs/${encode(id)}/logs?${params}`);
    if (request !== logState.request || route()[2] !== "logs") return;
    const html = (data.records || [])
      .map(
        (r) =>
          `<div class="workspace-logrow"><span class="log-origin">${escape(r.timestamp || "No timestamp")} ${escape(r.role || "Unknown")} · ${escape(r.pod)}/${escape(r.container)}${r.attempt === null || r.attempt === undefined ? "" : ` #${r.attempt}`}</span>${escape(r.message)}</div>`,
      )
      .join("");
    if (more) document.getElementById("log-lines").insertAdjacentHTML("beforeend", html);
    else document.getElementById("log-lines").innerHTML = html || empty("No collected logs match");
    logState.cursor = data.next_cursor;
    document.getElementById("log-status").textContent =
      data.error || `${data.source === "gke" ? "GKE logs" : "Collected logs"} · Newest first · History may be incomplete${data.source_note ? " · " + data.source_note : ""}`;
    document.getElementById("log-more").innerHTML = logState.cursor ? button("Older logs", 'data-older="true"') : "";
  } catch (error) {
    if (request === logState.request && document.getElementById("log-status")) document.getElementById("log-status").textContent = error.message;
  }
}
