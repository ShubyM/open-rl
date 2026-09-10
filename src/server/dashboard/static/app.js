// Entry point: routing, the poll loop, and the page-wide event handlers.
// Each page renders itself from the shared store; this file only decides
// which page runs and keeps the snapshot fresh.

import { bindTimeRange } from "./time-range.js";
import { encode, morph } from "./ui.js";
import { runs, scheduler, health } from "./views.js";
import { disposeMetricCharts } from "./charts.js";
import { root, content, ui, route, get, nodeNow } from "./store.js";
import { renderNodes, loadGpu, paintGpu } from "./nodes.js";
import { runPage, loadRunMetrics, loadLogs, runView, logState, resetRunWindow } from "./run.js";
import { experimentsPage } from "./experiments.js";

const pageOf = (page) => (!page || ["run", "runs"].includes(page) ? "overview" : page === "diagnostics" ? "health" : page);

let renderedPage = null;
function render() {
  if (!ui.state) return;
  ++logState.request;
  const [page, id, tab] = route();
  const currentPage = pageOf(page);
  // A page change starts clean. Within a page the markup is patched in place,
  // so polling never resets scroll, focus, or an open allocation panel.
  if (currentPage !== renderedPage || page === "run") {
    disposeMetricCharts(content);
    ui.gpuView = null;
    ui.runMetricView = null;
    if (currentPage !== renderedPage) content.innerHTML = "";
    renderedPage = currentPage;
  }
  root.querySelectorAll(".appbar nav a").forEach((link) => {
    if (link.hash === `#${currentPage}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (!page || page === "overview" || page === "runs") morph(content, runs(ui.state));
  else if (page === "scheduler") morph(content, scheduler(ui.state));
  else if (page === "run") runPage(id, tab);
  else if (page === "diagnostics") location.replace("#health");
  else if (page === "health") morph(content, health(ui.state));
  else if (page === "experiments") experimentsPage();
  else {
    renderNodes();
    if (ui.expanded) loadGpu(ui.expanded, true);
  }
}
ui.render = render;

bindTimeRange(
  root,
  () => ui.nodeSelection,
  (selection) => {
    ui.nodeSelection = selection;
    render();
  },
  nodeNow,
);

// ---- interactions ----------------------------------------------------------

root.addEventListener("click", (event) => {
  const allocation = event.target.closest("a[data-scheduler-placement]");
  if (allocation) {
    ui.nodeSelection = { duration: ui.nodeSelection.duration, end: null };
    ui.expanded = allocation.dataset.schedulerPlacement;
    ui.device = "all";
  }

  const incident = event.target.closest("[data-event-at], [data-all-logs]");
  if (incident) {
    event.preventDefault();
    runView.eventAt = incident.dataset.eventAt ? Number(incident.dataset.eventAt) : null;
    logState.cursor = null;
    if (route()[2] === "logs") render();
    else location.hash = `run/${encode(route()[1])}/logs`;
    return;
  }

  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.placement) {
    ui.expanded = ui.expanded === target.dataset.placement ? null : target.dataset.placement;
    ui.device = "all";
    render();
  }
  if (target.dataset.device) {
    ui.device = target.dataset.device;
    root.querySelectorAll("[data-device]").forEach((item) => item.setAttribute("aria-pressed", String(item.dataset.device === ui.device)));
    if (ui.gpuView) paintGpu(ui.gpuView);
  }
  if (target.dataset.latest) {
    resetRunWindow();
    render();
  }
  if (target.dataset.older) loadLogs(route()[1], true);
});

root.addEventListener("keydown", (event) => {
  if (!event.defaultPrevented && event.key === "Escape" && ui.expanded) {
    const previous = ui.expanded;
    ui.expanded = null;
    render();
    root.querySelector(`[data-placement="${CSS.escape(previous)}"]`)?.focus();
  }
});

root.addEventListener("input", (event) => {
  if (event.target.id === "log-search") {
    clearTimeout(logState.timer);
    logState.timer = setTimeout(() => loadLogs(route()[1]), 250);
  }
});

root.addEventListener("change", (event) => {
  if (event.target.id === "log-source") loadLogs(route()[1]);
  if (event.target.id === "event-range") {
    runView.windowMinutes = Number(event.target.value);
    resetRunWindow();
    render();
  }
});

window.addEventListener("hashchange", render);

// ---- polling -----------------------------------------------------------------

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    ui.state = await get("/api/v1/dashboard/snapshot");
    document.getElementById("connection").textContent = ui.state.demo ? "Demo" : ui.state.cluster.available ? "Connected" : "Cluster unavailable";
    // Keep an open time editor stable; a closed rolling picker can refresh.
    const focusedTimeAction = document.activeElement?.closest("[data-node-time]")?.dataset.nodeTime;
    const closedTimePicker = focusedTimeAction && root.querySelector(".node-time-popover")?.hidden;
    const focusedDevice = document.activeElement?.dataset.device;
    // Preserve text selection, logs and focused controls while polling.
    if (!["run", "diagnostics"].includes(route()[0]) && (!content.contains(document.activeElement) || closedTimePicker || focusedDevice)) {
      render();
      if (closedTimePicker) root.querySelector(`[data-node-time="${focusedTimeAction}"]`)?.focus();
      else if (focusedDevice) root.querySelector(`[data-device="${CSS.escape(focusedDevice)}"]`)?.focus();
    } else if (content.textContent === "Loading cluster…") render();
    else if (route()[0] === "run" && document.getElementById("operation-metrics")) loadRunMetrics(route()[1]);
  } catch (error) {
    document.getElementById("connection").textContent = error.message;
  } finally {
    refreshing = false;
  }
}
refresh();
setInterval(refresh, 10000);
