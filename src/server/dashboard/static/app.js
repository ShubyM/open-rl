// Entry point: routing, the poll loop, and the page-wide event handlers.
// Every page is a function of the shared store; this file decides which page
// runs, keeps the snapshot fresh, and turns user input into state changes.

import { encode, morph } from "./ui.js";
import { runs, scheduler, health, experiments } from "./views.js";
import { hoverChart, inspectChart } from "./charts.js";
import { root, content, ui, route, get, nodeNow } from "./store.js";
import { renderNodes } from "./nodes.js";
import { runPage, ensureLogs, loadLogs, runView, resetRunWindow, syncRunRoute, setLogFilter, setLogEvent, toggleLogFollow } from "./run.js";
import { use, beginRender, endRender } from "./cache.js";

const pageOf = (page) => (!page || ["run", "runs"].includes(page) ? "overview" : page);

function render() {
  if (!ui.state) return;
  const [page, id, tab] = route();
  syncRunRoute(page, id, tab);
  const current = pageOf(page);
  root.querySelectorAll(".appbar nav a").forEach((link) => {
    if (link.hash === `#${current}`) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  beginRender();
  try {
    if (page === "nodes") renderNodes();
    else if (page === "scheduler") morph(content, scheduler(ui.state));
    else if (page === "health") morph(content, health(ui.state));
    else if (page === "experiments") morph(content, experiments(use("/api/v1/dashboard/experiments")));
    else if (page === "run") morph(content, runPage(id, tab));
    else morph(content, runs(ui.state));
  } finally {
    endRender();
  }
  if (page === "run" && tab === "logs") ensureLogs(id);
}
ui.render = render;

// ---- interactions ----------------------------------------------------------

root.addEventListener("click", (event) => {
  const jump = event.target.closest("a[data-scheduler-placement]");
  if (jump) {
    ui.nodeSelection = { duration: ui.nodeSelection.duration, end: null };
    ui.expanded = jump.dataset.schedulerPlacement;
    ui.gpuGroup = null;
    ui.device = "all";
  }
  const incident = event.target.closest("[data-event-at], [data-all-logs]");
  if (incident) {
    event.preventDefault();
    setLogEvent(incident.dataset.eventAt ? Number(incident.dataset.eventAt) : null);
    if (route()[2] === "logs") render();
    else location.hash = `run/${encode(route()[1])}/logs`;
    return;
  }
  const allocation = event.target.closest("[data-placement]");
  if (allocation) {
    if (!ui.expanded) {
      ui.gpuGroup = null;
      ui.device = "all";
    }
    ui.expanded = ui.expanded === allocation.dataset.placement && !allocation.matches(".hold, .activity-label, .activity-block") ? null : allocation.dataset.placement;
    render();
    return;
  }
  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.device) {
    ui.device = target.dataset.device;
    render();
  }
  if (target.dataset.activityView) {
    ui.activityView = target.dataset.activityView;
    render();
  }
  if (target.dataset.timeLive) {
    ui.nodeSelection = { ...ui.nodeSelection, end: null };
    render();
  }
  if (target.dataset.timeShift) {
    const now = nodeNow();
    const end = (ui.nodeSelection.end ?? now) + Number(target.dataset.timeShift) * ui.nodeSelection.duration;
    ui.nodeSelection = { ...ui.nodeSelection, end: end >= now ? null : end };
    render();
  }
  if (target.dataset.logFollow !== undefined) toggleLogFollow();
  if (target.dataset.older) loadLogs(route()[1], true);
});

root.addEventListener("keydown", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot && inspectChart(plot, event.key)) {
    event.preventDefault();
    return;
  }
  if (event.key === "Escape") {
    const picker = root.querySelector(".time-picker[open]");
    if (picker) {
      picker.open = false;
      picker.querySelector("summary").focus();
      event.preventDefault();
      return;
    }
  }
  if (!event.defaultPrevented && event.key === "Escape" && ui.expanded) {
    const previous = ui.expanded;
    ui.expanded = null;
    ui.gpuGroup = null;
    render();
    root.querySelector(`[data-placement="${CSS.escape(previous)}"]`)?.focus();
  }
});

let searchTimer;
root.addEventListener("input", (event) => {
  if (event.target.id === "log-search") {
    setLogFilter("q", event.target.value);
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      if (route()[0] === "run" && route()[2] === "logs") loadLogs(route()[1]);
    }, 250);
  }
});

root.addEventListener("change", (event) => {
  const { target } = event;
  if (target.id === "log-source") {
    setLogFilter("pod", target.value);
    loadLogs(route()[1]);
  }
  if (target.id === "event-range") {
    runView.windowMinutes = Number(target.value);
    resetRunWindow();
    render();
  }
  if (target.dataset.timeDuration !== undefined) {
    ui.nodeSelection = { ...ui.nodeSelection, duration: Number(target.value) };
    render();
  }
  if (target.dataset.timeEnd !== undefined) {
    if (!target.validity.valid) return;
    const end = target.value ? Date.parse(`${target.value}Z`) / 1000 : null;
    ui.nodeSelection = { ...ui.nodeSelection, end: Number.isFinite(end) ? Math.min(end, nodeNow()) : null };
    render();
  }
});

root.addEventListener("pointermove", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot) hoverChart(plot, event.clientX);
});
root.addEventListener("pointerleave", (event) => event.target.classList?.contains("chart-plot") && (event.target.querySelector(".chart-hover").hidden = true), true);
root.addEventListener("focusout", (event) => {
  if (event.target.matches(".chart-plot")) event.target.querySelector(".chart-hover")?.setAttribute("hidden", "");
});
window.addEventListener("resize", () => root.querySelectorAll(".chart-hover").forEach((hover) => { hover.hidden = true; }));
document.addEventListener("click", (event) => {
  root.querySelectorAll(".time-picker[open]").forEach((picker) => {
    if (!picker.contains(event.target)) picker.open = false;
  });
});

window.addEventListener("hashchange", () => {
  clearTimeout(searchTimer);
  syncRunRoute(...route());
  content.innerHTML = "";
  render();
});

// ---- polling -----------------------------------------------------------------

let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    ui.state = await get("/api/v1/dashboard/snapshot");
    document.getElementById("connection").textContent = ui.state.recorded_at ? "Recording" : ui.state.demo ? "Demo" : ui.state.cluster.available ? "Connected" : "Cluster unavailable";
    document.getElementById("connection").title = ui.state.recorded_at ? `Captured ${ui.state.recorded_at}` : "";
    render();
  } catch (error) {
    document.getElementById("connection").textContent = error.message;
  } finally {
    refreshing = false;
  }
}
refresh();
setInterval(() => {
  if (!document.hidden && !ui.state?.recorded_at) refresh();
}, 10000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
