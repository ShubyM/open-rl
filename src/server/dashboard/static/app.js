// Entry point: routing, the poll loop, and the page-wide event handlers.
// Every page is a function of the shared store; this file decides which page
// runs, keeps the snapshot fresh, and turns user input into state changes.

import { encode, morph } from "./ui.js";
import { runs, scheduler, health, experiments } from "./views.js";
import { hoverChart } from "./charts.js";
import { root, content, ui, route, get } from "./store.js";
import { renderNodes } from "./nodes.js";
import { runPage, ensureLogs, loadLogs, runView, logState, resetRunWindow } from "./run.js";
import { use } from "./cache.js";

const pageOf = (page) => (!page || ["run", "runs"].includes(page) ? "overview" : page);

function render() {
  if (!ui.state) return;
  const [page, id, tab] = route();
  const current = pageOf(page);
  root.querySelectorAll(".appbar nav a").forEach((link) => link.toggleAttribute("aria-current", link.hash === `#${current}`));
  if (page === "nodes") renderNodes();
  else if (page === "scheduler") morph(content, scheduler(ui.state));
  else if (page === "health") morph(content, health(ui.state));
  else if (page === "experiments") morph(content, experiments(use("/api/v1/dashboard/experiments")));
  else if (page === "run") {
    morph(content, runPage(id, tab));
    if (tab === "logs") ensureLogs(id);
  } else morph(content, runs(ui.state));
}
ui.render = render;

// ---- interactions ----------------------------------------------------------

root.addEventListener("click", (event) => {
  const jump = event.target.closest("a[data-scheduler-placement]");
  if (jump) {
    ui.nodeSelection = { duration: ui.nodeSelection.duration, end: null };
    ui.expanded = jump.dataset.schedulerPlacement;
    ui.device = "all";
  }
  const incident = event.target.closest("[data-event-at], [data-all-logs]");
  if (incident) {
    event.preventDefault();
    runView.eventAt = incident.dataset.eventAt ? Number(incident.dataset.eventAt) : null;
    logState.key = null;
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
    render();
  }
  if (target.dataset.timeLive) {
    ui.nodeSelection = { ...ui.nodeSelection, end: null };
    render();
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

let searchTimer;
root.addEventListener("input", (event) => {
  if (event.target.id === "log-search") {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadLogs(route()[1]), 250);
  }
});

root.addEventListener("change", (event) => {
  const { target } = event;
  if (target.id === "log-source") loadLogs(route()[1]);
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
    const end = target.value ? Date.parse(`${target.value}:00Z`) / 1000 : null;
    ui.nodeSelection = { ...ui.nodeSelection, end: Number.isFinite(end) ? end : null };
    render();
  }
});

root.addEventListener("pointermove", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot) hoverChart(plot, event.clientX);
});
root.addEventListener("pointerleave", (event) => event.target.classList?.contains("chart-plot") && (event.target.querySelector(".chart-hover").hidden = true), true);

window.addEventListener("hashchange", () => {
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
    document.getElementById("connection").textContent = ui.state.cluster.available ? "Connected" : "Cluster unavailable";
    // A focused text field keeps its page still; everything else is patched in place.
    const typing = content.contains(document.activeElement) && ["INPUT", "SELECT"].includes(document.activeElement.tagName);
    if (!typing || !ui.state || content.textContent === "Loading cluster…") render();
  } catch (error) {
    document.getElementById("connection").textContent = error.message;
  } finally {
    refreshing = false;
  }
}
refresh();
setInterval(refresh, 10000);
