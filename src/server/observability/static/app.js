// OpenRL cluster dashboard entrypoint: shell, inventory polling, routing, and event delegation.
import { hoverActivity, resetActivity } from "./activity.js";
import { hoverChart, inspectChart, resetCharts } from "./charts.js";
import { copyLink, morph } from "./dom.js";
import { escape } from "./format.js";
import { attachNodeTime } from "./node-time.js";
import { beginRender, currentView, endRender, get, inspectUrl, now, parseView, rangeOf, retryQuery, saved, setCrosshair, syncHash, timeAt, ui, view, viewLink } from "./state.js";
import { adaptInventory } from "./visualization.js";
import { followLogs, loadLogs, logsState, pauseLogs, renderHealth, renderNodes, renderOverview, renderRun, stopLogs } from "./views.js";

const root = document.getElementById("app");
let inventoryTimer = null;
let inventoryController = null;
let copyTimer = null;

async function pollInventory() {
  if (ui.inventory.pending) return;
  clearTimeout(inventoryTimer);
  inventoryController?.abort();
  const controller = inventoryController = new AbortController();
  ui.inventory.pending = true;
  let retry = false;
  try {
    const { data, etag } = await get(inspectUrl("inventory"), controller.signal, ui.inventory.etag ? ui.inventory : null);
    if (controller.signal.aborted) return;
    ui.inventory.failures = 0;
    ui.inventory.error = data.error || null;
    ui.inventory.status = null;
    ui.inventory.stale = false;
    ui.inventory.etag = etag;
    if (!ui.inventory.error || !ui.inventory.data) ui.inventory.data = adaptInventory(data);
  } catch (error) {
    if (controller.signal.aborted) return;
    ui.inventory.failures += 1;
    retry = (!error.status || error.status >= 500 || error.status === 429) && ui.inventory.failures < 4;
    if (retry) ui.inventory.stale = Boolean(ui.inventory.data);
    else {
      ui.inventory.error = error.message;
      ui.inventory.status = error.status || null;
      ui.inventory.stale = false;
    }
  } finally {
    if (!controller.signal.aborted) {
      ui.inventory.pending = false;
      inventoryController = null;
      render();
      const delay = retry ? 2000 * 2 ** (ui.inventory.failures - 1) : 10000;
      inventoryTimer = setTimeout(() => { if (!document.hidden) pollInventory(); }, delay);
    }
  }
}

function render() {
  syncHash();
  beginRender();
  resetCharts();
  resetActivity();

  const inv = ui.inventory;
  const section = view.page === "run" ? (view.back ? "nodes" : "overview") : view.page;
  const connection = inv.error ? (inv.data ? "Stale snapshot" : "Unavailable") : inv.stale ? "Reconnecting…" : !inv.data ? "Connecting…" : inv.data.cluster.available ? "Connected" : "Cluster unavailable";
  const nav = [["overview", "Overview"], ["nodes", "Nodes"], ["health", "Health"]]
    .map(([page, label]) => `<a href="${escape(saved[page] || `#${page}`)}"${section === page ? ' aria-current="page"' : ""}>${escape(label)}</a>`).join("");

  let main;
  if (inv.data) {
    if (view.page === "run") main = renderRun(inv.data);
    else if (view.page === "nodes") { stopLogs(); main = renderNodes(inv.data); }
    else { stopLogs(); main = view.page === "health" ? renderHealth(inv.data) : renderOverview(inv.data); }
  } else {
    main = inv.error ? "" : "Loading cluster…";
  }

  endRender();
  morph(root, `<div class="product">
    <header class="appbar"><a href="${escape(saved.overview)}" class="brand">OpenRL</a><nav aria-label="Main">${nav}</nav><span id="connection" role="status"${inv.error ? ` title="${escape(inv.error)}"` : ""}>${escape(connection)}</span></header>
    <div id="snapshot-error" class="snapshot-error" role="status"${inv.error ? "" : " hidden"}><span>${escape(inv.error || "")}${inv.error && inv.data ? escape(` Showing the snapshot from ${inv.data.observed_at}.`) : ""}</span><button type="button" class="chip" data-retry-snapshot>Retry</button></div>
    <main id="content" class="pad">${main}</main>
  </div>`);

  if (view.page === "nodes" && view.node && ui.scrollSelection) {
    const detail = document.getElementById("placement-detail");
    if (detail) { ui.scrollSelection = false; detail.scrollIntoView({ block: "start" }); }
  }
  if (logsState.needsScroll) {
    const lines = document.getElementById("log-lines");
    if (lines) { lines.scrollTop = lines.scrollHeight - lines.clientHeight - logsState.fromBottom; }
    logsState.fromBottom = 0;
    logsState.needsScroll = false;
  }
}

ui.render = render;

function keepViewport() {
  root.style.setProperty("min-height", `${window.scrollY + innerHeight}px`);
}
function closeNodeDetails() {
  const prev = view.node;
  view.node = view.placement = null;
  root.style.removeProperty("min-height");
  render();
  if (prev) root.querySelector(`[data-node-details="${CSS.escape(prev)}"]`)?.focus({ preventScroll: true });
}

root.addEventListener("click", async (event) => {
  const target = event.target;
  if (target.closest("[data-retry-snapshot]")) { ui.inventory.failures = 0; pollInventory(); return; }

  const retryBtn = target.closest("[data-retry-query]");
  if (retryBtn) { retryQuery(retryBtn.dataset.retryQuery); return; }

  const filterBtn = target.closest("[data-run-filter]");
  if (filterBtn) { view.status = filterBtn.dataset.runFilter; render(); return; }

  const copyBtn = target.closest("[data-copy-view]");
  if (copyBtn) {
    copyBtn.textContent = await copyLink(new URL(currentView(true), location.href).href);
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { if (copyBtn.isConnected) copyBtn.textContent = "Copy link"; }, 2000);
    return;
  }

  const shiftBtn = target.closest("[data-time-shift]");
  if (shiftBtn) {
    const observed = now(ui.inventory.data), range = rangeOf(ui.inventory.data, view), dur = range.now - range.start;
    const end = range.now + Number(shiftBtn.dataset.timeShift) * dur;
    view.end = end >= observed ? null : end;
    render();
    return;
  }
  if (target.closest("[data-time-live]")) { view.end = null; render(); return; }

  const modeBtn = target.closest("[data-activity-mode]");
  if (modeBtn) { ui.activityMode = modeBtn.dataset.activityMode; render(); return; }

  const gpuToggle = target.closest("[data-toggle-gpu]");
  if (gpuToggle) {
    const key = gpuToggle.dataset.toggleGpu;
    if (ui.openGpus.has(key)) ui.openGpus.delete(key);
    else ui.openGpus.add(key);
    render();
    return;
  }

  const chartModeBtn = target.closest("[data-chart-mode]");
  if (chartModeBtn) { ui.chartModes.set(chartModeBtn.dataset.chartMode, chartModeBtn.dataset.mode); render(); return; }

  const nodeBtn = target.closest("[data-node-details]");
  if (nodeBtn) {
    const name = nodeBtn.dataset.nodeDetails;
    if (view.node === name) closeNodeDetails();
    else { root.style.removeProperty("min-height"); view.node = name; view.placement = null; view.gpu = "all"; ui.scrollSelection = false; render(); }
    return;
  }
  if (target.closest("[data-close-details]")) { closeNodeDetails(); return; }

  const devBtn = target.closest("[data-device]");
  if (devBtn) { keepViewport(); view.gpu = devBtn.dataset.device; render(); return; }

  const placementEl = target.closest("[data-placement]");
  if (placementEl && view.page === "nodes") {
    const id = placementEl.dataset.placement, nodeName = placementEl.dataset.placementNode || view.node;
    if (view.node !== nodeName) root.style.removeProperty("min-height");
    if (placementEl.closest(".capacity-allocation")) view.gpu = "all";
    view.node = nodeName;
    view.placement = id;
    render();
    return;
  }

  const pickTrack = target.closest(".activity-track[data-pick-logs]");
  if (pickTrack && view.page === "run") {
    const range = rangeOf(ui.inventory.data, view);
    const box = pickTrack.getBoundingClientRect();
    const at = Math.round(range.start + Math.max(0, Math.min(1, (event.clientX - box.left) / (box.width || 1))) * (range.now - range.start));
    location.hash = viewLink(`run/${encodeURIComponent(view.id)}/logs`, { duration: view.duration, end: range.now, back: view.back, event: at, q: view.q, pod: view.pod });
    return;
  }

  if (target.closest("[data-log-follow]")) { if (view.end === null) pauseLogs(); else followLogs(); return; }
  if (target.closest("[data-log-retry]") && logsState.params) { loadLogs(logsState.params, logsState.failedCursor); return; }
  const olderBtn = target.closest("[data-log-older]");
  if (olderBtn && logsState.params) {
    const cursor = olderBtn.dataset.logOlder;
    if (view.end === null) {
      view.end = logsState.data?.windowEnd ?? logsState.rangeNow;
      logsState.queryKey = `${logsState.filterKey}:${view.end}`;
      if (logsState.data) logsState.data = { ...logsState.data, queryKey: logsState.queryKey };
    }
    loadLogs(logsState.params, cursor);
  }
});

root.addEventListener("input", (event) => {
  const el = event.target;
  if (el.id === "run-search" || el.id === "log-search") { view.q = el.value; render(); }
});

root.addEventListener("change", (event) => {
  const el = event.target;
  if (el.id === "log-source") { view.pod = el.value; render(); }
  else if (el.matches("[data-log-everything]")) { logsState.everything = el.checked; render(); }
  else if (el.matches("[data-time-duration]")) {
    const range = rangeOf(ui.inventory.data, view);
    view.duration = Number(el.value);
    if (view.end !== null) view.end = range.now;
    render();
  } else if (el.matches("[data-time-end]")) {
    if (!el.validity.valid) { el.reportValidity(); return; }
    const observed = now(ui.inventory.data);
    const end = el.value ? Date.parse(`${el.value}Z`) / 1000 : null;
    view.end = Number.isFinite(end) ? Math.min(end, observed) : null;
    render();
  }
});

root.addEventListener("pointermove", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot) { hoverChart(plot, event); return; }
  const section = event.target.closest("section[data-crosshair-host]");
  if (section) { hoverActivity(section, event, rangeOf(ui.inventory.data, view)); return; }
  const track = event.target.closest(".capacity-track");
  if (track) setCrosshair(event.buttons ? null : timeAt({ currentTarget: track, clientX: event.clientX }, rangeOf(ui.inventory.data, view)), rangeOf(ui.inventory.data, view));
});

root.addEventListener("pointerout", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot && !plot.contains(event.relatedTarget)) {
    const hover = plot.querySelector(".chart-hover");
    if (hover) hover.hidden = true;
    setCrosshair(null);
  }
  const section = event.target.closest("section[data-crosshair-host]");
  if (section && !section.contains(event.relatedTarget)) {
    const tip = section.querySelector(".activity-tooltip");
    if (tip) tip.hidden = true;
    setCrosshair(null);
  }
  const track = event.target.closest(".capacity-track");
  if (track && !track.contains(event.relatedTarget)) setCrosshair(null);
});

root.addEventListener("scroll", (event) => {
  const box = event.target;
  if (box.id === "log-lines" && view.end === null && box.scrollHeight - box.clientHeight - box.scrollTop > 24) pauseLogs();
}, true);

root.addEventListener("keydown", (event) => {
  const plot = event.target.closest(".chart-plot");
  if (plot && inspectChart(plot, event.key)) { event.preventDefault(); event.stopPropagation(); return; }
  const placementEl = event.target.closest("[data-placement]");
  if (placementEl && view.page === "nodes" && ["Enter", " "].includes(event.key)) {
    event.preventDefault();
    view.placement = placementEl.dataset.placement;
    render();
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || event.defaultPrevented) return;
  const picker = root.querySelector("details.time-picker[open]");
  if (picker) { picker.open = false; picker.querySelector("summary")?.focus(); event.preventDefault(); return; }
  if (view.page === "nodes" && (view.node || view.placement)) closeNodeDetails();
});

window.addEventListener("click", (event) => {
  const picker = root.querySelector("details.time-picker[open]");
  if (picker && !picker.contains(event.target)) picker.open = false;
});

window.addEventListener("scroll", () => {
  for (const tip of root.querySelectorAll(".activity-tooltip")) tip.hidden = true;
  setCrosshair(null);
});
window.addEventListener("resize", () => {
  for (const el of root.querySelectorAll(".activity-tooltip, .chart-hover")) el.hidden = true;
  setCrosshair(null);
});

window.addEventListener("hashchange", () => {
  const next = parseView();
  if (view.page !== next.page || view.id !== next.id) {
    window.scrollTo(0, 0);
    ui.scrollSelection = true;
    root.style.removeProperty("min-height");
  }
  Object.assign(view, next);
  render();
});

document.addEventListener("visibilitychange", () => { if (!document.hidden) pollInventory(); });

attachNodeTime(root);
render();
pollInventory();
