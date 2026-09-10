// The Nodes page: one lane per node, one row per GPU, allocation bars over
// the selected window from the recorded placement history, and an expansion
// with the GPU chart and a legend of everything on that node.

import { escape, encode, empty, button, morph } from "./ui.js";
import { chart } from "./charts.js";
import { ui, content, nodeNow, nodeTime, family } from "./store.js";
import { use } from "./cache.js";

const LANE = 26;
export const WINDOWS = [
  [600, "10 minutes"],
  [1800, "30 minutes"],
  [3600, "1 hour"],
  [10800, "3 hours"],
  [21600, "6 hours"],
  [86400, "24 hours"],
];

// ---- window ----------------------------------------------------------------------

export function timeWindow() {
  const now = nodeNow();
  const end = ui.nodeSelection.end ?? now;
  return { start: end - ui.nodeSelection.duration, now: Math.min(end, now), live: ui.nodeSelection.end === null };
}

function timeControl() {
  const { end, duration } = ui.nodeSelection;
  const local = (at) => new Date(at * 1000).toISOString().slice(0, 16);
  return `<div class="time-control"><label>Window <select data-time-duration>${WINDOWS.map(([seconds, label]) => `<option value="${seconds}" ${seconds === duration ? "selected" : ""}>${label}</option>`).join("")}</select></label>
    <label>Until <input type="datetime-local" data-time-end value="${end === null ? "" : local(end)}" step="60"> UTC</label>
    ${end === null ? '<span class="chip chip-static">Live</span>' : button("Live", 'data-time-live="true"')}</div>`;
}

// ---- history -> segments -----------------------------------------------------------

// Every placement that touched the window, as [start, end] on its node. Live
// placements run to now; a placement not recorded yet shows from now.
function segments({ start, now }) {
  const live = new Map(ui.state.placements.map((p) => [p.id, p]));
  const seen = new Set();
  const found = [];
  for (const entry of ui.state.history) {
    seen.add(entry.id);
    const end = live.has(entry.id) ? now : entry.last_seen;
    if (end < start || entry.first_seen > now) continue;
    found.push({ ...entry, ...(live.get(entry.id) || {}), start: entry.first_seen, end });
  }
  for (const placement of ui.state.placements) if (!seen.has(placement.id)) found.push({ ...placement, start: now, end: now });
  return found;
}

function duty(node, nodeSegments, { start, now }) {
  if (!node.gpu_capacity) return "—";
  if (nodeSegments.some((s) => s.devices.length !== s.device_count)) return "—";
  const busy = nodeSegments.reduce((sum, s) => sum + s.devices.length * Math.max(0, Math.min(now, s.end) - Math.max(start, s.start)), 0);
  return `${Math.round((100 * busy) / ((now - start) * node.gpu_capacity))}%`;
}

// ---- who held a shared GPU ---------------------------------------------------------

// Time-sliced GPUs run one job at a time. The workers' own operation records
// say who held the device when, so shared lanes paint those intervals.
const holdUrl = (runId, { start, now }) =>
  `/api/v1/dashboard/runs/${encode(runId)}/metrics?${new URLSearchParams({ since: new Date(start * 1000).toISOString(), until: new Date(now * 1000).toISOString() })}`;

function holdIntervals(placement, range) {
  const intervals = [];
  for (const runId of placement.run_ids || []) {
    for (const sample of use(holdUrl(runId, range)).data?.samples || []) {
      if (sample.role !== placement.role || (sample.node && sample.node !== placement.node)) continue;
      const from = Math.max(range.start, placement.start, sample.started_at ?? sample.at - (sample.elapsed_seconds || 0));
      const to = Math.min(range.now, placement.end, sample.at);
      if (to > from) intervals.push([from, to]);
    }
  }
  intervals.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [from, to] of intervals) {
    const last = merged.at(-1);
    if (last && from - last[1] < 3) last[1] = Math.max(last[1], to);
    else merged.push([from, to]);
  }
  return merged;
}

// ---- lane markup ------------------------------------------------------------------

const acceleratorLabel = (node) =>
  node.accelerator
    ? node.accelerator
        .replace(/^nvidia[- ]/i, "")
        .replace(/-\d+gb$/i, "")
        .replace(/-/g, " ")
        .toUpperCase()
    : node.gpu_capacity
      ? "GPU"
      : "CPU";

function claimLabel(node, placements, live) {
  if (live && !node.ready) return "Offline";
  if (!node.gpu_capacity) return "CPU";
  if (!placements.every((p) => p.devices.length === p.device_count)) return "— claimed";
  return `${new Set(placements.flatMap((p) => p.devices)).size}/${node.gpu_capacity} claimed`;
}

function deviceGroups(indexes) {
  const groups = [];
  for (const i of indexes) {
    const last = groups.at(-1);
    if (last && last.at(-1) === i - 1) last.push(i);
    else groups.push([i]);
  }
  return groups;
}

function trackBars(node, devices, nodeSegments, range) {
  const { start, now } = range;
  const duration = now - start || 1;
  const span = (from, to) => `left:${(Math.max(0, Math.max(start, from) - start) / duration) * 100}%;width:${(Math.max(0, Math.min(now, to) - Math.max(start, from)) / duration) * 100}%`;
  const sharersOf = (i) => nodeSegments.filter((s) => s.devices.length === 1 && s.devices[0] === devices[i].id).sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
  const shared = new Set(devices.map((_, i) => i).filter((i) => new Set(sharersOf(i).map((s) => s.id)).size > 1));
  const sharedBars = [...shared]
    .map((i) => {
      const sharers = sharersOf(i);
      const from = Math.max(start, Math.min(...sharers.map((s) => s.start)));
      const to = Math.min(now, Math.max(...sharers.map((s) => s.end)));
      const chosen = sharers.find((s) => s.id === ui.expanded) || sharers[0];
      const open = sharers.some((s) => s.id === ui.expanded);
      const within = (a, b) => `left:${((Math.max(a, from) - from) / (to - from || 1)) * 100}%;width:${((Math.min(b, to) - Math.max(a, from)) / (to - from || 1)) * 100}%`;
      const holds = sharers
        .flatMap((s) => holdIntervals(s, range).map(([a, b]) => `<span class="hold ${family(s.runtime_id)}" style="${within(a, b)}" title="${escape(s.label)} · ${escape(s.role || "")}"></span>`))
        .join("");
      const title = `Shared GPU: ${[...new Map(sharers.map((s) => [s.id, s])).values()].map((s) => `${s.label} ${s.role || ""}`.trim()).join(", ")}`;
      return `<button type="button" class="capacity-allocation shared ${open ? "selected" : ""}" data-placement="${escape(chosen.id)}" aria-expanded="${open}" title="${escape(title)}" style="${span(from, to)};top:${i * LANE + 2}px;height:${LANE - 4}px">${holds}</button>`;
    })
    .join("");
  const solid = nodeSegments
    .flatMap((s) => {
      const indexes = s.devices
        .map((id) => devices.findIndex((d) => d.id === id))
        .filter((i) => i >= 0 && !(s.devices.length === 1 && shared.has(i)))
        .sort((a, b) => a - b);
      return deviceGroups(indexes).map((group) => {
        const title = `${s.label}${s.role ? " · " + s.role : ""} · ${group.length} GPU${group.length === 1 ? "" : "s"}`;
        return `<button type="button" class="capacity-allocation ${family(s.runtime_id)} ${s.id === ui.expanded ? "selected" : ""}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}" title="${escape(title)}" style="${span(s.start, s.end)};top:${group[0] * LANE + 2}px;height:${group.length * LANE - 4}px"></button>`;
      });
    })
    .join("");
  return sharedBars + solid;
}

function lane(node, all, range) {
  const devices = node.devices || [];
  const nodeSegments = all.filter((s) => s.node === node.name);
  const placements = ui.state.placements.filter((p) => p.node === node.name);
  const current = nodeSegments.find((s) => s.id === ui.expanded);
  const bars = trackBars(node, devices, nodeSegments, range);
  const accelerator = acceleratorLabel(node);
  const track = devices.length
    ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-auto-rows:${LANE}px">${devices.map((d) => `<span title="${escape(d.id)}">${escape(d.name)}</span>`).join("")}</div><div class="capacity-track" style="height:${devices.length * LANE}px;--gpu-lane-height:${LANE}px">${bars}</div></div>`
    : "";
  return `<div class="node-placement-group"><div class="node-lane"><div class="node-lane-label" title="${escape(node.name)}"><span class="node-accelerator">${escape(accelerator)}</span><span class="claim-label">${escape(claimLabel(node, placements, range.live))}</span></div><div>${track}${!devices.length ? empty("No GPUs") : ""}</div><span class="node-duty" title="GPU allocation time over the selected window">${duty(node, nodeSegments, range)}</span></div>${current ? detail(current, range) : ""}</div>`;
}

// ---- expansion ---------------------------------------------------------------------

function detail(placement, range) {
  if (ui.device !== "all" && !placement.devices.includes(ui.device)) ui.device = "all";
  const { state } = ui;
  const run = state.runs.find((r) => (placement.run_ids || []).includes(r.run_id));
  const devices = state.cluster.nodes.find((n) => n.name === placement.node)?.devices || [];
  const workload = run?.workloads?.find((w) => w.uid === placement.id);
  const title = [
    (run?.model || placement.runtime_id || placement.label).split("/").at(-1),
    (placement.owner_id || run?.run_id)?.slice(0, 8),
    { trainer: "Trainer", sampler: "Sampler" }[placement.role] || "Unknown process",
    placement.role === "trainer" ? { lora: "LoRA", fft: "FFT" }[workload?.training_kind] : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const neighbours = state.placements.filter((p) => p.node === placement.node);
  const legend = (neighbours.some((p) => p.id === placement.id) ? neighbours : [placement, ...neighbours])
    .map(
      (p) =>
        `<button type="button" class="legend-entry ${family(p.runtime_id)}" data-placement="${escape(p.id)}" aria-pressed="${p.id === placement.id}"><span class="legend-swatch"></span><span class="legend-name">${escape(p.label)}</span><span class="legend-meta">${escape(p.role || "process")} · ${p.device_count} GPU${p.device_count === 1 ? "" : "s"}</span></button>`,
    )
    .join("");
  const picker = [button("All GPUs", `data-device="all" aria-pressed="${ui.device === "all"}"`)]
    .concat(placement.devices.map((id) => button(devices.find((d) => d.id === id)?.name || id.split("/").at(-1), `data-device="${escape(id)}" aria-pressed="${ui.device === id}"`)))
    .join("");
  const metrics = use(
    `/api/v1/dashboard/allocations/${encode(placement.id)}/metrics?${new URLSearchParams({ since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() })}`,
  );
  return `<section class="allocation-expansion" id="placement-detail"><div class="allocation-detail-head"><h2>${run ? `<a href="#run/${encode(run.run_id)}/metrics">${escape(title)} ↗</a>` : escape(title)}</h2><div class="allocation-legend" aria-label="Allocations on this node">${legend}</div></div>
    <div class="allocation-device-picker" aria-label="GPU selection">${picker}</div>
    <div>${gpuChart(metrics, range)}${run?.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}</div>
    <div><p class="muted">GPU memory</p><p>${gpuMemory(metrics)}</p></div></section>`;
}

const selectedDevices = (metrics) => (metrics.data?.devices || []).filter((d) => ui.device === "all" || ui.device === d.id);

function gpuChart(metrics, range) {
  if (!metrics.data) return `<p class="muted">${escape(metrics.error || "Loading GPU metrics…")}</p>`;
  const selected = selectedDevices(metrics);
  const byTime = new Map();
  for (const device of selected) for (const [at, value] of device.utilization || []) byTime.set(at, [...(byTime.get(at) || []), value]);
  const points = [...byTime].map(([at, values]) => [at, values.reduce((a, b) => a + b, 0) / values.length]);
  const reason = metrics.data.reason || selected.find((d) => d.reason)?.reason || "No samples for this GPU";
  return chart({ title: "GPU utilization", unit: "%", points, start: range.start, end: range.now, min: 0, max: 100, gapSeconds: 60, empty: reason });
}

function gpuMemory(metrics) {
  const latest = selectedDevices(metrics).map((d) => d.memory_mib?.at(-1)?.[1]);
  return latest.length && latest.every(Number.isFinite) ? `${(latest.reduce((a, b) => a + b, 0) / 1024).toFixed(1)} GiB` : "—";
}

// ---- page -----------------------------------------------------------------------------

export function renderNodes() {
  const range = timeWindow();
  const all = segments(range);
  const { cluster } = ui.state;
  const axis = [0, 1, 2, 3].map((tick) => `<span>${nodeTime(range.start + ((range.now - range.start) * tick) / 3)}</span>`).join("");
  morph(
    content,
    `<div class="nodes-heading"><h1 class="heading">Kubernetes nodes</h1>${timeControl()}</div>${!cluster.available ? empty(cluster.error || "Kubernetes unavailable") : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${axis}</div><span class="node-duty" title="GPU allocation time divided by capacity over the selected window">Duty</span></div>
    ${cluster.nodes.map((node) => lane(node, all, range)).join("") || empty("No nodes available")}`,
  );
}
