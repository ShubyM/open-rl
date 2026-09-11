// The Nodes page: one lane per node, one row per GPU, allocation bars over
// the selected window from the recorded placement history, and an expansion
// with one GPU-group chart and the workloads that used those devices.

import { escape, encode, empty, button, morph } from "./ui.js";
import { chart } from "./charts.js";
import { ui, content, nodeNow, nodeTime, family } from "./store.js";
import { use } from "./cache.js";

const laneHeight = (devices) => (devices.length === 1 ? 30 : 22);
export const WINDOWS = [
  [60, "1 minute"],
  [600, "10 minutes"],
  [1800, "30 minutes"],
  [3600, "1 hour"],
  [10800, "3 hours"],
  [21600, "6 hours"],
  [86400, "24 hours"],
];

// ---- window ----------------------------------------------------------------------

export function timeWindow() {
  const observed = nodeNow();
  const now = Number.isFinite(ui.nodeSelection.end) ? Math.min(ui.nodeSelection.end, observed) : observed;
  const duration = Math.min(86400, Math.max(60, Number(ui.nodeSelection.duration) || 1800));
  return { start: now - duration, now, live: ui.nodeSelection.end === null };
}

function timeControl(range) {
  const { end, duration } = ui.nodeSelection;
  const local = (at) => new Date(at * 1000).toISOString().slice(0, 16);
  const chevron = (rotation = 0) =>
    `<svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(${rotation} 8 8)"/></svg>`;
  const label =
    end === null
      ? `Last ${WINDOWS.find(([seconds]) => seconds === duration)?.[1] || `${duration / 60} minutes`}`
      : `${local(range.now).slice(5, 10)} · ${nodeTime(range.start)} – ${nodeTime(range.now)}`;
  return `<div class="time-control" aria-label="Node time range">
    <button type="button" class="time-shift" data-time-shift="-1" aria-label="Previous time window">${chevron(90)}</button>
    <details class="time-picker" data-key="node-time-picker"><summary class="time-summary">${escape(label)}${chevron()}</summary>
      <div class="time-popover"><label>Window <select data-time-duration>${WINDOWS.map(([seconds, text]) => `<option value="${seconds}" ${seconds === duration ? "selected" : ""}>${text}</option>`).join("")}</select></label>
      <label>Until (UTC)<input type="datetime-local" data-time-end value="${end === null ? "" : local(end)}" max="${local(nodeNow())}" step="60"></label>
      ${end === null ? '<span class="muted">Following current time</span>' : button("Return to live", 'data-time-live="true"')}</div>
    </details><button type="button" class="time-shift" data-time-shift="1" aria-label="Next time window" ${range.live ? "disabled" : ""}>${chevron(-90)}</button></div>`;
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
    found.push({
      ...entry,
      ...(live.get(entry.id) || {}),
      start: entry.first_seen,
      end,
      ended: !live.has(entry.id),
    });
  }
  for (const placement of ui.state.placements) if (!seen.has(placement.id) && now >= nodeNow()) found.push({ ...placement, start: now, end: now });
  return found;
}

function mergeIntervals(intervals) {
  const merged = [];
  for (const [from, to] of intervals.sort((a, b) => a[0] - b[0])) {
    const last = merged.at(-1);
    if (last && from <= last[1]) last[1] = Math.max(last[1], to);
    else merged.push([from, to]);
  }
  return merged;
}

function duty(node, nodeSegments, { start, now }) {
  if (!node.gpu_capacity || now <= start || nodeSegments.some((s) => s.devices.length !== s.device_count || s.devices.some((id) => !node.devices.some((d) => d.id === id))))
    return "—";
  // A shared GPU counts once, even when several placements claim it.
  const byDevice = new Map();
  for (const s of nodeSegments)
    for (const id of s.devices) {
      if (!byDevice.has(id)) byDevice.set(id, []);
      byDevice.get(id).push([Math.max(start, s.start), Math.min(now, s.end)]);
    }
  const busy = [...byDevice.values()].flatMap((intervals) => mergeIntervals(intervals)).reduce((sum, [from, to]) => sum + Math.max(0, to - from), 0);
  return `${Math.min(100, Math.round((100 * busy) / ((now - start) * node.gpu_capacity)))}%`;
}

// ---- who held a shared GPU ---------------------------------------------------------

// Shared lanes paint completed operation activity. Leave gaps unpainted:
// another workload can run between even closely spaced operations.
const holdUrl = (runId, { start, now }) =>
  `/api/v1/dashboard/runs/${encode(runId)}/metrics?${new URLSearchParams({ since: new Date(start * 1000).toISOString(), until: new Date(now * 1000).toISOString() })}`;

function holdIntervals(placement, range, errors) {
  const intervals = [];
  for (const runId of placement.run_ids || []) {
    const entry = use(holdUrl(runId, range), range.live ? `holds:${runId}:${ui.nodeSelection.duration}` : undefined);
    if (entry.error) errors?.add(entry.error);
    for (const sample of entry.data?.samples || []) {
      if (sample.role !== placement.role || (sample.node && sample.node !== placement.node) || (sample.runtime_id && sample.runtime_id !== placement.runtime_id)) continue;
      const from = Math.max(range.start, placement.start, sample.started_at ?? sample.at - (sample.elapsed_seconds || 0));
      const to = Math.min(range.now, placement.end, sample.at);
      if (to > from) intervals.push([from, to]);
    }
  }
  return mergeIntervals(intervals);
}

// ---- lane markup ------------------------------------------------------------------

// DRA drivers name devices as they like ("gpu-0" from NVIDIA, "GPU 0" in
// fixtures). The lane shows the index the name ends in; a name without one
// is shown whole rather than mangled.
const deviceLabel = (name) => {
  const index = /(\d+)$/.exec(String(name ?? ""));
  return index ? index[1] : String(name ?? "");
};

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
  if (!placements.every((p) => p.devices.length === p.device_count && p.devices.every((id) => node.devices.some((d) => d.id === id)))) return "— claimed";
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

function trackBars(devices, nodeSegments, range) {
  const { start, now } = range;
  const height = laneHeight(devices);
  const duration = now - start || 1;
  const span = (from, to) =>
    `left:${(Math.max(0, Math.max(start, from) - start) / duration) * 100}%;width:${(Math.max(0, Math.min(now, to) - Math.max(start, from)) / duration) * 100}%`;
  const sharersOf = (i) => nodeSegments.filter((s) => s.devices.includes(devices[i].id)).sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
  const shared = new Set(
    devices
      .map((_, i) => i)
      .filter((i) => {
        const sharers = sharersOf(i);
        return sharers.some((a, index) => sharers.slice(index + 1).some((b) => Math.min(a.end, b.end) > Math.max(a.start, b.start)));
      }),
  );
  const sharedBars = [...shared]
    .map((i) => {
      const sharers = sharersOf(i);
      const from = Math.max(start, Math.min(...sharers.map((s) => s.start)));
      const to = Math.min(now, Math.max(...sharers.map((s) => s.end)));
      const chosen = sharers.find((s) => s.id === ui.expanded) || sharers[0];
      const open = sharers.some((s) => s.id === ui.expanded);
      const within = (a, b) => `left:${((Math.max(a, from) - from) / (to - from || 1)) * 100}%;width:${((Math.min(b, to) - Math.max(a, from)) / (to - from || 1)) * 100}%`;
      const holds = sharers
        .flatMap((s) =>
          holdIntervals(s, range).map(
            ([a, b]) => `<span class="hold ${family(s.runtime_id)} ${s.id === ui.expanded ? "selected" : ""}" data-placement="${escape(s.id)}" style="${within(a, b)}" title="${escape(s.label)} · ${escape(s.role || "")}"></span>`,
          ),
        )
        .join("");
      const title = `Shared GPU operation activity: ${sharers.map((s) => `${s.label} ${s.role || ""}`.trim()).join(", ")}`;
      return `<button type="button" class="capacity-allocation shared ${open ? "selected" : ""}" data-key="shared:${escape(devices[i].id)}" data-placement="${escape(chosen.id)}" aria-expanded="${open}" aria-label="${escape(title)}" title="${escape(title)}" style="${span(from, to)};top:${i * height + 2}px;height:${height - 4}px">${holds}</button>`;
    })
    .join("");
  const solid = nodeSegments
    .flatMap((s) => {
      const indexes = s.devices
        .map((id) => devices.findIndex((d) => d.id === id))
        .filter((i) => i >= 0 && !shared.has(i))
        .sort((a, b) => a - b);
      return deviceGroups(indexes).map((group) => {
        const title = `${s.label}${s.role ? " · " + s.role : ""} · ${group.length} GPU${group.length === 1 ? "" : "s"}`;
        return `<button type="button" class="capacity-allocation ${family(s.runtime_id)} ${s.ended ? "ended" : ""} ${s.id === ui.expanded ? "selected" : ""}" data-key="${escape(s.id)}:${group[0]}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}" aria-label="${escape(title)}" title="${escape(title)}" style="${span(s.start, s.end)};top:${group[0] * height + 2}px;height:${group.length * height - 4}px"><span class="allocation-name">${escape(s.label)}</span><span class="allocation-count">${group.length} GPU${group.length === 1 ? "" : "s"}</span></button>`;
      });
    })
    .join("");
  return sharedBars + solid;
}

function lane(node, all, range) {
  const devices = node.devices || [];
  const nodeSegments = all.filter((s) => s.node === node.name);
  const placements = range.live ? ui.state.placements.filter((p) => p.node === node.name) : nodeSegments.filter((p) => p.start <= range.now && p.end >= range.now);
  const current = nodeSegments.find((s) => s.id === ui.expanded);
  const bars = trackBars(devices, nodeSegments, range);
  const accelerator = acceleratorLabel(node);
  const height = laneHeight(devices);
  const unmapped = nodeSegments.filter((s) => !s.devices.length || s.devices.some((id) => !devices.some((d) => d.id === id)));
  const unknown = unmapped
    .map(
      (s) =>
        `<button type="button" class="capacity-allocation unknown-mapping ${family(s.runtime_id)}" data-key="unmapped:${escape(s.id)}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}"><span class="allocation-name">${escape(s.label)}</span><span class="allocation-count">${s.device_count} GPU${s.device_count === 1 ? "" : "s"}</span></button>`,
    )
    .join("");
  const track = devices.length
    ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-auto-rows:${height}px">${devices.map((d) => `<span title="${escape(d.id)}">${escape(deviceLabel(d.name))}</span>`).join("")}</div><div class="capacity-track" style="height:${devices.length * height}px;--gpu-lane-height:${height}px">${bars}</div></div>`
    : "";
  return `<div class="node-placement-group" data-key="${escape(node.name)}"><div class="node-lane"><div class="node-lane-label" title="${escape(node.name)}"><span class="node-accelerator">${escape(accelerator)}</span><span class="claim-label">${escape(claimLabel(node, placements, range.live))}</span></div><div>${track}${unknown}${unmapped.length ? '<span class="muted micro">Device mapping unavailable</span>' : !devices.length ? empty(node.gpu_capacity ? "GPUs without DRA devices" : "No GPUs") : ""}</div><span class="node-duty" title="Recorded GPU allocation time over the selected window">${duty(node, nodeSegments, range)}</span></div>${current ? detail(current, range, nodeSegments) : ""}</div>`;
}

// ---- expansion ---------------------------------------------------------------------

function detail(placement, range, neighbours) {
  const anchor = neighbours.find((p) => p.id === ui.gpuGroup && p.devices.some((id) => placement.devices.includes(id))) || placement;
  ui.gpuGroup = anchor.id;
  if (ui.device !== "all" && !anchor.devices.includes(ui.device)) ui.device = "all";
  if (anchor.devices.length === 1) ui.device = anchor.devices[0];
  const { state } = ui;
  const node = state.cluster.nodes.find((n) => n.name === anchor.node);
  const devices = (node?.devices || []).filter((d) => anchor.devices.includes(d.id));
  const group = neighbours.filter((p) => p.id === anchor.id || p.devices.some((id) => anchor.devices.includes(id)));
  const visible = group.filter((p) => ui.device === "all" || p.devices.includes(ui.device));
  const run = visible.some((p) => p.id === placement.id) && state.runs.find((r) => (placement.run_ids || []).includes(r.run_id));
  const workload = run?.workloads?.find((w) => w.uid === placement.id);
  const title = [
    (run?.model || placement.runtime_id || placement.label).split("/").at(-1),
    (placement.owner_id || run?.run_id)?.slice(0, 8),
    { trainer: "Trainer", sampler: "Sampler" }[placement.role] || "Unknown process",
    placement.role === "trainer" ? { lora: "LoRA", fft: "FFT" }[workload?.training_kind] : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const legend = visible
    .map(
      (p) =>
        `<button type="button" class="legend-entry ${family(p.runtime_id)}" data-key="${escape(p.id)}" data-placement="${escape(p.id)}" aria-pressed="${p.id === placement.id}" title="${escape(p.label)}"><span class="legend-swatch"></span><span class="legend-name">${escape(p.label)}</span><span class="legend-meta">${escape(p.role || "process")}${p.ended ? " · Ended" : ""}</span></button>`,
    )
    .join("");
  const metrics = use(
    `/api/v1/dashboard/allocations/${encode(anchor.id)}/metrics?${new URLSearchParams({ since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() })}`,
    range.live ? `allocation:${anchor.id}:${ui.nodeSelection.duration}` : undefined,
  );
  const picker = (anchor.devices.length > 1 ? [button("All GPUs", `data-device="all" aria-pressed="${ui.device === "all"}"`)] : [])
    .concat(
      anchor.devices.map((id) => {
        const name = deviceLabel(devices.find((d) => d.id === id)?.name || id.split("/").at(-1));
        const value = metrics.data?.devices?.find((d) => d.id === id)?.utilization?.at(-1)?.[1];
        return button(
          `GPU ${name}${Number.isFinite(value) ? ` · ${Math.round(value)}%` : ""}`,
          `data-device="${escape(id)}" aria-pressed="${ui.device === id}"`,
        );
      }),
    )
    .join("");
  const activityErrors = new Set();
  const activities = visible.map((p) => ({ id: p.id, label: p.label, tone: family(p.runtime_id), selected: p.id === placement.id, intervals: holdIntervals(p, range, activityErrors) }));
  const label = devices.length ? `${acceleratorLabel(node)} · GPU${devices.length === 1 ? "" : "s"} ${devices.map((d) => deviceLabel(d.name)).join(", ")}` : "GPU activity";
  return `<section class="allocation-expansion" id="placement-detail" data-key="${escape(anchor.id)}"><div class="allocation-detail-head"><h2 title="${escape(anchor.node)}">${escape(label)}</h2><span class="allocation-memory">GPU memory <strong>${gpuMemory(metrics)}</strong></span></div>
    <div class="allocation-device-picker" aria-label="GPU selection">${picker}</div>
    <div class="allocation-legend" aria-label="Workloads on these GPUs">${legend}</div>
    <div>${gpuChart(metrics, range, activities)}${activityErrors.size ? `<p class="activity-error" role="status">Operation history unavailable: ${escape([...activityErrors].join(" · "))}</p>` : ""}</div>
    <div class="allocation-detail-footer"><span>Colors show recorded operations; utilization is GPU-wide.</span>${run ? `<a href="#run/${encode(run.run_id)}/metrics">${escape(title)} ↗</a>` : ""}</div></section>`;
}

const selectedDevices = (metrics) => [...new Map((metrics.data?.devices || []).filter((d) => ui.device === "all" || ui.device === d.id).map((d) => [d.uuid || d.id, d])).values()];

function gpuChart(metrics, range, activities) {
  if (!metrics.data) return chart({ title: "GPU utilization", start: range.start, end: range.now, empty: metrics.error || "Loading GPU metrics…" });
  const selected = selectedDevices(metrics);
  const byTime = new Map();
  for (const device of selected)
    for (const [at, value] of device.utilization || []) {
      if (!byTime.has(at)) byTime.set(at, new Map());
      byTime.get(at).set(device.id, value);
    }
  const points = [...byTime].map(([at, values]) => [
    at,
    values.size === selected.length && [...values.values()].every(Number.isFinite) ? [...values.values()].reduce((a, b) => a + b, 0) / selected.length : null,
  ]);
  const reason = metrics.data.reason || selected.find((d) => d.reason)?.reason || "No samples for this GPU";
  return `${chart({
    title: "GPU utilization",
    unit: "%",
    points,
    start: range.start,
    end: range.now,
    min: 0,
    max: 100,
    gapSeconds: 60,
    activities,
    empty: reason,
  })}${metrics.error ? `<p class="muted" role="status">${escape(metrics.error)} · Showing the last available samples</p>` : ""}`;
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
  const axis = [0, 1, 2, 3].map((tick) => `<span>${nodeTime(range.start + ((range.now - range.start) * tick) / 3, range.now - range.start <= 120)}</span>`).join("");
  morph(
    content,
    `<div class="nodes-heading"><h1 class="heading">Kubernetes nodes</h1>${timeControl(range)}</div>${!cluster.available ? empty(cluster.error || "Kubernetes unavailable") : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${axis}</div><span class="node-duty" title="GPU allocation time divided by capacity over the selected window">Duty</span></div>
    ${cluster.nodes.map((node) => lane(node, all, range)).join("") || empty("No nodes available")}`,
  );
}
