// The Nodes page: one lane per node, one row per GPU, allocation bars over
// the selected time range, and an expansion with the GPU chart and a legend.

import { resolveTimeRange, timeRangeControl } from "./time-range.js";
import { escape, encode, empty, button, morph } from "./ui.js";
import { renderMetricChart, disposeMetricCharts } from "./charts.js";
import { ui, content, route, nodeNow, nodeTime, family } from "./store.js";
import { cached, isFresh, metricData, loadMetricView } from "./cache.js";

const LANE_HEIGHT = 26;
const LIVE_PHASES = ["Running", "Bound", "Placed"];

// ---- observations -------------------------------------------------------

// Fold the gateway's history samples into the pieces the page draws:
// observed windows with their live placements, the gaps between them, and
// contiguous per-placement segments.
function observe(start, now) {
  const history = ui.state.history;
  const observations = history.flatMap((sample, index) => {
    const from = Math.max(start, sample.at);
    const to = Math.min(now, history[index + 1]?.at ?? sample.at + 15, sample.at + 30);
    if (!sample.available || to <= from) return [];
    return [{ start: from, end: to, placements: sample.placements.filter((p) => LIVE_PHASES.includes(p.phase)) }];
  });
  const missing = [];
  let observedThrough = start;
  for (const sample of observations) {
    if (sample.start > observedThrough) missing.push([observedThrough, sample.start]);
    observedThrough = Math.max(observedThrough, sample.end);
  }
  if (observedThrough < now) missing.push([observedThrough, now]);
  const segments = [];
  observations.forEach((sample) =>
    sample.placements.forEach((placement) => {
      const key = `${placement.id}:${placement.node}:${placement.devices.join(",")}`;
      const previous = segments.findLast((segment) => segment.key === key);
      if (previous && previous.end === sample.start) previous.end = sample.end;
      else segments.push({ ...placement, key, start: sample.start, end: sample.end });
    }),
  );
  const endpoint = history.findLast((sample) => sample.available && sample.at <= now && now - sample.at <= 30);
  return { observations, missing, segments, endpoint };
}

// Allocated GPU-time over the time this gateway actually observed, so a
// recent restart shortens the denominator instead of blanking the column.
function duty(node, observations) {
  if (!node.gpu_capacity || !observations.length) return "—";
  const devices = new Set((node.devices || []).map((item) => item.id));
  if (devices.size !== node.gpu_capacity) return "—";
  let allocated = 0;
  let observed = 0;
  for (const sample of observations) {
    observed += sample.end - sample.start;
    const placements = sample.placements.filter((p) => p.node === node.name);
    if (placements.some((p) => p.devices.length !== p.device_count || p.devices.some((id) => !devices.has(id)))) return "—";
    allocated += new Set(placements.flatMap((p) => p.devices)).size * (sample.end - sample.start);
  }
  return observed > 0 ? `${Math.round((100 * allocated) / (observed * node.gpu_capacity))}%` : "—";
}

// ---- who held a shared GPU ----------------------------------------------

// Time-sliced GPUs run one job at a time. The workers' own operation records
// say who held the device when, so shared lanes paint those intervals instead
// of one flat allocation block.
function holdUrl(runId, start, end) {
  return `/api/v1/dashboard/runs/${encode(runId)}/metrics?${new URLSearchParams({
    since: new Date(start * 1000).toISOString(),
    until: new Date(end * 1000).toISOString(),
  })}`;
}

function wantHolds(placements, start, end) {
  new Set(placements.flatMap((p) => p.run_ids || [])).forEach((runId) => {
    const url = holdUrl(runId, start, end);
    if (isFresh(url)) return;
    metricData(url)
      .then(() => {
        if (route()[0] === "nodes") ui.render();
      })
      .catch(() => {});
  });
}

function holdIntervals(placement, start, end) {
  const intervals = [];
  (placement.run_ids || []).forEach((runId) => {
    (cached(holdUrl(runId, start, end))?.samples || []).forEach((sample) => {
      if (sample.role !== placement.role) return;
      if (sample.node && sample.node !== placement.node) return;
      const from = Math.max(start, placement.start, sample.started_at ?? sample.at - (sample.elapsed_seconds || 0));
      const to = Math.min(end, placement.end, sample.at);
      if (to > from) intervals.push([from, to]);
    });
  });
  intervals.sort((a, b) => a[0] - b[0]);
  const merged = [];
  intervals.forEach(([from, to]) => {
    const last = merged.at(-1);
    if (last && from - last[1] < 3) last[1] = Math.max(last[1], to);
    else merged.push([from, to]);
  });
  return merged;
}

// ---- lane markup ---------------------------------------------------------

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

function claimLabel(node, placements, endpoint) {
  const live = ui.nodeSelection.end === null;
  if (!live && !endpoint) return "— claimed";
  if (live && !node.ready) return "Offline";
  if (!node.gpu_capacity) return "CPU";
  if (!placements.every((p) => p.devices.length === p.device_count)) return "— claimed";
  return `${new Set(placements.flatMap((p) => p.devices)).size}/${node.gpu_capacity} claimed`;
}

// Consecutive device indexes of one placement form one block.
function deviceGroups(indexes) {
  const groups = [];
  indexes.forEach((i) => {
    const last = groups.at(-1);
    if (last && last.at(-1) === i - 1) last.push(i);
    else groups.push([i]);
  });
  return groups;
}

// Bars for one node's GPU track. A GPU held by several placements at once
// gets a single neutral bar carrying colored hold blocks; every other
// placement gets a solid block in its own color.
function trackBars(node, devices, segments, range) {
  const { start, now } = range;
  const duration = now - start;
  const pct = (at) => `${Math.max(0, ((Math.min(now, Math.max(start, at)) - start) / duration) * 100)}%`;
  const span = (from, to) => `left:${pct(from)};width:${Math.max(0, ((Math.min(now, to) - Math.max(start, from)) / duration) * 100)}%`;
  const sharersOf = (i) =>
    segments.filter((q) => q.devices.length === 1 && q.devices[0] === devices[i].id).sort((a, b) => a.start - b.start || a.id.localeCompare(b.id));
  const sharedDevices = new Set(devices.map((_, i) => i).filter((i) => new Set(sharersOf(i).map((q) => q.id)).size > 1));
  const sharedPlacements = [...sharedDevices].flatMap((i) => sharersOf(i));
  if (sharedPlacements.length) wantHolds(sharedPlacements, start, now);

  const shared = [...sharedDevices]
    .map((i) => {
      const sharers = sharersOf(i);
      const from = Math.min(...sharers.map((q) => q.start));
      const to = Math.max(...sharers.map((q) => q.end));
      const chosen = sharers.find((q) => q.id === ui.expanded) || sharers[0];
      const open = sharers.some((q) => q.id === ui.expanded);
      // Hold blocks sit inside the shared bar, so they are placed against
      // the bar's own span rather than the whole track.
      const barFrom = Math.max(start, from);
      const barTo = Math.min(now, to);
      const within = (a, b) => `left:${((Math.max(a, barFrom) - barFrom) / (barTo - barFrom)) * 100}%;width:${((Math.min(b, barTo) - Math.max(a, barFrom)) / (barTo - barFrom)) * 100}%`;
      const holds = sharers
        .flatMap((q) =>
          holdIntervals(q, start, now).map(
            ([a, b]) => `<span class="hold ${family(q.runtime_id)}" style="${within(a, b)}" title="${escape(q.label)}${q.role ? " · " + escape(q.role) : ""}"></span>`,
          ),
        )
        .join("");
      const distinct = [...new Map(sharers.map((q) => [q.id, q])).values()];
      const title = `Shared GPU: ${distinct.map((q) => `${q.label}${q.role ? " " + q.role : ""}`).join(", ")}`;
      return `<button type="button" class="capacity-allocation shared ${open ? "selected" : ""}" data-placement="${escape(chosen.id)}" aria-expanded="${open}" title="${escape(title)}" aria-label="${escape(title)} on ${escape(node.name)}" style="${span(from, to)};top:${i * LANE_HEIGHT + 2}px;height:${LANE_HEIGHT - 4}px">${holds}</button>`;
    })
    .join("");

  const solid = segments
    .flatMap((p) => {
      const indexes = p.devices
        .map((id) => devices.findIndex((d) => d.id === id))
        .filter((i) => i >= 0 && !(p.devices.length === 1 && sharedDevices.has(i)))
        .sort((a, b) => a - b);
      return deviceGroups(indexes).map((group) => {
        const title = `${p.label}${p.role ? " · " + p.role : ""} · ${group.length} GPU${group.length === 1 ? "" : "s"}`;
        return `<button type="button" class="capacity-allocation ${family(p.runtime_id)} ${p.id === ui.expanded ? "selected" : ""}" data-placement="${escape(p.id)}" aria-expanded="${p.id === ui.expanded}" title="${escape(title)}" aria-label="${escape(title)} on ${escape(node.name)}" style="${span(p.start, p.end)};top:${group[0] * LANE_HEIGHT + 2}px;height:${group.length * LANE_HEIGHT - 4}px"></button>`;
      });
    })
    .join("");
  return shared + solid;
}

function lane(node, observed, range) {
  const { observations, segments, endpoint, gaps } = observed;
  const devices = node.devices || [];
  const live = ui.nodeSelection.end === null;
  const placements = (live ? ui.state.placements : endpoint?.placements || []).filter((p) => p.node === node.name);
  const current = placements.find((p) => p.id === ui.expanded) || segments.find((p) => p.id === ui.expanded && p.node === node.name);
  const unmapped = (live ? placements : []).filter((p) => !p.devices.length || p.devices.some((d) => !devices.find((n) => n.id === d)));
  const bars = trackBars(
    node,
    devices,
    segments.filter((p) => p.node === node.name),
    range,
  );
  const accelerator = acceleratorLabel(node);
  const track = devices.length
    ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-auto-rows:${LANE_HEIGHT}px">${devices.map((d) => `<span title="${escape(d.id)}">${escape(d.name)}</span>`).join("")}</div><div class="capacity-track" style="height:${devices.length * LANE_HEIGHT}px;--gpu-lane-height:${LANE_HEIGHT}px">${gaps}${bars}</div></div>`
    : "";
  const unmappedChips = unmapped
    .map(
      (p) =>
        `<button type="button" class="capacity-allocation unknown-mapping ${family(p.runtime_id)}" data-placement="${escape(p.id)}" aria-expanded="${p.id === ui.expanded}"><span class="allocation-name">${escape(p.label)}</span><span class="allocation-count">${p.device_count} GPUs</span></button>`,
    )
    .join("");
  return `<div class="node-placement-group"><div class="node-lane"><div class="node-lane-label" title="${escape(node.name)} · ${live ? "Current node status" : "At selected range end"}" aria-label="${escape(accelerator)}, ${escape(node.name)}"><span class="node-accelerator">${escape(accelerator)}</span><span class="claim-label">${escape(claimLabel(node, placements, endpoint))}</span></div><div>
        ${track}
        ${unmappedChips}
        ${unmapped.length ? '<span class="muted micro">Device mapping unavailable</span>' : ""}
        ${!bars && !placements.length && !node.gpu_capacity ? empty("No GPUs") : ""}</div><span class="node-duty" title="GPU allocation time over the selected range; unavailable when observations or device mappings are incomplete">${duty(node, observations)}</span></div>${current ? detail(current) : ""}</div>`;
}

// ---- expansion -----------------------------------------------------------

function detail(placement) {
  if (ui.device !== "all" && !placement.devices.includes(ui.device)) ui.device = "all";
  const { state } = ui;
  const run = state.runs.find((r) => placement.run_ids.includes(r.run_id));
  const devices = state.cluster.nodes.find((n) => n.name === placement.node)?.devices || [];
  const workload = run?.workloads?.find((w) => w.uid === placement.id);
  const identity = placement.owner_id || run?.run_id;
  const kind = placement.role === "trainer" ? { lora: "LoRA", fft: "FFT" }[workload?.training_kind] : null;
  const title = [
    (run?.model || placement.runtime_id || placement.label).split("/").at(-1),
    identity?.slice(0, 8),
    { trainer: "Trainer", sampler: "Sampler" }[placement.role] || "Unknown process",
    kind,
  ]
    .filter(Boolean)
    .join(" · ");
  const neighbours = state.placements.filter((p) => p.node === placement.node);
  const legend = (neighbours.some((p) => p.id === placement.id) ? neighbours : [placement, ...neighbours])
    .map(
      (p) =>
        `<button type="button" class="legend-entry ${family(p.runtime_id)}" data-placement="${escape(p.id)}" aria-pressed="${p.id === placement.id}"><span class="legend-swatch"></span><span class="legend-name">${escape(p.label)}</span><span class="legend-meta">${escape({ trainer: "trainer", sampler: "sampler" }[p.role] || "process")} · ${p.device_count} GPU${p.device_count === 1 ? "" : "s"}</span></button>`,
    )
    .join("");
  const picker = [button("All GPUs", `data-device="all" aria-pressed="${ui.device === "all"}"`)]
    .concat(placement.devices.map((id) => button(devices.find((d) => d.id === id)?.name || id.split("/").at(-1), `data-device="${escape(id)}" aria-pressed="${ui.device === id}"`)))
    .join("");
  return `<section class="allocation-expansion" id="placement-detail"><div class="allocation-detail-head"><h2>${run ? `<a href="#run/${encode(run.run_id)}/overview">${escape(title)} ↗</a>` : escape(title)}</h2><div class="allocation-legend" aria-label="Allocations on this node">${legend}</div></div>
    <div class="allocation-device-picker" aria-label="GPU selection">${picker}</div>
    <div><div id="gpu-chart"></div><p id="gpu-status" class="muted" role="status"></p>${run?.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}</div>
    <div><p class="muted">GPU memory</p><p id="gpu-memory">—</p><p class="muted">Run MFU</p><p id="gpu-mfu">—</p></div></section>`;
}

// ---- page ------------------------------------------------------------------

export function renderNodes() {
  const observedAt = nodeNow();
  const { start, end: now } = resolveTimeRange(ui.nodeSelection, observedAt);
  const duration = now - start;
  const observed = observe(start, now);
  observed.gaps = observed.missing
    .map(
      ([from, to]) =>
        `<span class="allocation-unobserved" style="left:${((from - start) / duration) * 100}%;width:${((to - from) / duration) * 100}%" aria-label="No allocation observations"></span>`,
    )
    .join("");
  const { cluster } = ui.state;
  const axis = [0, 1, 2, 3].map((tick) => `<span>${nodeTime(start + (duration * tick) / 3)}</span>`).join("");
  morph(
    content,
    `<div class="nodes-heading"><h1 class="heading">Kubernetes nodes</h1>${timeRangeControl(ui.nodeSelection, observedAt)}</div>${observed.missing.length ? `<p class="history-coverage muted">Hatched areas have no allocation observations.</p>` : ""}${!cluster.available ? empty(cluster.error || "Kubernetes unavailable") : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${axis}</div><span class="node-duty" title="GPU allocation time divided by capacity over the selected range">Duty</span></div>
    ${cluster.nodes.map((node) => lane(node, observed, { start, now })).join("") || empty("No nodes available")}`,
  );
}

// ---- GPU chart in the expansion ------------------------------------------

export function paintGpu(view) {
  if (ui.gpuView !== view || !view.element.isConnected || !view.data) return;
  const { start, end, data, element: chart } = view;
  const selected = (data.devices || []).filter((item) => ui.device === "all" || ui.device === item.id);
  const points = new Map();
  selected.forEach((item) =>
    (item.utilization || [])
      .filter(([at]) => at >= start && at <= end)
      .forEach(([at, value]) => {
        const values = points.get(at) || [];
        values.push(Number.isFinite(value) ? value : null);
        points.set(at, values);
      }),
  );
  const samples = Array.from(points, ([at, values]) => {
    const reported = values.filter(Number.isFinite);
    return [at, reported.length ? reported.reduce((sum, value) => sum + value, 0) / reported.length : null];
  }).sort((a, b) => a[0] - b[0]);
  const memory = selected.map((item) => item.memory_mib?.filter(([at]) => at >= start && at <= end).at(-1)?.[1]);
  view.memory.textContent = memory.length && memory.every(Number.isFinite) ? `${(memory.reduce((sum, value) => sum + value, 0) / 1024).toFixed(1)} GiB` : "—";
  const mfu = data.mfu;
  view.mfu.textContent = Number.isFinite(mfu?.value) && mfu.value >= 0 && mfu.value <= 1 ? `${(mfu.value * 100).toFixed(1)}%` : "—";
  view.mfu.title =
    mfu && view.mfu.textContent !== "—"
      ? `${mfu.estimated ? "Estimated · " : ""}${mfu.scope === "run" ? "Entire run" : "Reported MFU"}${mfu.device_count ? ` · ${mfu.device_count} GPUs` : ""}`
      : "MFU not reported";
  if (!samples.some(([, value]) => Number.isFinite(value))) {
    disposeMetricCharts(chart);
    chart.innerHTML = empty(data.reason || selected[0]?.reason || "No samples for this GPU");
    return;
  }
  renderMetricChart(chart, { samples, start, end, title: "GPU utilization", unit: "%", min: 0, max: 100, tone: "neutral", gapSeconds: 30 });
}

export function loadGpu(id, retainSamples = false) {
  const element = document.getElementById("gpu-chart");
  if (!element) return;
  const { start, end } = resolveTimeRange(ui.nodeSelection, nodeNow());
  const url = `/api/v1/dashboard/allocations/${encode(id)}/metrics?${new URLSearchParams({
    since: new Date(start * 1000).toISOString(),
    until: new Date(end * 1000).toISOString(),
  })}`;
  if (ui.gpuView?.element !== element || ui.gpuView.url !== url)
    ui.gpuView = {
      id,
      url,
      start,
      end,
      element,
      memory: document.getElementById("gpu-memory"),
      mfu: document.getElementById("gpu-mfu"),
      status: document.getElementById("gpu-status"),
      data: retainSamples && ui.gpuView?.id === id ? ui.gpuView.data : null,
    };
  return loadMetricView(ui.gpuView, paintGpu, () => ui.gpuView);
}
