// The Nodes page: one lane per node, one row per GPU, allocation bars over
// the selected window from the recorded placement history, and an expansion
// with run activity lanes and GPU utilization for those devices.

import { escape, encode, empty, button, morph, shortNodeName } from "./ui.js";
import { chart } from "./charts.js";
import { ui, content, nodeNow, nodeTime, nodeLink, viewLink } from "./store.js";
import { use } from "./cache.js";
import { timeWindow, timeControl } from "./node-time.js";

const laneHeight = (devices) => (devices.length === 1 ? 30 : 22);
const axisTime = (at, range) => `${range.now - range.start >= 86400 ? `${new Date(at * 1000).toISOString().slice(5, 10)} ` : ""}${nodeTime(at, range.now - range.start <= 120)}`;
// ---- history -> segments -----------------------------------------------------------

// Keep the selected placement available when panning beyond its lifetime.
// Lanes clip history to the visible window; live placements run to now.
function segments({ now }) {
  const live = new Map(ui.state.placements.map((p) => [p.id, p]));
  const seen = new Set();
  const found = [];
  for (const entry of ui.state.history) {
    seen.add(entry.id);
    const end = live.has(entry.id) ? now : entry.last_seen;
    found.push({
      ...entry,
      ...(live.get(entry.id) || {}),
      start: entry.first_seen,
      end,
      ended: !live.has(entry.id),
    });
  }
  for (const placement of ui.state.placements) if (!seen.has(placement.id)) found.push({ ...placement, start: nodeNow(), end: now });
  // A shared LoRA process can serve several runs on the same allocation.
  return found.flatMap((placement) => (placement.run_ids?.length > 1 ? placement.run_ids.map((id) => {
    const run = ui.state.runs.find((r) => r.run_id === id);
    return { ...placement, id: `${placement.id}:${id}`, allocation_id: placement.id, run_ids: [id], label: run?.name || `${placement.label.split("/").at(-1)} ${id.slice(0, 8)}` };
  }) : [placement]));
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

// Fetch retained operations through the snapshot, then clip locally: an
// operation can finish beyond the viewed window while overlapping its edge.
// Leave gaps unpainted; another workload can run between recorded operations.
const holdUrl = (runId) =>
  `/api/v1/dashboard/runs/${encode(runId)}/metrics?${new URLSearchParams({ since: new Date((nodeNow() - 86400) * 1000).toISOString(), until: new Date(nodeNow() * 1000).toISOString() })}`;

const turnsUrl = (placementId) =>
  `/api/v1/dashboard/allocations/${encode(placementId)}/turns?${new URLSearchParams({ since: new Date((nodeNow() - 86400) * 1000).toISOString(), until: new Date(nodeNow() * 1000).toISOString() })}`;

// Workers record every GPU turn the time-slicer grants them. Those intervals
// are exclusive by construction; operation timings (below) include time
// spent waiting for the turn and overlapping in-flight requests, so they are
// only the fallback for workers that predate turn recording.
const activityCache = new Map();
function operationActivity(placement, range) {
  if (activityCache.has(placement.id)) return activityCache.get(placement.id);
  const save = (value) => { activityCache.set(placement.id, value); return value; };
  const allocationId = placement.allocation_id || placement.id;
  const turns = use(turnsUrl(allocationId), `turns:${allocationId}`);
  // A shared worker's turn serves several runs; its request ops carry the run
  // they served, so a run's lane on that worker is its own ops in those turns.
  if (turns.data?.samples?.length) {
    const runId = placement.allocation_id ? placement.run_ids[0] : null;
    const clip = (from, to) => [Math.max(range.start, placement.start, from), Math.min(range.now, placement.end, to)];
    const intervals = runId ? [] : turns.data.samples.map((t) => clip(t.started_at, t.at)).filter(([from, to]) => to > from);
    // What the GPU was doing inside each turn, recorded by the worker: one interval list per op name.
    const ops = {};
    for (const turn of turns.data.samples)
      for (const op of turn.ops || []) {
        if (runId && op.run_id !== runId) continue;
        const [from, to] = clip(op.start, op.end);
        if (to > from) {
          (ops[op.name] ||= []).push([from, to]);
          if (runId) intervals.push([from, to]);
        }
      }
    for (const name of Object.keys(ops)) ops[name] = mergeIntervals(ops[name]);
    return save({ intervals: mergeIntervals(intervals), ops, error: turns.error || "", errorStatus: turns.errorStatus, loading: false, exact: true });
  }
  const intervals = [];
  let error = "", errorStatus = null, loading = turns.pending && !turns.data && !turns.error;
  for (const runId of placement.run_ids || []) {
    const entry = use(holdUrl(runId), `holds:${runId}`);
    if (entry.error) { error = entry.error; errorStatus = entry.errorStatus; }
    if (!entry.data && entry.pending && !entry.error) loading = true;
    for (const sample of entry.data?.samples || []) {
      if (sample.run_id && sample.run_id !== runId) continue;
      if (sample.role !== placement.role || (sample.node && sample.node !== placement.node) || (sample.runtime_id && sample.runtime_id !== placement.runtime_id)) continue;
      const from = Math.max(range.start, placement.start, sample.started_at ?? sample.at - (sample.elapsed_seconds || 0));
      const to = Math.min(range.now, placement.end, sample.at);
      if (to > from) intervals.push([from, to]);
    }
  }
  const warning = turns.error && turns.errorStatus !== 404 ? `GPU turns unavailable: ${turns.error}` : "";
  return save({ intervals: mergeIntervals(intervals), error, errorStatus, loading, warning, exact: false });
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
  // Turns painted inside a bar spanning [from, to], coloured by the op the GPU ran; a turn with
  // no recorded op stays hatched. A bar still fetching turns is striped instead of looking idle.
  const holdsWithin = (s, from, to, classes = "") => {
    const within = (a, b) => `left:${((Math.max(a, from) - from) / (to - from || 1)) * 100}%;width:${((Math.min(b, to) - Math.max(a, from)) / (to - from || 1)) * 100}%`;
    const { intervals, ops = {} } = operationActivity(s, range);
    const who = `${s.label}${s.role ? `, ${s.role}` : ""}`;
    const turns = intervals.map(([a, b]) => `<span class="hold ${classes}" data-placement="${escape(s.id)}" style="${within(a, b)}" title="${escape(who)}"></span>`);
    const work = Object.entries(ops)
      .sort(([a], [b]) => opRank(a) - opRank(b))
      .flatMap(([op, spans]) => spans.map(([a, b]) => `<span class="hold op ${classes}" data-op="${escape(op)}" data-placement="${escape(s.id)}" style="${within(a, b)}" title="${escape(`${opLabel(op)}: ${who}`)}"></span>`));
    return turns.join("") + work.join("");
  };
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
      const holds = sharers.map((s) => holdsWithin(s, from, to, s.id === ui.expanded ? "selected" : "")).join("");
      const loading = sharers.some((s) => operationActivity(s, range).loading);
      const title = `Shared GPU operation activity: ${sharers.map((s) => `${s.label} ${s.role || ""}`.trim()).join(", ")}${loading ? ". Loading GPU turns…" : ""}`;
      return `<button type="button" class="capacity-allocation shared ${open ? "selected" : ""} ${loading ? "loading" : ""}" data-key="shared:${escape(devices[i].id)}" data-placement="${escape(chosen.id)}" aria-expanded="${open}" aria-label="${escape(title)}" title="${escape(title)}" style="${span(from, to)};top:${i * height + 2}px;height:${height - 4}px">${holds}</button>`;
    })
    .join("");
  const solid = nodeSegments
    .flatMap((s) => {
      const indexes = s.devices
        .map((id) => devices.findIndex((d) => d.id === id))
        .filter((i) => i >= 0 && !shared.has(i))
        .sort((a, b) => a - b);
      const loading = operationActivity(s, range).loading;
      return deviceGroups(indexes).map((group) => {
        const title = `${s.label}: ${s.role ? `${s.role}, ` : ""}${group.length} GPU${group.length === 1 ? "" : "s"}${loading ? ". Loading GPU turns…" : ""}`;
        return `<button type="button" class="capacity-allocation ${s.ended ? "ended" : ""} ${s.id === ui.expanded ? "selected" : ""} ${loading ? "loading" : ""}" data-key="${escape(s.id)}:${group[0]}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}" aria-label="${escape(title)}" title="${escape(title)}" style="${span(s.start, s.end)};top:${group[0] * height + 2}px;height:${group.length * height - 4}px">${holdsWithin(s, Math.max(start, s.start), Math.min(now, s.end))}<span class="allocation-name">${escape(s.label)}</span><span class="allocation-count">${group.length} GPU${group.length === 1 ? "" : "s"}</span></button>`;
      });
    })
    .join("");
  return sharedBars + solid;
}

function lane(node, all, range) {
  const devices = node.devices || [];
  const neighbours = all.filter((s) => s.node === node.name);
  const nodeSegments = neighbours.filter((s) => s.start <= range.now && s.end >= range.start);
  const placements = range.live ? ui.state.placements.filter((p) => p.node === node.name) : nodeSegments.filter((p) => p.start <= range.now && p.end >= range.now);
  const bars = trackBars(devices, nodeSegments, range);
  const accelerator = acceleratorLabel(node);
  const height = laneHeight(devices);
  const unmapped = nodeSegments.filter((s) => !s.devices.length || s.devices.some((id) => !devices.some((d) => d.id === id)));
  const unknown = unmapped
    .map(
      (s) =>
        `<button type="button" class="capacity-allocation unknown-mapping" data-key="unmapped:${escape(s.id)}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}"><span class="allocation-name">${escape(s.label)}</span><span class="allocation-count">${s.device_count} GPU${s.device_count === 1 ? "" : "s"}</span></button>`,
    )
    .join("");
  const track = devices.length
    ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-auto-rows:${height}px">${devices.map((d) => `<span title="${escape(d.id)}">${escape(deviceLabel(d.name))}</span>`).join("")}</div><div class="capacity-track" style="height:${devices.length * height}px;--gpu-lane-height:${height}px">${bars}</div></div>`
    : "";
  return `<div class="node-placement-group" data-key="${escape(node.name)}"><div class="node-lane"><button type="button" class="node-lane-label" data-node-details="${escape(node.name)}" aria-expanded="${ui.inspectorNode === node.name}" title="${escape(node.name)}"><span class="node-accelerator">${escape(accelerator)} <span class="muted">${escape(shortNodeName(node.name, ui.state.cluster.nodes))}</span></span>${node.gpu_capacity ? `<span class="claim-label">${escape(claimLabel(node, placements, range.live))}</span>` : ""}</button><div>${track}${unknown}${unmapped.length ? '<span class="muted micro">Device mapping unavailable</span>' : !devices.length && node.gpu_capacity ? empty("GPUs without DRA devices") : ""}</div><span class="node-duty" title="Recorded GPU allocation time over the selected window">${duty(node, nodeSegments, range)}</span></div>${node.name === ui.inspectorNode ? detail(node, range, neighbours) : ""}</div>`;
}

// ---- expansion ---------------------------------------------------------------------

function processLabel(placement) {
  const node = ui.state.cluster.nodes.find((n) => n.name === placement.node);
  const devices = placement.devices.map((id) => node?.devices.find((d) => d.id === id));
  const gpus = devices.length && devices.every(Boolean) ? `GPU${devices.length === 1 ? "" : "s"} ${devices.map((d) => deviceLabel(d.name)).join(", ")}` : `${placement.device_count} GPU${placement.device_count === 1 ? "" : "s"}`;
  const role = placement.role || "Process";
  return `${role[0].toUpperCase() + role.slice(1)} on ${gpus}`;
}

const processOrder = (a, b) => ({ trainer: 0, sampler: 1 }[a.role] ?? 2) - ({ trainer: 0, sampler: 1 }[b.role] ?? 2) || String(a.node || "").localeCompare(String(b.node || ""));

// Compute first, then the weight movement around it, so a weight sync inside a
// sample batch stays visible on top of it.
const OP_ORDER = ["forward_backward", "forward", "optim_step", "sample", "init", "wake_up", "sleep", "weight_sync"];
const opRank = (name) => (OP_ORDER.includes(name) ? OP_ORDER.indexOf(name) : OP_ORDER.length);
const opLabel = (name) => name.replace(/_/g, " ");
const opNames = (activities) => [...new Set(activities.flatMap((a) => Object.keys(a.ops || {})))].sort((a, b) => opRank(a) - opRank(b) || a.localeCompare(b));

function opLegend(activities) {
  const names = opNames(activities);
  return names.length
    ? `<div class="op-legend" aria-label="GPU ops">${names.map((name) => `<span data-op="${escape(name)}"><span class="op-swatch"></span>${escape(opLabel(name))}</span>`).join("")}<span><span class="op-swatch turn"></span>held, unlabelled</span></div>`
    : "";
}

function activityTimeline(placements, range, { acrossNodes = false, activeOnly = false } = {}) {
  const href = (p) => acrossNodes ? ui.state.cluster.nodes.some((n) => n.name === p.node) ? nodeLink(p.id, range) : null : p.run_ids?.[0] ? `#run/${encode(p.run_ids[0])}/activity` : null;
  const x = (at) => ((at - range.start) / (range.now - range.start)) * 1000;
  const block = (from, to) => `M${x(from)},0 H${x(to)} V24 H${x(from)} Z`;
  const path = (p, blocks) => `<path class="activity-block" ${!acrossNodes ? `data-placement="${escape(p.id)}"` : ""} data-selected="${!acrossNodes && p.id === ui.expanded}" data-label="${escape(acrossNodes ? `${p.label}: ${processLabel(p)} (${p.node})` : p.label)}" data-source="${p.exact ? "GPU turns" : "Recorded operations"}" data-intervals="${escape(JSON.stringify(p.intervals))}" d="${blocks}" vector-effect="non-scaling-stroke"/>`;
  const axis = [0, 1, 2, 3, 4].map((tick) => `<span>${axisTime(range.start + ((range.now - range.start) * tick) / 4, range)}</span>`).join("");
  const activities = [...placements].sort((a, b) => (acrossNodes ? processOrder(a, b) : 0) || a.start - b.start || a.id.localeCompare(b.id)).map((p) => ({ ...p, ...operationActivity(p, range) }));
  // A run's lane on a shared worker it never used in this window says nothing; drop it once its turns have loaded.
  const shown = activities.filter((p) => (activeOnly ? p.intervals.length : !p.allocation_id || p.loading || p.error || p.intervals.length));
  if (!shown.length) {
    const loading = activities.some((p) => p.loading);
    const unavailable = activities.some((p) => p.warning || (p.error && p.errorStatus !== 404));
    return empty(loading ? "Loading activity…" : unavailable ? "Activity unavailable" : "No recorded activity in this window");
  }
  const rows = shown.map((p) => {
    const { intervals, error, loading } = p;
    const status = error ? (intervals.length ? "Stale" : error.includes("not recorded") ? "Not recorded" : "Unavailable") : loading ? "Loading…" : !intervals.length ? "No recorded activity" : "";
    const name = acrossNodes ? processLabel(p) : p.label, meta = acrossNodes ? p.node ? shortNodeName(p.node, ui.state.cluster.nodes) : "Node unassigned" : p.role || "process";
    const tag = href(p) ? "a" : "span";
    const attrs = href(p) ? `href="${escape(href(p))}"` : 'aria-disabled="true"';
    const label = `<${tag} class="activity-label" data-key="label:${escape(p.id)}" ${attrs} title="${escape(acrossNodes ? `${name} (${p.node || "node unassigned"})${p.node && !href(p) ? ". Node no longer reported" : ""}` : p.label)}"><span class="activity-label-text"><span class="activity-name">${escape(name)}</span><span class="activity-meta">${escape(meta)}${p.ended ? ", ended" : ""}${!href(p) ? acrossNodes ? ", node no longer reported" : ", run ID unavailable" : ""}</span></span></${tag}>`;
    // One path per run, with separate blocks at their exact times. Dense
    // histories stay cheap, and labels remain selectable even for tiny bursts.
    const blocks = intervals.map(([from, to]) => block(from, to)).join(" ");
    const rowLabel = acrossNodes ? `${p.label}: ${processLabel(p)} (${p.node})` : p.label;
    const ops = Object.entries(p.ops || {})
      .sort(([a], [b]) => opRank(a) - opRank(b))
      .map(([op, spans]) => `<path class="activity-block op" data-op="${escape(op)}" data-label="${escape(`${opLabel(op)}: ${rowLabel}`)}" data-source="Worker-recorded op" data-intervals="${escape(JSON.stringify(spans))}" d="${spans.map(([from, to]) => block(from, to)).join(" ")}" vector-effect="non-scaling-stroke"/>`)
      .join("");
    return `<div class="activity-row" data-key="${escape(p.id)}" data-selected="${!acrossNodes && p.id === ui.expanded}">
      ${label}
      <div class="activity-track" data-start="${range.start}" data-end="${range.now}"><svg viewBox="0 0 1000 24" preserveAspectRatio="none" aria-label="Recorded activity">${path(p, blocks)}${ops}</svg>${status ? `<span class="activity-status ${error ? "unavailable" : ""}" title="${escape(error || status)}">${status}</span>` : ""}</div></div>`;
  }).join("");
  return `<div class="activity-timeline" aria-label="Recorded run activity"><div class="activity-header"><h3>${acrossNodes ? "Process" : "Run"}</h3><div class="activity-axis">${axis}</div></div>${rows}${opLegend(shown)}</div>`;
}

function activityNotes(placements, range) {
  const activities = placements.map((p) => operationActivity(p, range));
  const warnings = [...new Set(activities.flatMap((a) => [a.warning, a.errorStatus !== 404 && a.error]).filter(Boolean))];
  const source = activities.every((a) => a.exact) ? "GPU turns" : activities.some((a) => a.exact) ? "GPU turns and recorded operations" : "Recorded operations; may include GPU wait time";
  return { warnings: warnings.length ? `<p class="source-error" role="status">${escape(warnings.join("; "))}</p>` : "", source: activities.length ? `${source}. Gaps may include unrecorded activity.` : "" };
}

export function runActivity(runId, range) {
  activityCache.clear();
  const placements = segments(range).filter((p) => p.run_ids?.includes(runId));
  const error = ui.state.history_error;
  if (!placements.length) return empty(error ? `Placement history unavailable: ${error}` : "No placement history for this run");
  const notes = activityNotes(placements, range);
  return `<section class="cross-node-activity run-activity" data-key="activity:${escape(runId)}" aria-label="Run process activity">${activityTimeline(placements, range, { acrossNodes: true })}${error ? `<p class="source-error" role="status">Placement history unavailable: ${escape(error)}</p>` : ""}${notes.warnings}<p class="activity-source">${notes.source}</p></section>`;
}

// Allocation metrics cover their physical devices for the full requested window.
// Query only enough allocations to cover the node, including shared workers once.
function nodeMetrics(node, placements, range) {
  const sources = new Map(), query = ui.nodeQueryRange || range;
  const params = new URLSearchParams({ since: new Date(query.start * 1000).toISOString(), until: new Date(query.now * 1000).toISOString() });
  for (const p of [...placements].sort((a, b) => Number(!!a.ended) - Number(!!b.ended) || b.devices.length - a.devices.length)) {
    const ids = p.devices.filter((id) => node.devices.some((d) => d.id === id) && !sources.has(id));
    if (!ids.length) continue;
    const id = p.allocation_id || p.id;
    const entry = use(`/api/v1/dashboard/allocations/${encode(id)}/metrics?${params}`, `allocation:${id}`);
    for (const device of ids) sources.set(device, entry);
  }
  const devices = node.devices.map((device) => {
    const source = sources.get(device.id), data = source?.data?.devices?.find((d) => d.id === device.id);
    return data || { ...device, utilization: [], memory_mib: [], reason: source?.error || source?.data?.reason || (source?.pending ? "Loading GPU metrics…" : "No recorded telemetry for this GPU") };
  });
  const errors = [...new Set([...sources].filter(([id]) => ui.device === "all" || id === ui.device).map(([, source]) => source.error).filter(Boolean))];
  const entries = [...new Set(sources.values())];
  const workers = entries.map((entry) => entry.data?.worker).filter(Boolean);
  const source = entries.map((entry) => entry.data?.source).find(Boolean);
  return { data: { devices, workers, source, reason: !devices.length ? "GPU device mapping unavailable" : null }, error: errors.join("; ") };
}

function detail(node, range, placements) {
  const devices = node.devices || [];
  if (ui.device !== "all" && !devices.some((d) => d.id === ui.device)) ui.device = "all";
  if (devices.length === 1) ui.device = "all";
  const visible = placements.filter((p) => (ui.device === "all" || p.devices.includes(ui.device)) && p.start <= range.now && p.end >= range.start);
  const metrics = nodeMetrics(node, placements, range);
  const picker = (devices.length > 1 ? [button("All GPUs", `data-device="all" aria-pressed="${ui.device === "all"}"`)] : [])
    .concat(devices.map((d) => {
      const value = lastValue(metrics.data.devices.find((device) => device.id === d.id)?.utilization, range);
      return button(`GPU ${deviceLabel(d.name)}${Number.isFinite(value) ? ` (${Math.round(value)}%)` : ""}`, `data-device="${escape(d.id)}" aria-pressed="${devices.length === 1 || ui.device === d.id}"`);
    })).join("");
  const short = shortNodeName(node.name, ui.state.cluster.nodes);
  const notes = activityNotes(visible, range);
  return `<section class="allocation-expansion" id="placement-detail"><div class="allocation-detail-head inspector-heading"><h2 title="${escape(node.name)}">${escape(acceleratorLabel(node))} <span class="muted">${escape(short)}</span></h2><button type="button" class="detail-close" data-close-details aria-label="Close node details" title="Close (Esc)">×</button></div>
    ${devices.length ? `<div class="allocation-device-picker" aria-label="GPU selection">${picker}</div>` : ""}
    <div class="node-activity" data-key="node-timeline">${activityTimeline(visible, range, { activeOnly: true })}${notes.warnings}</div>
    ${node.gpu_capacity ? `<div class="node-gpu-detail" data-key="node-gpu-detail"><div>${gpuChart(metrics, range)}</div><div class="allocation-detail-footer"><span>${notes.source}</span><span class="allocation-memory">GPU memory <strong>${gpuMemory(metrics, range)}</strong></span></div></div>` : ""}
    ${metrics.data.workers.length ? `<div class="node-gpu-detail" data-key="node-worker-detail"><div>${workerChart(metrics, range)}</div><div class="allocation-detail-footer"><span>${escape(sourceNote(metrics.data.source))}</span><span class="allocation-memory">Worker memory <strong>${workerMemory(metrics, range)}</strong></span></div></div>` : ""}</section>`;
}

const SOURCE_NOTES = { gke: "GPU: DCGM via Cloud Monitoring; CPU/memory: GKE container metrics", prometheus: "GPU: DCGM via Prometheus; CPU/memory: GKE container metrics", local: "GPU: nvidia-smi; CPU/memory: psutil, sampled on this host" };
const sourceNote = (source) => SOURCE_NOTES[source] || "";

// The OpenRL worker processes holding this node's GPUs, summed at each sample.
function sumSeries(workers, field) {
  const byTime = new Map();
  for (const worker of workers) for (const [at, value] of worker[field] || []) byTime.set(at, (byTime.get(at) || 0) + value);
  return [...byTime].sort((a, b) => a[0] - b[0]);
}

// Break a line only where samples are missing. Longer windows are queried at a
// coarser step (144s at 24 hours), so the gap is three typical sample spacings,
// never less than the source's own interval floor.
function gapFor(points, floor) {
  const times = points.map(([at]) => at).sort((a, b) => a - b);
  const spacings = times.slice(1).map((at, i) => at - times[i]).filter((d) => d > 0).sort((a, b) => a - b);
  return Math.max(floor, 3 * (spacings[Math.floor(spacings.length / 2)] || 0));
}

function workerChart(metrics, range) {
  const { workers } = metrics.data;
  const reason = workers.map((w) => w.reason).find(Boolean) || "No CPU samples for these workers";
  const points = sumSeries(workers, "cpu_cores");
  return chart({ title: `Worker CPU, ${workers.length} process${workers.length === 1 ? "" : "es"}`, unit: "cores", points, start: range.start, end: range.now, min: 0, gapSeconds: gapFor(points, 180), empty: reason });
}

function workerMemory(metrics, range) {
  const latest = lastValue(sumSeries(metrics.data.workers, "memory_bytes"), range);
  return Number.isFinite(latest) ? `${(latest / 2 ** 30).toFixed(1)} GiB` : "—";
}

const selectedDevices = (metrics) => [...new Map((metrics.data?.devices || []).filter((d) => ui.device === "all" || ui.device === d.id).map((d) => [d.uuid || d.id, d])).values()];

function gpuChart(metrics, range) {
  const title = "GPU utilization";
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
  const failures = [...new Set([metrics.error, ...selected.map((d) => d.reason)].filter(Boolean))];
  const hasSamples = points.some(([at, value]) => at >= range.start && at <= range.now && Number.isFinite(value));
  return `${chart({
    title,
    unit: "%",
    points,
    start: range.start,
    end: range.now,
    min: 0,
    max: 100,
    gapSeconds: gapFor(points, 60),
    empty: metrics.error || reason,
  })}${hasSamples && failures.length ? `<p class="source-error" role="status">${escape(failures.join("; "))}${metrics.error ? ". Showing previously fetched samples" : ""}</p>` : ""}`;
}

const lastValue = (points, range) => points?.findLast(([at]) => at >= range.start && at <= range.now)?.[1];

function gpuMemory(metrics, range) {
  const latest = selectedDevices(metrics).map((d) => lastValue(d.memory_mib, range));
  return latest.length && latest.every(Number.isFinite) ? `${(latest.reduce((a, b) => a + b, 0) / 1024).toFixed(1)} GiB` : "—";
}

// ---- page -----------------------------------------------------------------------------

// The Overview's view of the fleet: every GPU of every node as one thin row,
// painted with the ops it ran over the last 30 minutes.
export function fleetStrip() {
  activityCache.clear();
  const { cluster } = ui.state;
  const gpuNodes = cluster.nodes.filter((n) => n.gpu_capacity && n.devices?.length);
  if (!gpuNodes.length) return "";
  const range = { start: nodeNow() - 1800, now: nodeNow(), live: true };
  const all = segments(range).filter((s) => s.start <= range.now && s.end >= range.start);
  const at = (a, b) => `left:${((a - range.start) / 1800) * 100}%;width:${((b - a) / 1800) * 100}%`;
  let busy = 0, loading = false;
  const activities = [];
  const occupied = new Set(ui.state.placements.map((p) => p.node));
  const rows = [...gpuNodes].sort((a, b) => Number(occupied.has(b.name)) - Number(occupied.has(a.name))).map((node) => {
    const lanes = node.devices.map((device) => {
      const sharers = all.filter((s) => s.node === node.name && s.devices.includes(device.id));
      const turns = [], spans = [];
      for (const s of sharers) {
        const activity = operationActivity(s, range);
        loading ||= activity.loading;
        activities.push(activity);
        turns.push(...activity.intervals);
        const ops = Object.entries(activity.ops || {}).sort(([a], [b]) => opRank(a) - opRank(b));
        if (!ops.length) spans.push(...activity.intervals.map(([a, b]) => `<span style="${at(a, b)}"></span>`));
        for (const [op, list] of ops) spans.push(...list.map(([a, b]) => `<span data-op="${escape(op)}" style="${at(a, b)}"></span>`));
      }
      busy += mergeIntervals(turns).reduce((sum, [a, b]) => sum + (b - a), 0);
      return `<span class="fleet-gpu${sharers.length ? " claimed" : ""}">${spans.join("")}</span>`;
    });
    const label = `${acceleratorLabel(node)} ${shortNodeName(node.name, cluster.nodes)}`;
    return `<a class="fleet-node" href="${escape(viewLink("nodes", { node: node.name }))}" data-key="fleet:${escape(node.name)}" title="${escape(node.name)}"><span class="fleet-name">${escape(acceleratorLabel(node))} <span class="muted">${escape(shortNodeName(node.name, cluster.nodes))}</span></span><span class="fleet-lane" aria-label="${escape(`${label}, GPU activity over the last 30 minutes`)}">${lanes.join("")}</span></a>`;
  });
  const capacity = gpuNodes.reduce((sum, n) => sum + n.gpu_capacity, 0);
  const claimed = gpuNodes.reduce((sum, n) => sum + new Set(ui.state.placements.filter((p) => p.node === n.name).flatMap((p) => p.devices)).size, 0);
  const share = loading ? "…" : `${Math.round((100 * busy) / (1800 * capacity))}%`;
  return `<section class="fleet${loading ? " loading" : ""}" data-key="fleet" aria-label="GPU fleet"><p class="fleet-summary"><span><strong>${claimed}</strong> of ${capacity} GPUs claimed</span><span title="Share of all GPU time spent in recorded GPU turns"><strong>${share}</strong> busy over the last 30 minutes</span></p>${rows.join("")}${opLegend(activities)}</section>`;
}

export function renderNodes() {
  activityCache.clear();
  const range = timeWindow();
  const all = segments(range);
  if (ui.expanded && !all.some((p) => p.id === ui.expanded)) ui.expanded = all.find((p) => p.allocation_id === ui.expanded)?.id || ui.expanded;
  const { cluster } = ui.state;
  const selected = all.find((p) => p.id === ui.expanded);
  // Older shared links identify a placement; resolve it to its physical node.
  if (!ui.inspectorNode && selected) ui.inspectorNode = selected.node;
  const node = cluster.nodes.find((n) => n.name === ui.inspectorNode);
  const unavailable = (ui.inspectorNode || ui.expanded) && !node;
  if (node && selected?.node !== node.name) ui.expanded = null;
  const occupied = new Set(ui.state.placements.map((p) => p.node));
  const nodes = [...cluster.nodes].sort((a, b) => Number(occupied.has(b.name)) - Number(occupied.has(a.name)) || Number(b.gpu_capacity > 0) - Number(a.gpu_capacity > 0));
  const axis = [0, 1, 2, 3].map((tick) => `<span>${axisTime(range.start + ((range.now - range.start) * tick) / 3, range)}</span>`).join("");
  const errors = [...new Set([cluster.error, cluster.nodes_error, cluster.devices?.error, cluster.scheduler?.error, ui.state.history_error].filter(Boolean))];
  morph(
    content,
    `<div class="nodes-heading"><h1 class="heading">${ui.state.telemetry_sources?.mode === "local" ? "This host" : "Kubernetes nodes"}</h1>${timeControl(range)}</div>${errors.map((error) => `<p class="source-error" role="status">${escape(error)}</p>`).join("")}${!cluster.available && !errors.length ? empty("Kubernetes unavailable") : ""}
    ${unavailable ? `<div class="unavailable-selection" role="status"><span>${ui.inspectorNode ? "The selected node is no longer reported." : "The selected placement is not in retained history."}</span><button type="button" class="detail-close" data-close-details aria-label="Clear unavailable selection" title="Clear selection">×</button></div>` : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${axis}</div><span class="node-duty" title="Share of GPU time allocated over this window, not measured GPU utilization">Allocated</span></div>
    ${nodes.map((node) => lane(node, all, range)).join("") || (!errors.length && cluster.available ? empty("No nodes reported by Kubernetes") : "")}`,
  );
}
