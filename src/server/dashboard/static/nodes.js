// The Nodes page: one lane per node, one row per GPU, allocation bars over
// the selected window from the recorded placement history, and an expansion
// with run activity lanes and GPU utilization for those devices.

import { escape, encode, empty, button, morph, shortNodeName } from "./ui.js";
import { chart } from "./charts.js";
import { ui, content, nodeNow, nodeTime, family, nodeLink } from "./store.js";
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
    return { ...placement, id: `${placement.id}:${id}`, allocation_id: placement.id, run_ids: [id], label: `${(run?.model || placement.label).split("/").at(-1)} · ${id.slice(0, 8)}` };
  }) : [placement]));
}

const placementColor = (placement) => family(placement.run_ids?.[0] || placement.runtime_id);

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
  // A shared worker's turn has no logical run ID. Its operations remain the
  // per-run evidence; copying the worker's turns to every run would duplicate it.
  if (!placement.allocation_id && turns.data?.samples?.length) {
    const intervals = turns.data.samples
      .map((t) => [Math.max(range.start, placement.start, t.started_at), Math.min(range.now, placement.end, t.at)])
      .filter(([from, to]) => to > from);
    return save({ intervals: mergeIntervals(intervals), error: turns.error || "", loading: false, exact: true });
  }
  const intervals = [];
  let error = "", errorStatus = null, loading = false;
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
          operationActivity(s, range).intervals.map(
            ([a, b]) => `<span class="hold ${placementColor(s)} ${s.id === ui.expanded ? "selected" : ""}" data-placement="${escape(s.id)}" style="${within(a, b)}" title="${escape(s.label)} · ${escape(s.role || "")}"></span>`,
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
        return `<button type="button" class="capacity-allocation ${placementColor(s)} ${s.ended ? "ended" : ""} ${s.id === ui.expanded ? "selected" : ""}" data-key="${escape(s.id)}:${group[0]}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}" aria-label="${escape(title)}" title="${escape(title)}" style="${span(s.start, s.end)};top:${group[0] * height + 2}px;height:${group.length * height - 4}px"><span class="allocation-name">${escape(s.label)}</span><span class="allocation-count">${group.length} GPU${group.length === 1 ? "" : "s"}</span></button>`;
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
  const current = neighbours.find((s) => s.id === ui.expanded);
  const bars = trackBars(devices, nodeSegments, range);
  const accelerator = acceleratorLabel(node);
  const height = laneHeight(devices);
  const unmapped = nodeSegments.filter((s) => !s.devices.length || s.devices.some((id) => !devices.some((d) => d.id === id)));
  const unknown = unmapped
    .map(
      (s) =>
        `<button type="button" class="capacity-allocation unknown-mapping ${placementColor(s)}" data-key="unmapped:${escape(s.id)}" data-placement="${escape(s.id)}" aria-expanded="${s.id === ui.expanded}"><span class="allocation-name">${escape(s.label)}</span><span class="allocation-count">${s.device_count} GPU${s.device_count === 1 ? "" : "s"}</span></button>`,
    )
    .join("");
  const track = devices.length
    ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-auto-rows:${height}px">${devices.map((d) => `<span title="${escape(d.id)}">${escape(deviceLabel(d.name))}</span>`).join("")}</div><div class="capacity-track" style="height:${devices.length * height}px;--gpu-lane-height:${height}px">${bars}</div></div>`
    : "";
  return `<div class="node-placement-group" data-key="${escape(node.name)}"><div class="node-lane"><div class="node-lane-label" title="${escape(node.name)}"><span class="node-accelerator">${escape(accelerator)} <span class="muted">· ${escape(shortNodeName(node.name, ui.state.cluster.nodes))}</span></span>${node.gpu_capacity ? `<span class="claim-label">${escape(claimLabel(node, placements, range.live))}</span>` : ""}</div><div>${track}${unknown}${unmapped.length ? '<span class="muted micro">Device mapping unavailable</span>' : !devices.length && node.gpu_capacity ? empty("GPUs without DRA devices") : ""}</div><span class="node-duty" title="Recorded GPU allocation time over the selected window">${duty(node, nodeSegments, range)}</span></div>${current ? detail(current, range, neighbours, all) : ""}</div>`;
}

// ---- expansion ---------------------------------------------------------------------

function processLabel(placement) {
  const node = ui.state.cluster.nodes.find((n) => n.name === placement.node);
  const devices = placement.devices.map((id) => node?.devices.find((d) => d.id === id));
  const gpus = devices.length && devices.every(Boolean) ? `GPU${devices.length === 1 ? "" : "s"} ${devices.map((d) => deviceLabel(d.name)).join(", ")}` : `${placement.device_count} GPU${placement.device_count === 1 ? "" : "s"}`;
  const role = placement.role || "Process";
  const accelerator = node && acceleratorLabel(node);
  return [role[0].toUpperCase() + role.slice(1), accelerator === "GPU" ? null : accelerator, gpus].filter(Boolean).join(" · ");
}

function activityTimeline(placements, devices, range, { acrossNodes = false, navigate = false } = {}) {
  const byGpu = !acrossNodes && ui.activityView === "gpu";
  const canSelect = (p) => !acrossNodes || ui.state.cluster.nodes.some((n) => n.name === p.node);
  const x = (at) => ((at - range.start) / (range.now - range.start)) * 1000;
  const block = (from, to, top = 0, bottom = 24) => `M${x(from)},${top} H${x(to)} V${bottom} H${x(from)} Z`;
  const path = (p, blocks) => {
    const shape = `<path class="activity-block ${placementColor(p)}" ${canSelect(p) && !navigate ? `data-placement="${escape(p.id)}"` : ""} data-selected="${!navigate && p.id === ui.expanded}" data-label="${escape(acrossNodes ? `${p.label} · ${processLabel(p)} · ${p.node}` : p.label)}" data-source="${p.exact ? "GPU turns" : "Recorded operations"}" data-intervals="${escape(JSON.stringify(p.intervals))}" d="${blocks}" vector-effect="non-scaling-stroke"/>`;
    return navigate && canSelect(p) ? `<a href="${escape(nodeLink(p.id, range))}" aria-label="Inspect ${escape(processLabel(p))} on ${escape(p.node)}">${shape}</a>` : shape;
  };
  const axis = [0, 1, 2, 3, 4].map((tick) => `<span>${axisTime(range.start + ((range.now - range.start) * tick) / 4, range)}</span>`).join("");
  const activities = [...placements].sort((a, b) => (acrossNodes ? ({ trainer: 0, sampler: 1 }[a.role] ?? 2) - ({ trainer: 0, sampler: 1 }[b.role] ?? 2) || String(a.node || "").localeCompare(String(b.node || "")) : 0) || a.start - b.start || a.id.localeCompare(b.id)).map((p) => ({ ...p, ...operationActivity(p, range) }));
  const combined = byGpu ? devices.map((device) => {
    const jobs = activities.filter((p) => p.devices.includes(device.id));
    const events = jobs.flatMap((p, i) => p.intervals.flatMap(([from, to]) => [[from, i, true], [to, i, false]])).sort((a, b) => a[0] - b[0]);
    const active = new Set(), blocks = jobs.map(() => []);
    let previous = range.start;
    for (let i = 0; i < events.length;) {
      const at = events[i][0];
      // Concurrent operations share the strip vertically; no run hides another.
      if (at > previous) [...active].sort((a, b) => a - b).forEach((job, row) => blocks[job].push(block(previous, at, (row * 24) / active.size, ((row + 1) * 24) / active.size)));
      while (i < events.length && events[i][0] === at) {
        const [, job, starts] = events[i++];
        if (starts) active.add(job);
        else active.delete(job);
      }
      previous = at;
    }
    return `<div class="activity-row activity-combined" data-key="combined:${escape(device.id)}"><span class="activity-device">GPU ${escape(deviceLabel(device.name))}</span><div class="activity-track" data-start="${range.start}" data-end="${range.now}"><svg viewBox="0 0 1000 24" preserveAspectRatio="none" aria-hidden="true">${jobs.map((p, i) => path(p, blocks[i].join(" "))).join("")}</svg></div></div>`;
  }).join("") : "";
  const rows = activities.map((p) => {
    const { intervals, error, loading } = p;
    const status = error ? (intervals.length ? "Stale" : error.includes("not recorded") ? "Not recorded" : "Unavailable") : loading ? "Loading…" : !intervals.length ? "No recorded activity" : "";
    const name = acrossNodes ? processLabel(p) : p.label, meta = acrossNodes ? p.node ? shortNodeName(p.node, ui.state.cluster.nodes) : "Node unassigned" : p.role || "process";
    const tag = navigate ? canSelect(p) ? "a" : "span" : "button";
    const attrs = navigate ? canSelect(p) ? `href="${escape(nodeLink(p.id, range))}"` : 'aria-disabled="true"' : `type="button" data-placement="${escape(p.id)}" ${canSelect(p) ? "" : "disabled"} aria-pressed="${p.id === ui.expanded}"`;
    const label = `<${tag} class="activity-label ${placementColor(p)}" data-key="label:${escape(p.id)}" ${attrs} title="${escape(acrossNodes ? `${name} · ${p.node || "Node unassigned"}${p.node && !canSelect(p) ? " · Node no longer reported" : ""}` : p.label)}"><span class="activity-swatch"></span><span class="activity-label-text"><span class="activity-name">${escape(name)}</span><span class="activity-meta">${escape(meta)}${p.ended ? " · Ended" : ""}${p.node && !canSelect(p) ? " · Node no longer reported" : ""}</span></span>${byGpu && status ? `<span class="activity-notice ${error ? "unavailable" : ""}" title="${escape(error || status)}">${status}</span>` : ""}</${tag}>`;
    if (byGpu) return label;
    // One path per run, with separate blocks at their exact times. Dense
    // histories stay cheap, and labels remain selectable even for tiny bursts.
    const blocks = intervals.map(([from, to]) => block(from, to)).join(" ");
    return `<div class="activity-row ${placementColor(p)}" data-key="${escape(p.id)}" data-selected="${!navigate && p.id === ui.expanded}">
      ${label}
      <div class="activity-track" data-start="${range.start}" data-end="${range.now}"><svg viewBox="0 0 1000 24" preserveAspectRatio="none" ${navigate ? 'aria-label="Recorded activity"' : 'aria-hidden="true"'}>${path(p, blocks)}</svg>${status ? `<span class="activity-status ${error ? "unavailable" : ""}" title="${escape(error || status)}">${status}</span>` : ""}</div></div>`;
  }).join("");
  return `<div class="activity-timeline" aria-label="Recorded run activity"><div class="activity-header"><h3>${byGpu ? "GPU" : acrossNodes ? "Process" : "Run"}</h3><div class="activity-axis">${axis}</div></div>${byGpu ? `${combined || empty("Device mapping unavailable")}<div class="activity-legend" aria-label="Runs on these GPUs">${rows}</div>` : rows}</div>`;
}

function activityNotes(placements, range) {
  const activities = placements.map((p) => operationActivity(p, range));
  const warnings = [...new Set(activities.flatMap((a) => [a.warning, a.errorStatus !== 404 && a.error]).filter(Boolean))];
  const source = activities.every((a) => a.exact) ? "GPU turns" : activities.some((a) => a.exact) ? "GPU turns and recorded operations" : "Recorded operations; may include GPU wait time";
  return { warnings: warnings.length ? `<p class="source-error" role="status">${escape(warnings.join(" · "))}</p>` : "", source: `${source}. Gaps may include unrecorded activity.` };
}

export function runActivity(runId, range) {
  activityCache.clear();
  const placements = segments(range).filter((p) => p.run_ids?.includes(runId));
  const error = ui.state.history_error;
  if (!placements.length) return empty(error ? `Placement history unavailable: ${error}` : "No placement history for this run");
  const notes = activityNotes(placements, range);
  return `<section class="cross-node-activity run-activity" data-key="activity:${escape(runId)}" aria-label="Run process activity">${activityTimeline(placements, [], range, { acrossNodes: true, navigate: true })}${error ? `<p class="source-error" role="status">Placement history unavailable: ${escape(error)}</p>` : ""}${notes.warnings}<p class="activity-source">${notes.source}</p></section>`;
}

function runAcrossNodes(placement, all, range) {
  const runId = placement.run_ids?.[0];
  if (!runId) return "";
  const related = all.filter((p) => p.node && p.run_ids?.includes(runId));
  if (new Set(related.map((p) => p.node).filter(Boolean)).size < 2) return "";
  const warnings = [...new Set(related.map((p) => operationActivity(p, range)).flatMap((a) => [a.warning, a.errorStatus !== 404 && a.error]).filter(Boolean))];
  return `<div class="cross-node-activity" data-key="across:${escape(runId)}"><div class="allocation-detail-head"><h2>${escape(placement.label)} <span class="muted">· Across nodes</span></h2><a href="#run/${encode(runId)}/activity">Run details ↗</a></div>${activityTimeline(related, [], range, { acrossNodes: true })}${warnings.length ? `<p class="source-error" role="status">${escape(warnings.join(" · "))}</p>` : ""}</div>`;
}

function detail(placement, range, neighbours, all) {
  const anchor = neighbours.find((p) => p.id === ui.gpuGroup && p.devices.some((id) => placement.devices.includes(id))) || placement;
  ui.gpuGroup = anchor.id;
  if (ui.device !== "all" && !anchor.devices.includes(ui.device)) ui.device = "all";
  if (anchor.devices.length === 1) ui.device = anchor.devices[0];
  const { state } = ui;
  const node = state.cluster.nodes.find((n) => n.name === anchor.node);
  const devices = (node?.devices || []).filter((d) => anchor.devices.includes(d.id));
  const group = neighbours.filter((p) => p.id === anchor.id || p.id === placement.id || (p.start <= range.now && p.end >= range.start && p.devices.some((id) => anchor.devices.includes(id))));
  const visible = group.filter((p) => ui.device === "all" || p.devices.includes(ui.device));
  const run = visible.some((p) => p.id === placement.id) && state.runs.find((r) => (placement.run_ids || []).includes(r.run_id));
  const workload = run?.workloads?.find((w) => w.uid === (placement.allocation_id || placement.id));
  const title = [
    (run?.model || placement.runtime_id || placement.label).split("/").at(-1),
    (run?.run_id || placement.owner_id)?.slice(0, 8),
    { trainer: "Trainer", sampler: "Sampler" }[placement.role] || "Unknown process",
    placement.role === "trainer" ? { lora: "LoRA", fft: "FFT" }[workload?.training_kind] : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const allocationId = anchor.allocation_id || anchor.id;
  const query = ui.nodeQueryRange || range;
  const metrics = use(
    `/api/v1/dashboard/allocations/${encode(allocationId)}/metrics?${new URLSearchParams({ since: new Date(query.start * 1000).toISOString(), until: new Date(query.now * 1000).toISOString() })}`,
    `allocation:${allocationId}`,
  );
  const picker = (anchor.devices.length > 1 ? [button("All GPUs", `data-device="all" aria-pressed="${ui.device === "all"}"`)] : [])
    .concat(
      anchor.devices.map((id) => {
        const name = deviceLabel(devices.find((d) => d.id === id)?.name || id.split("/").at(-1));
        const value = lastValue(metrics.data?.devices?.find((d) => d.id === id)?.utilization, range);
        return button(
          `GPU ${name}${Number.isFinite(value) ? ` · ${Math.round(value)}%` : ""}`,
          `data-device="${escape(id)}" aria-pressed="${ui.device === id}"`,
        );
      }),
    )
    .join("");
  const label = `${node ? acceleratorLabel(node) : "GPU activity"} · ${shortNodeName(anchor.node, ui.state.cluster.nodes)}${devices.length ? ` · GPU${devices.length === 1 ? "" : "s"} ${devices.map((d) => deviceLabel(d.name)).join(", ")}` : ""}`;
  const notes = activityNotes(visible, range);
  const acrossNodes = runAcrossNodes(placement, all, range);
  return `<section class="allocation-expansion" id="placement-detail" data-key="${escape(anchor.id)}" data-view="${ui.activityView}"><div class="allocation-detail-head"><h2 title="${escape(anchor.node)}">${escape(label)}</h2><div class="allocation-detail-actions"><span class="allocation-memory">GPU memory <strong>${gpuMemory(metrics, range)}</strong></span><button type="button" class="detail-close" data-close-details aria-label="Close GPU details" title="Close (Esc)">×</button></div></div>
    <div class="allocation-controls"><div class="allocation-device-picker" aria-label="GPU selection">${picker}</div><div class="activity-view" role="group" aria-label="Activity layout">${[ ["gpu", "By GPU"], ["run", "By run"] ].map(([view, text]) => button(text, `data-activity-view="${view}" aria-pressed="${ui.activityView === view}"`)).join("")}</div></div>
    ${activityTimeline(visible, devices.filter((d) => ui.device === "all" || d.id === ui.device), range)}
    ${notes.warnings}
    ${acrossNodes}
    <div>${gpuChart(metrics, range)}</div>
    <div class="allocation-detail-footer"><span>${notes.source}</span>${run && !acrossNodes ? `<a href="#run/${encode(run.run_id)}/activity">${escape(title)} · Run details ↗</a>` : ""}</div></section>`;
}

const selectedDevices = (metrics) => [...new Map((metrics.data?.devices || []).filter((d) => ui.device === "all" || ui.device === d.id).map((d) => [d.uuid || d.id, d])).values()];

function gpuChart(metrics, range) {
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
  const failures = [...new Set([metrics.error, ...selected.map((d) => d.reason)].filter(Boolean))];
  const hasSamples = points.some(([at, value]) => at >= range.start && at <= range.now && Number.isFinite(value));
  return `${chart({
    title: "GPU utilization",
    unit: "%",
    points,
    start: range.start,
    end: range.now,
    min: 0,
    max: 100,
    gapSeconds: 60,
    empty: metrics.error || reason,
  })}${hasSamples && failures.length ? `<p class="source-error" role="status">${escape(failures.join(" · "))}${metrics.error ? " · Showing previously fetched samples" : ""}</p>` : ""}`;
}

const lastValue = (points, range) => points?.findLast(([at]) => at >= range.start && at <= range.now)?.[1];

function gpuMemory(metrics, range) {
  const latest = selectedDevices(metrics).map((d) => lastValue(d.memory_mib, range));
  return latest.length && latest.every(Number.isFinite) ? `${(latest.reduce((a, b) => a + b, 0) / 1024).toFixed(1)} GiB` : "—";
}

// ---- page -----------------------------------------------------------------------------

export function renderNodes() {
  activityCache.clear();
  const range = timeWindow();
  const all = segments(range);
  if (ui.expanded && !all.some((p) => p.id === ui.expanded)) ui.expanded = all.find((p) => p.allocation_id === ui.expanded)?.id || ui.expanded;
  const { cluster } = ui.state;
  const selected = all.find((p) => p.id === ui.expanded);
  const unavailable = ui.expanded && (!selected || !cluster.nodes.some((n) => n.name === selected.node));
  const occupied = new Set(ui.state.placements.map((p) => p.node));
  const nodes = [...cluster.nodes].sort((a, b) => Number(occupied.has(b.name)) - Number(occupied.has(a.name)) || Number(b.gpu_capacity > 0) - Number(a.gpu_capacity > 0));
  const axis = [0, 1, 2, 3].map((tick) => `<span>${axisTime(range.start + ((range.now - range.start) * tick) / 3, range)}</span>`).join("");
  const errors = [...new Set([cluster.error, cluster.nodes_error, cluster.devices?.error, cluster.scheduler?.error, ui.state.history_error].filter(Boolean))];
  morph(
    content,
    `<div class="nodes-heading"><h1 class="heading">Kubernetes nodes</h1>${timeControl(range)}</div>${errors.map((error) => `<p class="source-error" role="status">${escape(error)}</p>`).join("")}${!cluster.available && !errors.length ? empty("Kubernetes unavailable") : ""}
    ${unavailable ? `<div class="unavailable-selection" role="status"><span>${selected ? "The selected placement's node is no longer reported." : "The selected placement is not in retained history."}</span><button type="button" class="detail-close" data-close-details aria-label="Clear unavailable selection" title="Clear selection">×</button></div>` : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${axis}</div><span class="node-duty" title="GPU claimed time over this window, not measured GPU utilization">Claimed</span></div>
    ${nodes.map((node) => lane(node, all, range)).join("") || (!errors.length && cluster.available ? empty("No nodes reported by Kubernetes") : "")}`,
  );
}
