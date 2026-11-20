// GPU & process activity timelines (Run across-nodes + per-GPU Duty cycle, and Node activity).
import { renderChart } from "./charts.js";
import { escape } from "./format.js";
import { inspectUrl, nodeLink, setCrosshair, ui, use, viewLink } from "./state.js";
import { DCGM_READING_SECONDS, activitySource, adaptLanes, axisTime, chartTime, deviceLabel, dutyBands, opLabel, opRank, placementColor, processLabel, shortNodeName, valueText } from "./visualization.js";

const trackRegistry = new Map();
let trackSeq = 0;

export function resetActivity() {
  trackRegistry.clear();
  trackSeq = 0;
}

function renderGpuDuty(node, device, range, runs) {
  const bounds = { node_id: node, since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() };
  const opts = { poll: 15000, pollWhen: range.live, remember: !range.live };
  const utilization = use(inspectUrl("query", { name: "gpu_util", ...bounds }), { ...opts, scope: `gpu-util:${node}` });
  const holders = use(inspectUrl("activity", bounds), { ...opts, scope: `gpu-act:${node}` });
  const points = utilization.data?.series?.find((s) => s.labels.UUID === device.id)?.points || [];
  return renderChart({
    id: `gpu-duty:${node}:${device.id}`,
    title: `Duty cycle · ${device.label}`,
    unit: "%",
    points,
    start: range.start,
    end: range.now,
    min: 0,
    max: 100,
    readingSeconds: DCGM_READING_SECONDS,
    bands: dutyBands(adaptLanes(holders.data?.lanes || [], runs), device.id),
    empty: utilization.error || (!utilization.data && "Loading…") || "No samples for this GPU",
  });
}

export function renderActivity({
  placements = [],
  range,
  inventory,
  acrossNodes = false,
  activeOnly = false,
  selected = null,
  selectable = false,
  error = null,
  pending = false,
  showSource = true,
  back = null,
  pickLogs = false,
  duty = null,
  slots = null,
}) {
  const mode = acrossNodes ? ui.activityMode : "lanes";
  const activities = placements.toSorted((a, b) =>
    (acrossNodes ? (({ trainer: 0, sampler: 1 }[a.role] ?? 2) - ({ trainer: 0, sampler: 1 }[b.role] ?? 2) || String(a.node || "").localeCompare(String(b.node || ""))) : 0) ||
    a.start - b.start ||
    a.id.localeCompare(b.id));
  const shown = activeOnly ? activities.filter((p) => p.intervals.length || p.error) : activities;

  function gpuOf(p, id) {
    const device = inventory.cluster.nodes.find((n) => n.name === p.node)?.devices.find((d) => d.id === id);
    return { id, label: device ? `GPU ${deviceLabel(device.name)}` : slots?.[id] != null ? `GPU ${slots[id]}` : id.slice(0, 12) };
  }

  const rows = shown.flatMap((p) => (acrossNodes && p.devices?.length ? p.devices : [null]).map((id, i) => ({
    key: `${p.id}:${id ?? "none"}`,
    lane: p,
    device: id && gpuOf(p, id),
    first: i === 0,
    lead: i === 0 && p.devices.length > 1,
  })));
  const warnings = [...new Set(activities.map((p) => p.errorStatus !== 404 && p.error).filter(Boolean))];
  const ops = [...new Set(shown.flatMap((p) => Object.keys(p.ops || {})))].sort((a, b) => opRank(a) - opRank(b) || a.localeCompare(b));
  const empty = error ? `Activity unavailable: ${error}` : pending ? "Loading activity…" : acrossNodes && !activities.length ? "No worker processes for this run in this window" : "No recorded activity in this window";
  const href = (p) => acrossNodes
    ? inventory.cluster.nodes.some((n) => n.name === p.node) ? nodeLink(p.id, range) : null
    : inventory.runs.some((r) => r.run_id === p.run_ids?.[0]) ? viewLink(`run/${encodeURIComponent(p.run_ids[0])}/activity`, { duration: range.now - range.start, end: range.now, back }) : null;
  const blocks = (spans) => spans.map(([a, b]) => `M${1000 * (a - range.start) / (range.now - range.start)},0 H${1000 * (b - range.start) / (range.now - range.start)} V24 H${1000 * (a - range.start) / (range.now - range.start)} Z`).join(" ");
  const rowLabel = (p) => acrossNodes ? `${p.label} · ${processLabel(p, inventory.cluster.nodes)} · ${p.node}` : p.label;
  const dutyText = (device) => { const v = device && duty?.[device.id]; return Number.isFinite(v) ? `${v}%` : "—"; };

  const reg = (spans, label, source) => {
    const id = `tr-${trackSeq++}`;
    trackRegistry.set(id, { spans, label, source, range, pickLogs });
    return id;
  };

  const head = acrossNodes
    ? `<h2 class="scheduler-title activity-title">GPU activity</h2><p class="muted activity-intro">One row per accelerator. Colored bars are recorded work on that GPU; translucent bars are GPU turns or placements held. Duty is the share of time a kernel ran on the GPU, measured by DCGM.</p>
       <div class="activity-mode" role="group" aria-label="Lanes or duty cycle"><button type="button" class="chip" data-activity-mode="lanes" aria-pressed="${mode === "lanes"}">Lanes</button><button type="button" class="chip" data-activity-mode="duty" aria-pressed="${mode === "duty"}">Duty cycle</button></div>`
    : "";

  let body;
  if (!shown.length) {
    body = `<p class="empty-state">${escape(empty)}</p>`;
  } else {
    const axis = [0, 1, 2, 3, 4].map((tick) => `<span>${escape(axisTime(range.start + (range.now - range.start) * tick / 4, range))}</span>`).join("");
    const rowsHtml = rows.map(({ key, lane: p, device, first, lead }) => {
      const name = acrossNodes ? processLabel(p, inventory.cluster.nodes, { gpus: false }) : p.label;
      const meta = acrossNodes ? p.node ? shortNodeName(p.node, inventory.cluster.nodes) : p.ended && p.pod ? p.pod : "Node unassigned" : p.role || "process";
      const link = href(p);
      const status = p.error ? "Ops unavailable" : !p.intervals.length ? "No recorded activity" : "";
      const color = placementColor(p);
      const isOpen = ui.openGpus.has(key);
      const metaText = `${meta}${p.ended ? " · Process ended" : p.phase && p.phase !== "ready" ? ` · ${p.phase.replace("_", " ")}` : ""}${!link ? acrossNodes ? p.node ? " · Node no longer reported" : "" : p.run_ids?.length ? " · Run record unavailable" : " · No linked run" : ""}`;
      const labelEl = first
        ? `<${link ? `a href="${escape(link)}"` : 'span aria-disabled="true"'} class="activity-label ${color}" data-key="label:${escape(p.id)}" title="${escape(acrossNodes ? `${name} · ${p.node || "Node unassigned"}${p.node && !link ? " · Node no longer reported" : ""}` : p.label)}"><span class="activity-swatch"></span><span class="activity-label-text"><span class="activity-name">${escape(name)}</span><span class="activity-meta">${escape(metaText)}</span></span></${link ? "a" : "span"}>`
        : "<span></span>";
      const gpuEl = acrossNodes
        ? device && p.node
          ? `<button type="button" class="activity-gpu activity-gpu-toggle" data-toggle-gpu="${escape(key)}" aria-expanded="${isOpen}" title="${escape(`${device.id} · ${isOpen ? "Hide" : "Plot"} this GPU's duty cycle`)}"><span class="activity-chevron" aria-hidden="true">${isOpen ? "▾" : "▸"}</span>${escape(device.label)}</button>`
          : `<span class="activity-gpu" title="${escape(device ? device.id : "This process recorded no GPUs")}">${escape(device?.label ?? "GPUs not recorded")}</span>`
        : "";

      const rl = rowLabel(p);
      const waitPath = p.waits?.length ? `<path class="activity-block wait" data-track-id="${reg(p.waits, `Waiting for the GPU · ${rl}`, "From asking the time slicer to the grant")}" d="${blocks(p.waits)}" vector-effect="non-scaling-stroke" />` : "";
      const selectAttrs = selectable ? ` role="button" tabindex="0" data-placement="${escape(p.id)}" data-selected="${p.id === selected}"` : "";
      const basePath = `<path class="activity-block ${color}" aria-label="${escape(rl)}"${selectAttrs} data-track-id="${reg(p.intervals, rl, p.exact ? "GPU turns" : "Recorded operations")}" d="${blocks(p.intervals)}" vector-effect="non-scaling-stroke" />`;
      const opPaths = Object.entries(p.ops || {}).sort(([a], [b]) => opRank(a) - opRank(b)).map(([op, spans]) =>
        `<path class="activity-block op" aria-label="${escape(`${opLabel(op)} · ${rl}`)}"${selectable ? ` role="button" tabindex="0" data-placement="${escape(p.id)}"` : ""} data-op="${escape(op)}" data-track-id="${reg(spans, `${opLabel(op)} · ${rl}`, "Worker-recorded op")}" d="${blocks(spans)}" vector-effect="non-scaling-stroke" />`).join("");
      const svg = mode === "lanes" ? `<svg viewBox="0 0 1000 24" preserveAspectRatio="none" role="img" aria-label="Recorded activity">${waitPath}${basePath}${opPaths}</svg>` : "";
      const statusHtml = status && first ? `<span class="activity-status${p.error ? " unavailable" : ""}" title="${escape(p.error || status)}">${escape(status)}</span>` : "";
      const dutyHtml = acrossNodes ? `<span class="node-duty">${escape(dutyText(device))}</span>` : "";
      const dutyRow = (isOpen || mode === "duty") && device && p.node ? `<div class="activity-duty" data-key="duty:${escape(key)}">${renderGpuDuty(p.node, device, range, inventory.runs)}</div>` : "";
      return `<div class="activity-row ${color}" data-key="${escape(key)}" data-selected="${!acrossNodes && p.id === selected}"${first ? "" : " data-continued"}${lead ? " data-lead" : ""}>${labelEl}${gpuEl}<div class="activity-track"${pickLogs ? ' data-pick-logs="true"' : ""}>${svg}${statusHtml}</div>${dutyHtml}</div>${dutyRow}`;
    }).join("");

    const legend = ops.length && mode === "lanes"
      ? `<div class="op-legend" aria-label="GPU ops">${ops.map((op) => `<span data-op="${escape(op)}"><span class="op-swatch"></span>${escape(opLabel(op))}</span>`).join("")}<span><span class="op-swatch turn"></span>held, unlabelled</span>${shown.some((p) => p.waits?.length) ? '<span><span class="op-swatch wait"></span>waiting for the GPU</span>' : ""}</div>`
      : "";
    body = `<div class="activity-timeline" aria-label="Recorded run activity"><div class="activity-header"><h3>${acrossNodes ? "Process" : "Run"}</h3>${acrossNodes ? "<h3>GPU</h3>" : ""}<div class="activity-axis">${axis}</div>${acrossNodes ? '<span class="node-duty" title="Duty cycle: mean GPU utilization (share of time a kernel was running) over this window, from DCGM">Duty</span>' : ""}</div>${rowsHtml}${legend}</div>`;
  }

  const errHtml = error && shown.length ? `<p class="source-error" role="status">Activity unavailable: ${escape(error)}</p>` : "";
  const warnHtml = warnings.length ? `<p class="source-error" role="status">${escape(warnings.join(" · "))}</p>` : "";
  const srcHtml = showSource && activities.length ? `<p class="activity-source">${escape(activitySource(activities))}</p>` : "";
  return `<section class="${acrossNodes ? "cross-node-activity run-activity" : ""}" aria-label="${acrossNodes ? "Run process activity" : "Node process activity"}" data-crosshair-host>${head}${body}${errHtml}${warnHtml}${srcHtml}<div class="activity-tooltip" role="tooltip" hidden><strong class="activity-tooltip-label"></strong><div class="activity-tooltip-time"></div><div class="activity-tooltip-detail"></div></div></section>`;
}

export function hoverActivity(section, event, range) {
  const tip = section.querySelector(".activity-tooltip");
  if (!tip) return;
  if (ui.navigating || event.buttons) {
    tip.hidden = true;
    setCrosshair(null, range);
    return;
  }
  const track = event.target.closest(".activity-track");
  if (track) {
    const box = track.getBoundingClientRect();
    setCrosshair(range.start + Math.max(0, Math.min(1, (event.clientX - box.left) / (box.width || 1))) * (range.now - range.start), range);
  }
  const block = event.target.closest(".activity-block[data-track-id]");
  const meta = block && trackRegistry.get(block.dataset.trackId);
  if (!meta) { tip.hidden = true; return; }
  const box = block.ownerSVGElement.getBoundingClientRect();
  const at = meta.range.start + (event.clientX - box.left) / (box.width || 1) * (meta.range.now - meta.range.start);
  const interval = meta.spans.find(([a, b]) => at >= a && at <= b);
  if (!interval) { tip.hidden = true; return; }
  const [a, b] = interval, date = (v) => new Date(v * 1000).toISOString().slice(0, 10);
  tip.querySelector(".activity-tooltip-label").textContent = meta.label;
  tip.querySelector(".activity-tooltip-time").textContent = `${date(a)} · ${chartTime(a, true)} – ${date(a) === date(b) ? "" : `${date(b)} · `}${chartTime(b, true)}`;
  tip.querySelector(".activity-tooltip-detail").textContent = `Visible interval · ${valueText(b - a, "s")}${meta.source ? ` · ${meta.source}` : ""}${meta.pickLogs ? " · Click for logs" : ""}`;
  tip.hidden = false;
  tip.style.left = `${Math.max(8, Math.min(event.clientX + 12, innerWidth - (tip.offsetWidth || 300) - 8))}px`;
  tip.style.top = `${Math.max(8, Math.min(event.clientY + 12, innerHeight - (tip.offsetHeight || 85) - 8))}px`;
}
