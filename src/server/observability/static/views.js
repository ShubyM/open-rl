// Overview, Health, Nodes, and Run (Activity, Metrics, Logs) views.
import { renderActivity } from "./activity.js";
import { renderChart, timeControlHtml } from "./charts.js";
import { KINDS, elapsedTime, escape, escapeRegExp, finishedAt, isFinished, runDescription, runLabel, shortNodeName, statusBadge } from "./format.js";
import { MAX_WINDOW, get, inspectUrl, now, rangeOf, saved, ui, use, view, viewLink } from "./state.js";
import { DCGM_READING_SECONDS, acceleratorLabel, activitySource, adaptLanes, axisTime, capacityBars, claimLabel, clipPlacements, deviceLabel, deviceSeries, dutyBands, gapFor, lastValue, opLabel, opRank, placementColor } from "./visualization.js";

const OVERVIEW_FILTERS = [
  ["all", "All", "Every recorded run"],
  ["active", "Active", "Runs whose workers are placed, or whose placement cannot be read"],
  ["queued", "Queued", "Waiting for a worker to be placed, or for the server to create one"],
  ["finished", "Finished", "Completed, or nothing serves the run and its session is gone"],
];
const runGroup = (run) => ({ ended: "finished", waiting: "queued" })[run.lifecycle] || "active";
const runLink = (run, tab = "activity") => viewLink(`run/${encodeURIComponent(run.run_id)}/${tab}`);
const healthStatus = (label, tone) => `<span class="health-status health-${tone}"><span class="health-dot" aria-hidden="true"></span>${escape(label)}</span>`;

export function renderOverview(inventory) {
  const summaries = use(inspectUrl("runs", { ended: "true" }), { scope: "overview:runs", poll: 15000 });
  const stepsByRun = new Map((summaries.data?.runs || []).map((r) => [r.run_id, r.steps]));
  const terms = view.q.toLowerCase().split(/\s+/).filter(Boolean);
  const shown = inventory.runs.filter((run) => {
    const text = [run.run_id, run.name, run.model, run.recipe_name, runLabel(run), KINDS[run.fine_tuning_type], ...run.nodes, ...run.pods.map((p) => p.name)].filter(Boolean).join(" ").toLowerCase();
    return (view.status === "all" || runGroup(run) === view.status) && terms.every((term) => text.includes(term));
  }).sort((a, b) => Number(runGroup(b) === "active") - Number(runGroup(a) === "active"));

  const filters = OVERVIEW_FILTERS.map(([key, label, title]) => {
    const count = key === "all" ? inventory.runs.length : inventory.runs.filter((r) => runGroup(r) === key).length;
    return `<button type="button" class="chip" data-run-filter="${key}" title="${escape(title)}" aria-pressed="${view.status === key}">${escape(label)} <span class="run-filter-count">${count}</span></button>`;
  }).join("");

  const rows = shown.map((run) => {
    const label = runDescription(run, null);
    const model = (run.model || "Run").split("/").at(-1);
    const kind = KINDS[run.fine_tuning_type] || run.fine_tuning_type;
    const session = run.session_ids?.at(-1) ?? null;
    const steps = run.steps ?? stepsByRun.get(run.run_id) ?? "—";
    return `<div class="job-list-row run-row" data-key="${escape(run.run_id)}">
      <span class="job-identity"><a href="${escape(runLink(run))}" title="${escape(run.run_id)}">${escape(model)} · <span class="mono">${escape(run.run_id.slice(0, 8))}</span>${kind ? escape(` · ${kind}`) : ""}</a>${label ? `<span class="muted micro">${escape(label)}</span>` : ""}</span>
      <span>${statusBadge(runLabel(run))}</span><span>${escape(steps)}</span><span>${escape(elapsedTime(run, inventory.observed_at))}</span>
      <span class="mono micro run-session"${run.session_ids?.length ? ` title="${escape(run.session_ids.join(", "))}"` : ""}>${escape(session ?? "—")}</span>
    </div>`;
  }).join("");

  return `<h1 class="heading">Overview</h1>
    <div class="overview-toolbar">
      <input id="run-search" type="search" value="${escape(view.q)}" placeholder="Search by run ID, model, name, recipe, node or pod" aria-label="Search runs" autocomplete="off">
      <div class="run-filters" role="group" aria-label="Filter by status">${filters}</div>
    </div>
    <div class="job-list run-list">${shown.length ? `<div class="job-list-head run-row"><span>Job</span><span>Status</span><span>Steps</span><span>Elapsed</span><span>Session</span></div>${rows}` : ""}</div>
    ${!shown.length ? `<div class="empty-state">${inventory.runs.length ? "No runs match the search and status filter" : "No runs recorded"}</div>` : ""}`;
}

export function renderHealth(inventory) {
  const errors = [...new Set(inventory.errors)];
  const pods = inventory.cluster.pods.map((pod) => {
    const run = inventory.runs.find((r) => r.pods.some((p) => p.name === pod.name));
    const [reason, detail = ""] = pod.problem.split(/:\s*(.*)/s);
    const evidence = [detail, pod.restarts ? `${pod.restarts} restarts` : ""].filter(Boolean).join(" · ") || "—";
    const tone = ["Pending", "Unschedulable"].includes(reason) ? "warning" : "error";
    return { issue: reason, resource: pod.name, evidence, tone, link: run && runLink(run, "logs"), label: "Logs" };
  });
  const nodes = (inventory.cluster.nodes || []).filter((n) => n.ready !== true).map((n) => ({
    issue: "Node not ready", resource: n.name, evidence: "Ready condition is false or unknown", tone: "error", link: viewLink("nodes", { node: n.name }), label: "Nodes",
  }));
  const issues = [...pods, ...nodes];
  const errRows = errors.map((e) => `<p class="health-message" role="status">${healthStatus("Source unavailable", "warning")}<span>${escape(e)}</span></p>`).join("");
  const table = issues.length
    ? `<div class="scheduler-table-wrap"><table class="scheduler-table"><thead><tr><th>Issue</th><th>Resource</th><th>Evidence</th><th></th></tr></thead><tbody>${issues.map((i) => `<tr><td>${healthStatus(i.issue, i.tone)}</td><td>${escape(i.resource)}</td><td>${escape(i.evidence)}</td><td>${i.link ? `<a href="${escape(i.link)}">${escape(i.label)} ↗</a>` : ""}</td></tr>`).join("")}</tbody></table></div>`
    : inventory.cluster.available === true && !errors.length
      ? `<p class="health-message">${healthStatus("Healthy", "success")}<span>No pod problems, node problems or source errors.</span></p>`
      : "";
  return `<h1 class="heading">Health</h1>${errRows}${table}<p class="run-json-link"><a href="${escape(inspectUrl("inventory"))}">Diagnostic JSON ↗</a> · <a href="/docs">API reference ↗</a></p>`;
}

export function renderNodes(inventory) {
  const observed = now(inventory);
  const range = rangeOf(inventory, view);
  const duration = range.now - range.start;
  if (view.duration !== duration) view.duration = duration;
  if (view.end !== null && view.end !== range.now) view.end = range.now;

  const bounds = { since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString() };
  const live = view.end === null;
  const queryOpts = { poll: 15000, pollWhen: live, remember: !live };
  const history = use(inspectUrl("activity", bounds), { ...queryOpts, scope: "activity:nodes" });
  const fleet = use(inspectUrl("query", { name: "gpu_busy", points: 1, ...bounds }), { ...queryOpts, scope: "query:fleet-busy" });

  const busyByGpu = Object.fromEntries((fleet.data?.series || []).filter((s) => s.points.length).map((s) => [s.labels.UUID, s.points.at(-1)[1]]));
  const dutyCycle = (n) => {
    const values = (n.devices || []).map((d) => busyByGpu[d.id]).filter(Number.isFinite);
    return values.length ? `${Math.round(values.reduce((a, b) => a + b, 0) / values.length)}%` : "—";
  };

  const loading = !history.data && !history.error;
  const all = clipPlacements(adaptLanes(history.data?.lanes || [], inventory.runs), range, history.error, history.status);
  const selectedPlacement = all.find((p) => p.id === view.placement);
  if (!view.node && selectedPlacement) view.node = selectedPlacement.node;
  const node = inventory.cluster.nodes.find((n) => n.name === view.node);
  if (node && selectedPlacement && selectedPlacement.node !== node.name) view.placement = null;
  if (node && (node.devices.length === 1 || (view.gpu !== "all" && !node.devices.some((d) => d.id === view.gpu)))) view.gpu = "all";

  const utilQuery = use(node ? inspectUrl("query", { name: "gpu_util", node_id: node.name, ...bounds }) : null, { ...queryOpts, scope: `node-util:${node?.name || ""}` });
  const memQuery = use(node ? inspectUrl("query", { name: "gpu_memory_mib", node_id: node.name, ...bounds }) : null, { ...queryOpts, scope: `node-mem:${node?.name || ""}` });

  const errors = [...new Set([...inventory.errors, history.error].filter(Boolean))];
  const busy = (n) => Number(inventory.placements.some((p) => p.node === n.name));
  const nodes = inventory.cluster.nodes.filter((n) => n.gpu_capacity > 0 || n.devices?.length).sort((a, b) => busy(b) - busy(a));
  const unavailable = !node && (view.node || (view.placement && history.data));
  const back = viewLink("nodes", { duration, end: range.now, node: view.node, placement: view.placement, gpu: view.node && view.gpu !== "all" ? view.gpu : null });

  const axis = [0, 1, 2, 3].map((step) => `<span>${escape(axisTime(range.start + (range.now - range.start) * step / 3, range))}</span>`).join("");

  const groupsHtml = loading
    ? `<p class="empty-state">Loading activity…</p>`
    : !nodes.length
      ? (!errors.length && inventory.cluster.available ? `<p class="empty-state">No accelerator nodes reported by Kubernetes</p>` : "")
      : nodes.map((item) => {
          const devices = item.devices || [];
          const height = devices.length === 1 ? 30 : 22;
          const placements = all.filter((p) => p.node === item.name && p.start <= range.now && p.end >= range.start);
          const current = range.live ? inventory.placements.filter((p) => p.node === item.name) : placements.filter((p) => p.start <= range.now && p.end >= range.now);
          const unmapped = placements.filter((p) => !p.devices.length || p.devices.some((id) => !devices.some((d) => d.id === id)));

          const bars = capacityBars(devices, placements, range, view.placement).map((bar) => {
            const holds = bar.holds.map((h) => `<span class="hold ${placementColor(h.placement)}${h.placement.id === view.placement ? " selected" : ""}" data-placement="${escape(h.placement.id)}" data-placement-node="${escape(item.name)}" style="${h.style}" title="${escape(`${h.placement.label} · ${h.placement.role || ""}`)}"></span>`).join("");
            return `<button type="button" class="capacity-allocation held${bar.selected ? " selected" : ""}" data-key="${escape(bar.key)}" data-placement="${escape(bar.placement.id)}" data-placement-node="${escape(item.name)}" aria-expanded="${bar.selected}" aria-label="${escape(bar.title)}" title="${escape(bar.title)}" style="${bar.style}">${holds}</button>`;
          }).join("");

          const track = devices.length
            ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-auto-rows:${height}px">${devices.map((d) => `<span title="${escape(d.id)}">${escape(deviceLabel(d.name))}</span>`).join("")}</div><div class="capacity-track" style="height:${devices.length * height}px;--gpu-lane-height:${height}px">${bars}</div></div>`
            : "";
          const unmappedHtml = unmapped.map((p) => `<button type="button" class="capacity-allocation unknown-mapping ${placementColor(p)}" data-key="unmapped:${escape(p.id)}" data-placement="${escape(p.id)}" data-placement-node="${escape(item.name)}" aria-expanded="${p.id === view.placement}"><span class="allocation-name">${escape(p.label)}</span><span class="allocation-count">${p.device_count} GPU${p.device_count === 1 ? "" : "s"}</span></button>`).join("");
          const noteHtml = unmapped.length ? `<span class="muted micro">Device mapping unavailable</span>` : !devices.length && item.gpu_capacity ? `<p class="empty-state">GPUs without DRA devices</p>` : "";

          let detailHtml = "";
          if (item.name === view.node) {
            const found = new Map();
            for (const [field, q] of [["utilization", utilQuery], ["memory_mib", memQuery]]) {
              for (const s of q.data?.series || []) found.set(s.labels.UUID, { id: s.labels.UUID, ...found.get(s.labels.UUID), [field]: s.points });
            }
            const metricDevices = [...found.values()];
            const loaded = Boolean(utilQuery.data || utilQuery.error || memQuery.data || memQuery.error);
            const selectedDevices = metricDevices.filter((d) => view.gpu === "all" || view.gpu === d.id);
            const gpuPoints = deviceSeries(selectedDevices, "utilization");
            const gpuMemory = lastValue(deviceSeries(selectedDevices, "memory_mib"), range);
            const gpuError = utilQuery.error || utilQuery.data?.error || "";
            const gpuMemoryError = memQuery.error || memQuery.data?.error || "";
            const visible = all.filter((p) => p.node === item.name && (view.gpu === "all" || p.devices.includes(view.gpu)) && p.start <= range.now && p.end >= range.start);
            const label = `${acceleratorLabel(item)} · ${shortNodeName(item.name, inventory.cluster.nodes)}`;

            const picker = devices.length > 1
              ? `<div class="allocation-device-picker" aria-label="GPU selection"><button type="button" class="chip" data-device="all" aria-pressed="${view.gpu === "all"}">All GPUs</button>${devices.map((d) => {
                  const v = lastValue(deviceSeries(metricDevices.filter((m) => m.id === d.id), "utilization"), range);
                  return `<button type="button" class="chip" data-device="${escape(d.id)}" title="GPU ${escape(deviceLabel(d.name))}" aria-pressed="${view.gpu === d.id}">${loaded ? escape(`${deviceLabel(d.name)}${Number.isFinite(v) ? ` · ${Math.round(v)}%` : ""}`) : ""}</button>`;
                }).join("")}</div>`
              : "";

            const gpuDetail = item.gpu_capacity
              ? `<div class="node-gpu-detail" data-key="node-gpu-detail"><div>${renderChart({
                  id: `node-duty:${item.name}:${view.gpu}`,
                  title: `Duty cycle · ${label}`,
                  unit: "%",
                  points: gpuPoints,
                  start: range.start,
                  end: range.now,
                  min: 0,
                  max: 100,
                  readingSeconds: DCGM_READING_SECONDS,
                  bands: dutyBands(visible),
                  gapSeconds: gapFor(gpuPoints, 60),
                  empty: gpuError || (!loaded && "Loading…") || "No samples for this GPU",
                  note: gpuError ? `${gpuError} · Showing previously fetched samples` : "",
                })}${utilQuery.error ? `<button type="button" class="chip" data-retry-query="node-util:${escape(item.name)}">Retry GPU metrics</button>` : ""}</div>
                <div class="allocation-detail-footer"><span>${escape(activitySource(visible))}</span><span class="allocation-memory">GPU memory <strong>${Number.isFinite(gpuMemory) ? `${(gpuMemory / 1024).toFixed(1)} GiB` : "—"}</strong>${gpuMemoryError ? `<span class="source-error" role="status"> · ${Number.isFinite(gpuMemory) ? "Stale" : "Unavailable"}: ${escape(gpuMemoryError)}</span>` : ""}</span></div></div>`
              : "";

            detailHtml = `<section class="allocation-expansion" id="placement-detail">
              <div class="allocation-detail-head inspector-heading"><h2 title="${escape(item.name)}">${escape(label)}</h2><button type="button" class="detail-close" data-close-details aria-label="Close node details" title="Close (Esc)">×</button></div>
              ${picker}
              <div class="node-activity" data-key="node-timeline">${renderActivity({ placements: visible, range, inventory, selected: view.placement, selectable: true, back, activeOnly: true, showSource: false })}</div>
              ${gpuDetail}
            </section>`;
          }

          return `<div class="node-placement-group" data-key="${escape(item.name)}">
            <div class="node-lane">
              <button type="button" class="node-lane-label" data-node-details="${escape(item.name)}" aria-expanded="${view.node === item.name}" title="${escape(item.name)}"><span class="node-accelerator">${escape(acceleratorLabel(item))} <span class="muted">· ${escape(shortNodeName(item.name, inventory.cluster.nodes))}</span></span>${item.gpu_capacity ? `<span class="claim-label">${escape(claimLabel(item, current, range.live))}</span>` : ""}</button>
              <div>${track}${unmappedHtml}${noteHtml}</div>
              <span class="node-duty">${escape(dutyCycle(item))}</span>
            </div>
            ${detailHtml}
          </div>`;
        }).join("");

  return `<div data-crosshair-host>
    <div class="nodes-heading"><h1 class="heading">Accelerator nodes</h1><div class="time-control" aria-label="Node time range">${timeControlHtml(range, observed)}<button type="button" class="chip" data-copy-view>Copy link</button></div></div>
    ${errors.map((e) => `<p class="source-error" role="status">${escape(e)}</p>`).join("")}
    ${history.error ? '<button type="button" class="chip" data-retry-query="activity:nodes">Retry activity</button>' : ""}
    ${!inventory.cluster.available && !errors.length ? '<p class="empty-state">Kubernetes unavailable</p>' : ""}
    ${unavailable ? `<div class="unavailable-selection" role="status"><span>${view.node ? "The selected node is no longer reported." : "The selected placement is not in retained history."}</span><button type="button" class="detail-close" data-close-details aria-label="Clear unavailable selection" title="Clear selection">×</button></div>` : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${axis}</div><span class="node-duty" title="Duty cycle: mean GPU utilization (share of time a kernel was running) over this window, from DCGM">Duty</span></div>
    ${groupsHtml}
  </div>`;
}

// --- Run Logs state machine (tailing, pause, older cursor paging, search debounce) ---
export const logsState = {
  data: null,
  pending: false,
  error: null,
  everything: false,
  filterKey: null,
  queryKey: null,
  lastQ: "",
  failedCursor: null,
  fromBottom: 0,
  needsScroll: false,
  controller: null,
  timer: null,
  params: null,
  rangeNow: 0,
};

export function stopLogs() {
  clearTimeout(logsState.timer);
  logsState.timer = null;
  logsState.controller?.abort();
  logsState.controller = null;
  logsState.pending = false;
}

export async function loadLogs(query, page = null) {
  stopLogs();
  const controller = logsState.controller = new AbortController();
  const windowEnd = logsState.rangeNow;
  logsState.pending = true;
  logsState.failedCursor = page;
  ui.render();
  try {
    const { data: response } = await get(inspectUrl("logs", { ...query, ...(page ? { cursor: page } : {}) }), controller.signal);
    if (controller.signal.aborted) return;
    if (response.error) throw new Error(response.error);
    const lines = document.getElementById("log-lines");
    logsState.fromBottom = page && lines ? lines.scrollHeight - lines.clientHeight - lines.scrollTop : 0;
    const prev = page ? logsState.data?.records || [] : [];
    const merged = [...new Map([...prev, ...(response.records || [])].map((r) => [r.id, r])).values()].slice(0, 2000);
    logsState.data = { ...response, queryKey: logsState.queryKey, windowEnd, records: merged };
    logsState.error = null;
    logsState.needsScroll = true;
  } catch (failure) {
    if (!controller.signal.aborted) {
      logsState.error = failure.message;
      if (page && failure.status === 400) logsState.failedCursor = null;
    }
  } finally {
    if (!controller.signal.aborted) {
      logsState.pending = false;
      logsState.controller = null;
      if (view.page === "run" && view.tab === "logs" && view.end === null) {
        logsState.timer = setTimeout(() => {
          if (!document.hidden && !logsState.pending && view.page === "run" && view.tab === "logs" && view.end === null && logsState.params) {
            loadLogs(logsState.params);
          }
        }, 5000);
      }
      ui.render();
    }
  }
}

export function pauseLogs() {
  stopLogs();
  view.end = logsState.data?.windowEnd ?? logsState.rangeNow;
  if (logsState.data) logsState.data = { ...logsState.data, queryKey: `${logsState.filterKey}:${view.end}` };
  ui.render();
}

export function followLogs() {
  view.event = view.end = null;
  if (logsState.data) logsState.data = { ...logsState.data, next_cursor: null, queryKey: null };
  ui.render();
}

function renderRunLogs(run, range, allLogs) {
  const scope = view.q || (run.shared_runtime && !logsState.everything ? run.run_id : "");
  const logRange = {
    since: new Date(Math.max(range.start, view.event === null ? range.start : view.event - 120) * 1000).toISOString(),
    until: new Date(Math.min(range.now, view.event === null ? range.now : view.event + 120) * 1000).toISOString(),
  };
  const params = { run_id: view.id, q: scope, limit: "200", ...logRange, ...(view.pod ? { pod: view.pod } : {}) };
  const filterKey = JSON.stringify([view.id, scope, view.pod, view.event, view.duration]);
  const queryKey = `${filterKey}:${view.end}`;
  logsState.params = params;
  logsState.rangeNow = range.now;

  if (!ui.navigating && queryKey !== logsState.data?.queryKey && queryKey !== logsState.queryKey) {
    const changed = filterKey !== logsState.filterKey;
    const searched = changed && logsState.filterKey !== null && params.q !== logsState.lastQ;
    if (changed) { logsState.data = logsState.error = null; }
    logsState.filterKey = filterKey;
    logsState.queryKey = queryKey;
    logsState.lastQ = params.q;
    stopLogs();
    logsState.pending = true;
    logsState.timer = setTimeout(() => loadLogs(params), searched ? 250 : 0);
  }

  const records = logsState.data?.records || [];
  const cursor = records.length < 2000 ? logsState.data?.next_cursor : null;
  const pods = new Map((run.pods || []).map((p) => [p.name, p]));
  for (const r of [...(logsState.data?.sources || []), ...records]) if (r.pod && !pods.has(r.pod)) pods.set(r.pod, { ...r, name: r.pod });
  if (view.pod && !pods.has(view.pod)) pods.set(view.pod, { name: view.pod });
  if (!pods.has("client")) pods.set("client", { name: "client" });
  const sources = [...pods.values()].sort((a, b) => (a.name === "client") - (b.name === "client") || a.name.localeCompare(b.name));

  const [first = "", ...rest] = sources.map((p) => p.name).filter((name) => name !== "client");
  let cutLen = 0;
  while (rest.length && cutLen < first.length && rest.every((name) => name[cutLen] === first[cutLen])) cutLen++;
  const cut = rest.length ? first.lastIndexOf("-", cutLen - 1) + 1 : first.length;
  const hues = new Map(sources.map((p, i) => [p.name, `hue-${i % 8}`]));
  const tag = (name) => name === "client" ? name : name?.slice(cut);
  const tagWidth = Math.min(20, Math.max(0, ...sources.map((p) => tag(p.name).length)));
  const match = view.q ? new RegExp(`(${escapeRegExp(view.q)})`, "i") : null;
  const shown = records.toReversed();
  const paging = logsState.pending && view.end !== null;
  const sourceName = logsState.data?.source?.startsWith("cloud-logging") ? "Cloud Logging" : logsState.data?.source || "Logs";
  const status = [
    sourceName,
    records.length || !logsState.error ? `${records.length.toLocaleString()} ${records.length === 1 ? "record" : "records"}` : "",
    logsState.pending && !records.length ? "Loading…" : logsState.error ? "" : view.end === null ? "Live" : "Paused",
  ].filter(Boolean).join(" · ");

  const formatMsg = (msg) => match
    ? msg.split(match).map((part, i) => i % 2 ? `<mark>${escape(part)}</mark>` : escape(part)).join("")
    : escape(msg);
  const sev = (r) => ["ERROR", "CRITICAL", "ALERT", "EMERGENCY"].includes(r.severity) ? "error" : r.severity === "WARNING" ? "warning" : "";

  const rowsHtml = shown.length
    ? shown.map((r) => {
        const at = Number.isFinite(Date.parse(r.timestamp)) ? new Date(r.timestamp).toISOString() : "";
        const s = sev(r);
        return `<div class="workspace-logrow" data-key="${escape(r.id)}"${s ? ` data-severity="${s}"` : ""}><time class="log-time" datetime="${escape(at)}" title="${escape(at || "No timestamp")}">${escape(at ? at.slice(11, 23) : "—")}</time><span class="log-pod ${hues.get(r.pod) || ""}" title="${escape(r.pod)}">${escape(tag(r.pod))}</span><pre class="log-message">${formatMsg(r.message)}${r.message_truncated ? '<span class="muted">\n[Message truncated by source]</span>' : ""}</pre></div>`;
      }).join("")
    : (!logsState.pending && !logsState.error ? `<p class="empty-state">${escape(logsState.data?.note || "No logs match this time range and filter")}</p>` : "");

  return `${view.event !== null ? `<div class="log-time-scope"><span>${escape(`${logRange.since.slice(0, 10)} · ${logRange.since.slice(11, 19)}–${logRange.until.slice(11, 19)}`)}</span><a href="${escape(allLogs)}">All logs</a></div>` : ""}
    <div class="log-toolbar">
      <input id="log-search" type="search" value="${escape(view.q)}" placeholder="Search logs" aria-label="Search logs" />
      <select id="log-source" aria-label="Pod"><option value="">All pods</option>${sources.map((p) => `<option value="${escape(p.name)}"${view.pod === p.name ? " selected" : ""}>${escape(tag(p.name) || p.name)}</option>`).join("")}</select>
      ${run.shared_runtime ? `<label class="log-scope" title="These pods also serve other runs on this base model. Lines that do not name this run are hidden unless this is on."><input type="checkbox" data-log-everything${logsState.everything ? " checked" : ""} />Include other runs' lines</label>` : ""}
      <button type="button" class="chip" data-log-follow>${view.end === null ? "Pause" : "Jump to latest"}</button>
    </div>
    <div id="log-status" class="log-status" role="status"><span>${escape(status)}</span>${logsState.error ? `<span class="log-error">${escape(logsState.error)}${records.length ? " · Showing previously fetched records" : ""}</span><span><button type="button" class="chip" data-log-retry${logsState.pending ? " disabled" : ""}>Retry logs</button></span>` : ""}</div>
    <div id="log-lines" tabindex="0" role="region" aria-label="Run logs" style="--tag-width:${tagWidth}ch">
      ${cursor ? `<button type="button" class="chip log-older" data-log-older="${escape(cursor)}"${paging ? " disabled" : ""}>${paging ? "Loading…" : "Older logs"}</button>` : records.length === 2000 ? '<p class="muted log-older">2,000 records shown. Narrow the time range or search to inspect more.</p>' : ""}
      ${rowsHtml}
    </div>`;
}

function renderRunMetrics(run, finished) {
  const steps = use(inspectUrl(`runs/${encodeURIComponent(view.id)}/steps`, { last: 1000 }), { scope: `steps:${view.id}`, poll: 30000, pollWhen: !finished });
  const rows = steps.data?.steps || [];
  const loading = !steps.data && !steps.error;
  const stepRange = rows.length ? { start: rows[0].step, end: rows.at(-1).step } : null;
  const completedSteps = steps.data ? rows.filter((r) => r.complete).length : run.steps ?? "—";
  const byStep = (fn) => rows.map((row) => [row.step, fn(row)]).filter(([, v]) => Number.isFinite(v));

  const trainerCharts = [...new Map(rows.flatMap((row) => Object.entries(row.ops).flatMap(([op, entry]) => Object.keys(entry.metrics).filter((name) => !name.includes(":") || name.endsWith(":mean")).map((name) => [`${op} ${name}`, [op, name]])))).values()]
    .sort(([a, x], [b, y]) => opRank(a) - opRank(b) || x.localeCompare(y));
  const opNames = [...new Set(rows.flatMap((row) => Object.keys(row.ops)))].sort((a, b) => opRank(a) - opRank(b) || a.localeCompare(b));
  const placeholders = (n, prefix) => `<div class="chart-grid">${Array.from({ length: n }, (_, i) => renderChart({ id: `${prefix}-${i}`, title: "", points: [], empty: "Loading…" })).join("")}</div>`;

  const trainerHtml = trainerCharts.length
    ? `<div class="chart-grid">${trainerCharts.map(([op, name]) => renderChart({ id: `tm:${op}:${name}`, title: `${name} · ${opLabel(op)}`, steps: true, points: byStep((r) => r.ops[op]?.metrics[name]?.mean), start: stepRange.start, end: stepRange.end, tone: "accent", empty: "No values" })).join("")}</div>`
    : loading ? placeholders(3, "tm-load") : steps.data ? '<p class="muted">0 metrics</p>' : "";
  const opsHtml = opNames.map((op) => renderChart({ id: `op:${op}`, title: `${opLabel(op)} duration`, unit: "s", steps: true, points: byStep((r) => r.ops[op] && r.ops[op].seconds / r.ops[op].count), start: stepRange.start, end: stepRange.end, min: 0, tone: "accent", empty: "No values" })).join("");

  return `<div class="run-metrics" data-run-metrics="${escape(view.id)}">
    ${steps.error ? `<p class="source-error" role="status">${escape(steps.error)}${rows.length ? " · Showing previously fetched steps" : ""}</p>` : ""}
    <h2 class="scheduler-title">Trainer metrics</h2>${trainerHtml}
    <h2 class="scheduler-title">Operations</h2>
    <div class="run-metric-summary"><span>Completed steps <strong>${escape(completedSteps)}</strong></span></div>
    ${loading ? placeholders(3, "op-load") : ""}
    <div class="chart-grid">${opsHtml}${steps.data && !rows.length ? '<p class="empty-state">No operations recorded</p>' : ""}</div>
  </div>`;
}

function setupOf(s) {
  const devices = s["openrl.trainer_device_count"] ?? s["openrl.trainer_gpus"], cp = s["openrl.trainer_cp"], samplers = s["openrl.sampler_replicas"];
  return [
    { automodel: "Automodel", pytorch: "PyTorch" }[s["openrl.trainer_backend"]] || s["openrl.trainer_backend"],
    devices != null && `${devices} trainer device${devices === 1 ? "" : "s"}${cp > 1 ? ` (CP ${cp})` : ""}`,
    samplers != null && `${samplers} sampler${samplers === 1 ? "" : "s"}`,
    s["openrl.exclusive"] && "exclusive GPUs",
    s["openrl.fine_tuning_type"] === "full" && s["openrl.weight_sync"] && `${s["openrl.weight_sync"]} weight sync`,
  ];
}

export function renderRun(inventory) {
  const run = inventory.runs.find((r) => r.run_id === view.id);
  const observed = now(inventory);
  const finished = isFinished(run);
  if (run && !ui.runOpened.has(view.id)) {
    ui.runOpened.add(view.id);
    const stop = finishedAt(run);
    if (finished && !view.explicitWindow && view.end === null && view.duration === 1800 && Number.isFinite(stop)) {
      const span = Math.max(0, stop - (run.created_at || stop));
      view.duration = Math.max(1800, Math.min(MAX_WINDOW, Math.round(span * 1.05)));
      view.end = Math.min(observed, Math.round(stop + Math.max(60, span * 0.025)));
    }
  }

  const range = rangeOf(inventory, view);
  if (view.event !== null && (view.event < range.start || view.event > range.now)) view.event = null;
  const tab = ["metrics", "logs"].includes(view.tab) ? view.tab : "activity";
  if (tab !== "logs") stopLogs();

  const bounds = { since: new Date(range.start * 1000).toISOString(), until: new Date(range.now * 1000).toISOString(), run_id: view.id };
  const activityUrl = inspectUrl("activity", bounds);
  const live = view.end === null;
  const detail = use(tab === "activity" ? activityUrl : null, { scope: `run-activity:${view.id}`, poll: 15000, pollWhen: live && tab === "activity", remember: !live });

  const backHtml = `<p class="overview-back"><a href="${escape(view.back || saved.overview)}">← ${view.back ? "Nodes" : "Overview"}</a></p>`;
  if (!run) {
    return `${backHtml}<h1 class="heading">Run</h1>${detail.error ? `<p class="source-error" role="status">${escape(detail.error)}</p><button type="button" class="chip" data-retry-query="run-activity:${escape(view.id)}">Retry</button>` : `<p class="empty-state">${view.id ? "Loading run…" : "Choose a run from Overview"}</p>`}`;
  }

  const title = [(run.model || "Run").split("/").at(-1), run.run_id.slice(0, 8), KINDS[run.fine_tuning_type]].filter(Boolean).join(" · ");
  const description = [runDescription(run, title), ...setupOf(run.settings || {})].filter(Boolean).join(" · ");
  const link = (t, changes = {}) => viewLink(`run/${encodeURIComponent(view.id)}/${t}`, { duration: view.duration, end: view.end, back: view.back, event: view.event, q: view.q, pod: view.pod, ...changes });

  let panel;
  if (tab === "logs") {
    panel = renderRunLogs(run, range, link("logs", { event: null, end: null }));
  } else if (tab === "metrics") {
    panel = renderRunMetrics(run, finished);
  } else {
    const busy = use(inspectUrl("query", { name: "gpu_busy", ...bounds, points: 1 }), { scope: `run-busy:${view.id}`, poll: 15000, pollWhen: live, remember: !live });
    const duty = Object.fromEntries((busy.data?.series || []).filter((s) => s.points.length).map((s) => [s.labels.UUID, Math.round(s.points.at(-1)[1])]));
    const slots = Object.fromEntries((busy.data?.series || []).filter((s) => s.labels.gpu != null).map((s) => [s.labels.UUID, s.labels.gpu]));
    const clipped = clipPlacements(adaptLanes(detail.data?.lanes || [], inventory.runs, view.id), range, detail.error, detail.status);
    panel = `<div class="run-activity">${renderActivity({ placements: clipped, range, inventory, acrossNodes: true, duty, slots, error: detail.error, pending: detail.pending, pickLogs: true })}</div>`;
  }

  const problems = (run.pods || []).filter((p) => p.problem).map((p) => `<p class="source-error" role="status">${escape(`${p.name}: ${p.problem}`)}</p>`).join("");
  return `${backHtml}
    <div class="run-heading"><h1 class="heading" title="${escape(run.run_id)}">${escape(title)}</h1>${statusBadge(runLabel(run))}</div>
    ${detail.error ? `<p class="source-error" role="status">${escape(detail.error)}</p>` : ""}
    ${description ? `<p class="run-description">${escape(description)}</p>` : ""}
    <section class="run-state" aria-label="Run state">${problems}${!finished && run.shared_runtime ? '<p class="muted">Shared runtime: its workers also serve other runs</p>' : ""}</section>
    <div class="run-toolbar">
      <nav class="workspace-tabs" aria-label="Run views">
        <a href="${escape(link("activity"))}"${tab === "activity" ? ' aria-current="page"' : ""}>Activity</a>
        <a href="${escape(link("metrics"))}"${tab === "metrics" ? ' aria-current="page"' : ""}>Metrics</a>
        <a href="${escape(link("logs"))}"${tab === "logs" ? ' aria-current="page"' : ""}>Logs</a>
      </nav>
      <div class="time-control run-time-controls" aria-label="Run time range">${timeControlHtml(range, observed)}<button type="button" class="chip" data-copy-view>Copy link</button><a href="${escape(activityUrl)}">Agent JSON ↗</a></div>
    </div>
    <div id="run-panel">${panel}</div>`;
}
