import {
  resolveTimeRange,
  timeRangeControl,
  bindTimeRange,
} from "./time-range.js";
import { runIncidents } from "./timeline.js";
import { escape, encode, empty, button, runStatus } from "./ui.js";
import { runs, scheduler, health } from "./views.js";
import { renderMetricChart, disposeMetricCharts } from "./charts.js";

const root = document.getElementById("orl-design-lab");
const content = document.getElementById("content");
let nodeSelection = { duration: 1800, end: null };
let gpuView = null;
let runMetricView = null;
const metricCache = new Map();
const metricFreshFor = 15000;
const nodeNow = () =>
  Date.parse(state?.observed_at) / 1000 || Date.now() / 1000;
const nodeTime = (at) => new Date(at * 1000).toISOString().slice(11, 16);
bindTimeRange(
  root,
  () => nodeSelection,
  (selection) => {
    nodeSelection = selection;
    render();
  },
  nodeNow,
);
let state,
  expanded = null,
  device = "all",
  logRequest = 0,
  logTimer,
  logCursor = null;
const family = (id) =>
  ["math", "sql", "eval"][
    Array.from(id || "").reduce((s, c) => s + c.charCodeAt(0), 0) % 3
  ];
const route = () => location.hash.slice(1).split("/").map(decodeURIComponent);
async function get(url, signal) {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}
function metricEntry(url) {
  const entry = metricCache.get(url) || {
    data: null,
    fetchedAt: 0,
    pending: null,
    controller: null,
  };
  metricCache.delete(url);
  metricCache.set(url, entry);
  while (metricCache.size > 24) {
    const oldest = metricCache.keys().next().value;
    metricCache.get(oldest).controller?.abort();
    metricCache.delete(oldest);
  }
  return entry;
}
function metricData(url) {
  const entry = metricEntry(url);
  if (entry.pending) return entry.pending;
  if (entry.data && Date.now() - entry.fetchedAt < metricFreshFor)
    return Promise.resolve(entry.data);
  entry.controller = new AbortController();
  entry.pending = get(url, entry.controller.signal)
    .then((data) => {
      entry.data = data;
      entry.fetchedAt = Date.now();
      return data;
    })
    .finally(() => {
      entry.pending = null;
      entry.controller = null;
    });
  return entry.pending;
}
async function loadMetricView(view, paint, current) {
  const entry = metricEntry(view.url);
  const fresh = entry.data && Date.now() - entry.fetchedAt < metricFreshFor;
  if (fresh && view.data === entry.data && view.painted) return;
  view.data = entry.data || view.data;
  view.status.textContent = "";
  paint(view);
  view.painted = Boolean(view.data);
  if (fresh) return;
  view.status.textContent = view.data
    ? "Updating metrics…"
    : "Loading metrics…";
  const active = () => current() === view && view.element.isConnected;
  try {
    const data = await metricData(view.url);
    if (!active()) return;
    view.data = data;
    view.status.textContent = "";
    paint(view);
    view.painted = true;
  } catch (error) {
    if (active())
      view.status.textContent = view.data
        ? `Metrics refresh failed: ${error.message}. Showing previously received samples.`
        : error.message;
  }
}
function detail(placement) {
  if (device !== "all" && !placement.devices.includes(device)) device = "all";
  const run = state.runs.find((r) => placement.run_ids.includes(r.run_id));
  const devices =
    state.cluster.nodes.find((n) => n.name === placement.node)?.devices || [];
  const workload = run?.workloads?.find((w) => w.uid === placement.id);
  const identity = placement.owner_id || run?.run_id;
  const kind =
    placement.role === "trainer"
      ? { lora: "LoRA", fft: "FFT" }[workload?.training_kind]
      : null;
  const title = [
    (run?.model || placement.runtime_id || placement.label).split("/").at(-1),
    identity?.slice(0, 8),
    { trainer: "Trainer", sampler: "Sampler" }[placement.role] ||
      "Unknown process",
    kind,
  ]
    .filter(Boolean)
    .join(" · ");
  return `<section class="allocation-expansion" id="placement-detail"><div class="allocation-detail-head"><h2>${run ? `<a href="#run/${encode(run.run_id)}/overview">${escape(title)} ↗</a>` : escape(title)}</h2></div>
    <div class="allocation-device-picker" aria-label="GPU selection">${button("All GPUs", `data-device="all" aria-pressed="${device === "all"}"`)}${placement.devices.map((id) => button(devices.find((d) => d.id === id)?.name || id.split("/").at(-1), `data-device="${escape(id)}" aria-pressed="${device === id}"`)).join("")}</div>
    <div><div id="gpu-chart"></div><p id="gpu-status" class="muted" role="status"></p>${run?.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}</div>
    <div><p class="muted">GPU memory</p><p id="gpu-memory">—</p><p class="muted">Run MFU</p><p id="gpu-mfu">—</p></div></section>`;
}
function nodes() {
  const observed = nodeNow();
  const { start, end: now } = resolveTimeRange(nodeSelection, observed);
  const duration = now - start;
  const endpoint = state.history.findLast(
    (sample) => sample.available && sample.at <= now && now - sample.at <= 30,
  );
  const history = state.history;
  const observations = history.flatMap((sample, index) => {
    const from = Math.max(start, sample.at);
    const to = Math.min(
      now,
      history[index + 1]?.at ?? sample.at + 15,
      sample.at + 30,
    );
    return sample.available && to > from
      ? [
          {
            start: from,
            end: to,
            placements: sample.placements.filter((placement) =>
              ["Running", "Bound", "Placed"].includes(placement.phase),
            ),
          },
        ]
      : [];
  });
  const missing = [];
  let observedThrough = start;
  for (const sample of observations) {
    if (sample.start > observedThrough)
      missing.push([observedThrough, sample.start]);
    observedThrough = Math.max(observedThrough, sample.end);
  }
  if (observedThrough < now) missing.push([observedThrough, now]);
  const gaps = missing
    .map(
      ([from, to]) =>
        `<span class="allocation-unobserved" style="left:${((from - start) / duration) * 100}%;width:${((to - from) / duration) * 100}%" aria-label="No allocation observations"></span>`,
    )
    .join("");
  const duty = (node) => {
    if (missing.length || !node.gpu_capacity) return "—";
    const devices = new Set((node.devices || []).map((item) => item.id));
    if (devices.size !== node.gpu_capacity) return "—";
    let allocated = 0;
    for (const sample of observations) {
      const placements = sample.placements.filter(
        (placement) => placement.node === node.name,
      );
      if (
        placements.some(
          (placement) =>
            placement.devices.length !== placement.device_count ||
            placement.devices.some((id) => !devices.has(id)),
        )
      )
        return "—";
      allocated +=
        new Set(placements.flatMap((placement) => placement.devices)).size *
        (sample.end - sample.start);
    }
    return `${Math.round((100 * allocated) / (duration * node.gpu_capacity))}%`;
  };
  const segments = [];
  // Successful observations define both allocation blocks and duty; gaps stay unknown.
  observations.forEach((sample) =>
    sample.placements.forEach((placement) => {
      const key = `${placement.id}:${placement.node}:${placement.devices.join(",")}`;
      const previous = segments.findLast((segment) => segment.key === key);
      if (previous && previous.end === sample.start) previous.end = sample.end;
      else
        segments.push({
          ...placement,
          key,
          start: sample.start,
          end: sample.end,
        });
    }),
  );
  content.innerHTML = `<div class="nodes-heading"><h1 class="heading">Kubernetes nodes</h1>${timeRangeControl(nodeSelection, observed)}</div>${missing.length ? `<p class="history-coverage muted">Hatched areas have no allocation observations.</p>` : ""}${!state.cluster.available ? empty(state.cluster.error || "Kubernetes unavailable") : ""}
    <div class="node-time-header"><span>Node</span><div class="node-axis">${[0, 1, 2, 3].map((tick) => `<span>${nodeTime(start + (duration * tick) / 3)}</span>`).join("")}</div><span class="node-duty" title="GPU allocation time divided by capacity over the selected range">Duty</span></div>
    ${
      state.cluster.nodes
        .map((node) => {
          const height = node.gpu_capacity === 1 ? 30 : 22;
          const devices = node.devices || [];
          const placements = (
            nodeSelection.end === null
              ? state.placements
              : endpoint?.placements || []
          ).filter((p) => p.node === node.name);
          const current =
            placements.find((p) => p.id === expanded) ||
            segments.find((p) => p.id === expanded && p.node === node.name);
          const claimed = new Set(placements.flatMap((p) => p.devices)).size;
          const knownClaims = placements.every(
            (p) => p.devices.length === p.device_count,
          );
          const claimLabel =
            nodeSelection.end !== null && !endpoint
              ? "— claimed"
              : nodeSelection.end === null && !node.ready
                ? "Offline"
                : !node.gpu_capacity
                  ? "CPU"
                  : !knownClaims
                    ? "— claimed"
                    : `${claimed}/${node.gpu_capacity} claimed`;
          const accelerator = node.accelerator
            ? node.accelerator
                .replace(/^nvidia[- ]/i, "")
                .replace(/-/g, " ")
                .toUpperCase()
            : node.gpu_capacity
              ? "GPU"
              : "CPU";
          const unmapped = (
            nodeSelection.end === null ? placements : []
          ).filter(
            (p) =>
              !p.devices.length ||
              p.devices.some((d) => !devices.find((n) => n.id === d)),
          );
          const nodeSegments = segments.filter((p) => p.node === node.name);
          // Shared seats put several allocations on one GPU at once. Each gets
          // its own sub-lane so bars never paint over each other's text.
          const slotOf = new Map();
          const slotCounts = devices.map((device) => {
            const ends = [];
            nodeSegments
              .filter((p) => p.devices.includes(device.id))
              .sort((a, b) => a.start - b.start)
              .forEach((p) => {
                let slot = ends.findIndex((end) => end <= p.start);
                if (slot < 0) slot = ends.push(0) - 1;
                ends[slot] = p.end;
                slotOf.set(`${p.key}:${device.id}`, slot);
              });
            return Math.max(1, ends.length);
          });
          const laneTops = slotCounts.reduce((tops, count, i) => [...tops, tops[i] + count * height], [0]);
          const bars = nodeSegments
            .flatMap((p) => {
              const indexes = p.devices
                .map((id) => devices.findIndex((d) => d.id === id))
                .filter((i) => i >= 0)
                .sort((a, b) => a - b);
              const groups = [];
              indexes.forEach((i) => {
                const last = groups.at(-1);
                if (last && last.at(-1) === i - 1) last.push(i);
                else groups.push([i]);
              });
              return groups.map(
                (group) =>
                  `<button type="button" class="capacity-allocation ${family(p.runtime_id)} ${p.id === expanded ? "selected" : ""}" data-placement="${escape(p.id)}" aria-expanded="${p.id === expanded}" aria-label="${escape(p.label)}, ${group.length} GPUs on ${escape(node.name)}" style="left:${Math.max(0, ((p.start - start) / duration) * 100)}%;width:${Math.max(0, ((Math.min(now, p.end) - Math.max(start, p.start)) / duration) * 100)}%;top:${laneTops[group[0]] + (slotOf.get(`${p.key}:${devices[group[0]].id}`) || 0) * height + 2}px;height:${group.length === 1 ? height - 4 : laneTops[group.at(-1) + 1] - laneTops[group[0]] - 4}px"><span class="allocation-name">${escape(p.label)}</span><span class="allocation-count">${group.length} GPU${group.length === 1 ? "" : "s"}</span></button>`,
              );
            })
            .join("");
          return `<div class="node-placement-group"><div class="node-lane"><div class="node-lane-label" title="${escape(node.name)} · ${nodeSelection.end === null ? "Current node status" : "At selected range end"}" aria-label="${escape(accelerator)}, ${escape(node.name)}"><span class="node-accelerator">${escape(accelerator)}</span><span class="claim-label">${escape(claimLabel)}</span></div><div>
        ${devices.length ? `<div class="gpu-capacity"><div class="gpu-lane-ids" style="grid-template-rows:${slotCounts.map((count) => `${count * height}px`).join(" ")}">${devices.map((d) => `<span title="${escape(d.id)}">${escape(d.name)}</span>`).join("")}</div><div class="capacity-track" style="height:${laneTops.at(-1)}px;--gpu-lane-height:${height}px">${gaps}${bars}</div></div>` : ""}
        ${unmapped.map((p) => `<button type="button" class="capacity-allocation unknown-mapping ${family(p.runtime_id)}" data-placement="${escape(p.id)}" aria-expanded="${p.id === expanded}"><span class="allocation-name">${escape(p.label)}</span><span class="allocation-count">${p.device_count} GPUs</span></button>`).join("")}
        ${unmapped.length ? '<span class="muted micro">Device mapping unavailable</span>' : ""}
        ${!bars && !placements.length && !node.gpu_capacity ? empty("No GPUs") : ""}</div><span class="node-duty" title="GPU allocation time over the selected range; unavailable when observations or device mappings are incomplete">${duty(node)}</span></div>${current ? detail(current) : ""}</div>`;
        })
        .join("") || empty("No nodes available")
    }`;
}
function paintGpu(view) {
  if (gpuView !== view || !view.element.isConnected || !view.data) return;
  const { start, end, data, element: chart } = view;
  const selected = (data.devices || []).filter(
    (item) => device === "all" || device === item.id,
  );
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
    return [
      at,
      reported.length
        ? reported.reduce((sum, value) => sum + value, 0) / reported.length
        : null,
    ];
  }).sort((a, b) => a[0] - b[0]);
  const memory = selected.map(
    (item) =>
      item.memory_mib?.filter(([at]) => at >= start && at <= end).at(-1)?.[1],
  );
  view.memory.textContent =
    memory.length && memory.every(Number.isFinite)
      ? `${(memory.reduce((sum, value) => sum + value, 0) / 1024).toFixed(1)} GiB`
      : "—";
  const mfu = data.mfu;
  view.mfu.textContent =
    Number.isFinite(mfu?.value) && mfu.value >= 0 && mfu.value <= 1
      ? `${(mfu.value * 100).toFixed(1)}%`
      : "—";
  view.mfu.title =
    mfu && view.mfu.textContent !== "—"
      ? `${mfu.estimated ? "Estimated · " : ""}${mfu.scope === "run" ? "Entire run" : "Reported MFU"}${mfu.device_count ? ` · ${mfu.device_count} GPUs` : ""}`
      : "MFU not reported";
  if (!samples.some(([, value]) => Number.isFinite(value))) {
    disposeMetricCharts(chart);
    chart.innerHTML = empty(
      data.reason || selected[0]?.reason || "No samples for this GPU",
    );
    return;
  }
  renderMetricChart(chart, {
    samples,
    start,
    end,
    title: "GPU utilization",
    unit: "%",
    min: 0,
    max: 100,
    tone: "neutral",
    gapSeconds: 30,
  });
}
function loadGpu(id, retainSamples = false) {
  const element = document.getElementById("gpu-chart");
  if (!element) return;
  const { start, end } = resolveTimeRange(nodeSelection, nodeNow());
  const url = `/api/v1/dashboard/allocations/${encode(id)}/metrics?${new URLSearchParams(
    {
      since: new Date(start * 1000).toISOString(),
      until: new Date(end * 1000).toISOString(),
    },
  )}`;
  if (gpuView?.element !== element || gpuView.url !== url)
    gpuView = {
      id,
      url,
      start,
      end,
      element,
      memory: document.getElementById("gpu-memory"),
      mfu: document.getElementById("gpu-mfu"),
      status: document.getElementById("gpu-status"),
      data: retainSamples && gpuView?.id === id ? gpuView.data : null,
    };
  return loadMetricView(gpuView, paintGpu, () => gpuView);
}
let inspectedRun = null;
let runWindowMinutes = 30;
let runWindowEnd = 0;
let selectedEventAt = null;
function runRange(nearEvent = false) {
  const start = runWindowEnd - runWindowMinutes * 60;
  const eventAt = nearEvent ? selectedEventAt : null;
  return {
    since: new Date(
      Math.max(start, eventAt === null ? start : eventAt - 120) * 1000,
    ).toISOString(),
    until: new Date(
      Math.min(runWindowEnd, eventAt === null ? runWindowEnd : eventAt + 120) *
        1000,
    ).toISOString(),
  };
}
function runPage(id, tab = "metrics") {
  if (inspectedRun !== id) {
    inspectedRun = id;
    selectedEventAt = null;
    runWindowEnd = Date.parse(state.observed_at) / 1000;
  }
  const run = state.runs.find((r) => r.run_id === id);
  if (!run) {
    content.innerHTML = empty("Run not found");
    return;
  }
  if (tab === "overview") tab = "metrics";
  if (tab === "events") {
    location.replace(`#run/${encode(id)}/logs`);
    return;
  }
  const tabs = ["metrics", "logs"];
  const title = [
    (run.model || "Run").split("/").at(-1),
    run.run_id.slice(0, 8),
    { lora: "LoRA", full: "FFT", fft: "FFT" }[run.fine_tuning_type],
  ]
    .filter(Boolean)
    .join(" · ");
  const description = [run.display_name, run.recipe_name]
    .filter(
      (value, index, values) =>
        value && value !== title && values.indexOf(value) === index,
    )
    .join(" · ");
  content.innerHTML = `<p class="overview-back"><a href="#overview">← Overview</a></p><div class="run-heading"><h1 class="heading" title="${escape(run.run_id)}">${escape(title)}</h1>${runStatus(run.display_status || run.status)}</div>${description ? `<p class="run-description">${escape(description)}</p>` : ""}<div class="run-toolbar"><nav class="workspace-tabs" aria-label="Run views">${tabs.map((t) => `<a href="#run/${encode(id)}/${t}" ${tab === t ? 'aria-current="page"' : ""}>${t[0].toUpperCase() + t.slice(1)}</a>`).join("")}</nav><div class="run-time-controls"><label>Time range <select id="event-range" aria-label="Time range">${[10, 30, 60].map((value) => `<option value="${value}" ${runWindowMinutes === value ? "selected" : ""}>Last ${value} minutes</option>`).join("")}</select></label>${button("Latest", 'data-latest="true"')}</div></div>${runIncidents(run, runWindowMinutes, runWindowEnd)}<div id="run-panel"></div><p class="run-json-link"><a href="/api/v1/dashboard/runs/${encode(id)}">Agent JSON ↗</a></p>`;
  const panel = document.getElementById("run-panel");
  if (tab === "logs") {
    const range = runRange(true);
    const scope =
      selectedEventAt === null
        ? ""
        : `<div class="log-time-scope"><span>${escape(range.since.slice(0, 10))} · ${escape(range.since.slice(11, 19))}–${escape(range.until.slice(11, 19))} UTC</span><a href="#run/${encode(id)}/logs" data-all-logs="true">All logs</a></div>`;
    panel.innerHTML = `${run.shared_runtime ? '<p class="muted">These pods serve a shared LoRA runtime. Their logs can include other runs.</p>' : ""}${scope}<div class="log-toolbar"><input id="log-search" type="search" placeholder="Search logs" aria-label="Search logs"><select id="log-source" aria-label="Log source"><option value="">All pods</option>${run.pods.map((p) => `<option value="${escape(p.name)}">${escape(p.role || "Unknown")} · ${escape(p.node)} / ${escape(p.name)}</option>`).join("")}</select></div><div id="log-status" role="status"></div><div id="log-lines"></div><div id="log-more"></div>`;
    loadLogs(id);
    return;
  }
  if (tab === "agent") {
    panel.innerHTML = `<p><a href="/api/v1/dashboard/runs/${encode(id)}">Open JSON endpoint ↗</a></p><pre>${escape(JSON.stringify(run, null, 2))}</pre>`;
    return;
  }
  const processes = `${run.shared_runtime ? '<p class="muted">Shared LoRA runtime</p>' : ""}<table class="run-table"><thead><tr><th>Process</th><th>Kind</th><th>Node</th><th>State</th><th>Restarts</th></tr></thead><tbody>${run.pods.map((p) => `<tr><td>${escape(p.name)}</td><td>${escape(p.role ? p.role[0].toUpperCase() + p.role.slice(1) : "Unknown")}</td><td>${escape(p.node)}</td><td>${escape(p.problem || p.phase)}</td><td>${p.restarts}</td></tr>`).join("")}</tbody></table>${!run.pods.length ? empty("No current pods") : ""}`;
  panel.innerHTML = `<p>Completed steps: ${escape(run.steps)}</p><p id="operation-metric-status" class="muted" role="status"></p><div id="operation-metrics"></div><h2 class="scheduler-title">Processes</h2>${processes}`;
  loadRunMetrics(id);
}
function paintRunMetrics(view) {
  if (runMetricView !== view || !view.element.isConnected || !view.data) return;
  const { data, element: panel, start, end } = view;
  const samples = (data.samples || []).filter(
    (sample) => sample.at >= start && sample.at <= end,
  );
  const names = Array.from(
    new Set(samples.flatMap((sample) => Object.keys(sample.metrics || {}))),
  ).slice(0, 12);
  const charts = [
    [
      "Operation duration (s)",
      samples.map((sample) => [sample.at, sample.elapsed_seconds]),
    ],
    ...names.map((name) => [
      name,
      samples
        .filter((sample) => sample.metrics[name] !== undefined)
        .map((sample) => [sample.at, sample.metrics[name]]),
    ]),
    ...(data.gke?.series || []).map((series) => [
      `${series.name} (${series.unit}) · ${series.role}${series.device ? " · " + series.device : ""}`,
      series.points,
    ]),
  ]
    .map(([name, points]) => [
      name,
      (points || []).filter(
        ([at, value]) => at >= start && at <= end && Number.isFinite(value),
      ),
    ])
    .filter(([, points]) => points.length);
  view.status.textContent = data.gke?.error || data.error || "";
  if (!charts.length) {
    disposeMetricCharts(panel);
    panel.innerHTML = empty("No metrics reported in this time range");
    view.keys = null;
    return;
  }
  const keys = JSON.stringify(charts.map(([name]) => name));
  if (keys !== view.keys) {
    disposeMetricCharts(panel);
    panel.innerHTML = charts
      .map(() => '<section class="operation-chart"></section>')
      .join("");
    view.keys = keys;
  }
  charts.forEach(([title, points], index) =>
    renderMetricChart(panel.children[index], {
      samples: points,
      start,
      end,
      title,
      tone: "accent",
    }),
  );
}
function loadRunMetrics(id) {
  const element = document.getElementById("operation-metrics");
  if (!element) return;
  const range = runRange();
  const url = `/api/v1/dashboard/runs/${encode(id)}/metrics?${new URLSearchParams(range)}`;
  if (runMetricView?.element !== element || runMetricView.url !== url)
    runMetricView = {
      element,
      url,
      status: document.getElementById("operation-metric-status"),
      start: Date.parse(range.since) / 1000,
      end: Date.parse(range.until) / 1000,
      keys: runMetricView?.element === element ? runMetricView.keys : null,
    };
  return loadMetricView(runMetricView, paintRunMetrics, () => runMetricView);
}
async function loadLogs(id, more = false) {
  const request = ++logRequest;
  const params = new URLSearchParams({
    q: document.getElementById("log-search").value,
    limit: "200",
    ...runRange(true),
  });
  const pod = document.getElementById("log-source").value;
  if (pod) params.set("pod", pod);
  if (more && logCursor) params.set("cursor", logCursor);
  document.getElementById("log-status").textContent = "Loading…";
  try {
    const data = await get(
      `/api/v1/dashboard/runs/${encode(id)}/logs?${params}`,
    );
    if (request !== logRequest || route()[2] !== "logs") return;
    const records = data.records || [];
    const html = records
      .map(
        (r) =>
          `<div class="workspace-logrow"><span class="log-origin">${escape(r.timestamp || "No timestamp")} ${escape(r.role || "Unknown")} · ${escape(r.pod)}/${escape(r.container)}${r.attempt === null || r.attempt === undefined ? "" : ` #${r.attempt}`}</span>${escape(r.message)}</div>`,
      )
      .join("");
    if (more)
      document
        .getElementById("log-lines")
        .insertAdjacentHTML("beforeend", html);
    else
      document.getElementById("log-lines").innerHTML =
        html || empty("No collected logs match");
    logCursor = data.next_cursor;
    document.getElementById("log-status").textContent =
      data.error ||
      `${data.source === "gke" ? "GKE logs" : "Collected logs"} · Newest first · History may be incomplete`;
    document.getElementById("log-more").innerHTML = logCursor
      ? button("Older logs", 'data-older="true"')
      : "";
  } catch (error) {
    if (request === logRequest && document.getElementById("log-status"))
      document.getElementById("log-status").textContent = error.message;
  }
}
function render({ preserveAllocation = false } = {}) {
  if (!state) return;
  ++logRequest;
  const [page, id, tab] = route();
  const retained =
    preserveAllocation && page === "nodes" && gpuView?.id === expanded
      ? content.querySelector(".allocation-expansion")
      : null;
  retained?.remove();
  disposeMetricCharts(content);
  if (!retained) gpuView = null;
  runMetricView = null;
  const currentPage =
    !page || ["run", "runs"].includes(page)
      ? "overview"
      : page === "diagnostics"
        ? "health"
        : page;
  root.querySelectorAll(".appbar nav a").forEach((link) => {
    if (link.hash === `#${currentPage}`)
      link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (!page || page === "overview" || page === "runs")
    content.innerHTML = runs(state);
  else if (page === "scheduler") content.innerHTML = scheduler(state);
  else if (page === "run") runPage(id, tab);
  else if (page === "diagnostics") {
    location.replace("#health");
    return;
  } else if (page === "health") content.innerHTML = health(state);
  else {
    nodes();
    const replacement = document.getElementById("placement-detail");
    if (retained && replacement) {
      retained.querySelector(".allocation-detail-head").innerHTML =
        replacement.querySelector(".allocation-detail-head").innerHTML;
      const nextDevices = replacement.querySelector(
        ".allocation-device-picker",
      ).innerHTML;
      const picker = retained.querySelector(".allocation-device-picker");
      if (picker.innerHTML !== nextDevices) picker.innerHTML = nextDevices;
      replacement.replaceWith(retained);
    } else if (retained) {
      disposeMetricCharts(retained);
      gpuView = null;
    }
    if (expanded) loadGpu(expanded, Boolean(retained && replacement));
  }
}
root.addEventListener("click", (event) => {
  const allocation = event.target.closest("a[data-scheduler-placement]");
  if (allocation) {
    nodeSelection = { duration: nodeSelection.duration, end: null };
    expanded = allocation.dataset.schedulerPlacement;
    device = "all";
  }

  const incident = event.target.closest("[data-event-at], [data-all-logs]");
  if (incident) {
    event.preventDefault();
    selectedEventAt = incident.dataset.eventAt
      ? Number(incident.dataset.eventAt)
      : null;
    logCursor = null;
    if (route()[2] === "logs") render();
    else location.hash = `run/${encode(route()[1])}/logs`;
    return;
  }

  const target = event.target.closest("button");
  if (!target) return;
  if (target.dataset.placement) {
    expanded =
      expanded === target.dataset.placement ? null : target.dataset.placement;
    device = "all";
    render();
  }
  if (target.dataset.device) {
    device = target.dataset.device;
    root
      .querySelectorAll("[data-device]")
      .forEach((item) =>
        item.setAttribute(
          "aria-pressed",
          String(item.dataset.device === device),
        ),
      );
    if (gpuView) paintGpu(gpuView);
  }
  if (target.dataset.latest) {
    selectedEventAt = null;
    runWindowEnd = Date.parse(state.observed_at) / 1000;
    logCursor = null;
    render();
  }
  if (target.dataset.older) loadLogs(route()[1], true);
});
root.addEventListener("keydown", (event) => {
  if (!event.defaultPrevented && event.key === "Escape" && expanded) {
    const previous = expanded;
    expanded = null;
    render();
    root.querySelector(`[data-placement="${CSS.escape(previous)}"]`)?.focus();
  }
});
root.addEventListener("input", (event) => {
  if (event.target.id === "log-search") {
    clearTimeout(logTimer);
    logTimer = setTimeout(() => loadLogs(route()[1]), 250);
  }
});
root.addEventListener("change", (event) => {
  if (event.target.id === "log-source") loadLogs(route()[1]);
  if (event.target.id === "event-range") {
    runWindowMinutes = Number(event.target.value);
    selectedEventAt = null;
    runWindowEnd = Date.parse(state.observed_at) / 1000;
    logCursor = null;
    render();
  }
});
window.addEventListener("hashchange", render);
let refreshing = false;
async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    state = await get("/api/v1/dashboard/snapshot");
    document.getElementById("connection").textContent = state.demo
      ? "Demo"
      : state.cluster.available
        ? "Connected"
        : "Cluster unavailable";
    // Keep an open time editor stable; a closed rolling picker can refresh.
    const focusedTimeAction =
      document.activeElement?.closest("[data-node-time]")?.dataset.nodeTime;
    const closedTimePicker =
      focusedTimeAction && root.querySelector(".node-time-popover")?.hidden;
    const focusedDevice = document.activeElement?.dataset.device;
    // Preserve text selection, logs and focused controls while polling.
    if (
      !["run", "diagnostics"].includes(route()[0]) &&
      (!content.contains(document.activeElement) ||
        closedTimePicker ||
        focusedDevice)
    ) {
      render({ preserveAllocation: true });
      if (closedTimePicker)
        root.querySelector(`[data-node-time="${focusedTimeAction}"]`)?.focus();
      else if (focusedDevice)
        root
          .querySelector(`[data-device="${CSS.escape(focusedDevice)}"]`)
          ?.focus();
    } else if (content.textContent === "Loading cluster…") render();
    else if (
      route()[0] === "run" &&
      document.getElementById("operation-metrics")
    )
      loadRunMetrics(route()[1]);
  } catch (error) {
    document.getElementById("connection").textContent = error.message;
  } finally {
    refreshing = false;
  }
}
refresh();
setInterval(refresh, 10000);
