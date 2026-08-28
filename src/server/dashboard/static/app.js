// open-rl operations dashboard. Vanilla ES module, no build step. Views poll the JSON API
// and reconcile the DOM in place so refreshes never rebuild the page or lose canvas
// position, selection, or the open log panel.

const POLL_MS = 8000;
const LOG_POLL_MS = 3000;

const $ = (id) => document.getElementById(id);

const S = {
  tab: sessionStorage.getItem("tab") || "cluster",
  pool: sessionStorage.getItem("pool") || null,
  cluster: null,
  runs: null,
  health: null,
  problems: null,
  canvas: JSON.parse(sessionStorage.getItem("canvas") || '{"x":0,"y":0,"k":1}'),
  panel: { pod: null, container: null },
  openRun: sessionStorage.getItem("open-run") || null,
  runDetails: new Map(),
  stopConfirm: null,
  stopNotes: new Map(),
};

// *** Helpers ***

async function fetchJSON(url, opts) {
  const resp = await fetch(url, opts);
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.error || `${resp.status} ${resp.statusText}`);
  return body;
}

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function setClass(el, cls) {
  if (el.className !== cls) el.className = cls;
}

// Keyed child reconciliation: create/update/reorder/remove without rebuilding untouched nodes.
// Duplicate keys get a deterministic suffix so repeated items reconcile instead of leaking.
function sync(parent, items, keyOf, create, update) {
  const existing = new Map();
  for (const child of [...parent.children]) {
    if (!child.dataset.key) continue;
    if (existing.has(child.dataset.key)) child.remove();
    else existing.set(child.dataset.key, child);
  }
  const claimed = new Set();
  let prev = null;
  for (const item of items) {
    let key = String(keyOf(item));
    while (claimed.has(key)) key += "#";
    claimed.add(key);
    let el = existing.get(key);
    if (el) {
      existing.delete(key);
    } else {
      el = create(item);
      el.dataset.key = key;
    }
    const want = prev ? prev.nextElementSibling : parent.firstElementChild;
    if (want !== el) parent.insertBefore(el, prev ? prev.nextElementSibling : parent.firstElementChild);
    update(el, item);
    prev = el;
  }
  for (const el of existing.values()) el.remove();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function copyText(value) {
  if (navigator.clipboard?.writeText) return navigator.clipboard.writeText(value);
  const input = el("textarea");
  input.value = value;
  input.style.position = "fixed";
  input.style.opacity = "0";
  document.body.append(input);
  input.select();
  document.execCommand("copy");
  input.remove();
}

function relTime(iso) {
  if (!iso) return "";
  const secs = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (secs < 90) return `${Math.round(secs)}s ago`;
  if (secs < 5400) return `${Math.round(secs / 60)}m ago`;
  if (secs < 129600) return `${Math.round(secs / 3600)}h ago`;
  return `${Math.round(secs / 86400)}d ago`;
}

function duration(seconds) {
  seconds = Math.max(0, Math.round(seconds || 0));
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function latency(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 10) return `${seconds.toFixed(2)}s`;
  return duration(seconds);
}

function compactNumber(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const number = Number(value);
  if (number === 0) return "0";
  if (Math.abs(number) >= 1000 || Math.abs(number) < 0.001) return number.toExponential(2);
  return Number(number.toPrecision(4)).toString();
}

function bytes(value) {
  if (!Number.isFinite(Number(value))) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let amount = Number(value);
  let unit = 0;
  while (Math.abs(amount) >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
}

function cores(value) {
  return Number.isFinite(Number(value)) ? `${compactNumber(value)} cores` : "—";
}

function revisionLabel(value) {
  if (!value || value === "unknown") return "unknown";
  const dirty = value.endsWith("-dirty");
  const revision = dirty ? value.slice(0, -6) : value;
  return `${revision.slice(0, 12)}${dirty ? "-dirty" : ""}`;
}

// *** Theme ***

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("theme", theme);
}
applyTheme(localStorage.getItem("theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
$("theme-toggle").addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

// *** Navigation: three tabs, plus a per-pool drill-in screen under Cluster ***

function activeViewId() {
  return S.pool && S.tab === "cluster" ? "pool" : S.tab;
}

function updateViews() {
  for (const btn of document.querySelectorAll(".tab")) btn.classList.toggle("active", btn.dataset.tab === S.tab);
  for (const view of document.querySelectorAll(".view")) view.hidden = view.id !== `view-${activeViewId()}`;
}

function showTab(tab) {
  S.tab = tab;
  S.pool = null;
  sessionStorage.setItem("tab", tab);
  sessionStorage.removeItem("pool");
  updateViews();
}

function openPool(poolId) {
  S.tab = "cluster";
  S.pool = poolId;
  sessionStorage.setItem("tab", "cluster");
  sessionStorage.setItem("pool", poolId);
  updateViews();
  renderPool();
}

for (const btn of document.querySelectorAll(".tab")) btn.addEventListener("click", () => showTab(btn.dataset.tab));
$("pool-back").addEventListener("click", () => showTab("cluster"));
updateViews();

// *** Canvas pan & zoom ***

const canvas = $("canvas");
const world = $("world");

function applyTransform() {
  // Edges live inside the transformed world, so panning/zooming never needs a redraw.
  world.style.transform = `translate(${S.canvas.x}px, ${S.canvas.y}px) scale(${S.canvas.k})`;
  sessionStorage.setItem("canvas", JSON.stringify(S.canvas));
}
applyTransform();

let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  if (e.target.closest("button, a, input, .pod")) return;
  drag = { x: e.clientX, y: e.clientY, ox: S.canvas.x, oy: S.canvas.y };
  canvas.setPointerCapture(e.pointerId);
  canvas.classList.add("panning");
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  S.canvas.x = drag.ox + (e.clientX - drag.x);
  S.canvas.y = drag.oy + (e.clientY - drag.y);
  applyTransform();
});
canvas.addEventListener("pointerup", () => {
  drag = null;
  canvas.classList.remove("panning");
});

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      const k = Math.min(1.6, Math.max(0.4, S.canvas.k * (e.deltaY < 0 ? 1.1 : 0.9)));
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      S.canvas.x = cx - ((cx - S.canvas.x) / S.canvas.k) * k;
      S.canvas.y = cy - ((cy - S.canvas.y) / S.canvas.k) * k;
      S.canvas.k = k;
    } else {
      S.canvas.x -= e.deltaX;
      S.canvas.y -= e.deltaY;
    }
    applyTransform();
  },
  { passive: false }
);

$("zoom-in").addEventListener("click", () => { S.canvas.k = Math.min(1.6, S.canvas.k * 1.15); applyTransform(); });
$("zoom-out").addEventListener("click", () => { S.canvas.k = Math.max(0.4, S.canvas.k / 1.15); applyTransform(); });
$("zoom-reset").addEventListener("click", () => { S.canvas = { x: 0, y: 0, k: 1 }; applyTransform(); });

// *** Cluster view ***

function kvRow(key) {
  const row = el("div", "kv");
  row.append(el("span", "k", key), el("span", "v"));
  return row;
}

function makeCard(id) {
  const card = el("div", "card");
  card.id = id;
  const head = el("div", "card-head");
  head.append(el("span", "card-title"), el("span", "card-tag"));
  card.append(head, el("div", "card-rows"));
  return card;
}

function updateCard(card, title, tag, rows) {
  setText(card.querySelector(".card-title"), title);
  setText(card.querySelector(".card-tag"), tag);
  sync(
    card.querySelector(".card-rows"),
    rows,
    (r) => r.k,
    (r) => kvRow(r.k),
    (rowEl, r) => {
      const v = rowEl.querySelector(".v");
      setText(v, r.v);
      setClass(v, r.bad ? "v bad" : "v");
    }
  );
}

// *** Pool screen: GPU duty-cycle chart ***
// Allocation duty (claimed GPUs / pool capacity) stacked by the job holding the GPUs, as
// step areas on a fixed 0–100% scale with a hover crosshair and per-job tooltip. Redrawn in
// place on every poll and window resize. Palette is CVD-validated on both themes; "other"
// and jobs beyond the palette fall back to grey, and the legend names every band so color
// is never the only identity cue.

const CHART_H = 240;
const CHART_MARGIN = { top: 10, right: 12, bottom: 22, left: 44 };
const JOB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"];
const OTHER_JOB_COLOR = "#8C8C8C";
const jobColorIndex = new Map();

function jobColor(job) {
  if (job === "other") return OTHER_JOB_COLOR;
  if (!jobColorIndex.has(job)) jobColorIndex.set(job, jobColorIndex.size);
  const index = jobColorIndex.get(job);
  return index < JOB_PALETTE.length ? JOB_PALETTE[index] : OTHER_JOB_COLOR;
}

function jobLabel(job) {
  const run = S.runs?.runs.find((r) => r.run_id === job);
  if (run) return run.name;
  return job.length > 20 ? `${job.slice(0, 8)}…` : job;
}

function timeLabel(ts) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function sampleTotal(claims) {
  let total = 0;
  for (const gpus of Object.values(claims)) total += gpus;
  return total;
}

function dutyReadout(duty, sample) {
  const [at, claims] = sample || duty.series[duty.series.length - 1];
  const total = sampleTotal(claims);
  return `${Math.round((total / duty.capacity) * 100)}% · ${total}/${duty.capacity} GPU · ${sample ? timeLabel(at) : "now"}`;
}

let chartGeometry = null;

function renderDutyChart(duty) {
  const svg = $("duty-chart");
  const empty = $("chart-empty");
  chartGeometry = null;
  if (!duty || !duty.series.length) {
    svg.style.display = "none";
    empty.hidden = false;
    setText(empty, duty ? "No duty samples yet — they accumulate while the gateway is polled." : "No GPUs in this pool, so it has no duty cycle.");
    setText($("chart-readout"), "");
    renderChartLegend(null);
    $("chart-tip").hidden = true;
    return;
  }
  svg.style.display = "block";
  empty.hidden = true;

  const width = Math.max(320, Math.floor(svg.parentElement.clientWidth) - 32);
  svg.setAttribute("viewBox", `0 0 ${width} ${CHART_H}`);
  svg.setAttribute("width", width);
  svg.setAttribute("height", CHART_H);
  const series = duty.series;
  const start = series[0][0];
  const span = Math.max(1, series[series.length - 1][0] - start);

  // Stack the jobs bottom-up in first-appearance order; each band is a step area whose
  // bottom edge is the top of the band below it. Time-sliced pools can be overcommitted
  // (claims beyond capacity), so the y-domain grows past 100% instead of clipping.
  const times = series.map((sample) => sample[0]);
  const stacks = duty.jobs.map(() => ({ lower: [], upper: [] }));
  let yMax = 1;
  for (const [, claims] of series) {
    let cumulative = 0;
    duty.jobs.forEach((job, j) => {
      stacks[j].lower.push(cumulative);
      cumulative += (claims[job] || 0) / duty.capacity;
      stacks[j].upper.push(cumulative);
    });
    yMax = Math.max(yMax, cumulative);
  }

  const x = (t) => CHART_MARGIN.left + ((t - start) / span) * (width - CHART_MARGIN.left - CHART_MARGIN.right);
  const y = (v) => CHART_MARGIN.top + (1 - v / yMax) * (CHART_H - CHART_MARGIN.top - CHART_MARGIN.bottom);
  chartGeometry = { x, y, duty, start, span };

  const parts = [];
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    const capacity = v === 1 && yMax > 1;
    parts.push(`<line class="${capacity ? "capacity-line" : "grid-line"}" x1="${x(start)}" y1="${y(v)}" x2="${x(start + span)}" y2="${y(v)}"></line>`);
  }
  for (const v of [0, 0.5, 1]) {
    parts.push(`<text class="axis-label" x="${CHART_MARGIN.left - 8}" y="${y(v) + 3}" text-anchor="end">${v * 100}%</text>`);
  }
  if (yMax > 1) {
    parts.push(`<text class="axis-label" x="${CHART_MARGIN.left - 8}" y="${y(yMax) + 3}" text-anchor="end">${Math.round(yMax * 100)}%</text>`);
  }
  const ticks = Math.min(4, series.length);
  for (let i = 0; i < ticks; i++) {
    const t = start + (span * i) / Math.max(1, ticks - 1);
    const anchor = i === 0 ? "start" : i === ticks - 1 ? "end" : "middle";
    parts.push(`<text class="axis-label" x="${x(t)}" y="${CHART_H - 6}" text-anchor="${anchor}">${timeLabel(t)}</text>`);
  }
  duty.jobs.forEach((job, j) => {
    const { lower, upper } = stacks[j];
    const n = times.length;
    let d = `M ${x(times[0]).toFixed(1)} ${y(upper[0]).toFixed(1)}`;
    for (let i = 1; i < n; i++) d += ` H ${x(times[i]).toFixed(1)} V ${y(upper[i]).toFixed(1)}`;
    d += ` V ${y(lower[n - 1]).toFixed(1)}`;
    for (let i = n - 2; i >= 0; i--) d += ` V ${y(lower[i]).toFixed(1)} H ${x(times[i]).toFixed(1)}`;
    parts.push(`<path class="chart-band" fill="${jobColor(job)}" d="${d} Z"></path>`);
  });
  const lastTotal = sampleTotal(series[series.length - 1][1]) / duty.capacity;
  parts.push(`<circle class="chart-end" r="2.5" cx="${x(times[times.length - 1]).toFixed(1)}" cy="${y(lastTotal).toFixed(1)}"></circle>`);
  parts.push('<line class="cross-line" visibility="hidden"></line><circle class="cross-dot" r="3" visibility="hidden"></circle>');
  svg.innerHTML = parts.join("");
  setText($("chart-readout"), dutyReadout(duty));
  renderChartLegend(duty);
}

function renderChartLegend(duty) {
  const lastClaims = duty?.series.length ? duty.series[duty.series.length - 1][1] : {};
  sync(
    $("chart-legend"),
    duty?.jobs || [],
    (job) => job,
    () => {
      const chip = el("span", "legend-chip");
      chip.append(el("span", "legend-swatch"), el("span", "legend-name"), el("span", "legend-count"));
      return chip;
    },
    (chip, job) => {
      chip.querySelector(".legend-swatch").style.background = jobColor(job);
      setText(chip.querySelector(".legend-name"), jobLabel(job));
      const gpus = lastClaims[job] || 0;
      setText(chip.querySelector(".legend-count"), `${gpus} GPU`);
    }
  );
}

function scrubChart(e) {
  if (!chartGeometry) return;
  const { x, y, duty, start, span } = chartGeometry;
  const plotWidth = Math.max(1, x(start + span) - CHART_MARGIN.left);
  const frac = Math.min(1, Math.max(0, (e.offsetX - CHART_MARGIN.left) / plotWidth));
  const target = start + frac * span;
  let nearest = duty.series[0];
  for (const sample of duty.series) if (Math.abs(sample[0] - target) < Math.abs(nearest[0] - target)) nearest = sample;
  const [at, claims] = nearest;
  const svg = $("duty-chart");
  const cross = svg.querySelector(".cross-line");
  const dot = svg.querySelector(".cross-dot");
  cross.setAttribute("visibility", "visible");
  cross.setAttribute("x1", x(at));
  cross.setAttribute("x2", x(at));
  cross.setAttribute("y1", CHART_MARGIN.top);
  cross.setAttribute("y2", CHART_H - CHART_MARGIN.bottom);
  dot.setAttribute("visibility", "visible");
  dot.setAttribute("cx", x(at));
  dot.setAttribute("cy", y(sampleTotal(claims) / duty.capacity));
  setText($("chart-readout"), dutyReadout(duty, nearest));

  const tip = $("chart-tip");
  tip.replaceChildren(el("div", "tip-time", timeLabel(at)));
  for (const job of duty.jobs) {
    if (!claims[job]) continue;
    const row = el("div", "tip-row");
    const swatch = el("span", "legend-swatch");
    swatch.style.background = jobColor(job);
    row.append(swatch, el("span", "tip-name", jobLabel(job)), el("span", "tip-count", `${claims[job]} GPU`));
    tip.append(row);
  }
  if (tip.children.length === 1) tip.append(el("div", "tip-row", "nothing scheduled"));
  tip.hidden = false;
  const card = tip.parentElement;
  const cardRect = card.getBoundingClientRect();
  tip.style.left = `${Math.max(8, Math.min(e.clientX - cardRect.left + 16, cardRect.width - tip.offsetWidth - 8))}px`;
  tip.style.top = `${e.clientY - cardRect.top + 14}px`;
}

$("duty-chart").addEventListener("mousemove", scrubChart);
$("duty-chart").addEventListener("mouseleave", () => {
  const svg = $("duty-chart");
  svg.querySelector(".cross-line")?.setAttribute("visibility", "hidden");
  svg.querySelector(".cross-dot")?.setAttribute("visibility", "hidden");
  $("chart-tip").hidden = true;
  if (chartGeometry) setText($("chart-readout"), dutyReadout(chartGeometry.duty));
});
addEventListener("resize", () => {
  if (S.pool) renderPool();
});

function renderPool() {
  if (!S.pool || !S.cluster) return;
  const pool = S.cluster.pools.find((p) => p.id === S.pool);
  if (!pool) {
    setText($("pool-title"), S.pool);
    setText($("pool-meta"), "not visible in the latest snapshot");
    renderDutyChart(null);
    $("pool-node-list").replaceChildren();
    return;
  }
  setText($("pool-title"), pool.label);
  const gpus = pool.nodes.reduce((n, node) => n + node.gpu_capacity, 0);
  setText($("pool-meta"), `${pool.nodes.length} node${pool.nodes.length === 1 ? "" : "s"}${gpus ? ` · ${gpus} GPU` : ""}`);
  renderDutyChart(pool.duty);
  sync(
    $("pool-node-list"),
    pool.nodes,
    (node) => node.name,
    () => {
      const row = el("div", "pool-node");
      row.append(el("span", "pool-node-name"), el("span", "pool-node-gpus"), el("span", "pool-node-meta"));
      return row;
    },
    (row, node) => {
      setText(row.querySelector(".pool-node-name"), node.name);
      setText(row.querySelector(".pool-node-gpus"), node.gpu_capacity ? `${node.gpu_allocatable}/${node.gpu_capacity} GPU` : "");
      const meta = row.querySelector(".pool-node-meta");
      const pods = `${node.pods.length} pod${node.pods.length === 1 ? "" : "s"}`;
      const usage = node.usage ? `${compactNumber(node.usage.cpu_cores)} CPU · ${bytes(node.usage.memory_bytes)}` : null;
      meta.replaceChildren(el("span", "", [node.instance_type, pods, usage].filter(Boolean).join(" · ")));
      if (node.ready === false) meta.append(" ", el("span", "bad", "not ready"));
    }
  );
}

function podStateClass(pod) {
  if (pod.problem || pod.phase === "Failed") return "pod-state bad";
  if (pod.phase === "Pending" || pod.restarts >= 3) return "pod-state warn";
  return "pod-state";
}

function podStateText(pod) {
  const restarts = pod.restarts > 0 ? ` ↺${pod.restarts}` : "";
  return `${pod.phase}${restarts}`;
}

function renderCluster(data) {
  const podsByName = new Map(data.pods.map((p) => [p.name, p]));

  // Control column: gateway + configured services.
  const controlItems = [{ id: "gateway" }, { id: "scheduler" }, ...data.services];
  sync(
    $("control-col"),
    controlItems,
    (item) => item.id,
    (item) => makeCard(`card-${item.id}`),
    (card, item) => {
      if (item.id === "gateway") {
        const g = data.gateway;
        updateCard(card, "gateway", g.mode, [
          { k: "build", v: revisionLabel(g.build?.revision) },
          { k: "uptime", v: duration(g.build?.uptime_seconds) },
          { k: "sampler", v: g.sampler_backend },
          { k: "fft", v: g.fft_enabled ? "enabled" : "off" },
          { k: "namespace", v: data.kubernetes.namespace || "—" },
        ]);
      } else if (item.id === "scheduler") {
        const scheduler = data.scheduler || {};
        const summary = scheduler.summary || {};
        const phases = Object.entries(summary.phase_counts || {}).map(([phase, count]) => `${count} ${phase}`).join(" · ") || "none";
        const state = scheduler.installed === false ? "not installed" : scheduler.available ? "active" : "unavailable";
        updateCard(card, "scheduler", state, [
          { k: "workloads", v: scheduler.available ? String(summary.workloads || 0) : "—" },
          {
            k: "phases",
            v: scheduler.available ? phases : scheduler.error || "optional placement API",
            bad: scheduler.installed !== false && !scheduler.available,
          },
          { k: "ledgers", v: scheduler.available ? `${summary.ledgers || 0} · ${summary.seats || 0} seats` : "—" },
        ]);
      } else {
        const state = item.ok === false ? "unreachable" : item.ok === true ? "connected" : item.configured ? "configured" : "not configured";
        updateCard(card, item.id, state, [{ k: "detail", v: item.detail, bad: item.ok === false }]);
      }
    }
  );

  // Pools column: real nodes grouped by GPU pool, or a note when Kubernetes is absent.
  const pools = data.kubernetes.available ? data.pools : [];
  const poolsCol = $("pools-col");
  let note = poolsCol.querySelector(".note-card");
  if (!data.kubernetes.available || pools.length === 0) {
    if (!note) {
      note = el("div", "card note-card");
      note.append(el("div", "note-title"), el("div", "note-body"));
      poolsCol.append(note);
    }
    if (!data.kubernetes.available) {
      setText(note.querySelector(".note-title"), "Kubernetes not connected");
      setText(note.querySelector(".note-body"), `${data.kubernetes.error || "unavailable"}. Showing gateway-local components only.`);
    } else {
      setText(note.querySelector(".note-title"), "No pods or nodes visible");
      setText(note.querySelector(".note-body"), `Namespace ${data.kubernetes.namespace} is empty.`);
    }
  } else if (note) {
    note.remove();
  }

  sync(
    poolsCol,
    pools,
    (pool) => pool.id,
    () => {
      const card = el("div", "card pool");
      const head = el("button", "card-head pool-head");
      head.title = "Open pool";
      head.append(el("span", "card-title"), el("span", "card-tag"), el("span", "pool-chevron", "›"));
      head.addEventListener("click", () => openPool(card.dataset.key));
      card.append(head, el("div", "pool-nodes"));
      return card;
    },
    (card, pool) => {
      const gpus = pool.nodes.reduce((n, node) => n + node.gpu_capacity, 0);
      setText(card.querySelector(".card-title"), pool.label);
      const duty = pool.duty?.series.length ? ` · ${Math.round(pool.duty.current * 100)}% duty` : "";
      setText(card.querySelector(".card-tag"), `${pool.nodes.length} node${pool.nodes.length === 1 ? "" : "s"}${gpus ? ` · ${gpus} GPU` : ""}${duty}`);
      sync(
        card.querySelector(".pool-nodes"),
        pool.nodes,
        (node) => node.name,
        () => {
          const nodeEl = el("div", "node");
          const head = el("div", "node-head");
          head.append(el("span", "node-name"), el("span", "node-meta"));
          nodeEl.append(head, el("div", "node-pods"));
          return nodeEl;
        },
        (nodeEl, node) => {
          setText(nodeEl.querySelector(".node-name"), node.name);
          const meta = nodeEl.querySelector(".node-meta");
          const bits = [];
          if (node.instance_type) bits.push(node.instance_type);
          if (node.gpu_capacity) bits.push(`${node.gpu_allocatable}/${node.gpu_capacity} GPU`);
          if (node.usage) bits.push(`${compactNumber(node.usage.cpu_cores)} CPU · ${bytes(node.usage.memory_bytes)}`);
          meta.replaceChildren(el("span", "", bits.join(" · ")));
          if (node.ready === false) meta.append(" ", el("span", "bad", "not ready"));
          sync(
            nodeEl.querySelector(".node-pods"),
            node.pods.map((name) => podsByName.get(name)).filter(Boolean),
            (pod) => pod.name,
            () => {
              const btn = el("button", "pod");
              btn.append(el("span", "pod-name"), el("span", "pod-state"));
              btn.addEventListener("click", () => openPanel(btn.dataset.key));
              return btn;
            },
            (btn, pod) => {
              setText(btn.querySelector(".pod-name"), pod.name);
              const state = btn.querySelector(".pod-state");
              setText(state, podStateText(pod));
              setClass(state, podStateClass(pod));
              btn.classList.toggle("selected", S.panel.pod === pod.name);
            }
          );
        }
      );
    }
  );

  requestAnimationFrame(drawEdges);
}

function drawEdges() {
  const svg = $("edges");
  const edges = S.cluster?.edges || [];
  const paths = [];
  for (const edge of edges) {
    const from = $(`card-${edge.from}`);
    const to = $(`card-${edge.to}`);
    if (!from || !to) continue;
    const x1 = from.offsetLeft + from.offsetWidth / 2;
    const y1 = from.offsetTop + from.offsetHeight;
    const x2 = to.offsetLeft + to.offsetWidth / 2;
    const y2 = to.offsetTop;
    const service = S.cluster.services.find((s) => s.id === edge.to);
    const down = service && service.ok === false;
    paths.push(`<path class="${down ? "down" : ""}" d="M ${x1} ${y1} C ${x1} ${y1 + 20}, ${x2} ${y2 - 20}, ${x2} ${y2}"><title>${edge.reason}</title></path>`);
  }
  svg.innerHTML = paths.join("");
}

// *** Runs view ***

function renderRuns(data) {
  const runs = data.runs || [];
  if (S.openRun && !runs.some((run) => run.run_id === S.openRun)) {
    S.openRun = null;
    sessionStorage.removeItem("open-run");
  }
  setText($("runs-count"), String(runs.length || ""));
  $("runs-empty").hidden = runs.length > 0;
  sync(
    $("runs-list"),
    runs,
    (run) => run.run_id,
    (run) => {
      const row = el("div", "run-row");
      const main = el("div", "run");
      const info = el("div", "run-info");
      const meta = el("div", "run-meta");
      meta.append(el("span", "run-state"), el("span", "run-reason"));
      info.append(el("div", "run-name"), el("div", "run-ids"), meta);
      const actions = el("div", "run-actions");
      main.append(info, actions);
      row.append(main, el("div", "run-note"), el("div", "run-detail"));
      return row;
    },
    (row, run) => {
      setText(row.querySelector(".run-name"), run.name);
      setText(row.querySelector(".run-ids"), run.base_model ? `${run.run_id} · ${run.base_model}` : run.run_id);
      const state = run.state || { phase: "unknown", status: "off", reason: "state unavailable" };
      const stateEl = row.querySelector(".run-state");
      setText(stateEl, state.phase);
      setClass(stateEl, `run-state ${state.status}`);
      setText(row.querySelector(".run-reason"), state.reason);
      const actions = row.querySelector(".run-actions");

      let inspect = actions.querySelector(".inspect");
      if (!inspect) {
        inspect = el("button", "inspect btn", "Inspect");
        inspect.addEventListener("click", () => toggleRunDetail(run.run_id));
        actions.prepend(inspect);
      }
      setText(inspect, S.openRun === run.run_id ? "Close" : "Inspect");

      let wandb = actions.querySelector(".wandb");
      if (run.wandb_url) {
        if (!wandb || wandb.tagName !== "A") {
          wandb?.remove();
          wandb = el("a", "wandb link-btn", "W&B");
          wandb.target = "_blank";
          wandb.rel = "noopener";
          actions.prepend(wandb);
        }
        if (wandb.href !== run.wandb_url) wandb.href = run.wandb_url;
      } else {
        if (!wandb || wandb.tagName !== "SPAN") {
          wandb?.remove();
          wandb = el("span", "wandb no-link", "W&B —");
          wandb.title = "No W&B link recorded for this run";
          actions.prepend(wandb);
        }
      }

      let stop = actions.querySelector(".stop");
      if (run.stoppable) {
        if (!stop) {
          stop = el("button", "stop btn", "Stop");
          stop.addEventListener("click", () => onStopClick(run.run_id, stop));
          actions.append(stop);
        }
        const confirming = S.stopConfirm === run.run_id;
        setText(stop, confirming ? "Confirm stop" : "Stop");
        stop.classList.toggle("confirm", confirming);
      } else {
        stop?.remove();
      }

      const note = row.querySelector(".run-note");
      setText(note, S.stopNotes.get(run.run_id) || "");
      note.hidden = !S.stopNotes.get(run.run_id);

      const detail = row.querySelector(".run-detail");
      detail.hidden = S.openRun !== run.run_id;
      if (!detail.hidden) renderRunDetail(detail, run, S.runDetails.get(run.run_id));
    }
  );
}

function metric(label, value, detail = "") {
  const item = el("div", "run-metric");
  item.append(el("div", "run-metric-value", value), el("div", "run-metric-label", label));
  if (detail) item.append(el("div", "run-metric-detail", detail));
  return item;
}

function renderRunDetail(container, run, detail) {
  if (!detail) {
    container.replaceChildren(el("div", "run-detail-loading", "Loading run state, pods, and logs…"));
    return;
  }
  if (detail.error) {
    const retry = el("button", "btn", "Retry");
    retry.addEventListener("click", () => loadRunDetail(run.run_id));
    container.replaceChildren(el("div", "run-detail-error", `Inspection failed: ${detail.error}`), retry);
    return;
  }

  const head = el("div", "run-detail-head");
  head.append(el("div", "eyebrow", "Live inspection"));
  const refreshButton = el("button", "btn", "Refresh inspection");
  refreshButton.addEventListener("click", () => loadRunDetail(run.run_id));
  head.append(refreshButton);

  const claims = Object.entries(detail.gpu_claims || {});
  const claimsText = claims.length ? claims.map(([pool, count]) => `${count} on ${pool}`).join(" · ") : "none visible";
  const metrics = el("div", "run-metrics");
  const resources = detail.resources || {};
  const requests = resources.requests || {};
  const limits = resources.limits || {};
  const usage = resources.usage || {};
  const workloadPhases = Object.entries(detail.state?.workload_phase_counts || {}).map(([phase, count]) => `${count} ${phase}`).join(" · ") || "none";
  metrics.append(
    metric("Queue", String(detail.queue_depth || 0), detail.queue_depth ? `oldest waiting ${duration(detail.queue_oldest_seconds)}` : "requests"),
    metric("GPU claims", String(detail.gpu_devices || 0), claimsText),
    metric("Pods", String(detail.pods?.length || 0), Object.entries(detail.state?.pod_phase_counts || {}).map(([phase, count]) => `${count} ${phase}`).join(" · ") || "none visible"),
    metric("Workloads", String(detail.workloads?.length || 0), workloadPhases),
    metric("Sources", String(detail.sources?.length || 0), (detail.sources || []).join(" · ") || "none")
  );
  if (usage.total_pods) {
    const measured = usage.measured_pods ? `${usage.measured_pods}/${usage.total_pods} pods measured` : "usage unavailable";
    const cpuLimit = limits.cpu_cores == null ? "no limit" : `${cores(limits.cpu_cores)} limit`;
    const memoryLimit = limits.memory_bytes == null ? "no limit" : `${bytes(limits.memory_bytes)} limit`;
    metrics.append(
      metric(
        "CPU",
        usage.measured_pods ? `${cores(usage.cpu_cores)} used` : `${cores(requests.cpu_cores)} requested`,
        `${cores(requests.cpu_cores)} requested · ${cpuLimit} · ${measured}`
      ),
      metric(
        "Memory",
        usage.measured_pods ? `${bytes(usage.memory_bytes)} used` : `${bytes(requests.memory_bytes)} requested`,
        `${bytes(requests.memory_bytes)} requested · ${memoryLimit} · ${measured}`
      )
    );
  }

  const diagnostics = el("div", "run-diagnostics");
  if (detail.diagnostics?.length) {
    diagnostics.append(el("div", "eyebrow", "Diagnostics"));
    for (const diagnostic of detail.diagnostics) {
      const row = el("div", `run-diagnostic ${diagnostic.severity}`);
      row.append(el("span", "run-diagnostic-code", diagnostic.code), el("span", "run-diagnostic-message", diagnostic.message));
      diagnostics.append(row);
    }
  }

  const telemetrySection = el("div", "run-telemetry");
  const telemetry = detail.telemetry || {};
  if (telemetry.requests_completed || telemetry.active_request) {
    telemetrySection.append(el("div", "eyebrow", "Gateway request telemetry"));
    const requestMetrics = el("div", "run-metrics");
    const failures = telemetry.requests_failed || 0;
    if (telemetry.active_request) {
      const active = telemetry.active_request;
      requestMetrics.append(metric("Executing", active.operation || "request", `${duration(active.age_seconds)} active · queued ${latency(active.queue_wait_seconds)}`));
    }
    requestMetrics.append(
      metric("Completed", String(telemetry.requests_completed || 0), Object.entries(telemetry.operation_counts || {}).map(([op, count]) => `${op} ${count}`).join(" · ")),
      metric("Failures", String(failures), `${((telemetry.failure_rate || 0) * 100).toFixed(1)}% of requests`)
    );
    if (telemetry.requests_completed) {
      requestMetrics.append(
        metric("Last latency", latency(telemetry.last_latency_seconds), telemetry.last_operation || "request"),
        metric("Mean latency", latency(telemetry.mean_latency_seconds), `max ${latency(telemetry.max_latency_seconds)}`)
      );
    }
    telemetrySection.append(requestMetrics);

    const latest = Object.entries(telemetry.latest_metrics || {}).sort(([a], [b]) => a.localeCompare(b));
    if (latest.length) {
      telemetrySection.append(el("div", "eyebrow", "Latest worker metrics"));
      const workerMetrics = el("div", "run-metrics worker-metrics");
      for (const [name, value] of latest) {
        const points = telemetry.metric_series?.[name] || [];
        const trend = points.length > 1 ? `${compactNumber(points[0].value)} → ${compactNumber(points.at(-1).value)} · ${points.length} samples` : `${points.length || 1} sample`;
        workerMetrics.append(metric(name, compactNumber(value), trend));
      }
      telemetrySection.append(workerMetrics);
    }
  }

  const workloadSection = el("div", "run-workloads");
  if (detail.workloads?.length) {
    workloadSection.append(el("div", "eyebrow", "Scheduler placement"));
    for (const workload of detail.workloads) {
      const card = el("div", "run-workload-card");
      const head = el("div", "run-workload-head");
      const statusClass = workload.phase === "Failed" ? "bad" : ["Pending", "Placing"].includes(workload.phase) ? "warn" : "";
      head.append(el("span", "run-workload-name", workload.name), el("span", `pod-state ${statusClass}`.trim(), workload.phase));
      const ledger = (detail.claim_ledgers || []).find((item) => item.claim_name === workload.claim_name);
      const placement = [
        workload.role,
        workload.requested_memory ? `${workload.requested_memory} requested` : null,
        workload.device_count ? `${workload.device_count} device${workload.device_count === 1 ? "" : "s"}` : null,
        workload.node_name || "node pending",
        workload.claim_name || "claim pending",
        ledger ? `${ledger.seat_count} seat${ledger.seat_count === 1 ? "" : "s"}` : null,
      ].filter(Boolean).join(" · ");
      card.append(head, el("div", "run-workload-meta", placement));
      if (workload.reason) card.append(el("div", "run-workload-reason", workload.reason));
      workloadSection.append(card);
    }
  }

  const podSection = el("div", "run-pods");
  podSection.append(el("div", "eyebrow", "Pods and recent logs"));
  if (!detail.pods?.length) {
    podSection.append(el("div", "run-detail-empty", "No run pods are visible."));
  }
  for (const pod of detail.pods || []) {
    const card = el("div", "run-pod-card");
    const podButton = el("button", "run-pod");
    podButton.append(el("span", "run-pod-name", pod.name), el("span", podStateClass(pod), podStateText(pod)));
    podButton.addEventListener("click", () => openPanel(pod.name));
    const podMeta = el(
      "div",
      "run-pod-meta",
      [pod.node || "not scheduled", pod.ready ? `${pod.ready} ready` : null, pod.gpus ? `${pod.gpus} GPU` : null, pod.restarts ? `${pod.restarts} restarts` : null]
        .filter(Boolean)
        .join(" · ")
    );
    card.append(podButton, podMeta);
    if (pod.problem) card.append(el("div", "run-pod-problem", pod.problem));
    if (detail.logs && Object.hasOwn(detail.logs, pod.name)) card.append(el("pre", "run-log-preview", detail.logs[pod.name] || "(no log output)"));
    podSection.append(card);
  }
  container.replaceChildren(head, metrics, diagnostics, telemetrySection, workloadSection, podSection);
}

async function loadRunDetail(runId) {
  S.runDetails.delete(runId);
  if (S.runs) renderRuns(S.runs);
  try {
    const detail = await fetchJSON(`/api/v1/dashboard/runs/${encodeURIComponent(runId)}?logs=120`);
    S.runDetails.set(runId, detail);
  } catch (err) {
    S.runDetails.set(runId, { error: err.message });
  }
  if (S.runs && S.openRun === runId) renderRuns(S.runs);
}

function toggleRunDetail(runId) {
  if (S.openRun === runId) {
    S.openRun = null;
    sessionStorage.removeItem("open-run");
  } else {
    S.openRun = runId;
    sessionStorage.setItem("open-run", runId);
    if (!S.runDetails.has(runId)) loadRunDetail(runId);
  }
  if (S.runs) renderRuns(S.runs);
}

$("launch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("launch-model");
  const button = $("launch-button");
  const baseModel = input.value.trim();
  if (!baseModel) return;

  localStorage.setItem("openrl-base-model", baseModel);
  button.disabled = true;
  setText($("launch-note"), "Launching…");
  try {
    const result = await fetchJSON("/api/v1/dashboard/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_model: baseModel }),
    });
    setText($("launch-note"), result.launched === false ? "Demo mode — nothing launched" : `Queued ${result.request_id}`);
    if (result.launched !== false) await refresh();
  } catch (err) {
    setText($("launch-note"), `Launch failed: ${err.message}`);
  } finally {
    button.disabled = false;
  }
});
$("launch-model").value = localStorage.getItem("openrl-base-model") || "";

async function onStopClick(runId, btn) {
  if (S.stopConfirm !== runId) {
    S.stopConfirm = runId;
    renderRuns(S.runs);
    setTimeout(() => {
      if (S.stopConfirm === runId) {
        S.stopConfirm = null;
        if (S.runs) renderRuns(S.runs);
      }
    }, 4000);
    return;
  }
  S.stopConfirm = null;
  btn.disabled = true;
  try {
    const result = await fetchJSON(`/api/v1/dashboard/runs/${encodeURIComponent(runId)}/stop`, { method: "POST" });
    const errors = result.errors?.length ? ` Errors: ${result.errors.join("; ")}` : "";
    S.stopNotes.set(runId, `Stopped: ${result.actions.join("; ")}.${errors}`);
  } catch (err) {
    S.stopNotes.set(runId, `Stop failed: ${err.message}`);
  }
  btn.disabled = false;
  await refresh();
}

// *** Health view ***

function renderHealth(health, problems) {
  const list = problems?.problems || [];
  $("problems-empty").hidden = list.length > 0;
  $("problem-dot").hidden = !list.some((p) => p.severity === "error");
  sync(
    $("problems-list"),
    list,
    (p) => p.id || `${p.source}:${p.message}`,
    () => {
      const row = el("div", "problem");
      const body = el("div", "problem-body");
      const summary = el("div", "problem-summary");
      summary.append(el("span", "problem-src"), el("span", "problem-msg"));
      body.append(summary, el("div", "problem-actions"));
      row.append(el("span", "problem-sev"), body);
      return row;
    },
    (row, p) => {
      const sev = row.querySelector(".problem-sev");
      setText(sev, p.severity);
      setClass(sev, `problem-sev ${p.severity}`);
      setText(row.querySelector(".problem-src"), p.source);
      setText(row.querySelector(".problem-msg"), p.message);
      sync(
        row.querySelector(".problem-actions"),
        p.actions || [],
        (action, i) => `${action.method || "command"}:${action.path || action.command || i}`,
        () => {
          const button = el("button", "problem-action");
          button.addEventListener("click", async () => {
            const action = button._action;
            const problem = button._problem;
            if (problem.resource?.kind === "Pod" && action.path?.includes("/pods/")) {
              openPanel(problem.resource.name);
              return;
            }
            if (action.command) {
              await copyText(action.command);
              const old = button.textContent;
              setText(button, "Copied command");
              setTimeout(() => setText(button, old), 1200);
              return;
            }
            if (action.path) window.open(action.path, "_blank", "noopener");
          });
          return button;
        },
        (button, action) => {
          button._action = action;
          button._problem = p;
          setText(button, p.resource?.kind === "Pod" && action.path?.includes("/pods/") ? "Open logs" : action.command ? `Copy: ${action.label}` : action.label);
          button.title = action.command || action.path || "";
        }
      );
    }
  );

  sync(
    $("stats-grid"),
    health?.stats || [],
    (s) => s.id,
    () => {
      const tile = el("div", "stat");
      tile.append(el("div", "stat-value"), el("div", "stat-label"), el("div", "stat-detail"));
      return tile;
    },
    (tile, s) => {
      setClass(tile, `stat ${s.status || "ok"}`);
      setText(tile.querySelector(".stat-value"), s.value);
      setText(tile.querySelector(".stat-label"), s.label);
      setText(tile.querySelector(".stat-detail"), s.detail || "");
    }
  );
  const queues = health?.queues || [];
  $("queues-block").hidden = queues.length === 0;
  sync(
    $("queue-list"),
    queues,
    (q) => q.model_id,
    () => {
      const row = el("div", "queue-row");
      row.append(el("span", "queue-model"), el("span", "queue-depth"));
      return row;
    },
    (row, q) => {
      setText(row.querySelector(".queue-model"), q.model_id);
      const wait = q.oldest_age_seconds == null ? "" : ` · oldest ${duration(q.oldest_age_seconds)}`;
      setText(row.querySelector(".queue-depth"), `${q.depth} request${q.depth === 1 ? "" : "s"}${wait}`);
    }
  );

  const groups = [];
  for (const check of health?.checks || []) {
    let group = groups.find((g) => g.name === check.group);
    if (!group) groups.push((group = { name: check.group, checks: [] }));
    group.checks.push(check);
  }
  sync(
    $("checks"),
    groups,
    (g) => g.name,
    (g) => {
      const section = el("div", "check-group");
      section.append(el("div", "eyebrow", g.name), el("div", "check-rows"));
      return section;
    },
    (section, g) => {
      sync(
        section.querySelector(".check-rows"),
        g.checks,
        (c) => c.id,
        () => {
          const row = el("div", "check");
          row.append(el("span", "check-label"), el("span", "check-status"), el("span", "check-detail"));
          return row;
        },
        (row, c) => {
          setText(row.querySelector(".check-label"), c.label);
          const status = row.querySelector(".check-status");
          setText(status, c.status);
          setClass(status, `check-status ${c.status}`);
          setText(row.querySelector(".check-detail"), c.detail);
        }
      );
    }
  );
}

// *** Pod side panel ***

const panel = $("panel");
let logTimer = null;

function openPanel(podName) {
  S.panel.pod = podName;
  S.panel.container = null;
  $("log-previous").checked = false;
  panel.hidden = false;
  $("panel-logs").textContent = "loading…";
  updatePanel();
  pollLogs();
  if (S.cluster) renderCluster(S.cluster);
}

function closePanel() {
  S.panel.pod = null;
  panel.hidden = true;
  clearTimeout(logTimer);
  logTimer = null;
  if (S.cluster) renderCluster(S.cluster);
}

$("panel-close").addEventListener("click", closePanel);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!panel.hidden) closePanel();
  else if (S.pool) showTab("cluster");
});
document.addEventListener("pointerdown", (e) => {
  if (!panel.hidden && !e.target.closest("#panel") && !e.target.closest(".pod, .run-pod")) closePanel();
});

function updatePanel() {
  if (!S.panel.pod || !S.cluster) return;
  const pod = S.cluster.pods.find((p) => p.name === S.panel.pod);
  setText($("panel-title"), S.panel.pod);
  const sub = $("panel-sub");
  if (!pod) {
    sub.replaceChildren(el("span", "bad", "pod no longer exists"));
    $("panel-meta").replaceChildren();
    $("panel-containers").replaceChildren();
    $("panel-evidence").replaceChildren();
    return;
  }
  sub.replaceChildren(el("span", pod.problem ? "bad" : "", pod.problem || pod.phase));
  const requests = pod.resources?.requests || {};
  const limits = pod.resources?.limits || {};
  sync(
    $("panel-meta"),
    [
      { k: "node", v: pod.node || "—" },
      { k: "app", v: pod.app || "—" },
      { k: "ready", v: pod.ready },
      { k: "restarts", v: String(pod.restarts) },
      {
        k: "cpu",
        v: [
          pod.usage ? `${cores(pod.usage.cpu_cores)} used` : null,
          `${cores(requests.cpu_cores || 0)} requested`,
          limits.cpu_cores == null ? "no limit" : `${cores(limits.cpu_cores)} limit`,
        ].filter(Boolean).join(" · "),
      },
      {
        k: "memory",
        v: [
          pod.usage ? `${bytes(pod.usage.memory_bytes)} used` : null,
          `${bytes(requests.memory_bytes || 0)} requested`,
          limits.memory_bytes == null ? "no limit" : `${bytes(limits.memory_bytes)} limit`,
        ].filter(Boolean).join(" · "),
      },
      { k: "created", v: pod.created_at ? `${relTime(pod.created_at)}` : "—" },
    ],
    (r) => r.k,
    (r) => kvRow(r.k),
    (rowEl, r) => setText(rowEl.querySelector(".v"), r.v)
  );
  const containers = pod.containers.length > 1 ? pod.containers : [];
  sync(
    $("panel-containers"),
    containers,
    (c) => c.name,
    (c) => {
      const chip = el("button", "container-chip", c.name);
      chip.addEventListener("click", () => {
        S.panel.container = c.name;
        updatePanel();
        pollLogs();
      });
      return chip;
    },
    (chip, c) => {
      setText(chip, c.name);
      chip.classList.toggle("active", (S.panel.container || pod.containers[0].name) === c.name);
      chip.classList.toggle("bad", Boolean(c.reason || c.last_termination?.reason));
      chip.title = [c.kind, c.state, c.reason, c.message].filter(Boolean).join(" · ");
    }
  );
  const selected = pod.containers.find((container) => container.name === S.panel.container) || pod.containers.find((container) => container.kind === "app") || pod.containers[0];
  const evidence = [];
  if (selected) {
    const current = [selected.state, selected.reason, selected.exit_code != null ? `exit ${selected.exit_code}` : null, selected.message].filter(Boolean).join(" · ");
    evidence.push({ key: `container/${selected.name}`, value: current || "state unavailable" });
    if (selected.image) evidence.push({ key: "image", value: selected.image });
    if (selected.image_id) evidence.push({ key: "image ID", value: selected.image_id });
    if (selected.last_termination) {
      const prior = selected.last_termination;
      evidence.push({ key: "previous", value: [prior.reason, prior.exit_code != null ? `exit ${prior.exit_code}` : null, prior.message].filter(Boolean).join(" · ") });
    }
  }
  for (const event of (pod.events || []).slice(-5).reverse()) {
    evidence.push({ key: event.reason || event.type || "event", value: `${event.message || ""}${event.count > 1 ? ` · ×${event.count}` : ""}` });
  }
  sync(
    $("panel-evidence"),
    evidence,
    (item) => `${item.key}:${item.value}`,
    () => {
      const row = el("div", "panel-evidence-row");
      row.append(el("span", "panel-evidence-key"), el("span", "panel-evidence-value"));
      return row;
    },
    (row, item) => {
      setText(row.querySelector(".panel-evidence-key"), item.key);
      setText(row.querySelector(".panel-evidence-value"), item.value);
    }
  );
  const previousWrap = $("log-previous-wrap");
  previousWrap.hidden = !selected?.last_termination;
  if (previousWrap.hidden) $("log-previous").checked = false;
}

$("log-previous").addEventListener("change", pollLogs);

async function pollLogs() {
  clearTimeout(logTimer);
  if (!S.panel.pod) return;
  const pod = S.panel.pod;
  const params = new URLSearchParams({ tail: "500" });
  const podInfo = S.cluster?.pods.find((p) => p.name === pod);
  const container = S.panel.container || (podInfo?.containers.length > 1 ? podInfo.containers[0].name : null);
  if (container) params.set("container", container);
  if ($("log-previous").checked) params.set("previous", "true");
  try {
    const data = await fetchJSON(`/api/v1/dashboard/pods/${encodeURIComponent(pod)}/logs?${params}`);
    if (S.panel.pod !== pod) return;
    const logsEl = $("panel-logs");
    const atBottom = logsEl.scrollHeight - logsEl.scrollTop - logsEl.clientHeight < 40;
    setText(logsEl, data.text || "(no log output)");
    if ($("log-follow").checked && atBottom) logsEl.scrollTop = logsEl.scrollHeight;
    setText($("log-updated"), new Date().toLocaleTimeString());
  } catch (err) {
    if (S.panel.pod === pod) setText($("panel-logs"), `logs unavailable: ${err.message}`);
  }
  logTimer = setTimeout(pollLogs, LOG_POLL_MS);
}

// *** Polling ***

let refreshing = false;

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  try {
    const snapshot = await fetchJSON("/api/v1/dashboard/snapshot");
    S.cluster = snapshot.cluster;
    S.runs = snapshot.runs;
    S.health = snapshot.health;
    S.problems = snapshot.problems;
    renderCluster(S.cluster);
    updatePanel();
    renderPool();
    renderRuns(S.runs);
    renderHealth(S.health, S.problems);
    $("demo-banner").hidden = !snapshot.demo;
    const build = snapshot.cluster?.gateway?.build;
    setText($("build-at"), `build ${revisionLabel(build?.revision)}`);
    $("build-at").title = build ? `${build.hostname} · Python ${build.python_version} · started ${build.started_at}` : "build identity unavailable";
    const updated = snapshot.generated_at ? new Date(snapshot.generated_at) : new Date();
    const observation = snapshot.cluster?.kubernetes?.observation;
    const observed = observation
      ? ` · k8s ${observation.source} · ${compactNumber(observation.age_seconds)}s old · ${compactNumber(observation.collection_ms)} ms collect`
      : "";
    setText($("updated-at"), `updated ${updated.toLocaleTimeString()}${observed}`);
  } catch (err) {
    setText($("updated-at"), `refresh failed: ${err.message}`);
  } finally {
    refreshing = false;
  }
}

$("refresh").addEventListener("click", refresh);
refresh();
setInterval(() => {
  if (!document.hidden) refresh();
}, POLL_MS);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
