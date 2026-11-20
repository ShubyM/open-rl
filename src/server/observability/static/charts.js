// Stretched SVG charts (Running / Readings / Duty 1 min modes, band clipPaths, pointer/keyboard inspection) and time picker.
import { escape } from "./format.js";
import { MAX_WINDOW, WINDOWS, setCrosshair, ui } from "./state.js";
import { axisTime, chartNumber, chartTime, valueText } from "./visualization.js";

const ROLLING_SECONDS = 60;
const chartRegistry = new Map();
let chartSeq = 0;

export function resetCharts() {
  chartRegistry.clear();
  chartSeq = 0;
}

export function renderChart({
  id,
  title,
  unit = "",
  points = [],
  start,
  end,
  min,
  max,
  tone = "neutral",
  gapSeconds = Infinity,
  empty = "No samples in this time range",
  note = "",
  steps = false,
  readingSeconds = 0,
  bands = [],
}) {
  const chartKey = id || title || `chart-${chartSeq}`;
  const uid = `chart-${chartSeq++}`;
  const mode = ui.chartModes.get(chartKey) ?? (bands.length ? "running" : "readings");

  const running = [];
  if (mode === "running") {
    const raw = bands.flatMap((band) => band.spans).map(([a, b]) => [Math.max(start, a), Math.min(end, b)]).filter(([a, b]) => b > a).toSorted((p, q) => p[0] - q[0]);
    for (const [a, b] of raw) {
      if (running.length && a <= running.at(-1)[1]) running.at(-1)[1] = Math.max(running.at(-1)[1], b);
      else running.push([a, b]);
    }
  }

  let shown = points;
  if (mode === "running") {
    shown = [[start, 0], ...running.flatMap(([a, b]) => [[a, 0], [a, 100], [b, 100], [b, 0]]), [end, 0]];
  } else if (mode === "rolling") {
    const sorted = points.filter(([at, value]) => Number.isFinite(at) && Number.isFinite(value)).toSorted((a, b) => a[0] - b[0]);
    let first = 0, sum = 0;
    shown = sorted.map(([at, value], i) => {
      sum += value;
      while (sorted[first][0] <= at - ROLLING_SECONDS) sum -= sorted[first++][1];
      return [at, sum / (i - first + 1)];
    });
  }

  const samples = shown.filter(([at]) => Number.isFinite(at) && at >= start && at <= end).sort((a, b) => a[0] - b[0]);
  const data = samples.filter(([, value]) => Number.isFinite(value));
  const values = data.map(([, value]) => value);

  const modeButtons = readingSeconds
    ? `<span class="chart-mode" role="group" aria-label="What to plot">${bands.length ? `<button type="button" class="chip" data-chart-mode="${escape(chartKey)}" data-mode="running" aria-pressed="${mode === "running"}">Running</button>` : ""}<button type="button" class="chip" data-chart-mode="${escape(chartKey)}" data-mode="readings" aria-pressed="${mode === "readings"}">Readings</button><button type="button" class="chip" data-chart-mode="${escape(chartKey)}" data-mode="rolling" aria-pressed="${mode === "rolling"}">Duty (1 min)</button></span>`
    : "";
  const latest = data.length && mode !== "running"
    ? `<span class="chart-latest" title="${escape(steps ? `Step ${data.at(-1)[0]}` : new Date(data.at(-1)[0] * 1000).toISOString())}"><span>Latest</span> <strong>${escape(valueText(data.at(-1)[1], unit))}</strong></span>`
    : "";
  const head = `<figcaption class="chart-head"><h2>${escape(title)}</h2>${modeButtons}${latest}</figcaption>`;

  if (!data.length) {
    return `<figure class="chart" data-key="${escape(chartKey)}" data-tone="${escape(tone)}">${head}<p class="chart-empty">${escape(empty)}</p></figure>`;
  }

  let low = Number.isFinite(min) ? min : Math.min(...values);
  let high = Number.isFinite(max) ? max : Math.max(...values);
  if (high <= low) {
    const pad = Math.max(Math.abs(low) * 0.1, 0.1);
    if (!Number.isFinite(min)) low -= pad;
    high += pad;
  }
  let ticks = [high, (high + low) / 2, low];
  if (!Number.isFinite(min) && !Number.isFinite(max)) {
    const raw = (high - low) / 3, magnitude = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].find((n) => n >= raw / magnitude) * magnitude;
    low = Math.floor(low / step) * step;
    high = Math.ceil(high / step) * step;
    ticks = Array.from({ length: Math.round((high - low) / step) + 1 }, (_, i) => Number((high - i * step).toPrecision(12)));
  }

  const segments = [];
  let current = null;
  for (const sample of samples) {
    if (!Number.isFinite(sample[1])) { current = null; continue; }
    if (!current || (mode !== "running" && sample[0] - current.at(-1)[0] > gapSeconds)) segments.push((current = []));
    current.push(sample);
  }
  const spans = segments.map((segment) => [segment[0][0], segment.at(-1)[0]]);
  const xAt = (at) => (Math.max(start, Math.min(end, at)) - start) / (end - start || 1) * 1000;
  const x = (at) => ((at - start) / (end - start || 1) * 1000).toFixed(1);
  const y = (value) => (1000 - (Math.max(low, Math.min(high, value)) - low) / (high - low) * 1000).toFixed(1);

  const paths = segments.map((segment) => {
    const line = segment.map(([at, value], i) => !i ? `M${x(at)},${y(value)}` : readingSeconds && mode === "readings" ? `H${x(at)} V${y(value)}` : `L${x(at)},${y(value)}`).join(" ");
    return { line: segment.length === 1 ? `${line} h0.01` : line, area: segment.length > 1 ? `${line} L${x(segment.at(-1)[0])},1000 L${x(segment[0][0])},1000 Z` : null };
  });

  const gridSvg = ticks.map((tick) => `<line class="chart-grid" x1="0" x2="1000" y1="${y(tick)}" y2="${y(tick)}" vector-effect="non-scaling-stroke" />`).join("");
  const areasSvg = paths.filter((p) => p.area).map((p) => `<path class="chart-area" d="${p.area}" />`).join("");
  const bandsSvg = bands.map((band, i) => {
    const rects = band.spans.map(([a, b]) => `<rect x="${xAt(a)}" width="${Math.max(0, xAt(b) - xAt(a))}" y="0" height="1000" />`).join("");
    const fills = paths.filter((p) => p.area).map((p) => `<path class="chart-band ${escape(band.hue)}${band.role === "sampler" ? " sampler" : ""}" d="${p.area}" clip-path="url(#${uid}-${i})" />`).join("");
    return `<clipPath id="${uid}-${i}">${rects}</clipPath>${fills}`;
  }).join("");
  const linesSvg = paths.map((p) => `<path class="${p.area ? "chart-line" : "chart-dot"}" d="${p.line}" vector-effect="non-scaling-stroke" />`).join("");
  const crosshairSvg = steps ? "" : `<line class="chart-crosshair" visibility="hidden" x1="0" x2="0" y1="0" y2="1000" vector-effect="non-scaling-stroke" />`;

  const tickCount = end === start ? 0 : 4;
  const label = (at) => steps ? chartNumber(Math.round(at)) : axisTime(at, { start, now: end });
  const xLabels = Array.from({ length: tickCount + 1 }, (_, i) => `<span>${escape(label(start + (end - start) * i / (tickCount || 1)))}</span>`).join("");
  const yLabels = ticks.map((tick) => `<span>${escape(chartNumber(tick))}</span>`).join("");

  const summary = mode === "running"
    ? `<span>Something ran <strong>${escape(valueText(100 * running.reduce((sum, [a, b]) => sum + b - a, 0) / (end - start || 1), "%"))}</strong> of the range</span>`
    : `<span title="Arithmetic mean of the reported samples">Average <strong>${escape(valueText(values.reduce((sum, v) => sum + v, 0) / values.length, unit))}</strong></span><span>Peak <strong>${escape(valueText(Math.max(...values), unit))}</strong></span>${readingSeconds ? `<span>${escape(mode === "rolling" ? `Mean of the last minute's ${readingSeconds} s readings` : `One reading per ${readingSeconds} s`)}</span>` : ""}`;
  const bandKeys = bands.map((band) => `<span class="chart-band-key ${escape(band.hue)}${band.role === "sampler" ? " sampler" : ""}"><span class="chart-band-swatch"></span>${escape(band.label)}</span>`).join("");
  const noteHtml = note ? `<span class="chart-note" role="status">${escape(note)}</span>` : "";

  chartRegistry.set(uid, { data, spans, start, end, low, high, unit, steps, index: -1 });

  return `<figure class="chart" data-key="${escape(chartKey)}" data-tone="${escape(tone)}">${head}
    <div class="chart-plot" tabindex="0" role="group" aria-label="${escape(title)}. Use left and right arrow keys to inspect samples." data-chart-id="${uid}" data-start="${start}" data-end="${end}" data-min="${low}" data-max="${high}">
      <div class="chart-y">${yLabels}</div>
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label="${escape(title)}. Latest ${escape(valueText(data.at(-1)[1], unit))}.">${gridSvg}${areasSvg}${bandsSvg}${linesSvg}${crosshairSvg}</svg>
      <div class="chart-x">${xLabels}</div>
      <div class="chart-hover" hidden><div class="chart-cursor"></div><div class="chart-point"></div><div class="chart-tip"></div></div>
    </div>
    <div class="chart-summary">${summary}${bandKeys}${noteHtml}</div>
  </figure>`;
}

function indexAt(data, at) {
  let left = 0, right = data.length - 1;
  while (left < right) {
    const middle = Math.floor((left + right) / 2);
    if (data[middle][0] < at) left = middle + 1;
    else right = middle;
  }
  return left > 0 && Math.abs(data[left - 1][0] - at) < Math.abs(data[left][0] - at) ? left - 1 : left;
}

function showSample(plot, entry, index) {
  const hover = plot.querySelector(".chart-hover"), svg = plot.querySelector("svg");
  if (!entry.data.length || !hover || !svg) return;
  entry.index = Math.max(0, Math.min(entry.data.length - 1, index));
  const [at, value] = entry.data[entry.index], box = svg.getBoundingClientRect();
  const left = (at - entry.start) / (entry.end - entry.start || 1) * box.width;
  hover.hidden = false;
  hover.style.left = `${box.left - plot.getBoundingClientRect().left + left}px`;
  hover.querySelector(".chart-point").style.top = `${Math.max(0, Math.min(100, (entry.high - value) / (entry.high - entry.low) * 100))}%`;
  const tip = hover.querySelector(".chart-tip");
  tip.textContent = `${entry.steps ? `Step ${at}` : chartTime(at, true)} · ${valueText(value, entry.unit)}`;
  tip.style.left = `${Math.max(-left, Math.min(8, box.width - left - (tip.offsetWidth || 160)))}px`;
}

export function hoverChart(plot, event) {
  const entry = chartRegistry.get(plot.dataset.chartId);
  const svg = plot.querySelector("svg"), hover = plot.querySelector(".chart-hover");
  if (!entry || !svg || !hover) return;
  const box = svg.getBoundingClientRect();
  const at = entry.start + (event.clientX - box.left) / (box.width || 1) * (entry.end - entry.start);
  const outside = event.clientX < box.left || event.clientX > box.right;
  if (!entry.steps) setCrosshair(ui.navigating || event.buttons || outside ? null : at, { start: entry.start, now: entry.end });
  if (!entry.data.length || ui.navigating || event.buttons) { hover.hidden = true; return; }
  const idx = indexAt(entry.data, at);
  const near = Math.abs(entry.data[idx][0] - at) / (entry.end - entry.start || 1) * box.width < 6;
  if (outside || (!near && !entry.spans.some(([a, b]) => at >= a && at <= b))) hover.hidden = true;
  else showSample(plot, entry, idx);
}

export function inspectChart(plot, key) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(key)) return false;
  const entry = chartRegistry.get(plot.dataset.chartId);
  if (!entry) return false;
  if (key === "Escape") {
    const hover = plot.querySelector(".chart-hover");
    if (hover) hover.hidden = true;
    plot.blur();
    return true;
  }
  const idx = key === "Home" ? 0 : key === "End" ? entry.data.length - 1 : (entry.index < 0 ? entry.data.length - 1 : entry.index) + (key === "ArrowLeft" ? -1 : 1);
  showSample(plot, entry, idx);
  return true;
}

export function timeControlHtml(range, observed) {
  const duration = range.now - range.start;
  const custom = [[Math.floor(duration / 3600), "h"], [Math.floor(duration / 60) % 60, "m"], [duration % 60, "s"]].filter(([n]) => n).map(([n, unit]) => `${n}${unit}`).join(" ");
  const options = WINDOWS.some(([n]) => n === duration) ? WINDOWS : [[duration, `${custom} (custom)`], ...WINDOWS];
  const local = (at) => new Date(at * 1000).toISOString().slice(0, duration <= 120 ? 19 : 16);
  const timeLabel = range.live
    ? `Last ${WINDOWS.find(([n]) => n === duration)?.[1] || custom}`
    : `${local(range.start).slice(5, 10)} · ${chartTime(range.start, duration <= 120)} – ${local(range.start).slice(0, 10) === local(range.now).slice(0, 10) ? "" : `${local(range.now).slice(5, 10)} · `}${chartTime(range.now, duration <= 120)}`;
  return `<button type="button" class="time-shift" data-time-shift="-1" aria-label="Previous time window" ${range.start <= observed - MAX_WINDOW ? "disabled" : ""}><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(90 8 8)" /></svg></button>
    <details class="time-picker" data-key="node-time-picker"><summary class="time-summary" title="${escape(timeLabel)}"><span>${escape(timeLabel)}</span><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" /></svg></summary>
      <div class="time-popover"><label>Window <select data-time-duration>${options.map(([n, text]) => `<option value="${n}" ${n === duration ? "selected" : ""}>${escape(text)}</option>`).join("")}</select></label>
        <label>Until (UTC)<input type="datetime-local" data-time-end value="${range.live ? "" : escape(local(range.now))}" min="${escape(local(observed - MAX_WINDOW + duration))}" max="${escape(local(observed))}" step="any"></label>
        ${range.live ? '<span class="muted">Following current time</span>' : '<button type="button" class="chip" data-time-live="true">Return to live</button>'}
        <span class="muted">W/S zoom · A/D pan</span>
      </div>
    </details>
    <button type="button" class="time-shift" data-time-shift="1" aria-label="Next time window" ${range.now >= observed ? "disabled" : ""}><svg class="time-chevron" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 5.5 5 5 5-5" transform="rotate(-90 8 8)" /></svg></button>`;
}
