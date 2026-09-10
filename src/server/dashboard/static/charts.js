// One line chart as a string of markup. The plot is an SVG stretched to its
// box, so it needs no measuring and no resize handling; axis labels are HTML
// so they never distort. Hover is handled once, in app.js, from the data
// attributes written here.

import { escape } from "./ui.js";

export const chartNumber = (value) =>
  new Intl.NumberFormat(undefined, { maximumSignificantDigits: 4, notation: Math.abs(value) >= 10000 ? "compact" : "standard" }).format(value);
export const chartTime = (value, seconds = false) =>
  new Date(value * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", ...(seconds ? { second: "2-digit" } : {}), hour12: false, timeZone: "UTC" });
export const valueText = (value, unit) =>
  unit === "%" ? `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)}%` : `${chartNumber(value)}${unit ? ` ${unit}` : ""}`;

const SCALE = 1000;

export function chart({ title, unit = "", points = [], start, end, min, max, tone = "neutral", xFormat = "time", gapSeconds = Infinity, empty = "No samples in this time range" }) {
  const data = points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && x >= start && x <= end).sort((a, b) => a[0] - b[0]);
  const head = (latest) =>
    `<figcaption class="chart-head"><h2>${escape(title)}</h2>${latest ? `<span class="chart-latest"><span>Latest</span> <strong>${escape(valueText(latest[1], unit))}</strong></span>` : ""}</figcaption>`;
  if (!data.length) return `<figure class="chart" data-tone="${tone}">${head(null)}<p class="chart-empty">${escape(empty)}</p></figure>`;
  const values = data.map(([, y]) => y);
  let low = Number.isFinite(min) ? min : Math.min(...values);
  let high = Number.isFinite(max) ? max : Math.max(...values);
  if (!Number.isFinite(min) && !Number.isFinite(max)) {
    const pad = (high - low || Math.abs(high) || 1) * 0.08;
    low -= pad;
    high += pad;
  }
  if (high === low) high = low + 1;
  const x = (at) => (((at - start) / (end - start || 1)) * SCALE).toFixed(1);
  const y = (value) => (SCALE - ((value - low) / (high - low)) * SCALE).toFixed(1);
  const segments = [];
  data.forEach((point) => {
    const last = segments.at(-1);
    if (last && point[0] - last.at(-1)[0] <= gapSeconds) last.push(point);
    else segments.push([point]);
  });
  const paths = segments
    .map((segment) => {
      if (segment.length === 1) return `<path class="chart-dot" d="M${x(segment[0][0])},${y(segment[0][1])} h0.01" vector-effect="non-scaling-stroke"/>`;
      const line = segment.map(([at, value], i) => `${i ? "L" : "M"}${x(at)},${y(value)}`).join(" ");
      return `<path class="chart-area" d="${line} L${x(segment.at(-1)[0])},${SCALE} L${x(segment[0][0])},${SCALE} Z"/><path class="chart-line" d="${line}" vector-effect="non-scaling-stroke"/>`;
    })
    .join("");
  const ticks = 4;
  const label = (at) => (xFormat === "step" ? `step ${Math.round(at)}` : chartTime(at));
  const xLabels = Array.from({ length: ticks + 1 }, (_, i) => `<span>${escape(label(start + ((end - start) * i) / ticks))}</span>`).join("");
  const yLabels = [high, (high + low) / 2, low].map((value) => `<span>${escape(chartNumber(value))}</span>`).join("");
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const compact = data.map(([a, b]) => [Math.round(a * 1000) / 1000, Math.round(b * 1000) / 1000]);
  return `<figure class="chart" data-tone="${tone}">${head(data.at(-1))}
    <div class="chart-plot" data-points="${escape(JSON.stringify(compact))}" data-start="${start}" data-end="${end}" data-unit="${escape(unit)}" data-xformat="${xFormat}">
      <div class="chart-y">${yLabels}</div>
      <svg viewBox="0 0 ${SCALE} ${SCALE}" preserveAspectRatio="none" role="img" aria-label="${escape(title)}. Latest ${escape(valueText(data.at(-1)[1], unit))}."><line class="chart-grid" x1="0" x2="${SCALE}" y1="${SCALE / 2}" y2="${SCALE / 2}" vector-effect="non-scaling-stroke"/>${paths}</svg>
      <div class="chart-x">${xLabels}</div>
      <div class="chart-hover" hidden><div class="chart-cursor"></div><div class="chart-tip"></div></div>
    </div>
    <div class="chart-summary"><span>Average <strong>${escape(valueText(average, unit))}</strong></span><span>Peak <strong>${escape(valueText(Math.max(...values), unit))}</strong></span><span class="chart-zone">${xFormat === "step" ? "per step" : "UTC"}</span></div>
  </figure>`;
}

// Hover for every chart on the page: the sample nearest the pointer.
export function hoverChart(plot, clientX) {
  const points = JSON.parse(plot.dataset.points || "[]");
  const svg = plot.querySelector("svg");
  const hover = plot.querySelector(".chart-hover");
  if (!points.length || !svg || !hover) return;
  const box = svg.getBoundingClientRect();
  const start = Number(plot.dataset.start);
  const end = Number(plot.dataset.end);
  const at = start + ((clientX - box.left) / (box.width || 1)) * (end - start);
  let nearest = points[0];
  for (const point of points) if (Math.abs(point[0] - at) < Math.abs(nearest[0] - at)) nearest = point;
  const left = ((nearest[0] - start) / (end - start || 1)) * 100;
  hover.hidden = false;
  hover.style.left = `${Math.max(0, Math.min(100, left))}%`;
  const when = plot.dataset.xformat === "step" ? `step ${Math.round(nearest[0])}` : `${chartTime(nearest[0], true)} UTC`;
  const tip = hover.querySelector(".chart-tip");
  tip.textContent = `${when} · ${valueText(nearest[1], plot.dataset.unit)}`;
  tip.classList.toggle("flip", left > 60);
}
