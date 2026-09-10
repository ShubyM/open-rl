import { escape } from "./ui.js";

const metricCharts = new WeakMap();
const chartNumber = (value) =>
  new Intl.NumberFormat(undefined, {
    maximumSignificantDigits: 4,
    notation: Math.abs(value) >= 10000 ? "compact" : "standard",
  }).format(value);
const chartTime = (value, seconds = false) =>
  new Date(value * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    ...(seconds ? { second: "2-digit" } : {}),
    hour12: false,
    timeZone: "UTC",
  });

// Instances survive updates to the same container. Call destroy before replacing
// their page; the observer also disconnects when a rendered container is removed.
export function renderMetricChart(container, options) {
  const existing = metricCharts.get(container);
  if (existing) {
    existing.update(options);
    return existing;
  }
  let settings = {},
    width = 0,
    frame = 0,
    observer,
    destroyed = false;
  let points = [],
    x,
    y,
    plotLeft,
    plotRight,
    selected = -1;
  const cleanupListeners = [];
  const instance = {
    update(next) {
      settings = { ...settings, ...next };
      draw();
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      observer.disconnect();
      cancelAnimationFrame(frame);
      cleanupListeners.splice(0).forEach((cleanup) => cleanup());
      metricCharts.delete(container);
      container.classList.remove("metric-chart");
    },
  };
  function listen(element, event, callback) {
    element.addEventListener(event, callback);
    cleanupListeners.push(() => element.removeEventListener(event, callback));
  }
  function measureWidth() {
    const style = getComputedStyle(container);
    return Math.max(
      160,
      Math.round(
        (container.clientWidth || 320) -
          parseFloat(style.paddingLeft || 0) -
          parseFloat(style.paddingRight || 0),
      ),
    );
  }
  function draw() {
    if (destroyed) return;
    cleanupListeners.splice(0).forEach((cleanup) => cleanup());
    width = measureWidth();
    const {
      title = "Metric",
      unit = "",
      tone = "neutral",
      gapSeconds = Infinity,
      height = 180,
    } = settings;
    const samples = (settings.samples || [])
      .filter((sample) => Array.isArray(sample) && Number.isFinite(sample[0]))
      .sort((a, b) => a[0] - b[0]);
    const start = Number.isFinite(settings.start)
      ? settings.start
      : samples[0]?.[0];
    const end = Number.isFinite(settings.end)
      ? settings.end
      : samples.at(-1)?.[0];
    const inRange = samples.filter(([at]) => at >= start && at <= end);
    points = inRange.filter(([, value]) => Number.isFinite(value));
    selected = -1;
    const valueText = (value) =>
      `${unit === "%" ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value) : chartNumber(value)}${unit === "%" ? "%" : unit ? ` ${unit}` : ""}`;
    const last = points.at(-1);
    container.classList.add("metric-chart");
    container.dataset.chartTone = tone === "accent" ? "accent" : "neutral";
    container.innerHTML = `<div class="metric-chart-head"><h2>${escape(title)}</h2>${last ? `<span class="metric-chart-latest" title="Latest sample at ${escape(settings.xFormat ? settings.xFormat(last[0]) : new Date(last[0] * 1000).toISOString())}"><span>Latest</span> <strong>${escape(valueText(last[1]))}</strong></span>` : ""}</div>`;
    if (!points.length) {
      container.insertAdjacentHTML(
        "beforeend",
        '<p class="metric-chart-empty">No samples in this time range</p>',
      );
      return;
    }
    let low = Number.isFinite(settings.min)
      ? settings.min
      : Math.min(...points.map((point) => point[1]));
    let high = Number.isFinite(settings.max)
      ? settings.max
      : Math.max(...points.map((point) => point[1]));
    if (high <= low) {
      const padding = Math.max(Math.abs(low) * 0.1, 0.1);
      low -= padding;
      high += padding;
    }
    let ticks = [high, (high + low) / 2, low];
    if (!Number.isFinite(settings.min) && !Number.isFinite(settings.max)) {
      const rawStep = (high - low) / 3;
      const magnitude = 10 ** Math.floor(Math.log10(rawStep));
      const step =
        [1, 2, 5, 10].find((value) => value >= rawStep / magnitude) * magnitude;
      low = Math.floor(low / step + 1e-9) * step;
      high = Math.ceil(high / step - 1e-9) * step;
      ticks = Array.from(
        { length: Math.round((high - low) / step) + 1 },
        (_, index) => Number((high - index * step).toPrecision(12)),
      );
    }
    plotLeft = Math.max(
      38,
      ...ticks.map((value) => chartNumber(value).length * 6.4 + 12),
    );
    plotRight = width - 10;
    const top = 12,
      bottom = height - 28;
    x = (at) =>
      plotLeft +
      ((at - start) / Math.max(1, end - start)) * (plotRight - plotLeft);
    y = (value) =>
      top +
      (1 - (Math.min(high, Math.max(low, value)) - low) / (high - low)) *
        (bottom - top);
    const segments = [];
    let segment = [];
    for (const sample of inRange) {
      if (!Number.isFinite(sample[1])) {
        if (segment.length) segments.push(segment);
        segment = [];
        continue;
      }
      if (segment.length && sample[0] - segment.at(-1)[0] > gapSeconds) {
        segments.push(segment);
        segment = [];
      }
      segment.push(sample);
    }
    if (segment.length) segments.push(segment);
    const grid = ticks
      .map(
        (value) =>
          `<line x1="${plotLeft}" x2="${plotRight}" y1="${y(value)}" y2="${y(value)}" class="metric-gridline"/><text x="${plotLeft - 10}" y="${y(value) + 4}" text-anchor="end">${escape(chartNumber(value))}</text>`,
      )
      .join("");
    const tickCount = width < 420 ? 2 : width < 760 ? 3 : 5;
    const labels = Array.from({ length: tickCount }, (_, index) => {
      const at = start + ((end - start) * index) / (tickCount - 1);
      const date =
        end - start >= 86400
          ? `${new Date(at * 1000).toISOString().slice(5, 10)} `
          : "";
      const label = settings.xFormat ? settings.xFormat(at) : date + chartTime(at);
      return `<text x="${x(at)}" y="${height - 5}" text-anchor="${index === 0 ? "start" : index === tickCount - 1 ? "end" : "middle"}">${escape(label)}</text>`;
    }).join("");
    const series = segments
      .map((part) => {
        const path = part
          .map(
            ([at, value], index) => `${index ? "L" : "M"}${x(at)},${y(value)}`,
          )
          .join(" ");
        if (part.length === 1)
          return `<circle class="metric-series-dot" cx="${x(part[0][0])}" cy="${y(part[0][1])}" r="2.5"/>`;
        return `<path class="metric-series-area" d="${path} L${x(part.at(-1)[0])},${bottom} L${x(part[0][0])},${bottom} Z"/><path class="metric-series-line" d="${path}"/>`;
      })
      .join("");
    const average =
      points.reduce((sum, point) => sum + point[1], 0) / points.length;
    const peak = Math.max(...points.map((point) => point[1]));
    container.insertAdjacentHTML(
      "beforeend",
      `<div class="metric-chart-plot"><svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" role="img" tabindex="0" aria-label="${escape(title)}. Latest ${escape(valueText(last[1]))}. Use left and right arrow keys to inspect samples."><title>${escape(title)}</title>${grid}${series}<circle class="metric-series-latest" cx="${x(last[0])}" cy="${y(last[1])}" r="3"/>${labels}<g class="metric-crosshair" visibility="hidden"><line y1="${top}" y2="${bottom}"/><circle r="3.5"/></g></svg><div class="metric-chart-tooltip" hidden></div></div><div class="metric-chart-summary"><span title="Arithmetic mean of the reported samples">Average <strong>${escape(valueText(average))}</strong></span><span>Peak <strong>${escape(valueText(peak))}</strong></span><span class="metric-chart-zone">UTC</span></div>`,
    );
    const svg = container.querySelector("svg");
    const crosshair = container.querySelector(".metric-crosshair");
    const tooltip = container.querySelector(".metric-chart-tooltip");
    function inspect(index) {
      selected = Math.max(0, Math.min(points.length - 1, index));
      const [at, value] = points[selected];
      crosshair.setAttribute("visibility", "visible");
      const line = crosshair.querySelector("line"),
        dot = crosshair.querySelector("circle");
      line.setAttribute("x1", x(at));
      line.setAttribute("x2", x(at));
      dot.setAttribute("cx", x(at));
      dot.setAttribute("cy", y(value));
      tooltip.textContent = `${chartTime(at, true)} UTC · ${valueText(value)}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(0, Math.min(width - tooltip.offsetWidth, x(at) + 12))}px`;
    }
    function hide() {
      if (document.activeElement === svg) return;
      crosshair.setAttribute("visibility", "hidden");
      tooltip.hidden = true;
      selected = -1;
    }
    listen(svg, "pointermove", (event) => {
      const bounds = svg.getBoundingClientRect();
      const pointer = ((event.clientX - bounds.left) * width) / bounds.width;
      if (pointer < plotLeft || pointer > plotRight) {
        hide();
        return;
      }
      const at =
        start + ((pointer - plotLeft) / (plotRight - plotLeft)) * (end - start);
      let left = 0,
        right = points.length - 1;
      while (left < right) {
        const middle = Math.floor((left + right) / 2);
        if (points[middle][0] < at) left = middle + 1;
        else right = middle;
      }
      inspect(
        left > 0 &&
          Math.abs(points[left - 1][0] - at) < Math.abs(points[left][0] - at)
          ? left - 1
          : left,
      );
    });
    listen(svg, "pointerleave", hide);
    listen(svg, "focus", () => inspect(points.length - 1));
    listen(svg, "blur", hide);
    listen(svg, "keydown", (event) => {
      if (
        !["ArrowLeft", "ArrowRight", "Home", "End", "Escape"].includes(
          event.key,
        )
      )
        return;
      event.preventDefault();
      if (event.key === "Escape") {
        svg.blur();
        return;
      }
      inspect(
        event.key === "Home"
          ? 0
          : event.key === "End"
            ? points.length - 1
            : (selected < 0 ? points.length - 1 : selected) +
              (event.key === "ArrowLeft" ? -1 : 1),
      );
    });
  }
  observer = new ResizeObserver(() => {
    if (!container.isConnected) {
      instance.destroy();
      return;
    }
    const next = measureWidth();
    if (next !== width) {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(draw);
    }
  });
  metricCharts.set(container, instance);
  observer.observe(container);
  instance.update(options);
  return instance;
}

export function disposeMetricCharts(root) {
  if (root.matches?.(".metric-chart")) metricCharts.get(root)?.destroy();
  root
    .querySelectorAll(".metric-chart")
    .forEach((container) => metricCharts.get(container)?.destroy());
}
