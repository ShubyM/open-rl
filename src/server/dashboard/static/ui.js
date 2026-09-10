export const escape = (value) =>
  String(value ?? "—").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const encode = encodeURIComponent;
export const empty = (text) => `<p class="empty-state">${escape(text)}</p>`;
export const button = (label, attrs = "") =>
  `<button type="button" class="appbutton" ${attrs}>${escape(label)}</button>`;
export const runStatus = (value) => {
  const label = String(value || "Unknown");
  const tone =
    {
      active: "running",
      running: "running",
      restarting: "pending",
      starting: "pending",
      "needs attention": "failed",
      queued: "pending",
      pending: "pending",
      failed: "failed",
      completed: "completed",
    }[label.toLowerCase()] || "unknown";
  return `<span class="run-state state-${tone}"><span aria-hidden="true"></span>${escape(label[0].toUpperCase() + label.slice(1))}</span>`;
};

export function elapsedTime(run, observedAt) {
  const timestamp = (value) => {
    if (value === null || value === undefined || value === "") return NaN;
    return typeof value === "number" ? value * 1000 : Date.parse(value);
  };
  const start = timestamp(run.created_at);
  const terminal = ["completed", "failed", "cancelled", "canceled"].includes(
    String(run.status || "").toLowerCase(),
  );
  const end = timestamp(terminal ? run.completed_at : observedAt);
  if (
    !Number.isFinite(start) ||
    start <= 0 ||
    !Number.isFinite(end) ||
    end < start
  )
    return "—";
  const minutes = Math.floor((end - start) / 60000);
  if (minutes < 1) return "<1m";
  return minutes < 60
    ? `${minutes}m`
    : `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

// morph patches a container toward new markup instead of replacing it, so a
// 10-second refresh keeps scroll position, focus, open panels, and the charts
// the chart module owns. Elements that carry the metric-chart class are
// JS-rendered and are left alone apart from their data attributes.
export function morph(container, html) {
  const template = document.createElement("template");
  template.innerHTML = html;
  morphChildren(container, template.content);
}
function morphChildren(from, to) {
  const current = Array.from(from.childNodes);
  const next = Array.from(to.childNodes);
  for (let i = 0; i < Math.max(current.length, next.length); i++) {
    if (!next[i]) current[i].remove();
    else if (!current[i]) from.appendChild(next[i]);
    else morphNode(current[i], next[i], from);
  }
}
function morphNode(node, next, parent) {
  if (node.nodeType !== next.nodeType || (node.nodeType === 1 && node.tagName !== next.tagName)) {
    parent.replaceChild(next, node);
    return;
  }
  if (node.nodeType !== 1) {
    if (node.data !== next.data) node.data = next.data;
    return;
  }
  const chart = node.classList.contains("metric-chart") && !next.classList.contains("metric-chart");
  syncAttributes(node, next, chart);
  if (!chart) morphChildren(node, next);
}
function syncAttributes(node, next, keepChartAttributes) {
  const owned = (name) => keepChartAttributes && (name === "class" || name === "data-chart-tone");
  for (const { name } of Array.from(node.attributes)) {
    if (!next.hasAttribute(name) && !owned(name)) node.removeAttribute(name);
  }
  for (const { name, value } of Array.from(next.attributes)) {
    if (!owned(name) && node.getAttribute(name) !== value) node.setAttribute(name, value);
  }
}
