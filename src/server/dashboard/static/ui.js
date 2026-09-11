export const escape = (value) =>
  String(value ?? "—").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const encode = encodeURIComponent;
export const empty = (text) => `<p class="empty-state">${escape(text)}</p>`;
export const button = (label, attrs = "") => `<button type="button" class="chip" ${attrs}>${escape(label)}</button>`;

export const runStatus = (value) => {
  const label = String(value || "unknown");
  const tone =
    { running: "running", starting: "running", queued: "pending", unassigned: "pending", "needs attention": "failed", failed: "failed", completed: "completed", ended: "completed", unknown: "pending" }[
      label.toLowerCase()
    ] || "pending";
  return `<span class="state state-${tone}"><span class="state-dot" aria-hidden="true"></span>${escape(label)}</span>`;
};

const seconds = (value) => (value === null || value === undefined || value === "" ? NaN : typeof value === "number" ? value : Date.parse(value) / 1000);

export function elapsedTime(run, observedAt) {
  const start = seconds(run.created_at);
  const end = ["completed", "failed"].includes(String(run.status || "").toLowerCase()) && run.completed_at ? seconds(run.completed_at) : seconds(observedAt);
  if (!Number.isFinite(start) || !Number.isFinite(end)) return "—";
  return duration(Math.max(0, end - start));
}

export const duration = (total) => {
  if (total < 90) return `${Math.round(total)}s`;
  if (total < 5400) return `${Math.round(total / 60)}m`;
  if (total < 172800) return `${(total / 3600).toFixed(1)}h`;
  return `${(total / 86400).toFixed(1)}d`;
};

// morph patches a container toward new markup instead of replacing it, so a
// refresh keeps scroll position, focus, open panels and typed input.
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
  for (const { name } of Array.from(node.attributes)) if (!next.hasAttribute(name)) node.removeAttribute(name);
  for (const { name, value } of Array.from(next.attributes)) if (node.getAttribute(name) !== value) node.setAttribute(name, value);
  // Live form values belong to the user, not the template.
  if (node.tagName === "INPUT" || node.tagName === "SELECT") return;
  morphChildren(node, next);
}
