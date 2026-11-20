export const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
export const duration = (total) => total < 90 ? `${Math.round(total)}s` : total < 5400 ? `${Math.round(total / 60)}m` : total < 172800 ? `${(total / 3600).toFixed(1)}h` : `${(total / 86400).toFixed(1)}d`;
const seconds = (value) => value === null || value === undefined || value === "" ? NaN : typeof value === "number" ? value : Date.parse(value) / 1000;
export const isFinished = (run) => run?.lifecycle === "ended";
export const finishedAt = (run) => [seconds(run?.completed_at), run?.last_completed_at].find(Number.isFinite) ?? NaN;
export const runLabel = (run) => run.client_status !== "active" ? run.client_status : { ended: "ended", failed: "failed", waiting: "queued", placed: "running" }[run.lifecycle] || "active";
export function elapsedTime(run, observedAt) {
  const stop = isFinished(run) ? finishedAt(run) : NaN;
  const start = seconds(run.created_at), end = Number.isFinite(stop) ? stop : seconds(observedAt);
  return Number.isFinite(start) && Number.isFinite(end) ? duration(Math.max(0, end - start)) : "—";
}
export function shortNodeName(name, nodes = []) {
  const full = String(name || "—"), parts = full.split("-");
  const others = nodes.map((n) => typeof n === "string" ? n : n.name).filter((n) => n && n !== full);
  for (let i = parts.length - 1; i >= 0; i--) {
    const suffix = parts.slice(i).join("-");
    if (suffix && !others.some((other) => other === suffix || other.endsWith(`-${suffix}`))) return suffix;
  }
  return full;
}
export const KINDS = { lora: "LoRA", full: "FFT", fft: "FFT" };
export const runDescription = (run, title = run.name) => [run.display_name, run.recipe_name].filter((v, i, values) => v && v !== title && values.indexOf(v) === i).join(" · ");
export const escapeRegExp = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
export const family = (id) => `hue-${Array.from(id || "").reduce((sum, c) => sum + c.charCodeAt(0), 0) % 8}`;
export function statusBadge(value) {
  const words = value === "waiting_for_placement" ? "queued" : String(value || "Unknown");
  const label = words[0].toUpperCase() + words.slice(1);
  const tone = { running: "running", ready: "running", live: "running", stalled: "failed", failed: "failed", completed: "completed", ended: "completed", stopped: "neutral" }[label.toLowerCase()] || "pending";
  return `<span class="state state-${tone}"><span class="state-dot" aria-hidden="true"></span>${escape(label)}</span>`;
}
