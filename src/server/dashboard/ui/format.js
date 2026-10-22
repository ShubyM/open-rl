export const duration = (total) => total < 90 ? `${Math.round(total)}s` : total < 5400 ? `${Math.round(total / 60)}m` : total < 172800 ? `${(total / 3600).toFixed(1)}h` : `${(total / 86400).toFixed(1)}d`;
const seconds = (value) => value === null || value === undefined || value === "" ? NaN : typeof value === "number" ? value : Date.parse(value) / 1000;
// A run is finished once the gateway marks it so, or once nothing serves it any more; a client
// that never deletes its model leaves the status active.
export const isFinished = (run) => run?.lifecycle === "ended";
// When a finished run stopped: when it was marked complete, else its last completed work.
export const finishedAt = (run) => [seconds(run.completed_at), run.last_completed_at].find(Number.isFinite) ?? NaN;
// One label: the gateway's status once it is final, else the lifecycle, unless the run's checks fail.
export const runLabel = (run, verdict) =>
  run.status !== "active" ? run.status : verdict === "fail" ? (run.lifecycle === "ended" ? "failed" : "unhealthy") : run.lifecycle === "ended" ? "ended" : { waiting: "queued", placed: "running" }[run.lifecycle] || "active";
// Failing then warning checks, symptoms before the causes that explain them.
export const flaggedChecks = (checks = []) => checks.filter((c) => c.verdict === "fail" || c.verdict === "warn").sort((a, b) => (a.verdict === "warn") - (b.verdict === "warn") || (a.kind === "cause") - (b.kind === "cause"));
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
