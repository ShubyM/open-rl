export const duration = (total) => total < 90 ? `${Math.round(total)}s` : total < 5400 ? `${Math.round(total / 60)}m` : total < 172800 ? `${(total / 3600).toFixed(1)}h` : `${(total / 86400).toFixed(1)}d`;
const seconds = (value) => value === null || value === undefined || value === "" ? NaN : typeof value === "number" ? value : Date.parse(value) / 1000;
export function elapsedTime(run, observedAt) {
  const start = seconds(run.created_at), end = ["completed", "failed", "ended"].includes(String(run.status || "").toLowerCase()) && run.completed_at ? seconds(run.completed_at) : seconds(observedAt);
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
export function samplerPodPattern(run, placements = []) {
  const names = [...(run?.pods || []).filter((p) => p.role === "sampler").map((p) => p.name),
    ...(run?.workloads || []).filter((w) => w.role === "sampler").map((w) => w.pod_name),
    ...placements.filter((p) => p.role === "sampler" && p.run_ids?.includes(run?.run_id)).map((p) => p.pod)];
  const pods = [...new Set(names.filter((name) => typeof name === "string" && name.length))].sort();
  return pods.length ? `^(${pods.map(escapeRegExp).join("|")})$` : null;
}
