import { untrack } from "svelte";

const queryString = (params) => new URLSearchParams(Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")).toString();
export const viewLink = (path, params = {}) => `#${path}${queryString(params) ? `?${queryString(params)}` : ""}`;
export const inspectUrl = (resource, params = {}) => `/api/v1/inspect/${resource}${queryString(params) ? `?${queryString(params)}` : ""}`;
export const nodeLink = (placement, range) => viewLink("nodes", { placement, duration: range.now - range.start, end: range.now });
export const now = (inventory) => Date.parse(inventory?.observed_at) / 1000 || Date.now() / 1000;
const cache = new Map();
// The longest window the pages show, matching the inspection API's seven-day limit.
export const MAX_WINDOW = 7 * 86400;
export const saved = $state({ overview: "#overview", nodes: "#nodes" });
const number = (params, key, fallback) => params.get(key)?.trim() && Number.isFinite(Number(params.get(key))) ? Number(params.get(key)) : fallback;
export function parseView(hash = location.hash) {
  const [path, search = ""] = hash.slice(1).split("?");
  const [page, id = "", tab = "activity"] = path.split("/").map((part) => { try { return decodeURIComponent(part); } catch { return part; } });
  const params = new URLSearchParams(search), duration = Math.max(60, Math.min(MAX_WINDOW, Math.round(number(params, "duration", 1800))));
  const end = number(params, "end", null), event = number(params, "event", null), back = params.get("back");
  return { page: ["nodes", "run", "health"].includes(page) ? page : "overview", id, tab,
    duration, end: end !== null && end >= duration && end <= 8.64e12 ? end : null,
    event: event !== null && (end === null || (event >= end - duration && event <= end)) ? event : null,
    back: back?.startsWith("#nodes?") ? back : null,
    node: params.get("node"), placement: params.get("placement"), gpu: params.get("gpu") || "all",
    q: params.get("q") || "", pod: params.get("pod") || "",
    status: ["active", "unassigned", "finished"].includes(params.get("status")) ? params.get("status") : "all" };
}
export const view = $state(parseView());
export const rangeOf = (inventory, selection = view) => {
  const end = selection.end ?? now(inventory);
  return { start: end - selection.duration, now: end, live: selection.end === null };
};
// Grafana's shared crosshair: hovering any chart or timeline marks the same instant on the others.
export const crosshair = $state({ at: null });
export const timeAt = (event, range) => {
  const box = event.currentTarget.getBoundingClientRect();
  return range.start + Math.max(0, Math.min(1, (event.clientX - box.left) / (box.width || 1))) * (range.now - range.start);
};
export const crosshairOffset = (range) => crosshair.at === null || crosshair.at < range.start || crosshair.at > range.now ? null : 100 * (crosshair.at - range.start) / (range.now - range.start || 1);
export function currentView(freeze = false, tab = view.tab) {
  const { page, id, duration, end, back, event, q, pod, node, placement, gpu, status } = view;
  if (page === "overview") return viewLink(page, { q, status: status === "all" ? null : status });
  if (page === "run") return viewLink(`run/${encodeURIComponent(id)}/${tab || "activity"}`, { duration, end: freeze ? end ?? now(cache.get(inspectUrl("inventory"))) : end, back, event, q, pod });
  if (page === "nodes") return viewLink(page, { duration, end: freeze ? end ?? now(cache.get(inspectUrl("inventory"))) : end, node, placement, gpu: node && gpu !== "all" ? gpu : null });
  return viewLink(page + (id ? `/${encodeURIComponent(id)}` : ""));
}
export async function get(url, signal) {
  const timeout = AbortSignal.timeout(15000);
  let response;
  try { response = await fetch(url, { cache: "no-store", signal: signal ? AbortSignal.any([signal, timeout]) : timeout }); }
  catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(timeout.aborted ? "Dashboard request timed out. Try again." : "Cannot reach the dashboard. Check the connection and try again.");
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    const detail = Array.isArray(data?.detail) ? data.detail.map((item) => `${(item.loc || []).filter((part) => part !== "body").join(".")}: ${item.msg}`).join("; ") : data?.detail;
    const error = new Error([detail, data?.error].find((v) => typeof v === "string") || ({ 401: "Authentication required", 403: "Access denied", 404: "Requested data not found", 429: "Too many requests. Try again shortly." }[response.status] || `Dashboard request failed (${response.status})`));
    error.status = response.status;
    throw error;
  }
  return response.json().catch(() => { throw new Error("The dashboard returned an invalid response. Try again."); });
}

// Reuse recent responses across navigation; changing the key or unmounting aborts old work.
// A live view keys on the user's selection, so its moving window is read at each poll instead of restarting it.
export function createQuery(url, { poll = 0, pollWhen = () => true, scope = () => true, key = url } = {}) {
  let previousScope;
  const result = $state({ data: null, error: null, status: null, pending: false, stale: false, retry: () => {} });
  $effect(() => {
    const requestedKey = key(), requestedScope = scope();
    return untrack(() => {
      if (!requestedKey || !url()) return;
      if (requestedScope !== previousScope) { result.data = result.error = result.status = null; }
      previousScope = requestedScope;
      result.data = cache.get(url()) ?? result.data;
      const controller = new AbortController();
      let failures = 0, retryTimer = null;
      const load = result.retry = async () => {
        const requested = url();
        if (!requested || result.pending) return;
        result.pending = true;
        try {
          const data = await get(requested, controller.signal);
          if (controller.signal.aborted) return;
          failures = 0;
          Object.assign(result, { error: data.error || null, status: null, stale: false });
          if (!result.error || !result.data) result.data = data;
          if (!result.error) {
            cache.set(requested, data);
            if (cache.size > 24) cache.delete([...cache.keys()].find((key) => key !== inspectUrl("inventory")));
          }
        } catch (error) {
          if (controller.signal.aborted) return;
          // A dropped tunnel or a busy server is usually brief, so keep what is shown and retry; report it only if it lasts.
          failures += 1;
          if ((!error.status || error.status >= 500 || error.status === 429) && failures < 4) {
            result.stale = Boolean(result.data);
            clearTimeout(retryTimer);
            retryTimer = setTimeout(load, 2000 * 2 ** (failures - 1));
          } else {
            Object.assign(result, { error: error.message, status: error.status || null, stale: false });
          }
        }
        finally { if (!controller.signal.aborted) { result.pending = false; } }
      };
      load();
      const timer = poll ? setInterval(() => { if (!document.hidden && pollWhen()) load(); }, poll) : null;
      return () => { controller.abort(); clearInterval(timer); clearTimeout(retryTimer); result.pending = false; };
    });
  });
  return result;
}

export async function copyLink(url, success = "Link copied") {
  try { await navigator.clipboard.writeText(url); return success; }
  catch {
    const focused = document.activeElement, input = Object.assign(document.createElement("textarea"), { value: url });
    input.style.cssText = "position:fixed;opacity:0";
    document.body.append(input);
    input.select();
    let copied = false;
    try { copied = document.execCommand("copy"); } catch {}
    input.remove();
    focused?.focus({ preventScroll: true });
    return copied ? success : "Copy failed";
  }
}
