// URL state, fetch client, and render-scoped query cache.

export const MAX_WINDOW = 7 * 86400;
export const WINDOWS = [[60, "1 minute"], [600, "10 minutes"], [1800, "30 minutes"], [3600, "1 hour"], [10800, "3 hours"], [21600, "6 hours"], [86400, "24 hours"], [259200, "3 days"], [MAX_WINDOW, "7 days"]];

const queryString = (params) => {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")).toString();
  return query ? `?${query}` : "";
};
export const viewLink = (path, params = {}) => `#${path}${queryString(params)}`;
export const inspectUrl = (resource, params = {}) => `/api/v1/inspect/${resource}${queryString(params)}`;
export const nodeLink = (placement, range) => viewLink("nodes", { placement, duration: range.now - range.start, end: range.now });
export const now = (inventory) => Date.parse(inventory?.observed_at) / 1000 || Date.now() / 1000;

export const saved = { overview: "#overview", nodes: "#nodes" };
export const ui = {
  inventory: { data: null, error: null, status: null, pending: false, stale: false, failures: 0, etag: null },
  navigating: false,
  activityMode: "lanes",
  openGpus: new Set(),
  chartModes: new Map(),
  runOpened: new Set(),
  scrollSelection: true,
  render: () => {},
};

const number = (params, key, fallback) => params.get(key)?.trim() && Number.isFinite(Number(params.get(key))) ? Number(params.get(key)) : fallback;

export function parseView(hash = location.hash) {
  const [path, search = ""] = hash.slice(1).split("?");
  const [page, id = "", tab = "activity"] = path.split("/").map((part) => { try { return decodeURIComponent(part); } catch { return part; } });
  const params = new URLSearchParams(search), duration = Math.max(60, Math.min(MAX_WINDOW, Math.round(number(params, "duration", 1800))));
  const end = number(params, "end", null), event = number(params, "event", null), back = params.get("back");
  return {
    page: ["nodes", "run", "health"].includes(page) ? page : "overview",
    id,
    tab,
    explicitWindow: params.has("duration") || params.has("end"),
    duration,
    end: end !== null && end >= duration && end <= 8.64e12 ? end : null,
    event: event !== null && (end === null || (event >= end - duration && event <= end)) ? event : null,
    back: back?.startsWith("#nodes?") ? back : null,
    node: params.get("node"),
    placement: params.get("placement"),
    gpu: params.get("gpu") || "all",
    q: params.get("q") || "",
    pod: params.get("pod") || "",
    status: ["active", "queued", "finished"].includes(params.get("status")) ? params.get("status") : "all",
  };
}

export const view = parseView();

export const rangeOf = (inventory, selection = view) => {
  const observed = now(inventory), duration = Math.max(60, Math.min(MAX_WINDOW, Math.round(selection.duration || 1800)));
  const end = Math.max(observed - MAX_WINDOW + duration, Math.min(observed, selection.end ?? observed));
  return { start: end - duration, now: end, live: selection.end === null };
};

export function withoutSelection(hash) {
  const [path, query = ""] = hash.split("?");
  const params = new URLSearchParams(query);
  for (const name of ["node", "placement", "gpu"]) params.delete(name);
  return params.size ? `${path}?${params}` : path;
}

export function currentView(freeze = false, tab = view.tab) {
  const { page, id, duration, end, back, event, q, pod, node, placement, gpu, status } = view;
  if (page === "overview") return viewLink(page, { q, status: status === "all" ? null : status });
  if (page === "run") return viewLink(`run/${encodeURIComponent(id)}/${tab || "activity"}`, { duration, end: freeze ? end ?? now(ui.inventory.data) : end, back, event, q, pod });
  if (page === "nodes") return viewLink(page, { duration, end: freeze ? end ?? now(ui.inventory.data) : end, node, placement, gpu: node && gpu !== "all" ? gpu : null });
  return viewLink(page + (id ? `/${encodeURIComponent(id)}` : ""));
}

export function syncHash() {
  const hash = currentView();
  if (location.hash !== hash) history.replaceState(history.state, "", hash);
  if (view.page === "overview") saved.overview = hash;
  if (view.page === "nodes" || (view.page === "run" && view.back)) saved.nodes = withoutSelection(view.page === "nodes" ? hash : view.back);
}

export async function get(url, signal, previous) {
  const timeout = AbortSignal.timeout(15000);
  let response;
  try {
    response = await fetch(url, { cache: "no-store", headers: previous?.etag ? { "If-None-Match": previous.etag } : undefined, signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(timeout.aborted ? "Dashboard request timed out. Try again." : "Cannot reach the dashboard. Check the connection and try again.");
  }
  if (response.status === 304 && previous?.etag) return previous;
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    const detail = Array.isArray(data?.detail) ? data.detail.map((item) => `${(item.loc || []).filter((part) => part !== "body").join(".")}: ${item.msg}`).join("; ") : data?.detail;
    const error = new Error([detail, data?.error].find((v) => typeof v === "string") || ({ 401: "Authentication required", 403: "Access denied", 404: "Requested data not found", 429: "Too many requests. Try again shortly." }[response.status] || `Dashboard request failed (${response.status})`));
    error.status = response.status;
    throw error;
  }
  const data = await response.json().catch(() => { throw new Error("The dashboard returned an invalid response. Try again."); });
  return { data, etag: response.headers.get("ETag") };
}

// Render-scoped query cache: pages request a URL under a scope key; fetches run in the background and coalesce repaints.
const entries = new Map();
const remembered = new Map();
const MAX_ENTRIES = 24;
let generation = 0;
let repaint = 0;

function scheduleRepaint() {
  cancelAnimationFrame(repaint);
  const g = generation;
  repaint = requestAnimationFrame(() => { if (g === generation) ui.render(); });
}

export const beginRender = () => { generation++; };
export function endRender() {
  for (const [scope, entry] of entries) {
    if (entry.used === generation) continue;
    entry.controller?.abort();
    clearTimeout(entry.timer);
    entries.delete(scope);
  }
}

export function retryQuery(scope) {
  const entry = entries.get(scope);
  if (!entry) return;
  entry.controller?.abort();
  clearTimeout(entry.timer);
  entry.fetchedAt = 0;
  entry.failures = 0;
  entry.pending = false;
  ui.render();
}

export function use(url, { scope = url, poll = 15000, pollWhen = true, remember = false } = {}) {
  if (!url) return { data: null, error: null, status: null, pending: false, stale: false };
  let entry = entries.get(scope);
  if (!entry) {
    const cached = remembered.get(url);
    entry = { scope, url: null, data: cached?.data ?? null, etag: cached?.etag ?? null, error: null, status: null, fetchedAt: cached ? Date.now() : 0, pending: false, stale: false, failures: 0, controller: null, timer: null, used: generation };
    entries.set(scope, entry);
  }
  entry.used = generation;
  const urlChanged = entry.url !== url;
  if (urlChanged && remembered.has(url)) {
    const cached = remembered.get(url);
    entry.url = url;
    entry.data = cached.data;
    entry.etag = cached.etag;
    entry.error = entry.status = null;
    entry.stale = false;
    entry.fetchedAt = Date.now();
  }
  const due = urlChanged || (poll > 0 && pollWhen && Date.now() - entry.fetchedAt >= poll);
  if (!ui.navigating && !entry.pending && due) {
    entry.controller?.abort();
    clearTimeout(entry.timer);
    entry.url = url;
    entry.pending = true;
    const controller = entry.controller = new AbortController();
    get(url, controller.signal, entry.etag ? entry : null)
      .then(({ data, etag }) => {
        if (controller.signal.aborted) return;
        entry.failures = 0;
        entry.error = data.error || null;
        entry.status = null;
        entry.stale = false;
        if (!entry.error || !entry.data) entry.data = data;
        entry.etag = etag;
        if (!entry.error && remember) {
          remembered.set(url, { data: entry.data, etag });
          if (remembered.size > MAX_ENTRIES) remembered.delete(remembered.keys().next().value);
        }
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        entry.failures += 1;
        const transient = (!error.status || error.status >= 500 || error.status === 429) && entry.failures < 4;
        if (transient) {
          entry.stale = Boolean(entry.data);
          entry.timer = setTimeout(() => { if (entries.get(scope) === entry) { entry.fetchedAt = 0; ui.render(); } }, 2000 * 2 ** (entry.failures - 1));
        } else {
          entry.error = error.message;
          entry.status = error.status || null;
          entry.stale = false;
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          entry.pending = false;
          entry.controller = null;
          entry.fetchedAt = Date.now();
          if (entries.get(scope) === entry && entry.used === generation) {
            if (poll > 0 && pollWhen && !entry.timer) {
              entry.timer = setTimeout(() => { if (!document.hidden && entries.get(scope) === entry) ui.render(); }, poll);
            }
            scheduleRepaint();
          }
        }
      });
  }
  return entry;
}

export const timeAt = (event, range) => {
  const box = event.currentTarget.getBoundingClientRect();
  return range.start + Math.max(0, Math.min(1, (event.clientX - box.left) / (box.width || 1))) * (range.now - range.start);
};

// Update shared crosshair directly on DOM hosts (no full re-morph on pointermove).
export function setCrosshair(at, range) {
  const pct = at === null || !range || at < range.start || at > range.now ? null : 100 * (at - range.start) / (range.now - range.start || 1);
  for (const host of document.querySelectorAll("[data-crosshair-host]")) {
    if (pct === null) host.removeAttribute("data-crosshair");
    else {
      host.setAttribute("data-crosshair", "");
      host.style.setProperty("--crosshair", `${pct}%`);
    }
  }
  for (const line of document.querySelectorAll("line.chart-crosshair")) {
    if (pct === null) line.setAttribute("visibility", "hidden");
    else {
      line.removeAttribute("visibility");
      const x = (pct * 10).toFixed(1);
      line.setAttribute("x1", x);
      line.setAttribute("x2", x);
    }
  }
}
