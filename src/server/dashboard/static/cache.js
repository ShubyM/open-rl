// A small LRU of fetched metric payloads, shared by every chart on the page.
// Views paint from whatever is cached, then refresh in the background.

import { get } from "./store.js";

const metricCache = new Map();
export const metricFreshFor = 15000;

export function metricEntry(url) {
  const entry = metricCache.get(url) || {
    data: null,
    fetchedAt: 0,
    pending: null,
    controller: null,
  };
  metricCache.delete(url);
  metricCache.set(url, entry);
  while (metricCache.size > 24) {
    const oldest = metricCache.keys().next().value;
    metricCache.get(oldest).controller?.abort();
    metricCache.delete(oldest);
  }
  return entry;
}

export const cached = (url) => metricCache.get(url)?.data;

export function isFresh(url) {
  const entry = metricCache.get(url);
  return Boolean(entry?.pending || (entry?.data && Date.now() - entry.fetchedAt < metricFreshFor));
}

export function metricData(url) {
  const entry = metricEntry(url);
  if (entry.pending) return entry.pending;
  if (entry.data && Date.now() - entry.fetchedAt < metricFreshFor) return Promise.resolve(entry.data);
  entry.controller = new AbortController();
  entry.pending = get(url, entry.controller.signal)
    .then((data) => {
      entry.data = data;
      entry.fetchedAt = Date.now();
      return data;
    })
    .finally(() => {
      entry.pending = null;
      entry.controller = null;
    });
  return entry.pending;
}

// Paint a view from cache immediately, then again when fresh data lands,
// unless the view has been replaced or detached in the meantime.
export async function loadMetricView(view, paint, current) {
  const entry = metricEntry(view.url);
  const fresh = entry.data && Date.now() - entry.fetchedAt < metricFreshFor;
  if (fresh && view.data === entry.data && view.painted) return;
  view.data = entry.data || view.data;
  view.status.textContent = "";
  paint(view);
  view.painted = Boolean(view.data);
  if (fresh) return;
  view.status.textContent = view.data ? "Updating metrics…" : "Loading metrics…";
  const active = () => current() === view && view.element.isConnected;
  try {
    const data = await metricData(view.url);
    if (!active()) return;
    view.data = data;
    view.status.textContent = "";
    paint(view);
    view.painted = true;
  } catch (error) {
    if (active())
      view.status.textContent = view.data ? `Metrics refresh failed: ${error.message}. Showing previously received samples.` : error.message;
  }
}
