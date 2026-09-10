// The one way pages fetch anything besides the snapshot. A page asks for a
// URL and gets whatever is cached, possibly nothing; the fetch runs in the
// background and re-renders when it lands. Rendering is idempotent, so no
// page tracks whether a reply is still wanted.

import { get, ui } from "./store.js";

const entries = new Map();
const FRESH_MS = 15000;
const MAX_ENTRIES = 24;

export function use(url) {
  let entry = entries.get(url);
  if (!entry) {
    entry = { data: null, error: null, fetchedAt: 0, pending: false };
    entries.set(url, entry);
    while (entries.size > MAX_ENTRIES) entries.delete(entries.keys().next().value);
  }
  if (!entry.pending && Date.now() - entry.fetchedAt >= FRESH_MS) {
    entry.pending = true;
    get(url)
      .then((data) => {
        entry.data = data;
        entry.error = null;
      })
      .catch((error) => {
        entry.error = error.message;
      })
      .finally(() => {
        entry.fetchedAt = Date.now();
        entry.pending = false;
        ui.render();
      });
  }
  return entry;
}
