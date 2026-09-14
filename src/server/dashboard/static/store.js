// Shared page state and the few helpers every module needs. There is one
// snapshot, one selection, and one render function; keeping them here means
// no module has to import the app to reach them.

export const root = document.getElementById("orl-design-lab");
export const content = document.getElementById("content");

export const ui = {
  state: null, // latest /snapshot
  expanded: null, // placement id open on the Nodes page
  gpuGroup: null, // allocation anchoring the expansion's physical GPUs
  device: "all", // GPU selected in the expansion
  activityView: "gpu", // combined GPU strips or individual run rows
  nodeSelection: { duration: 1800, end: null },
  nodeNavigating: false, // redraw cached data while a timeline gesture is active
  nodeQueryRange: null, // GPU query frozen until the gesture settles
  // Installed by app.js once the page is wired up.
  render: () => {},
};

export const route = () => location.hash.slice(1).split("/").map(decodeURIComponent);

export async function get(url, signal) {
  const timeout = AbortSignal.timeout(15000);
  let response;
  try {
    response = await fetch(url, { cache: "no-store", signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new Error(timeout.aborted ? "Dashboard request timed out. Try again." : "Cannot reach the dashboard. Check the connection and try again.");
  }
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    const message = [data?.detail, data?.error].find((message) => typeof message === "string");
    const error = new Error(message || ({ 401: "Authentication required", 403: "Access denied", 404: "Requested data not found", 429: "Too many requests. Try again shortly." }[response.status] || `Dashboard request failed (${response.status})`));
    error.status = response.status;
    throw error;
  }
  return response.json().catch(() => { throw new Error("The dashboard returned an invalid response. Try again."); });
}

export const nodeNow = () => Date.parse(ui.state?.observed_at) / 1000 || Date.now() / 1000;
export const nodeTime = (at, seconds = false) => new Date(at * 1000).toISOString().slice(11, seconds ? 19 : 16);

const PALETTE_SIZE = 8;
// A runtime keeps its color when other jobs arrive, end, or leave history.
export const family = (id) => {
  const hash = Array.from(id || "").reduce((s, c) => s + c.charCodeAt(0), 0);
  return `hue-${hash % PALETTE_SIZE}`;
};
