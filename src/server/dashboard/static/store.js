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
  nodeSelection: { duration: 1800, end: null },
  // Installed by app.js once the page is wired up.
  render: () => {},
};

export const route = () => location.hash.slice(1).split("/").map(decodeURIComponent);

export async function get(url, signal) {
  const timeout = AbortSignal.timeout(15000);
  const response = await fetch(url, { cache: "no-store", signal: signal ? AbortSignal.any([signal, timeout]) : timeout });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error([error?.detail, error?.error].find((message) => typeof message === "string") || `Request failed (${response.status})`);
  }
  return response.json();
}

export const nodeNow = () => Date.parse(ui.state?.observed_at) / 1000 || Date.now() / 1000;
export const nodeTime = (at, seconds = false) => new Date(at * 1000).toISOString().slice(11, seconds ? 19 : 16);

const PALETTE_SIZE = 8;
// A runtime keeps its color when other jobs arrive, end, or leave history.
export const family = (id) => {
  const hash = Array.from(id || "").reduce((s, c) => s + c.charCodeAt(0), 0);
  return `hue-${hash % PALETTE_SIZE}`;
};
