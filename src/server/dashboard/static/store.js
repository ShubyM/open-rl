// Shared page state and the few helpers every module needs. There is one
// snapshot, one selection, and one render function; keeping them here means
// no module has to import the app to reach them.

export const root = document.getElementById("orl-design-lab");
export const content = document.getElementById("content");

export const ui = {
  state: null, // latest /snapshot
  expanded: null, // placement id open on the Nodes page
  device: "all", // GPU selected in the expansion
  nodeSelection: { duration: 1800, end: null },
  gpuView: null,
  runMetricView: null,
  experimentData: null,
  // Installed by app.js once the page is wired up.
  render: () => {},
};

export const route = () => location.hash.slice(1).split("/").map(decodeURIComponent);

export async function get(url, signal) {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}

export const nodeNow = () => Date.parse(ui.state?.observed_at) / 1000 || Date.now() / 1000;
export const nodeTime = (at) => new Date(at * 1000).toISOString().slice(11, 16);

const PALETTE_SIZE = 8;
// One hue per runtime, so a job keeps its color on every node and in the
// legend. Current placements take the first hues in a stable order; anything
// only seen in history falls back to a hash.
export const family = (id) => {
  const ids = [...new Set((ui.state?.placements || []).map((p) => p.runtime_id))].sort();
  const index = ids.indexOf(id);
  const hash = Array.from(id || "").reduce((s, c) => s + c.charCodeAt(0), 0);
  return `hue-${(index >= 0 ? index : hash) % PALETTE_SIZE}`;
};
