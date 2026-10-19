import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { build } from 'esbuild';
import { compileModule } from 'svelte/compiler';

// Run the real rune-compiled query helper, including effect cleanup and batching.
globalThis.location = { hash: '' };
globalThis.document = { hidden: false };
const { outputFiles } = await build({
  stdin: { contents: 'export { createQuery, currentView, view } from "../../src/server/dashboard/ui/state.svelte.js"; export { flushSync } from "svelte"; export { effect_root as root, state as source, get, set } from "svelte/internal/client";', resolveDir: import.meta.dirname },
  bundle: true, write: false, format: 'esm', conditions: ['browser'], nodePaths: [import.meta.dirname + '/node_modules'],
  plugins: [{ name: 'runes', setup(build) {
    build.onLoad({ filter: /\.svelte\.js$/ }, async ({ path }) => ({
      contents: compileModule(await readFile(path, 'utf8'), { filename: path, generate: 'client' }).js.code, resolveDir: dirname(path),
    }));
  } }],
});
const { createQuery, currentView, view, flushSync, root, source, get, set } = await import('data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64'));
const settle = async () => { await new Promise(resolve => setImmediate(resolve)); flushSync(); };
let requests;
globalThis.fetch = (url, options) => new Promise(resolve => requests.push({
  url, signal: options.signal, respond: (data, status = 200) => resolve({ ok: status === 200, status, json: async () => data }),
}));
function open(url, scope = '', options = {}) {
  const target = source(url), identity = source(scope);
  let query;
  const close = root(() => { query = createQuery(() => get(target), { scope: () => get(identity), ...options }); });
  flushSync();
  return { query, close, navigate(url, scope = '') { flushSync(() => { set(target, url); set(identity, scope); }); } };
}

test('cached revisit renders immediately, refreshes, preserves data on failure and retries', async () => {
  requests = [];
  const first = open('/activity?run=a');
  requests.at(-1).respond({ samples: [1] }); await settle(); first.close();
  const revisit = open('/activity?run=a');
  assert.deepEqual(revisit.query.data.samples, [1]);
  assert.equal(revisit.query.pending, true);
  // A brief outage keeps the data and retries quietly; it is reported only once it lasts.
  requests.at(-1).respond({ detail: 'Offline' }, 503); await settle();
  assert.deepEqual(revisit.query.data.samples, [1]);
  assert.equal(revisit.query.error, null);
  assert.equal(revisit.query.stale, true);
  for (let i = 0; i < 3; i++) { revisit.query.retry(); requests.at(-1).respond({ detail: 'Offline' }, 503); await settle(); }
  assert.deepEqual(revisit.query.data.samples, [1]);
  assert.equal(revisit.query.error, 'Offline');
  assert.equal(revisit.query.status, 503);
  assert.equal(revisit.query.stale, false);
  revisit.query.retry();
  requests.at(-1).respond({ samples: [2] }); await settle();
  assert.deepEqual(revisit.query.data.samples, [2]);
  assert.equal(revisit.query.error, null);
  assert.equal(revisit.query.status, null);
  revisit.close();
});

test('a request the server rejects is reported at once, not retried', async () => {
  requests = [];
  const page = open('/activity?run=missing');
  requests.at(-1).respond({ detail: 'Requested data not found' }, 404); await settle();
  assert.equal(page.query.error, 'Requested data not found');
  assert.equal(page.query.stale, false);
  assert.equal(requests.length, 1);
  page.close();
});

test('changed scopes and cancelled responses never leak another node or time window', async () => {
  requests = [];
  const view = open('/series?node=a&end=1', 'a');
  const old = requests.at(-1);
  view.navigate('/series?node=b&end=1', 'b');
  assert.equal(old.signal.aborted, true);
  requests.at(-1).respond({ node: 'b' }); await settle();
  old.respond({ node: 'a' }); await settle(); // Even a transport ignoring abort cannot overwrite B.
  assert.equal(view.query.data.node, 'b');
  view.navigate('/series?node=b&end=2', 'b');
  assert.equal(view.query.data.node, 'b'); // Retain B while its new time window loads.
  const pending = requests.at(-1);
  view.navigate('/series?node=c&end=2', 'c');
  assert.equal(view.query.data, null);
  view.close();
  assert.equal(requests.at(-1).signal.aborted, true);
  pending.respond({ node: 'b-old-window' }); await settle();
  const exact = open('/series?node=b&end=2', 'b');
  assert.equal(exact.query.data, null); // Aborted response was not cached.
  exact.close();
});

test('partial errors are visible and never replace a successful cached response', async () => {
  requests = [];
  const view = open('/activity?run=partial');
  requests.at(-1).respond({ samples: [3] }); await settle();
  view.query.retry();
  requests.at(-1).respond({ samples: [], error: 'Telemetry unavailable' }); await settle();
  assert.deepEqual(view.query.data.samples, [3]);
  assert.equal(view.query.error, 'Telemetry unavailable');
  view.close();
  const revisit = open('/activity?run=partial');
  assert.deepEqual(revisit.query.data.samples, [3]);
  revisit.close();
});

test('cache is bounded while the inventory timestamp remains available for copied links', async () => {
  requests = [];
  const inventory = open('/api/v1/inspect/inventory');
  requests.at(-1).respond({ observed_at: '2026-01-02T12:00:00Z' }); await settle(); inventory.close();
  for (let i = 0; i < 30; i++) {
    const view = open('/activity?end=' + i);
    requests.at(-1).respond({ window: i }); await settle(); view.close();
  }
  const evicted = open('/activity?end=0');
  assert.equal(evicted.query.data, null); evicted.close();
  const recent = open('/activity?end=29');
  assert.equal(recent.query.data.window, 29); recent.close();
  const retained = open('/api/v1/inspect/inventory');
  assert.equal(retained.query.data.observed_at, '2026-01-02T12:00:00Z'); retained.close();
  Object.assign(view, { page: 'run', id: 'a', end: null });
  assert.equal(new URLSearchParams(currentView(true).split('?')[1]).get('end'), '1767355200');
});

test('live selection keys keep requests running and read the latest window on refresh', async () => {
  requests = [];
  const selection = source('run-a');
  const live = open('/live?end=1', '', { key: () => get(selection) });
  const initial = requests.at(-1);
  live.navigate('/live?end=2');
  assert.equal(initial.signal.aborted, false);
  assert.equal(requests.length, 1);
  initial.respond({ window: 1 }); await settle();
  live.query.retry();
  assert.equal(requests.at(-1).url, '/live?end=2');
  const active = requests.at(-1);
  flushSync(() => set(selection, false));
  assert.equal(active.signal.aborted, true);
  live.navigate('/live?end=3');
  assert.equal(requests.length, 2);
  flushSync(() => set(selection, 'run-a'));
  assert.equal(requests.at(-1).url, '/live?end=3');
  live.close();
});
