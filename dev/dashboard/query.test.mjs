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
  stdin: { contents: `export { createQuery } from "../../src/server/dashboard/ui/state.svelte.js"; export { mergeSeries } from "../../src/server/dashboard/ui/series.js"; export { flushSync } from "svelte"; export { effect_root as root, state as source, get, set } from "svelte/internal/client";`, resolveDir: import.meta.dirname },
  bundle: true, write: false, format: 'esm', conditions: ['browser'], nodePaths: [import.meta.dirname + '/node_modules'],
  plugins: [{ name: 'runes', setup(build) {
    build.onLoad({ filter: /\.svelte\.js$/ }, async ({ path }) => ({
      contents: compileModule(await readFile(path, 'utf8'), { filename: path, generate: 'client' }).js.code, resolveDir: dirname(path),
    }));
  } }],
});
const { createQuery, mergeSeries, flushSync, root, source, get, set } = await import('data:text/javascript;base64,' + Buffer.from(outputFiles[0].text).toString('base64'));
const settle = async () => { await new Promise(resolve => setImmediate(resolve)); flushSync(); };
let requests;
globalThis.fetch = (url, options) => new Promise(resolve => requests.push({
  url, signal: options.signal, headers: options.headers, respond: (data, status = 200, etag = null) => resolve({
    ok: status === 200, status, headers: new Headers(etag ? { ETag: etag } : {}),
    json: async () => { assert.notEqual(status, 304, 'a 304 must not be parsed'); return data; },
  }),
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
  // A brief outage keeps the data quietly; it is reported only once it lasts.
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
  revisit.query.retry();
  requests.at(-1).respond({ samples: [], error: 'Telemetry unavailable' }); await settle();
  assert.deepEqual(revisit.query.data.samples, [2]);
  assert.equal(revisit.query.error, 'Telemetry unavailable');
  revisit.close();
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
  view.navigate(null, null);
  assert.equal(view.query.data, null);
  assert.equal(view.query.pending, false);
  assert.equal(view.query.error, null);
  view.navigate('/series?node=c&end=2', 'c');
  view.close();
  assert.equal(requests.at(-1).signal.aborted, true);
  pending.respond({ node: 'b-old-window' }); await settle();
  const exact = open('/series?node=b&end=2', 'b');
  assert.equal(exact.query.data, null); // Aborted response was not cached.
  exact.close();
});

test('unchanged responses reuse chart data, recover from errors, and revalidate cached revisits', async () => {
  requests = [];
  const page = open('/history?run=etag-a');
  const first = { rows: [{ step: 1, metrics: { reward: 0.7 } }] };
  requests.at(-1).respond(first, 200, 'W/"r1"'); await settle();
  page.query.retry();
  assert.equal(requests.at(-1).headers['If-None-Match'], 'W/"r1"');
  requests.at(-1).respond(null, 304); await settle();
  assert.equal(page.query.data, first, 'unchanged data must not rebuild charts');
  page.query.retry();
  requests.at(-1).respond({ detail: 'Offline' }, 503); await settle();
  assert.equal(page.query.stale, true);
  page.query.retry();
  requests.at(-1).respond(null, 304); await settle();
  assert.equal(page.query.stale, false);
  assert.equal(page.query.error, null);
  assert.equal(page.query.data, first);
  page.close();
  const revisit = open('/history?run=etag-a');
  assert.equal(requests.at(-1).headers['If-None-Match'], 'W/"r1"');
  const corrected = { rows: [{ step: 1, metrics: { reward: 0.9 } }] };
  requests.at(-1).respond(corrected, 200, 'W/"r2"'); await settle();
  assert.equal(revisit.query.data, corrected);
  revisit.navigate('/history?run=etag-b');
  assert.equal(requests.at(-1).headers, undefined, 'validators belong to their exact URL');
  revisit.close();
});

test('polling and retries share one schedule and recovery cancels a pending retry', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  requests = [];
  const page = open('/scheduled', '', { poll: 1000 });
  t.after(page.close);
  requests.at(-1).respond({ value: 1 }); await settle();
  t.mock.timers.tick(1000);
  requests.at(-1).respond({ detail: 'Offline' }, 503); await settle();
  t.mock.timers.tick(1000);
  assert.equal(requests.length, 2, 'polling must not bypass retry backoff');
  page.query.retry();
  requests.at(-1).respond({ value: 2 }); await settle();
  t.mock.timers.tick(1000);
  assert.equal(requests.length, 4);
  requests.at(-1).respond({ value: 3 }); await settle();
  t.mock.timers.tick(1000);
  assert.equal(requests.length, 5, 'recovery must leave only the next poll scheduled');
});

test('one-shot queries stop retrying after recovery or a rejected request', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  requests = [];
  const page = open('/one-shot');
  t.after(page.close);
  requests.at(-1).respond({ detail: 'Offline' }, 503); await settle();
  page.query.retry();
  requests.at(-1).respond({ value: 1 }); await settle();
  t.mock.timers.tick(3000);
  assert.equal(requests.length, 2);
  page.query.retry();
  requests.at(-1).respond({ detail: 'Requested data not found' }, 404); await settle();
  assert.equal(page.query.error, 'Requested data not found');
  t.mock.timers.tick(10000);
  assert.equal(requests.length, 3);
});

test('gestures retain data, suspend requests, and fetch the final window once', async () => {
  requests = [];
  const selection = source('window-1');
  const page = open('/gesture?end=1', '', { key: () => get(selection) });
  const initial = requests.at(-1);
  page.navigate('/gesture?end=2');
  assert.equal(initial.signal.aborted, false, 'inventory time updates must not interrupt the current request');
  assert.equal(requests.length, 1);
  initial.respond({ window: 1 }); await settle();
  page.query.retry();
  assert.equal(requests.at(-1).url, '/gesture?end=2');
  const old = requests.at(-1);
  flushSync(() => set(selection, false));
  assert.equal(old.signal.aborted, true);
  for (const end of [2, 3, 4]) page.navigate('/gesture?end=' + end);
  assert.equal(requests.length, 2);
  assert.equal(page.query.data.window, 1);
  flushSync(() => set(selection, 'window-4'));
  assert.equal(requests.length, 3);
  assert.equal(requests.at(-1).url, '/gesture?end=4');
  old.respond({ window: 'obsolete' }); await settle();
  assert.equal(page.query.data.window, 1);
  page.close();
});

test('a paused poll resumes without changing the query key', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  requests = [];
  let live = false;
  const page = open('/pause-resume', '', { poll: 1000, pollWhen: () => live });
  t.after(page.close);
  requests.at(-1).respond({ value: 1 }); await settle();
  t.mock.timers.tick(3000);
  assert.equal(requests.length, 1);
  live = true;
  t.mock.timers.tick(1000);
  assert.equal(requests.length, 2);
});

test('replacing inventory objects does not restart an unchanged query selection', async () => {
  requests = [];
  const node = source({ name: 'a' });
  const page = open('/identity', '', { key: () => get(node).name, scope: () => get(node).name });
  const first = requests.at(-1);
  flushSync(() => set(node, { name: 'a' }));
  assert.equal(requests.length, 1);
  assert.equal(first.signal.aborted, false);
  flushSync(() => set(node, { name: 'b' }));
  assert.equal(first.signal.aborted, true);
  assert.equal(requests.length, 2);
  page.close();
});

test('partial metrics retain only failed fields, including across cached revisits', async () => {
  requests = [];
  const previous = {
    devices: [{ id: 'd', uuid: 'gpu', utilization: [[1, 60]], memory_mib: [[1, 2]] }],
    workers: [{ id: 'w', pod: 'pod', pod_uid: 'uid', cpu_cores: [[1, 2]], memory_bytes: [[1, 10]] }],
    samplers: { requests_waiting: [[1, 1]], requests_running: [[1, 4]] },
  };
  const next = {
    devices: [{ id: 'd', uuid: 'gpu', utilization: [], memory_mib: [], errors: { memory_mib: 'Offline' } }],
    workers: [{ id: 'w', pod: 'pod', pod_uid: 'uid', cpu_cores: [], memory_bytes: [[2, 20]], errors: { cpu_cores: 'Offline' } }],
    samplers: { requests_waiting: [], requests_running: [[2, 8]] },
    samplers_errors: { requests_waiting: 'Offline' },
  };
  const page = open('/partial-series', 'a', { merge: mergeSeries });
  requests.at(-1).respond(previous); await settle();
  page.query.retry();
  requests.at(-1).respond(next); await settle();
  assert.deepEqual(page.query.data.devices[0].utilization, []);
  assert.deepEqual(page.query.data.devices[0].memory_mib, [[1, 2]]);
  assert.deepEqual(page.query.data.workers[0].cpu_cores, [[1, 2]]);
  assert.deepEqual(page.query.data.workers[0].memory_bytes, [[2, 20]]);
  assert.deepEqual(page.query.data.samplers.requests_waiting, [[1, 1]]);
  assert.deepEqual(page.query.data.samplers.requests_running, [[2, 8]]);
  assert.deepEqual(next.devices[0].memory_mib, [], 'network response is not mutated');
  assert.deepEqual(previous.devices[0].utilization, [[1, 60]], 'old cached response is not mutated');
  page.close();
  const revisit = open('/partial-series', 'a', { merge: mergeSeries });
  assert.deepEqual(revisit.query.data.devices[0].memory_mib, [[1, 2]]);
  assert.equal(revisit.query.data.devices[0].errors.memory_mib, 'Offline');
  revisit.close();
});

test('partial failures never carry samples across GPU generations, pod generations or query scopes', async () => {
  const previous = {
    devices: [{ id: 'd', uuid: 'old-gpu', utilization: [[1, 60]] }],
    workers: [{ id: 'w', pod: 'pod', pod_uid: 'old-uid', cpu_cores: [[1, 2]] }],
    samplers: { requests_running: [[1, 4]] },
  };
  const next = {
    devices: [{ id: 'd', uuid: 'new-gpu', utilization: [], errors: { utilization: 'Offline' } }],
    workers: [{ id: 'w', pod: 'pod', pod_uid: 'new-uid', cpu_cores: [], errors: { cpu_cores: 'Offline' } }],
    samplers: { requests_running: [] }, samplers_errors: { requests_running: 'Offline' },
  };
  const merged = mergeSeries(next, previous);
  assert.deepEqual(merged.devices[0].utilization, []);
  assert.deepEqual(merged.workers[0].cpu_cores, []);
  requests = [];
  const page = open('/series-run-a', 'a', { merge: mergeSeries });
  requests.at(-1).respond(previous); await settle();
  page.navigate('/series-run-b', 'b');
  requests.at(-1).respond(next); await settle();
  assert.deepEqual(page.query.data.samplers.requests_running, []);
  page.close();
});
