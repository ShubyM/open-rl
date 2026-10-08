// Same Overview, styles and data; compare renderer CPU time, not network latency.
import { chromium } from "@playwright/test";
import { build } from "esbuild";
import { format } from "prettier";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { execFileSync, spawn } from "node:child_process";
import { gzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const here = fileURLToPath(new URL("./", import.meta.url));
const staticPath = "src/server/dashboard/static/";
const baseline = process.env.DASHBOARD_BASELINE || "5b42aeac";
const bundle = await build({ entryPoints: [here + "overview.preact.js"], bundle: true, minify: true, format: "esm", external: ["/dashboard/assets/*"], write: false });
const runtime = await build({ stdin: { contents: 'export {h, render} from "preact"; export {default as htm} from "htm";', resolveDir: here }, bundle: true, minify: true, format: "esm", write: false });
const source = await readFile(here + "overview.preact.js", "utf8");
const current = await readFile(root + staticPath + "views.js", "utf8");
const formattedLines = async (source) => (await format(source, { parser: "babel", printWidth: 120 })).trim().split("\n").length;
const overviewSource = current.slice(current.indexOf("// Status chips"), current.indexOf("export function scheduler"));
const helpers = await readFile(root + staticPath + "ui.js", "utf8");
const result = {
  baseline,
  scope: "Overview only. Same shared filters, formatting helpers, delegated events and CSS. Warm renderer CPU time including synchronous layout, excluding network and paint.",
  bytes: { preactHtmGzip: gzipSync(runtime.outputFiles[0].contents).length, prototypeBundleGzip: gzipSync(bundle.outputFiles[0].contents).length },
  sourceBytes: { productionOverview: Buffer.byteLength(overviewSource), preactOverview: Buffer.byteLength(source) },
  formattedLines: { productionOverview: await formattedLines(overviewSource), customMorph: await formattedLines(helpers.slice(helpers.indexOf("export function morph"))), preactOverview: await formattedLines(source) },
  scenarios: [],
};
const server = spawn("uv", ["run", "--frozen", "--no-sync", "python", "dev/dashboard_fixture.py", "--port", "9019"], { cwd: root, stdio: "ignore" });
let browser;
try {
  for (let i = 0; ; i++) {
    try { if ((await fetch("http://127.0.0.1:9019/dashboard")).ok) break; } catch {}
    if (i === 100) throw new Error("Fixture server did not start");
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  browser = await chromium.launch({ executablePath: process.env.DASHBOARD_CHROMIUM || undefined });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/dashboard/assets/app.js", (route) => route.fulfill({ body: "", contentType: "text/javascript" }));
  await page.route("**/benchmark/preact.js", (route) => route.fulfill({ body: bundle.outputFiles[0].text, contentType: "text/javascript" }));
  await page.route("**/baseline/*.js", (route) => {
    const name = new URL(route.request().url()).pathname.split("/").at(-1);
    const body = execFileSync("git", ["show", `${baseline}:${staticPath}${name}`], { cwd: root, encoding: "utf8" });
    return route.fulfill({ body, contentType: "text/javascript" });
  });
  await page.goto("http://127.0.0.1:9019/dashboard");
  result.browser = await browser.version();
  result.scenarios = await page.evaluate(async () => {
    const [{ ui }, { runs }, { morph }, preact, oldStore, oldViews, oldUI] = await Promise.all([
      import("/dashboard/assets/store.js"), import("/dashboard/assets/views.js"), import("/dashboard/assets/ui.js"), import("/benchmark/preact.js"),
      import("/baseline/store.js"), import("/baseline/views.js"), import("/baseline/ui.js"),
    ]);
    const recorded = await (await fetch("/api/v1/dashboard/snapshot")).json();
    delete recorded.recorded_at;
    const container = document.getElementById("content");
    const implementations = {
      baseline: { paint: (state) => oldUI.morph(container, oldViews.runs(state)), store: oldStore.ui },
      current: { paint: (state) => morph(container, runs(state)), store: ui },
      preact: { paint: (state) => preact.paint(container, state), store: ui },
    };
    const scenarios = [];
    for (const size of [12, 100, 1000]) for (const scenario of ["poll", "filter"]) {
      const data = { ...recorded, runs: Array.from({ length: size }, (_, i) => ({ ...recorded.runs[i % recorded.runs.length], run_id: `bench-${i}`, delete_blocker: null })) };
      for (const [implementation, { paint, store }] of Object.entries(implementations)) {
        preact.clear(container);
        container.replaceChildren();
        store.runFilter = { q: "", status: "all" };
        store.state = data;
        const times = [];
        for (let i = 0; i < 65; i++) {
          if (scenario === "poll") data.runs[0].steps = i;
          else store.runFilter.q = i % 2 ? "Qwen3-8B" : "";
          const before = performance.now();
          paint(data);
          void container.offsetHeight;
          if (i >= 15) times.push(performance.now() - before);
        }
        times.sort((a, b) => a - b);
        scenarios.push({ size, scenario, implementation, medianMs: +times[25].toFixed(2), p95Ms: +times[47].toFixed(2) });
      }
    }
    // Interaction parity: the prototype uses the existing delegated controls.
    preact.clear(container);
    container.replaceChildren();
    ui.state = recorded;
    ui.runFilter = { q: "", status: "all" };
    preact.paint(container, recorded);
    const search = container.querySelector("#run-search");
    search.focus();
    search.value = "Qwen3-8B";
    ui.runFilter.q = search.value;
    preact.paint(container, recorded);
    if (document.activeElement !== search || search.value !== ui.runFilter.q) throw new Error("Prototype lost input focus/state");
    return scenarios;
  });
  if (errors.length) throw new Error(errors.join("\n"));
  await mkdir(here + "test-results", { recursive: true });
  await writeFile(here + "test-results/benchmark.json", JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser?.close();
  server.kill();
}
