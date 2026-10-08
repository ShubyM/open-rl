import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

const snapshot = JSON.parse(readFileSync(new URL("../fixtures/dashboard/snapshot.json", import.meta.url)));
const runId = "1390ffaa-2cbb-499d-bfe6-8e94bed23a9d";
const run = snapshot.runs.find((r) => r.run_id === runId);
const clone = () => structuredClone(snapshot);

async function open(page, hash = "overview", state) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  if (state) await page.route("**/api/v1/dashboard/snapshot", (route) => route.fulfill({ json: state }));
  await page.goto(`/dashboard#${hash}`);
  await expect(page.locator("h1")).toBeVisible();
  return errors;
}

async function repaint(page, changes = {}) {
  await page.evaluate(async (changes) => {
    const { ui } = await import("/dashboard/assets/store.js");
    Object.assign(ui.state, changes);
    ui.render();
  }, changes);
}

for (const width of [1280, 800, 600, 390, 320]) {
  test(`pages fit at ${width}px with readable run identities`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    const errors = await open(page);
    for (const hash of ["overview", "nodes", "scheduler", "experiments", "health", `run/${runId}/metrics`, `run/${runId}/logs`]) {
      await page.evaluate((hash) => { location.hash = hash; }, hash);
      await expect(page.locator("h1")).toBeVisible();
      if (hash === "experiments") await expect(page.locator(".experiment-row").first()).toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    await page.goto("/dashboard#overview");
    await expect(page.locator(".run-row .job-identity").first()).toBeVisible();
    const identity = await page.locator(".run-row .job-identity").first().boundingBox();
    expect(identity.width).toBeGreaterThan(130);
    if (width <= 960) {
      expect((await page.locator(".job-list-row.run-row").first().boundingBox()).height).toBeLessThan(130);
      await expect(page.getByRole("checkbox", { name: "Select all shown runs that can be deleted" })).toBeVisible();
    }
    expect(errors).toEqual([]);
  });
}

test("search preserves focus and invalid status links recover", async ({ page }) => {
  await open(page, "overview?status=typo");
  await expect(page.locator(".job-list-row.run-row")).toHaveCount(snapshot.runs.length);
  const input = page.getByRole("searchbox", { name: "Search runs" });
  await input.pressSequentially("1390ff");
  await expect(input).toBeFocused();
  await expect(input).toHaveValue("1390ff");
  await expect(page.locator(".job-list-row.run-row")).toHaveCount(1);
  await repaint(page);
  await expect(input).toBeFocused();
});

test("recordings cannot offer deletion", async ({ page }) => {
  await open(page);
  await expect(page.locator("[data-delete-runs]")).toHaveCount(0);
  expect(await page.locator("[data-select-run]:enabled").count()).toBe(0);
});

test("bulk selection is mixed and pending deletion stays disabled across renders", async ({ page }) => {
  const state = clone();
  state.runs.forEach((r) => { r.delete_blocker = null; });
  await open(page, "overview", state);
  await page.locator("[data-select-run]").first().check();
  const all = page.locator("[data-select-all-runs]");
  expect(await all.evaluate((el) => el.indeterminate)).toBe(true);
  await all.check();
  await expect(page.locator("[data-delete-runs]")).toHaveText(`Delete ${state.runs.length} runs`);
  let finish;
  const pending = new Promise((resolve) => { finish = resolve; });
  await page.route("**/runs/delete", async (route) => {
    await pending;
    await route.fulfill({ json: { deleted: [], kept: state.runs.map((r) => ({ run_id: r.run_id, reason: "Still live" })) } });
  });
  page.on("dialog", (dialog) => dialog.accept());
  await page.locator("[data-delete-runs]").click();
  await repaint(page);
  await expect(page.locator("[data-delete-runs]")).toBeDisabled();
  finish();
  await expect(page.locator(".run-notice")).toContainText("Still live");
  await expect(page.locator("[data-select-run]:checked")).toHaveCount(state.runs.length);
});

test("a new run resolves from detail before it appears in a snapshot", async ({ page }) => {
  const state = clone();
  state.runs = state.runs.filter((r) => r.run_id !== runId);
  await page.route(`**/runs/${runId}`, (route) => route.fulfill({ json: run }));
  const errors = await open(page, `run/${runId}/activity`, state);
  await expect(page.locator("h1")).toContainText(runId.slice(0, 8));
  expect(errors).toEqual([]);
});

test("run and log failures offer a working retry", async ({ page }) => {
  const state = clone();
  state.runs = [];
  let available = false;
  await page.route(`**/runs/${runId}`, (route) => route.fulfill(available ? { json: run } : { status: 503, json: { detail: "Store temporarily unavailable" } }));
  await open(page, `run/${runId}/activity`, state);
  await expect(page.locator("#content")).toContainText("Store temporarily unavailable");
  available = true;
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(page.locator("h1")).toContainText(runId.slice(0, 8));
  let logsAvailable = false;
  await page.route(`**/runs/${runId}/logs?*`, (route) => route.fulfill(logsAvailable ? { json: { records: [] } } : { status: 503, json: { detail: "Logs temporarily unavailable" } }));
  await page.getByRole("link", { name: "Logs", exact: true }).click();
  await expect(page.locator("#log-status")).toContainText("Logs temporarily unavailable");
  logsAvailable = true;
  await page.getByRole("button", { name: "Retry logs" }).click();
  await expect(page.locator("#log-status")).not.toContainText("Logs temporarily unavailable");
  await expect(page.locator("#log-lines")).toContainText("No logs match");
});

test("opening a run from a scrolled list starts at its heading", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 500 });
  await open(page);
  await page.locator(".job-list-row.run-row").last().scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => scrollY)).toBeGreaterThan(0);
  await page.locator(".job-list-row.run-row").last().getByRole("link").click();
  await expect(page.locator(".run-heading")).toBeInViewport();
});

test("history keeps device mapping through claim teardown", async ({ page }) => {
  const state = clone(), placement = state.placements[0];
  const past = state.history.find((p) => p.id === placement.id);
  expect(past.devices.length).toBeGreaterThan(0);
  placement.devices = [];
  await open(page, `nodes?node=${encodeURIComponent(placement.node)}`, state);
  const group = page.locator(".node-placement-group").filter({ has: page.locator("#placement-detail") });
  await expect(group.locator(".unknown-mapping")).toHaveCount(0);
  await expect(group.locator(".capacity-track [data-placement]").first()).toBeVisible();
});

test("unavailable telemetry stays visible and expired run links are disabled", async ({ page }) => {
  const state = clone(), placement = state.placements[0];
  state.runs = [];
  await page.route("**/runs/*/metrics?*", (route) => route.fulfill({ status: 503, json: { detail: "Metrics offline" } }));
  await open(page, `nodes?node=${encodeURIComponent(placement.node)}`, state);
  const detail = page.locator("#placement-detail");
  await expect(detail.locator(".activity-row").first()).toBeVisible();
  await expect(detail.locator(".activity-label").first()).toContainText("Run record unavailable");
  await expect(detail.locator('a[href^="#run/"]')).toHaveCount(0);
  await expect(detail).toContainText("Metrics offline");
  await expect(detail).not.toContainText("No recorded activity in this window");
});

test("a scheduler outage does not label allocations ended", async ({ page }) => {
  const state = clone();
  state.placements = [];
  state.cluster.scheduler.available = false;
  await open(page, `run/${runId}/activity`, state);
  await expect(page.locator(".activity-meta").first()).toBeVisible();
  await expect(page.locator("#run-panel")).not.toContainText("Ended");
  await expect(page.locator("#run-panel")).not.toContainText("Allocation no longer reported");
});

test("clicking an operation selects its run without changing GPU or time", async ({ page }) => {
  const state = clone(), placement = state.placements.find((p) => p.run_ids.includes(runId));
  const now = Date.parse(state.observed_at) / 1000;
  await page.route("**/allocations/*/turns?*", (route) => route.fulfill({ json: { samples: [
    { started_at: now - 600, at: now - 60, ops: [{ name: "sample", run_id: runId, start: now - 550, end: now - 100 }] },
  ] } }));
  await open(page, `nodes?node=${encodeURIComponent(placement.node)}&duration=1800&end=${now}`, state);
  const op = page.locator("#placement-detail .activity-block.op").first();
  await expect(op).toBeVisible();
  const before = new URLSearchParams(new URL(page.url()).hash.split("?")[1]);
  await op.click();
  await expect(page.locator('#placement-detail .activity-row[data-selected="true"]')).toHaveCount(1);
  const after = new URLSearchParams(new URL(page.url()).hash.split("?")[1]);
  for (const name of ["node", "gpu", "duration", "end"]) expect(after.get(name)).toBe(before.get(name));
});

test("minute end times remain valid for custom durations", async ({ page }) => {
  const end = Date.parse(snapshot.observed_at) / 1000;
  await open(page, `nodes?duration=601&end=${end}`);
  await page.locator(".time-summary").click();
  const input = page.locator("[data-time-end]");
  await input.fill("2026-09-11T08:50");
  await input.press("Tab");
  expect(await input.evaluate((el) => el.validity.valid)).toBe(true);
  await expect(page).toHaveURL(/end=1789116600/);
  await repaint(page);
  await expect(page.locator(".time-picker")).toHaveAttribute("open", "");
});

test("a focused time selector keeps its value when the custom option disappears", async ({ page }) => {
  await open(page, "nodes?duration=876");
  await page.locator(".time-summary").click();
  const select = page.locator("[data-time-duration]");
  await select.focus();
  await select.selectOption("60");
  await expect(page).toHaveURL(/duration=60(?:&|$)/);
  await expect(select).toBeFocused();
  await expect(select).toHaveValue("60");
  await expect(select.locator("option:checked")).toHaveText("1 minute");
});

test("late logs from the previous run cannot replace the current run", async ({ page }) => {
  let release;
  const pending = new Promise((resolve) => { release = resolve; });
  await page.route(`**/runs/${runId}/logs?*`, async (route) => {
    await pending;
    await route.fulfill({ json: { records: [{ timestamp: snapshot.observed_at, message: "Wrong run" }] } });
  });
  await open(page, `run/${runId}/logs`);
  await page.evaluate(() => { location.hash = "run/f332cde2-3266-4aa2-9ed3-3739530b3f03/logs"; });
  release();
  await expect(page.locator("h1")).toContainText("f332cde2");
  await expect(page.locator("#log-status")).toContainText("Recorded logs");
  await expect(page.locator("#log-lines")).not.toContainText("Wrong run");
});
