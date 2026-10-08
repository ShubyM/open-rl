import { test, expect } from "@playwright/test";
import { build } from "esbuild";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

test("Preact Overview works with the existing dashboard interactions", async ({ page }) => {
  const bundle = await build({ entryPoints: [fileURLToPath(new URL("overview.preact.js", import.meta.url))], bundle: true, minify: true, format: "esm", external: ["/dashboard/assets/*"], write: false });
  const app = readFileSync(new URL("../../src/server/dashboard/static/app.js", import.meta.url), "utf8");
  await page.route("**/dashboard/assets/app.js", (route) => route.fulfill({ contentType: "text/javascript", body:
    'import { paint } from "/benchmark/preact.js";\n' + app.replace("else morph(content, runs(ui.state));", "else paint(content, ui.state);"),
  }));
  await page.route("**/benchmark/preact.js", (route) => route.fulfill({ contentType: "text/javascript", body: bundle.outputFiles[0].text }));
  const snapshot = JSON.parse(readFileSync(new URL("../fixtures/dashboard/snapshot.json", import.meta.url)));
  await page.route("**/api/v1/dashboard/snapshot", (route) => route.fulfill({ json: snapshot }));
  await page.goto("/dashboard#overview");
  await expect(page.locator(".job-list-row.run-row")).toHaveCount(snapshot.runs.length);
  const search = page.getByRole("searchbox", { name: "Search runs" });
  await search.pressSequentially("1390ff");
  await expect(search).toBeFocused();
  await expect(page.locator(".job-list-row.run-row")).toHaveCount(1);
  await search.fill("");
  await page.getByRole("button", { name: "Finished", exact: false }).click();
  await page.locator("[data-select-run]").first().check();
  await expect(page.locator("[data-delete-runs]")).toHaveText("Delete 1 run");
  expect(await page.locator("[data-select-all-runs]").evaluate((el) => el.indeterminate)).toBe(true);
});
