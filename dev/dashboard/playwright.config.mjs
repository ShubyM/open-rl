import { defineConfig } from "@playwright/test";
import { fileURLToPath } from "node:url";

export default defineConfig({
  testMatch: "*.spec.mjs",
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:9018",
    viewport: { width: 1280, height: 900 },
    colorScheme: "light",
    launchOptions: { executablePath: process.env.DASHBOARD_CHROMIUM || undefined },
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "uv run --frozen --no-sync python dev/dashboard_fixture.py --port 9018",
    cwd: fileURLToPath(new URL("../../", import.meta.url)),
    url: "http://127.0.0.1:9018/dashboard",
  },
});
