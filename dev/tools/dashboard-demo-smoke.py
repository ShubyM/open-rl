"""Browser checks for fleet/run navigation, live inspection, and diagnostic handoff.

Requires a gateway started with OPEN_RL_DASHBOARD_DEMO=1 and Playwright Chromium.
"""

import json
import os

from playwright.sync_api import expect, sync_playwright

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:9003").rstrip("/")


def main():
  with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 960})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script("Object.defineProperty(navigator, 'clipboard', {value: {writeText: async text => {window.copied = text}}})")
    page.goto(f"{BASE_URL}/dashboard")
    page.wait_for_function("document.querySelector('#updated-at').textContent.includes('updated')")
    expect(page.locator("#demo-banner")).to_be_visible()
    expect(page.locator(".placement-row")).to_have_count(3)
    expect(page.locator(".placement-row").first).to_contain_text("sft-gemma-warmup")
    page.locator("#copy-snapshot").click()
    page.wait_for_function("window.copied")
    snapshot = json.loads(page.evaluate("window.copied"))
    assert snapshot["schema_version"] == 1 and snapshot["demo"]
    page.locator(".placement-identity").filter(has_text="math-rl-qwen3-8b").click()
    expect(page.locator("#view-runs")).to_be_visible()
    expect(page.locator('.run-row[data-key="demo-run-1"] .run-detail-head')).to_be_visible()
    expect(page.locator('.run-row[data-key="demo-run-1"] .worker-metrics svg')).to_have_count(2)
    page.get_by_role("button", name="Copy run link", exact=True).click()
    page.wait_for_function("window.copied.includes('/dashboard?run=')")
    run_url = page.evaluate("window.copied")
    assert run_url.endswith("?run=demo-run-1")
    # The global refresh must update an already-open inspection, not only the list.
    with page.expect_response(lambda response: response.url.endswith("/runs/demo-run-1")):
      page.locator("#refresh").click()
    page.get_by_role("button", name="Copy diagnostic JSON", exact=True).click()
    page.wait_for_function("window.copied.startsWith('{')")
    detail = json.loads(page.evaluate("window.copied"))
    assert detail["run_id"] == "demo-run-1" and detail["pods"] and detail["inspected_at"]
    page.get_by_role("button", name="Logs", exact=True).click()
    expect(page.locator(".log-record").first).to_be_visible()
    page.locator("#run-logs-follow").uncheck()
    page.locator("#run-logs-search").fill("step=2")
    page.get_by_role("button", name="Apply", exact=True).click()
    expect(page.locator("#run-logs-status")).not_to_have_text("Loading logs…")
    assert all("step=2" in text for text in page.locator(".log-record > .log-record-message").all_text_contents())
    page.locator(".log-record-source").first.click()
    expect(page.locator(".log-record-metadata").first).to_be_visible()
    page.locator("#run-logs-json").click()
    page.wait_for_function("window.copied.includes('best_effort_polling')")
    assert json.loads(page.evaluate("window.copied"))["query"]["q"] == "step=2"
    page.locator("#run-logs-link").click()
    page.wait_for_function("window.copied.includes('view=logs')")
    log_link = page.evaluate("window.copied")
    page.goto(log_link)
    expect(page.locator("#run-logs-search")).to_have_value("step=2")
    expect(page.locator(".log-record").first).to_be_visible()
    page.locator("#run-logs-back").click()
    page.get_by_role("button", name="Logs near latest sample").first.click()
    assert page.locator("#run-logs-since").input_value()
    assert not page.locator("#run-logs-follow").is_checked()
    page.locator("#run-logs-back").click()
    page.locator("#run-search").fill("demo-l4-node-1")
    expect(page.locator(".run-row")).to_have_count(1)
    expect(page.locator(".run-row")).to_contain_text("sft-gemma-warmup")
    page.locator("#run-search").fill("no-such-job")
    expect(page.locator("#runs-empty")).to_have_text("No runs match these filters.")
    page.locator("#run-search").fill("")
    page.locator("#run-filter").select_option("attention")
    expect(page.locator(".run-row")).to_have_count(1)
    page.locator("#run-filter").select_option("running")
    expect(page.locator(".run-row")).to_have_count(1)
    expect(page.locator(".run-row")).to_contain_text("math-rl-qwen3-8b")
    page.goto(run_url)
    expect(page.locator('.run-row[data-key="demo-run-1"] .run-detail-head')).to_be_visible()
    for width in (1440, 390, 320):
      page.set_viewport_size({"width": width, "height": 844})
      for tab in ("cluster", "runs", "health"):
        page.locator(f'[data-tab="{tab}"]').click()
        assert page.evaluate("document.body.scrollWidth <= innerWidth && document.querySelector('main').scrollWidth <= innerWidth"), (width, tab)
      page.locator('[data-tab="cluster"]').click()
      page.locator(".placement-pod").first.click()
      expect(page.locator("#view-logs")).to_be_visible()
      expect(page.locator(".log-record").first).to_be_visible()
      assert page.evaluate("document.body.scrollWidth <= innerWidth"), (width, "logs")
      page.locator('[data-tab="cluster"]').click()
      page.locator(".pool-head").first.click()
      expect(page.locator("#view-pool")).to_be_visible()
      page.locator("#pool-back").click()
    assert not errors, errors
    browser.close()
  print("Dashboard demo browser checks passed (1440, 390, 320px).")


if __name__ == "__main__":
  main()
