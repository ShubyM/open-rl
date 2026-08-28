#!/usr/bin/env python3
"""Real-browser layout smoke for the Kind operations dashboard."""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:9014").rstrip("/")
SCREENSHOT_DIR = os.getenv("DASHBOARD_SCREENSHOT_DIR")
VIEWPORTS = (1440, 390, 320)


def dimensions(page) -> dict:
  return page.evaluate(
    """() => ({
      bodyClient: document.body.clientWidth,
      bodyScroll: document.body.scrollWidth,
      mainClient: document.querySelector("main").clientWidth,
      mainScroll: document.querySelector("main").scrollWidth,
      topClient: document.querySelector(".topbar").clientWidth,
      topScroll: document.querySelector(".topbar").scrollWidth,
      topHeight: document.querySelector(".topbar").getBoundingClientRect().height,
    })"""
  )


def assert_fits(label: str, measured: dict) -> None:
  assert measured["bodyScroll"] <= measured["bodyClient"], (label, measured)
  assert measured["mainScroll"] <= measured["mainClient"], (label, measured)
  assert measured["topScroll"] <= measured["topClient"], (label, measured)
  assert measured["topHeight"] == 48, (label, measured)


def main() -> None:
  screenshots = Path(SCREENSHOT_DIR) if SCREENSHOT_DIR else None
  if screenshots:
    screenshots.mkdir(parents=True, exist_ok=True)

  with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True)
    for width in VIEWPORTS:
      page = browser.new_page(viewport={"width": width, "height": 844})
      page.goto(f"{BASE_URL}/dashboard", wait_until="networkidle")
      page.wait_for_function("document.querySelector('#updated-at').textContent.includes('updated')")

      cards = page.locator("#control-col .card-title").all_text_contents()
      assert "gateway" in cards and "rollouts" in cards, cards
      gateway_card = page.locator("#control-col .card").filter(has=page.locator(".card-title", has_text="gateway")).first
      gateway_text = gateway_card.text_content()
      assert "traffic" in gateway_text and "5xx" in gateway_text, gateway_text
      freshness = page.locator("#updated-at")
      assert freshness.get_attribute("data-compact").startswith("k8s ")
      tooltip = freshness.get_attribute("title")
      assert "total" in tooltip and "pods" in tooltip and "rollouts" in tooltip, tooltip

      for tab in ("cluster", "runs", "health"):
        page.click(f'[data-tab="{tab}"]')
        assert_fits(f"{width}px {tab}", dimensions(page))
        if screenshots:
          page.screenshot(path=screenshots / f"{tab}-{width}.png", full_page=True)

      page.click('[data-tab="cluster"]')
      if page.locator(".pool-head").count():
        page.locator(".pool-head").first.click()
        assert_fits(f"{width}px pool", dimensions(page))
      page.close()
    browser.close()
  print(f"Dashboard browser smoke passed at {', '.join(f'{width}px' for width in VIEWPORTS)}")


if __name__ == "__main__":
  main()
