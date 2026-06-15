#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a real running Streamlit dashboard screenshot.")
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--output", default="outputs/demo_screenshots/streamlit_dashboard_home.png")
    parser.add_argument(
        "--browser_executable",
        default=str(ROOT / ".playwright-browsers/chromium-1223/chrome-linux64/chrome"),
        help="Chromium executable path. Install with PLAYWRIGHT_BROWSERS_PATH=.playwright-browsers python -m playwright install chromium.",
    )
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1300)
    parser.add_argument("--clip_x", type=int, default=320)
    parser.add_argument("--clip_y", type=int, default=36)
    parser.add_argument("--clip_width", type=int, default=1040)
    parser.add_argument("--clip_height", type=int, default=930)
    parser.add_argument("--no_click", action="store_true", help="Only capture the loaded dashboard, without scoring.")
    parser.add_argument("--full_page", action="store_true", help="Capture the full browser viewport instead of the main content crop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    browser_executable = Path(args.browser_executable)
    if not browser_executable.exists():
        raise FileNotFoundError(f"Chromium executable not found: {browser_executable}")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(browser_executable),
            headless=True,
            args=["--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        expect(page.get_by_text("中文金融舆情风险智能预警系统")).to_be_visible(timeout=60_000)

        if not args.no_click:
            page.get_by_role("button", name="风险识别").click(timeout=30_000)
            expect(page.locator(".risk-card").first).to_be_visible(timeout=180_000)
            expect(page.locator(".profile-card").first).to_be_visible(timeout=180_000)

        if args.full_page:
            page.screenshot(path=str(output), full_page=False)
        else:
            page.screenshot(
                path=str(output),
                full_page=False,
                clip={"x": args.clip_x, "y": args.clip_y, "width": args.clip_width, "height": args.clip_height},
            )
        browser.close()

    print(output)


if __name__ == "__main__":
    main()
