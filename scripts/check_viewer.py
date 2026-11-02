"""Run on the PC with: uv run --with playwright python scripts/check_viewer.py."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main(args):
    token = Path("access.token").read_text().strip()
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000")
        page.locator("#address").fill(args.url)
        page.locator("#token").fill(token)
        page.locator("#connect").click()
        page.wait_for_function("document.getElementById('status').textContent.startsWith('Live')", timeout=30000)
        page.wait_for_timeout(3000)
        assert page.locator("#placeholder").is_hidden()
        assert page.locator("#fps").inner_text() != "—"
        # The viewer sets canvas.width once for each decoded/drawn frame.
        page.evaluate("""() => {
            window.frameCounter = {count: 0, start: performance.now()};
            window.frameObserver = new MutationObserver(records => {
                window.frameCounter.count += records.length;
            });
            window.frameObserver.observe(document.getElementById('video'),
                {attributes: true, attributeFilter: ['width']});
        }""")
        page.wait_for_timeout(args.seconds * 1000)
        rate = page.evaluate("""() => {
            window.frameObserver.disconnect();
            const duration = (performance.now() - window.frameCounter.start) / 1000;
            return {frames: window.frameCounter.count, duration_s: duration,
                displayed_fps: window.frameCounter.count / duration};
        }""")
        measurements = {name: page.locator(f"#{name}").inner_text()
                        for name in ("fps", "hands", "inference", "pipeline", "roundtrip")}
        other = context.new_page()
        other.goto("http://127.0.0.1:8000")
        other.locator("#address").fill(args.url)
        other.locator("#token").fill(token)
        other.locator("#connect").click()
        other.wait_for_function("document.getElementById('status').textContent === 'Another receiver is connected'")
        other.locator("#token").fill("incorrect-token")
        other.locator("#connect").click()
        other.wait_for_function("document.getElementById('status').textContent === 'Invalid access token'")
        page.locator("#token").fill("")
        output = Path("runtime")
        output.mkdir(exist_ok=True)
        page.screenshot(path=str(output / "viewer.png"), full_page=True)
        page.locator("#disconnect").click()
        page.wait_for_function("document.getElementById('disconnect').disabled")
        assert not errors, errors
        browser.close()
        report = {"browser": "Chrome", "live_video": "passed", "second_receiver": "rejected",
                  "invalid_token": "rejected", "javascript_errors": errors,
                  "metrics": measurements, "measurement": rate}
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://192.168.1.36:8765/stream")
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--report")
    main(parser.parse_args())
