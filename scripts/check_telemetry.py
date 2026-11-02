"""Exercise live charts and inspect downloaded CSV/JSON/PNG in installed Chrome."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


def main(args):
    token = Path("access.token").read_text().strip()
    output = Path("runtime")
    output.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 1100}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://127.0.0.1:8000")
        assert page.locator("#export-json").is_disabled()
        page.locator("#address").fill(args.url)
        page.locator("#token").fill(token)
        page.locator("#connect").click()
        page.wait_for_function("Number(document.getElementById('telemetry-sample-count').textContent) >= 4", timeout=15000)
        assert page.locator("#telemetry-status").inner_text().startswith("Live")
        for metric in ("power-now", "cpu-now", "gpu-now", "cpu-clock-now", "gpu-clock-now"):
            assert page.locator(f"#{metric}").inner_text() != "—", metric
        downloads = {}
        for button, extension in (("export-csv", "csv"), ("export-json", "json"), ("export-charts", "png")):
            with page.expect_download() as event:
                page.locator(f"#{button}").click()
            download = event.value
            assert download.suggested_filename.endswith("." + extension)
            path = output / f"telemetry-export.{extension}"
            download.save_as(path)
            downloads[extension] = path
        report = json.loads(downloads["json"].read_text())
        samples = report["samples"]
        assert len(samples) >= 4
        assert report["units"] == {"power": "W", "clocks": "MHz", "utilization": "%"}
        assert 0 < samples[-1]["power"]["input_w"] < 100
        assert 0 <= samples[-1]["cpu"]["usage_pct"] <= 100
        assert 0 <= samples[-1]["gpu"]["usage_pct"] <= 100
        assert len(samples[-1]["cpu"]["cores"]) == 6
        assert token not in downloads["json"].read_text()
        assert token not in downloads["csv"].read_text()
        with downloads["csv"].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) >= 4
        assert "cpu5_clock_mhz" in rows[0]
        assert rows[0]["source_url"] == args.url
        with Image.open(downloads["png"]) as image:
            assert image.size == (1600, 760) and image.format == "PNG"
        page.locator("#token").fill("")
        page.locator("#telemetry").screenshot(path=str(output / "telemetry-dashboard.png"))
        page.locator("#disconnect").click()
        page.wait_for_function("document.getElementById('disconnect').disabled")
        assert page.locator("#export-json").is_enabled()
        count = int(page.locator("#telemetry-sample-count").inner_text())
        page.locator("#token").fill(token)
        page.locator("#connect").click()
        page.wait_for_function("count => Number(document.getElementById('telemetry-sample-count').textContent) > count", arg=count, timeout=10000)
        assert page.evaluate("new Set(telemetryDashboard.samples.map(s => s.session_id)).size") == 2
        page.locator("#disconnect").click()
        page.wait_for_function("document.getElementById('disconnect').disabled")
        # Retention is bounded even during a long capture. Avoid rendering 3601 times.
        retained = page.evaluate("""() => {
            const dashboard = telemetryDashboard;
            const render = dashboard.render;
            dashboard.render = () => {};
            dashboard.beginSession('ws://192.168.1.36:8765/stream');
            const sample = structuredClone(dashboard.samples.at(-1));
            for (let i = 0; i < 3602; i++) {
                dashboard.add({...sample, sample_id: 10000 + i, timestamp_ms: sample.timestamp_ms + i * 1000});
            }
            dashboard.render = render;
            dashboard.disconnect();
            return dashboard.samples.length;
        }""")
        assert retained == 3600
        page.locator("#clear-telemetry").click()
        assert page.locator("#telemetry-sample-count").inner_text() == "0"
        assert page.locator("#export-csv").is_disabled()
        assert not errors, errors
        browser.close()
    result = {"live_samples": len(samples), "cpu_cores": 6, "units": report["units"],
              "exports": ["CSV", "JSON", "PNG"], "token_in_exports": False,
              "history_after_disconnect": "passed", "reconnect": "passed", "retention_limit": retained,
              "javascript_errors": errors, "sample": samples[-1]}
    (output / "telemetry-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://192.168.1.36:8765/stream")
    main(parser.parse_args())
