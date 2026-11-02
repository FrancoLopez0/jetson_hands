import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_energy import analyze


DATA = Path(__file__).resolve().parents[1] / "measurements/2026-11-02-single-run"
CSV = DATA / "jetson-telemetry-2026-11-02T02-53-03-475Z.csv"
JSON = DATA / "jetson-telemetry-2026-11-02T02-53-02-491Z.json"


def test_single_recording_integration_and_export_consistency():
    report = analyze(CSV, JSON)
    assert report["independent_measurement_runs"] == 1
    assert report["csv_samples"] == 87 and report["matching_json_samples"] == 86
    assert report["duration_s"] == pytest.approx(86.52)
    assert report["energy_j"] == pytest.approx(662.171774)
    assert report["energy_wh"] == pytest.approx(0.183936603889)
    assert report["vdd_in_w"]["time_weighted_mean"] == pytest.approx(7.65339544614)
    assert report["gpu_clock_mhz_values"] == [306]


def test_archived_hashes_and_derived_report_are_reproducible():
    report = analyze(CSV, JSON)
    saved = json.loads((DATA / "analysis.json").read_text())
    for name, path in (("csv", CSV), ("json", JSON)):
        assert saved["sources"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report == saved


def test_battery_estimates_distinguish_measurement_from_assumptions():
    report = analyze(CSV, JSON)
    pack = next(pack for pack in report["packs_18650"] if pack["configuration"] == "3S2P")
    tool = next(pack for pack in report["tool_packs"] if pack["capacity_ah"] == 5)
    assert pack["nominal_wh"] == pytest.approx(75.6)
    assert pack["runtime_h_at_measured_module_power"] == pytest.approx(7.11213740137)
    assert tool["nominal_wh"] == 90
    assert tool["runtime_h_at_unmeasured_10w_scenario"] == pytest.approx(6.48)
    assert report["battery_assumptions"]["whole_system_10w_is_measured"] is False


def test_conflicting_exports_are_rejected(tmp_path):
    altered = json.loads(JSON.read_text())
    altered["samples"][0]["power"]["input_w"] += 0.5
    path = tmp_path / "conflicting.json"
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="disagree"):
        analyze(CSV, path)


def test_nonincreasing_monotonic_timestamps_are_rejected(tmp_path):
    with CSV.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    rows[1]["monotonic_ms"] = rows[0]["monotonic_ms"]
    path = tmp_path / "duplicate-time.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="Timestamps"):
        analyze(path, JSON)
