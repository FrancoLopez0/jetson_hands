"""Audit the single 2026-11-02 power capture and reproduce battery estimates."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


SEGMENTS = {
    "higher_gpu_before": (None, "2026-11-02T02:52:10"),
    "lower_gpu_plateau": ("2026-11-02T02:52:13", "2026-11-02T02:52:34"),
    "higher_gpu_after": ("2026-11-02T02:52:44", None),
}
FIELDS = ("power_input_w", "power_cpu_gpu_cv_w", "power_soc_w", "cpu_usage_pct",
          "cpu_clock_mean_mhz", "gpu_usage_pct", "gpu_clock_mhz")


def source_info(path):
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def analyze(csv_path, json_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    samples = exported["samples"]
    if len(rows) < 2 or not samples:
        raise ValueError("At least two CSV rows and one matching JSON sample are required")
    if len({row["session_id"] for row in rows}) != 1:
        raise ValueError("This analysis is for one recording session only")
    times = [int(row["monotonic_ms"]) / 1000 for row in rows]
    powers = [float(row["power_input_w"]) for row in rows]
    intervals = [b - a for a, b in zip(times, times[1:])]
    if any(value <= 0 for value in intervals) or any(not math.isfinite(p) or p < 0 for p in powers):
        raise ValueError("Timestamps must increase and power must be finite and nonnegative")
    lookup = {int(row["monotonic_ms"]): row for row in rows}
    for sample in samples:
        row = lookup.get(sample["monotonic_ms"])
        if row is None or row["session_id"] != sample["session_id"]:
            raise ValueError("JSON sample has no matching CSV sample/session")
        values = {"power_input_w": sample["power"]["input_w"],
                  "power_cpu_gpu_cv_w": sample["power"]["rails_w"]["VDD_CPU_GPU_CV"],
                  "power_soc_w": sample["power"]["rails_w"]["VDD_SOC"],
                  "cpu_usage_pct": sample["cpu"]["usage_pct"],
                  "cpu_clock_mean_mhz": sample["cpu"]["clock_mhz"],
                  "gpu_usage_pct": sample["gpu"]["usage_pct"],
                  "gpu_clock_mhz": sample["gpu"]["clock_mhz"]}
        for core in sample["cpu"]["cores"]:
            for field in ("usage_pct", "clock_mhz"):
                values[f"cpu{core['id']}_{field}"] = core[field]
        if any(not math.isclose(float(row[key]), value, abs_tol=1e-8) for key, value in values.items()):
            raise ValueError("CSV and JSON disagree on a numeric measurement")
    duration = times[-1] - times[0]
    energy_j = sum((a + b) / 2 * interval for a, b, interval in zip(powers, powers[1:], intervals))
    mean_power = energy_j / duration
    segments = {}
    for name, (start, end) in SEGMENTS.items():
        selected = [row for row in rows if (start is None or row["timestamp_utc"] >= start)
                    and (end is None or row["timestamp_utc"] < end)]
        if selected:
            segments[name] = {"samples": len(selected), "start_utc": selected[0]["timestamp_utc"],
                              "end_utc": selected[-1]["timestamp_utc"],
                              "sample_means": {field: statistics.mean(float(row[field]) for row in selected)
                                               for field in FIELDS}}
    efficiency, usable_fraction, cell_voltage, cell_ah = 0.9, 0.8, 3.6, 3.5

    def battery(energy):
        return {"nominal_wh": energy,
                "runtime_h_at_measured_module_power": energy * efficiency * usable_fraction / mean_power,
                "runtime_h_at_unmeasured_10w_scenario": energy * efficiency * usable_fraction / 10}

    return {
        "independent_measurement_runs": 1,
        "scope": "One short exploratory module-power recording; not whole-system or repeated-run validation",
        "sources": {"csv": source_info(csv_path), "json": source_info(json_path)},
        "csv_samples": len(rows), "json_samples": len(samples), "matching_json_samples": len(samples),
        "start_utc": rows[0]["timestamp_utc"], "end_utc": rows[-1]["timestamp_utc"],
        "duration_s": duration, "interval_s": {"min": min(intervals), "max": max(intervals),
                                                "mean": statistics.mean(intervals)},
        "vdd_in_w": {"time_weighted_mean": mean_power, "sample_mean": statistics.mean(powers),
                     "min": min(powers), "max": max(powers), "population_stddev": statistics.pstdev(powers)},
        "energy_j": energy_j, "energy_wh": energy_j / 3600,
        "gpu_clock_mhz_values": sorted({float(row["gpu_clock_mhz"]) for row in rows}),
        "segments": segments,
        "battery_assumptions": {"converter_efficiency": efficiency, "usable_energy_fraction": usable_fraction,
                                "cell_nominal_v": cell_voltage, "cell_typical_ah": cell_ah,
                                "whole_system_10w_is_measured": False},
        "packs_18650": [{"configuration": f"{series}S{parallel}P", "cells": series * parallel,
                         "nominal_v": series * cell_voltage, "capacity_ah": parallel * cell_ah,
                         **battery(series * parallel * cell_voltage * cell_ah)}
                        for series, parallel in ((2, 1), (3, 1), (4, 1), (3, 2), (4, 2), (5, 2), (6, 2))],
        "tool_packs": [{"nominal_v": 18, "capacity_ah": ah, **battery(18 * ah)} for ah in (2, 3, 4, 5, 6, 8)],
    }


def main(args):
    report = analyze(args.csv, args.json)
    if args.archive_dir:
        args.archive_dir.mkdir(parents=True, exist_ok=True)
        for source in (args.csv, args.json):
            destination = args.archive_dir / source.name
            if source.resolve() != destination.resolve():
                if destination.exists() and source.read_bytes() != destination.read_bytes():
                    raise FileExistsError(f"Refusing to replace a different archived input: {destination}")
                shutil.copyfile(source, destination)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        if not args.output.parent.is_dir():
            raise FileNotFoundError("Output parent directory must already exist")
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args())
