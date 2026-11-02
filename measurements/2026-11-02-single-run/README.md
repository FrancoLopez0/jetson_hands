# Single exploratory power recording

**Independent measurements: ONE.** The CSV and JSON are two exports of the same
recording, not separate experiments. Samples within this trace are not repeated
independent trials. The recording spans 86.52 seconds at approximately 1 Hz.

Original input exports are preserved byte-for-byte:

- `jetson-telemetry-2026-11-02T02-53-03-475Z.csv`: 87 samples.
- `jetson-telemetry-2026-11-02T02-53-02-491Z.json`: 86 samples; exported about one
  second earlier. All matching numeric fields, including per-core clocks and
  utilization, agree with the CSV.
- `analysis.json`: reproducible derived statistics, input SHA256 hashes,
  illustrative state windows and theoretical battery comparisons.

The exports retain the original timestamps, session ID and local source URL.
They contain telemetry only, not the access token, images or hand landmarks.

## Measurement boundary correction

The original export/UI called `VDD_IN` "board input". NVIDIA identifies this
channel as **total module power** for Orin Nano/NX. Preserve the original files
as evidence, but do not interpret their old wording as a measurement of the
entire developer kit or battery-side power. Carrier/peripheral consumption and
external conversion losses require a separate measurement.

The trace does not record `hand_count`, `palm_ran`, tracking state or FPS. The
lower-GPU interval is consistent with the user's report of hand tracking and the
code's operation, but these files alone do not establish the state transition.

## Reproduce from the repository root

```bash
uv run python scripts/analyze_energy.py \
  --csv measurements/2026-11-02-single-run/jetson-telemetry-2026-11-02T02-53-03-475Z.csv \
  --json measurements/2026-11-02-single-run/jetson-telemetry-2026-11-02T02-53-02-491Z.json \
  --output measurements/2026-11-02-single-run/analysis.json
```

Integration uses the trapezoidal rule over the exported monotonic timestamps,
not the assumption that every interval is exactly one second. Statistics are
descriptive of this one trace, not confidence intervals, worst-case power limits
or guaranteed operating autonomy. See [battery sizing](../../docs/battery-sizing.md).
