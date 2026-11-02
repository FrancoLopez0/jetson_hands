# TensorRT performance experiment

For the separate **single 86.52-second power recording** and theoretical battery
calculations, see [battery sizing](battery-sizing.md). These FPS experiments do
not establish repeated-run battery endurance. The original "board power" label
means VDD_IN module power, not a verified complete-kit or battery-side reading.

Measured on 2026-11-01 on the user's Jetson Orin Nano Super, Ubuntu 22.04 /
L4T R36.4.7, TensorRT 10.3, CUDA 12.6 and uv-managed Python 3.14.7.
Power mode remained MAXN_SUPER with the original `schedutil` governor;
`jetson_clocks` was not enabled. Both recognition strategies were set to one hand.

## Real OV5647 driver capture

Each case had three seconds of warmup and at least thirty seconds of measurement.
Capture uses V4L2 ioctl/MMAP; there is no network image source. Output width is
640, preserving aspect ratio. JPEG quality is 80, exposure 16000 microseconds
and gain 339. The 640×480 profile changes sensor mode, aspect ratio and field of
view as well as resolution; it is a speed-oriented profile.

These camera runs contained **no hands**. This exercises palm searching, not
continuous landmark tracking. Positive-hand correctness/performance was measured
separately with the reference-image sequence described below.

| Strategy | Capture | Requested / actual capture FPS | Renderer | Processing FPS | Dequeue → JPEG median / p95 |
|---|---|---:|---|---:|---:|
| MediaPipe | 1280×720 | 30 / 30.01 | Standard | 9.48 | 122.74 / 144.51 ms |
| MediaPipe | 640×480 | 30 / 30.02 | Standard | 9.86 | 119.14 / 144.27 ms |
| TensorRT FP16 | 1280×720 | 30 / 30.01 | Standard | 30.01 | 28.15 / 30.01 ms |
| TensorRT FP16 | 640×480 | 30 / 30.02 | Standard | 30.02 | 30.70 / 31.28 ms |
| TensorRT FP16 | 640×480 | 60 / 60.03 | Standard | 32.53 | 38.83 / 46.45 ms |
| TensorRT FP16 | 1280×720 | 30 / 30.01 | LUT | 30.02 | 24.99 / 26.12 ms |
| TensorRT FP16 | 640×480 | 30 / 30.02 | LUT | 30.02 | 22.88 / 25.78 ms |
| **TensorRT FP16** | **640×480** | **60 / 60.03** | **LUT** | **60.00** | **16.35 / 16.91 ms** |

The selected case processed 1801 frames in 30.02 seconds. Its median conversion,
recognition and JPEG times were 7.34, 6.30 and 2.56 ms. Before LUT rendering at
the same resolution/rate, conversion alone took 21.77 ms.

The LUT renderer debayers before resizing, preserves the existing levels/gamma
mapping, and avoids full-frame floating-point images. Eight equivalence cases
cover random RAW data, gradients, darkness, saturation and resizing, with a
maximum allowed discrepancy of one 8-bit intensity level. Work remains bounded
to the latest frame; no queue was added to manufacture a higher FPS number.

The changing CPU load and automatic clock scaling affect component timings, so
not every difference between rows should be attributed solely to pixel count.

## PC network and browser

Measured after deploying the 640×480@60 / TensorRT / LUT profile:

- WebSocket client on the PC: **1594 frames, 53.09 FPS**, thirty-second window
  after five seconds of warmup, decoding each JPEG and acknowledging it.
- In that network run: pipeline median **16.4 ms**, p95 **18.4 ms**;
  inference median **6.3 ms**, p95 **6.5 ms**.
- Chrome headless on the PC: **1728 decoded/drawn frames in 30.014 seconds**, or
  **57.57 FPS**, after warmup. The browser check counts canvas updates, not the
  instantaneous smoothed FPS label. It does not measure physical display refresh.
- The client and browser runs were separate; they should not have identical rates.
- Token rejection, second-receiver rejection, acknowledgement/reconnection, real
  canvas rendering and zero JavaScript errors were verified.
- Observed during operation: GPU utilization about 30%, temperatures about 54°C,
  board power about 7.8 W. These are samples, not a dedicated power benchmark.

These latency values start at driver buffer dequeue. Sensor exposure, time before
dequeue and display scanout are excluded. The Tegra hardware timestamp has a
different clock epoch from the host monotonic clock. No synchronized full
camera-to-screen latency is claimed.

## Positive-hand model and tracking validation

The public MediaPipe `right_hands.jpg` fixture is cropped to its upright right
hand, avoiding ambiguity between targets. Each strategy is tested on 100 frames
with controlled translation and rotation, followed by blank frames and reacquisition.

| Check | MediaPipe | TensorRT |
|---|---:|---:|
| Frames with a detected hand | 100 / 100 | 100 / 100 |
| Landmarks per detected hand | 21 | 21 |
| Median normalized consistency error | 0.0062 | 0.0082 |
| Blank-frame rejection | Passed | Passed |
| Reacquisition | Passed | Passed |
| Reference inference median / p95 | 70.53 / 74.98 ms | 5.99 / 6.34 ms |
| Reference inference throughput | 15.31 FPS | 161.43 FPS |

Reference throughput excludes camera conversion, JPEG and network transfer; it
is **not video FPS**. Consistency error compares landmarks with the initial
prediction transformed geometrically, not with manually annotated ground truth.
It is a regression check, not a comprehensive accuracy evaluation.

The TensorRT hand engine was also compared with FP32 OpenCV DNN on the exact
same crop. Maximum absolute pixel-landmark error was **0.030 pixels**; hand
presence differed by approximately **0.0000072**. This checks engine conversion
numerics, not all pose/lighting conditions.

The tracking rectangle uses palm/proximal landmarks. Expanding the bounds of all
21 points by two caused progressive ROI drift in an initial implementation;
the corrected subset convention is covered by a regression test.

## Reproduce

From the project directory on the Jetson (the suites temporarily disconnect
viewers and restore the service in a `finally` block):

```bash
uv run python scripts/setup_tensorrt.py
uv run python -m pytest -q
uv run python scripts/benchmark_suite.py
uv run python scripts/benchmark_suite.py --renderer lut --label lut --tensorrt-only --skip-validation
```

On the PC with the service running and the viewer disconnected:

```powershell
uv run python scripts/check_stream.py --url ws://192.168.1.36:8765/stream --seconds 30 --warmup 5 --report runtime/network-tensorrt.json
uv run --with playwright python scripts/check_viewer.py --seconds 30 --report runtime/browser-tensorrt.json
```

Raw per-case JSON, annotated images and `tegrastats` logs are in `runtime/` on
the Jetson. `comparison.json`, `comparison-lut.json`, both validation JSON files
and the PC reports are also available in the local `runtime/`. These generated
artifacts are excluded from version control. The native bridge and model setup
do not add a Python TensorRT dependency or alter the system Python environment.

## Deployed configuration

`/home/talos/jetson-hands/.env` was installed from
`deploy/tensorrt-640x480.env`. The service remains enabled at boot. No token,
network port or receiver-exclusivity behavior was changed. Refresh the PC page
to load the aspect-ratio-aware viewer and reconnect with the same token.

For higher capture resolution at 30 FPS, install
`deploy/tensorrt-1280x720.env` instead. To return to CPU inference at the current
resolution, change `HANDS_STRATEGY=mediapipe` in the active environment file and
restart the service; `HANDS_MAX_HANDS=1` remains the supported requirement.

## Tracking continuity correction

After the initial FPS experiment, the user reported a hand appearing briefly
with a 99% label and then disappearing. Inspection found a deterministic
failure mechanism: at age 60 frames, the tracker discarded its valid ROI and
replaced it with a new palm detection, even when that detector returned nothing.
At 60 FPS this introduced an unnecessary opportunity for loss once per second.

The tracker now retains a valid ROI until a real tracking failure. On failure it
tries palm detection and landmark inference again in the same frame, at most
once, and never returns stale landmarks when recovery fails. The 0.6 thresholds
were preserved. The overlay now says `presence 0.99`, identifying the model's
hand-presence score rather than suggesting 99% geometric accuracy or temporal
reliability. Metadata exposes score type and recovery state.

Five state-machine regression tests were added (30 total tests pass on Windows
and Jetson). The corrected backend also passes the 100-frame transformed-image
sequence, blank-frame rejection and reacquisition with the same 0.0082 normalized
consistency error. The FPS tables above describe the original benchmark; a
subsequent short hardware validation ran concurrently with the camera service
and should not be compared to the isolated throughput figures. Physical motion
and lighting conditions still require evaluation in the actual camera scene.
