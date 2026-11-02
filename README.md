# Jetson hand tracking

Direct OV5647 driver capture, OpenCV processing, TensorRT/MediaPipe hand tracking and a
token-authenticated live viewer for one simultaneous receiver.

## Current deployment

- Jetson: `ssh jetson`, application in `/home/talos/jetson-hands`.
- Camera: `/dev/video0`, sensor mode 3, BG10 RAW, **640 × 480 at 60 capture FPS**.
- Output: 640 × 480 JPEG with one-hand skeleton tracking.
- Environment: **uv**, Python **3.14.7**, MediaPipe **1.2.0**, OpenCV **4.14**.
- Active strategy: **TensorRT 10.3 FP16**, with LUT levels/gamma rendering.
- API: FastAPI served by one Uvicorn worker on port **8765**.
- PC viewer: **http://127.0.0.1:8000**, WebSocket
  **ws://192.168.1.36:8765/stream** (the address is editable).
- Token: `access.token` on both machines; generated using `secrets`, not committed.

The old `jetson-camera-web.service` is stopped/disabled because it uses the same
camera. Its application and calibration files remain in `/home/talos/camera-web`.

## Open the viewer on the PC

From this project directory in PowerShell:

```powershell
.\start-viewer.ps1
```

Open http://127.0.0.1:8000, paste the contents of `access.token`, and click
**Connect**. To copy the token without printing it:

```powershell
(Get-Content -Raw .\access.token).Trim() | Set-Clipboard
```

The viewer server binds only to loopback and serves only `web/`, so the token
file is not served. The viewer keeps the entered token in tab memory, not in a
URL or local storage. `ws://` authenticates access but does not encrypt traffic;
use it on the trusted LAN. For an SSH-encrypted connection:

```powershell
ssh -N -L 18765:127.0.0.1:8765 jetson
```

Then enter `ws://127.0.0.1:18765/stream` in the viewer.

## Exportable Jetson telemetry

Connect the viewer and scroll to **Jetson telemetry**. Four live charts show:

- Module input power, combined CPU/GPU/CV rail power and SoC power in watts.
- Mean CPU utilization and GPU utilization in percent.
- Mean CPU clock and GPU clock in MHz.
- Utilization of each CPU core.

The Jetson samples `tegrastats` once per second. On this Orin, GPU frequency is
not printed by `tegrastats`, so it is read from the GPU devfreq `cur_freq` sysfs
node and converted from Hz to MHz. This collector is independent of inference.
CPU summaries average online cores; each core's utilization and clock are also
included in the data exports. `VDD_IN` is total module power, not a measurement
of the complete developer kit, all peripherals or battery-side power;
the CPU/GPU/CV rail is a shared measurement, not independent device powers.

Export buttons:

- **Export CSV**: timestamped measurements with per-core utilization/clocks,
  session identifier, source URL and explicit unit-bearing column names.
- **Export JSON**: all retained samples, raw rail names, units and metric definitions.
- **Export charts PNG**: a 1600×760 image of all four charts and their legends.

Charts display up to the latest two minutes. The tab retains at most 3600 samples
(about one hour at 1 Hz); CSV/JSON include that retained history, not just the
visible plot. History survives disconnect/reconnect for export, but page reload
or **Clear history** removes it. Session boundaries and missing readings are
not drawn as continuous data. Unavailable sensors remain null/blank, not zero.
All exports are generated on the PC and exclude the token, video and landmarks.
No external chart library or CDN is required.

The browser requests telemetry with `{"token":"...","telemetry":true}` as its
first WebSocket message. Optional `type: "telemetry"` JSON messages are sent once
per new sample on the same exclusive authenticated socket, outside frame/JPEG
pairs. Clients that omit this flag retain the original video-only protocol.
Collection failures produce `telemetry_status` messages and do not stop video;
the collector retries automatically and its child process is cleaned up on shutdown.

Browser/export verification on the PC, with the viewer disconnected:

```powershell
uv run --with playwright python scripts/check_telemetry.py
```

This checks live units and all six cores, parses actual downloaded CSV/JSON,
validates PNG dimensions, checks that exports omit the token, and verifies
history retention, reconnect, the 3600-sample bound and clearing. Generated
files and screenshots are in `runtime/telemetry-*`.

## Architecture and latency control

```text
OV5647 -> V4L2 ioctl/MMAP -> latest RAW buffer
                            |
                   OpenCV Bayer conversion + resize + levels/gamma
                            |
                  RecognitionStrategy.process(BGR, timestamp)
                            |
                  OpenCV JPEG -> latest encoded frame
                            |
                  authenticated single WebSocket receiver
                            |
                  browser decode + canvas + acknowledgement
```

There is no IP camera or internal HTTP image fetch on the Jetson. `v4l2-ctl` is
used only to configure the OV5647-specific controls. Frame capture itself uses
`open`, `ioctl`, `mmap`, `select`, `DQBUF` and `QBUF` directly against the driver.
The actual driver-negotiated stride and buffer sizes are respected.

The capture thread continuously drains the driver. The inference thread takes
the latest frame rather than accumulating a queue. The network sender retains
one latest encoded frame and allows only one frame in flight, waiting for the
browser to decode/draw and acknowledge it. Slow viewers skip frames.

The first WebSocket message is `{"token":"..."}`. After authentication the
server atomically claims the receiver slot; a second authenticated connection
is rejected with code 4409. Incorrect tokens receive code 4401. Each frame is a
JSON metadata message followed by binary JPEG data. The receiver responds with
`{"type":"ack","sequence":123}`. Missing acknowledgements time out after
10 seconds. Disconnects release the slot. Idle connections use application
ping/pong messages as well as the server's WebSocket heartbeat.

**Run exactly one worker and one service instance.** The receiver slot and frame
hub are process-local, and the camera has a single owner.

## Recognition Strategy

- `jetson_hands/recognition.py`: abstract `RecognitionStrategy` with `process`
  and `close`; `MediaPipeHandsStrategy` uses the current Hand Landmarker API in
  VIDEO mode with tracking, the successor of legacy `mp.solutions.hands`.
- `PassthroughStrategy`: camera-only baseline without inference.
- `jetson_hands/tensorrt_hands.py`: `TensorRTHandsStrategy` runs palm acquisition
  and 21-landmark tracking on the GPU. It updates an oriented crop using palm/
  proximal landmarks and reacquires after actual tracking loss. A valid track is
  not reset on a timer; failed tracking gets one same-frame reacquisition attempt.
- Add a class implementing the interface and register it in `STRATEGIES`.
- Select it with `HANDS_STRATEGY=tensorrt`, `mediapipe` or `passthrough`;
  restart the service. `HANDS_MAX_HANDS` defaults to 1; TensorRT supports one.

The capture, network protocol and PC viewer do not depend on MediaPipe-specific
types. The metadata contains normalized x/y/z landmarks and handedness, as well
as timing information. MediaPipe uses the CPU XNNPACK delegate. The TensorRT
strategy uses CUDA and FP16 engines with FP32 input/output. Its `confidence`
is hand-presence confidence; handedness selects the Left/Right label.
The overlay labels it explicitly, for example `Right presence 0.99`. This is a
model score for the current crop, not calibrated landmark accuracy or a guarantee
of tracking stability. Palm and hand-presence acceptance thresholds remain 0.6.
Recognition metadata also reports tracking state, presence score and recovery
attempts; failed recovery returns no hand rather than stale landmarks.

The small native library in `native/` exposes a C ABI through `ctypes`; it is
independent of CPython's extension-module ABI. This lets Python 3.14 use the
system TensorRT 10.3 runtime even though the installed TensorRT Python package
is built for Python 3.10. Device buffers, pinned host buffers and a CUDA stream
are allocated once per engine and reused. The rest of the application is Python.

## Environment and configuration

`uv` manages Python, the virtual environment and the reproducible `uv.lock`.
Uvicorn is the ASGI server; it is not the environment manager.

On a new Jetson with the same working OV5647 driver, `v4l2-ctl`, and network access:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
# From the copied project directory:
uv python install 3.14
uv sync --locked
uv run python scripts/setup_jetson.py
uv run python -m uvicorn jetson_hands.server:create_app --factory \
  --host 0.0.0.0 --port 8765 --workers 1 --ws websockets-sansio \
  --ws-max-size 4096 --ws-max-queue 1 --ws-per-message-deflate false --no-access-log
```

Stop any existing camera owner before starting capture. The model setup script
downloads the versioned official float16 hand model and verifies its SHA256.
The old system Python/OpenCV installation is separate from this virtualenv.

### Build the TensorRT backend on the Jetson

With the system CUDA/TensorRT development libraries, CMake and g++ installed:

```bash
uv run python scripts/setup_tensorrt.py
```

This compiles `native/build/libhands_trt.so`, downloads checksum-pinned OpenCV
Zoo palm/landmark ONNX models and their license, and builds FP16 engines locally.
Build logs and the model/engine SHA256 manifest are in `models/tensorrt/`.
Allow several minutes for the first build. Use `--rebuild` after changing the
TensorRT/GPU environment. Engine binaries are not portable deployment assets.
See [THIRD_PARTY.md](THIRD_PARTY.md) for sources.

To use the deployed fast profile:

```bash
install -m 600 deploy/tensorrt-640x480.env .env
sudo systemctl restart jetson-hands
```

`deploy/tensorrt-1280x720.env` retains 720p capture and 640×360 output at 30 FPS
when spatial detail is preferred. Preserve any custom settings before replacing
an existing `.env`. Both profiles use the LUT renderer, whose output matches the
standard renderer within one intensity level per channel in equivalence tests.

The deployed service reads optional settings from
`/home/talos/jetson-hands/.env` using systemd's `EnvironmentFile` format:

```ini
HANDS_STRATEGY=tensorrt
HANDS_MAX_HANDS=1
HANDS_SENSOR_MODE=3
HANDS_WIDTH=640
HANDS_HEIGHT=480
HANDS_FPS=60
HANDS_OUTPUT_WIDTH=640
HANDS_JPEG_QUALITY=80
HANDS_EXPOSURE=16000
HANDS_GAIN=339
HANDS_RENDERER=lut
```

Additional variables: `HANDS_DEVICE`, `HANDS_SENSOR_MODE`, `HANDS_WIDTH`,
`HANDS_HEIGHT`, `HANDS_FPS`, `HANDS_MODEL`, `HANDS_TOKEN_FILE`, `HANDS_TOKEN`,
`HANDS_TRT_LIBRARY`, `HANDS_PALM_ENGINE`, `HANDS_HAND_ENGINE`.
Changing the sensor mode requires matching dimensions; configuration is
validated before opening the camera. The model and token paths default to the
application working directory. For a foreground launch, export these variables
in the shell; `.env` is loaded by the deployed systemd service, not by Python.

```powershell
ssh jetson "sudo systemctl restart jetson-hands"
ssh jetson "systemctl status jetson-hands --no-pager"
ssh jetson "journalctl -u jetson-hands -n 50 --no-pager"
```

`GET /health` returns readiness and receiver occupancy, without images,
recognition results or tokens. A healthy server responds 200; a stalled/failed
camera pipeline responds 503. If a runtime capture/inference error occurs,
restart the service after fixing the camera problem.

To return to the old camera calibration application:

```powershell
ssh jetson "sudo systemctl disable --now jetson-hands && sudo systemctl enable --now jetson-camera-web"
```

To return to hand tracking:

```powershell
ssh jetson "sudo systemctl disable --now jetson-camera-web && sudo systemctl enable --now jetson-hands"
```

## Verification and measured performance

```bash
uv run python -m pytest -q
uv run python -m scripts.check_recognition
uv run python scripts/validate_tensorrt.py --strategy tensorrt --seconds 30
uv run python scripts/check_stream.py --frames 100
```

The reference-image check downloads the official MediaPipe `right_hands.jpg`
fixture, requires detected hands with 21 landmarks each, and writes an annotated
image to `runtime/reference-hands.jpg`. It does not replace a physical hand test.

From the PC, while no viewer is connected:

```powershell
uv run python scripts/check_stream.py --url ws://192.168.1.36:8765/stream --seconds 30 --warmup 5 --report runtime/network-tensorrt.json
uv run --with playwright python scripts/check_viewer.py --seconds 30 --report runtime/browser-tensorrt.json
```

The browser check uses installed Chrome, verifies live canvas rendering,
incorrect-token rejection, second-viewer rejection and JavaScript errors. Its
screenshot is written to `runtime/viewer.png` with the token field cleared.

Observed on this Jetson with the live camera and the optimized profile:

| Measurement | Observed |
|---|---:|
| Jetson capture + recognition + JPEG pipeline | 60.00 FPS |
| Chrome on the PC, decoded/drawn frames over 30 seconds | 57.57 FPS |
| PC WebSocket smoke client over 30 seconds | 53.09 FPS |
| TensorRT inference median | 6.3 ms |
| Driver dequeue to encoded frame median / p95, network test | 16.4 / 18.4 ms |
| Original MediaPipe one-hand 720p pipeline | 9.48 FPS |

Full resolution/backend comparisons, validation scope and reproduction commands
are in [docs/performance.md](docs/performance.md).

The live scene during verification contained no hands. Physical tracking still
needs a hand placed in view. Thirty-nine tests passed on both Windows and
the Jetson. Live driver capture, JPEG decoding, token rejection, receiver
exclusivity and reconnection passed against the deployed service.

**Timing definitions:** the Tegra VI hardware timestamp was observed to have a
different epoch from the host monotonic clock despite its MONOTONIC flag. The
pipeline therefore measures from driver dequeue using the host clock and
exposes the raw hardware timestamp separately. Dequeue-to-encoded excludes
sensor exposure and time before dequeue. ACK-to-displayed on the browser
includes frame wait, transfer and decode; neither is a synchronized full
camera-to-screen latency measurement. The frame-rate figures count actual work
or decoded/drawn frames; they are not a guarantee of physical monitor refresh.

If measurements do not meet the application's needs, profile the camera
conversion and inference independently using the passthrough strategy before
porting the bottleneck to C++. The WebSocket protocol and viewer can be kept.
