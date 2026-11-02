import logging
import threading
import time

import cv2
import numpy as np

from .recognition import create_strategy


def render_raw(raw, output_width):
    # Debayer before resizing: scaling the mosaic would destroy its color phase.
    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBGGR2BGR)
    height = round(output_width * bgr.shape[0] / bgr.shape[1])
    bgr = cv2.resize(bgr, (output_width, height), interpolation=cv2.INTER_AREA)
    image = bgr.astype(np.float32)
    black = float(np.percentile(raw[::7, ::7], 1))
    image = np.maximum(image - black, 0)
    white = max(float(np.percentile(image[::4, ::4], 99.5)), 64)
    return (np.power(np.clip(image / white, 0, 1), 1 / 2.2) * 255).astype(np.uint8)


def render_raw_lut(raw, output_width):
    """Equivalent levels/gamma mapping without full-frame float temporaries."""
    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBGGR2BGR)
    if bgr.shape[1] != output_width:
        height = round(output_width * bgr.shape[0] / bgr.shape[1])
        bgr = cv2.resize(bgr, (output_width, height), interpolation=cv2.INTER_AREA)
    black = np.float32(np.percentile(raw[::7, ::7], 1))
    # Percentiles commute with subtract/clamp here, above the black floor.
    sample = np.maximum(bgr[::4, ::4].astype(np.float32) - black, 0)
    white = max(float(np.percentile(sample, 99.5)), 64)
    levels = np.maximum(np.arange(65536, dtype=np.float32) - black, 0)
    lut = (np.power(np.clip(levels / white, 0, 1), 1 / 2.2) * 255).astype(np.uint8)
    return lut[bgr]


class Pipeline:
    def __init__(self, settings, loop, hub):
        self.settings, self.loop, self.hub = settings, loop, hub
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="recognition", daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        from .camera import V4L2Camera
        camera = V4L2Camera(self.settings)
        strategy = None
        try:
            cv2.setNumThreads(2)
            strategy = create_strategy(self.settings)
            camera.start()
            renderer = render_raw_lut if self.settings.renderer == "lut" else render_raw
            previous, last_timestamp, last_finished = -1, -1, None
            fps = 0.0
            while not self.stop.is_set():
                raw = camera.read_latest(previous)
                if raw is None:
                    continue
                dropped = max(0, raw.sequence - previous - 1) if previous >= 0 else 0
                previous = raw.sequence
                started = time.monotonic()
                bgr = renderer(raw.pixels, self.settings.output_width)
                converted = time.monotonic()
                timestamp = max(last_timestamp + 1, int(raw.captured_at * 1000))
                last_timestamp = timestamp
                recognition = strategy.process(bgr, timestamp)
                inferred = time.monotonic()
                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.settings.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                finished = time.monotonic()
                if last_finished is not None:
                    current = 1 / max(finished - last_finished, 1e-6)
                    fps = current if fps == 0 else fps * 0.85 + current * 0.15
                last_finished = finished
                metadata = {
                    "type": "frame", "sequence": raw.sequence,
                    "width": bgr.shape[1], "height": bgr.shape[0],
                    "fps": round(fps, 1), "skipped_capture_frames": dropped,
                    "convert_ms": round((converted - started) * 1000, 1),
                    "inference_ms": round((inferred - converted) * 1000, 1),
                    "encode_ms": round((finished - inferred) * 1000, 1),
                    "pipeline_ms": round((finished - raw.captured_at) * 1000, 1),
                    "timestamp_source": raw.timestamp_source,
                    "driver_timestamp_s": raw.driver_timestamp,
                    "strategy": self.settings.strategy, **recognition,
                }
                self.loop.call_soon_threadsafe(self.hub.publish, metadata, jpeg.tobytes(), raw.captured_at)
        except Exception as error:
            logging.exception("Camera/recognition pipeline failed")
            self.loop.call_soon_threadsafe(self.hub.fail, str(error))
        finally:
            camera.close()
            if strategy is not None:
                strategy.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
