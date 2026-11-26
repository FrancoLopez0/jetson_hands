import logging
import threading
import time

import cv2
import numpy as np

from .recognition import INDEX_TIP, PointingGesture, create_strategy, draw_hand, is_pointing


def render_raw(raw, output_width, gamma=2.2):
    # Debayer before resizing: scaling the mosaic would destroy its color phase.
    bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBGGR2BGR)
    height = round(output_width * bgr.shape[0] / bgr.shape[1])
    bgr = cv2.resize(bgr, (output_width, height), interpolation=cv2.INTER_AREA)
    image = bgr.astype(np.float32)
    black = float(np.percentile(raw[::7, ::7], 1))
    image = np.maximum(image - black, 0)
    white = max(float(np.percentile(image[::4, ::4], 99.5)), 64)
    return (np.power(np.clip(image / white, 0, 1), 1 / gamma) * 255).astype(np.uint8)


def render_raw_lut(raw, output_width, gamma=2.2):
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
    lut = (np.power(np.clip(levels / white, 0, 1), 1 / gamma) * 255).astype(np.uint8)
    return lut[bgr]


def render_calibrated(raw, width, controls, renderer):
    if controls["auto_levels"] and not controls["auto_wb"] and all(
        controls[channel] == 1.0 for channel in ("red", "green", "blue")
    ):
        image = renderer(raw, width, controls["gamma"])
    else:
        bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBGGR2BGR)
        if bgr.shape[1] != width:
            height = round(width * bgr.shape[0] / bgr.shape[1])
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        black = (float(np.percentile(raw[::7, ::7], 1))
                 if controls["auto_levels"] else controls["black"])
        scaled = np.maximum(bgr.astype(np.float32) - black, 0)
        weights = np.array([controls[k] for k in ("blue", "green", "red")], dtype=np.float32)
        if controls["auto_wb"]:
            means = scaled[::8, ::8].mean(axis=(0, 1))
            if (means > 1).all():
                weights *= np.clip(means.mean() / means, .25, 4)
        scaled *= weights
        white = (max(float(np.percentile(scaled[::4, ::4], 99.5)), 64)
                 if controls["auto_levels"] else controls["white"] - black)
        image = (np.power(np.clip(scaled / white, 0, 1), 1 / controls["gamma"]) * 255).astype(np.uint8)
    if controls["denoise"]:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if controls["flip"]:
        image = cv2.flip(image, -1)
    return image


class Pipeline:
    def __init__(self, settings, loop, hub):
        self.settings, self.loop, self.hub = settings, loop, hub
        self.camera = None
        self.controls = {
            "exposure_us": settings.exposure, "gain": settings.gain, "fps": settings.fps,
            "gamma": 2.2, "black": 1024, "white": 16000, "red": 1.0, "green": 1.0, "blue": 1.0,
            "auto_levels": True, "auto_wb": False, "denoise": False, "flip": False,
            "jpeg_quality": settings.jpeg_quality, "preview_fps": 30,
        }
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, name="recognition", daemon=True)

    def set_image_controls(self, controls):
        camera = self.camera
        if camera is None:
            raise RuntimeError("Camera is not ready")
        hardware = ("exposure_us", "gain", "fps")
        if any(controls[key] != self.controls[key] for key in hardware):
            camera.set_image_controls(*(controls[key] for key in hardware))
        self.controls = controls

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
            self.camera = camera
            renderer = render_raw_lut if self.settings.renderer == "lut" else render_raw
            previous, last_timestamp, last_finished = -1, -1, None
            fps = 0.0
            pointing = PointingGesture()
            last_rendered_at = None
            while not self.stop.is_set():
                raw = camera.read_latest(previous)
                if raw is None:
                    continue
                controls = self.controls
                started = time.monotonic()
                dropped = max(0, raw.sequence - previous - 1) if previous >= 0 else 0
                previous = raw.sequence
                bgr = render_calibrated(raw.pixels, self.settings.output_width, controls, renderer)
                converted = time.monotonic()
                timestamp = max(last_timestamp + 1, int(raw.captured_at * 1000))
                last_timestamp = timestamp
                recognition = strategy.process(bgr, timestamp)
                hands = recognition["hands"]
                tip = hands[0]["landmarks"][INDEX_TIP] if hands else None
                if tip and not (0 <= tip["x"] < 1 and 0 <= tip["y"] < 1):
                    tip = None
                if self.settings.overlay == "gesture":
                    height, width = bgr.shape[:2]
                    candidate = tip is not None and is_pointing(hands[0]["landmarks"], width, height)
                    active = pointing.update(candidate, timestamp, max(300, 2_000 // controls["fps"]))
                    if not active:
                        tip = None
                        for hand in hands:
                            points = [(p["x"] * width, p["y"] * height) for p in hand["landmarks"]]
                            draw_hand(bgr, points, hand["handedness"], hand["confidence"],
                                      score_label="presence" if hand.get("confidence_kind") == "hand_presence" else "")
                if self.settings.overlay == "gesture":
                    self.loop.call_soon_threadsafe(self.hub.publish_pointing,
                        tip["x"] if tip else None, bgr.shape[1], raw.captured_at)
                inferred = time.monotonic()
                if last_rendered_at is not None and inferred - last_rendered_at < .9 / controls["preview_fps"]:
                    continue
                last_rendered_at = inferred
                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, controls["jpeg_quality"]])
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
                    "strategy": self.settings.strategy, "index_tip": tip, **recognition,
                }
                self.loop.call_soon_threadsafe(self.hub.publish, metadata, jpeg.tobytes(), raw.captured_at)
        except Exception as error:
            logging.exception("Camera/recognition pipeline failed")
            self.loop.call_soon_threadsafe(self.hub.fail, str(error))
        finally:
            self.camera = None
            camera.close()
            if strategy is not None:
                strategy.close()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
