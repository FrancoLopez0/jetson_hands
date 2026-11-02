"""Single-hand palm acquisition and oriented-ROI landmark tracking on TensorRT."""
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np

from .recognition import RecognitionStrategy, draw_hand
from .trt_engine import TensorRTEngine


@dataclass(frozen=True)
class HandROI:
    cx: float
    cy: float
    size: float
    angle: float

    def matrix(self, output_size=224):
        scale = output_size / self.size
        cosine, sine = math.cos(self.angle), math.sin(self.angle)
        return np.array([
            [scale * cosine, scale * sine, output_size / 2 - scale * (cosine * self.cx + sine * self.cy)],
            [-scale * sine, scale * cosine, output_size / 2 - scale * (-sine * self.cx + cosine * self.cy)],
        ], dtype=np.float32)


def hand_angle(wrist, middle):
    direction = middle - wrist
    angle = math.pi / 2 + math.atan2(float(direction[1]), float(direction[0]))
    return (angle + math.pi) % (2 * math.pi) - math.pi


def tracking_roi(points):
    # MediaPipe's tracking rectangle uses palm/proximal landmarks, not fingertips.
    # Expanding the full-hand bounds by 2 would shrink the hand on each iteration.
    direction = ((points[5] + points[13]) / 2 + points[9]) / 2
    angle = hand_angle(points[0], direction)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    partial = points[[0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]]
    oriented = partial @ rotation.T
    low, high = oriented.min(axis=0), oriented.max(axis=0)
    extent = high - low
    center = (low + high) / 2
    center[1] -= 0.1 * extent[1]
    center = center @ rotation
    return HandROI(float(center[0]), float(center[1]), max(float(extent.max()) * 2.0, 24), angle)


def palm_anchors():
    # MediaPipe SSD strides 8,16,16,16: two anchors on 24x24 and six on 12x12.
    return np.array([((x + 0.5) / grid, (y + 0.5) / grid)
                     for grid, repeat in ((24, 2), (12, 6))
                     for y in range(grid) for x in range(grid) for _ in range(repeat)], dtype=np.float32)


class TensorRTHandsStrategy(RecognitionStrategy):
    def __init__(self, settings):
        if settings.max_hands != 1:
            raise ValueError("TensorRTHandsStrategy tracks one hand")
        self.palm = TensorRTEngine(settings.palm_engine, settings.trt_library)
        try:
            self.hand = TensorRTEngine(settings.hand_engine, settings.trt_library)
            if self.palm.input_shape != (1, 192, 192, 3) or self.hand.input_shape != (1, 224, 224, 3):
                raise ValueError("Unexpected engine layout; use the checksum-pinned OpenCV Zoo models")
            if not {"Identity", "Identity_1", "Identity_2", "Identity_3"} <= self.hand.outputs.keys():
                raise ValueError(f"Unexpected landmark output names: {list(self.hand.outputs)}")
        except BaseException:
            self.palm.close()
            if hasattr(self, "hand"):
                self.hand.close()
            raise
        self.anchors = palm_anchors()
        self.roi = None
        self.age = 0
        self.previous_timestamp = None

    def _detect(self, bgr):
        height, width = bgr.shape[:2]
        scale = 192 / max(height, width)
        scaled_width, scaled_height = round(width * scale), round(height * scale)
        left, top = (192 - scaled_width) // 2, (192 - scaled_height) // 2
        image = np.zeros((192, 192, 3), dtype=np.uint8)
        image[top:top + scaled_height, left:left + scaled_width] = cv2.resize(bgr, (scaled_width, scaled_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        outputs = list(self.palm.infer(image[None]).values())
        boxes = next(value.reshape(2016, 18) for value in outputs if value.size == 2016 * 18)
        logits = next(value.reshape(2016) for value in outputs if value.size == 2016)
        # A single target needs only the highest-confidence candidate, not multi-hand NMS.
        index = int(np.argmax(logits))
        score = float(1 / (1 + np.exp(-np.clip(logits[index], -80, 80))))
        if score < 0.6:
            return None
        box = boxes[index]
        center = ((box[:2] / 192 + self.anchors[index]) * 192 - [left, top]) / scale
        keypoints = ((box[4:].reshape(7, 2) / 192 + self.anchors[index]) * 192 - [left, top]) / scale
        size = float(max(box[2], box[3]) / 192 * max(width, height))
        if not math.isfinite(size) or size < 4:
            return None
        angle = hand_angle(keypoints[0], keypoints[2])
        center += np.array([math.sin(angle), -math.cos(angle)]) * (0.5 * size)
        return HandROI(float(center[0]), float(center[1]), size * 2.9, angle)

    def process(self, bgr, timestamp_ms):
        if self.previous_timestamp is not None and timestamp_ms - self.previous_timestamp > 500:
            self.roi = None
        self.previous_timestamp = timestamp_ms
        was_tracking = self.roi is not None
        status = {"palm_ran": False, "palm_ms": 0.0, "presence_score": None,
                  "recovery_attempted": False, "state": "searching"}
        # Keep a valid ROI indefinitely. Never discard it on a frame-count timer.
        # If tracking fails, acquire a fresh palm and retry THIS frame once.
        for attempt in range(2):
            if self.roi is None:
                palm_started = time.perf_counter()
                self.roi = self._detect(bgr)
                self.age = 0
                status["palm_ran"] = True
                status["palm_ms"] = round((time.perf_counter() - palm_started) * 1000, 2)
            if self.roi is None:
                break
            matrix = self.roi.matrix()
            crop = cv2.warpAffine(bgr, matrix, (224, 224), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            outputs = self.hand.infer(image[None])
            confidence = float(outputs["Identity_1"].flat[0])
            landmarks = outputs["Identity"].reshape(21, 3).copy()
            handedness = float(outputs["Identity_2"].flat[0])
            status["presence_score"] = confidence if math.isfinite(confidence) else None
            valid = np.isfinite(landmarks).all() and math.isfinite(confidence) and confidence >= 0.6
            if valid:
                inverse = cv2.invertAffineTransform(matrix)
                xy = cv2.transform(landmarks[None, :, :2], inverse)[0]
                height, width = bgr.shape[:2]
                next_roi = tracking_roi(xy)
                valid = math.isfinite(next_roi.size) and 24 <= next_roi.size <= 3 * max(width, height)
            if not valid:
                self.roi = None
                if was_tracking and attempt == 0:
                    status["recovery_attempted"] = True
                    continue
                break
            self.roi = next_roi
            self.age += 1
            status["state"] = "reacquired" if status["palm_ran"] else "tracking"
            # z is in crop-pixel scale; normalize by original image width like MediaPipe.
            landmarks[:, :2] = xy / [width, height]
            landmarks[:, 2] /= float(np.linalg.norm(matrix[0, :2])) * width
            label = "Right" if handedness >= 0.5 else "Left"
            draw_hand(bgr, xy, label, confidence, score_label="presence")
            return {"hands": [{"handedness": label, "confidence": confidence,
                               "confidence_kind": "hand_presence",
                               "landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                                            for x, y, z in landmarks]}], "recognition": status}
        # No stale landmarks are drawn or reported after unsuccessful recovery.
        return {"hands": [], "recognition": status}

    def close(self):
        self.palm.close()
        self.hand.close()
