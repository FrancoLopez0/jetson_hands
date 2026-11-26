"""Recognition strategies share BGR input and return portable metadata."""
from abc import ABC, abstractmethod

import cv2
import numpy as np

INDEX_TIP = 8
POINTING_HOLD_MS = 200
MAX_GESTURE_GAP_MS = 300

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def _joint_angle(points, a, b, c):
    before = points[a] - points[b]
    after = points[c] - points[b]
    scale = np.linalg.norm(before) * np.linalg.norm(after)
    if scale <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(before, after) / scale, -1, 1))))


def is_pointing(landmarks, width, height):
    points = np.asarray([(p["x"] * width, p["y"] * height, p["z"] * width) for p in landmarks])
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return False
    angles = [(_joint_angle(points, i, i+1, i+2), _joint_angle(points, i+1, i+2, i+3))
              for i in (5, 9, 13, 17)]
    if any(angle is None for finger in angles for angle in finger):
        return False
    index, *others = angles
    return min(index) >= 150 and all(min(finger) <= 135 for finger in others)


class PointingGesture:
    def __init__(self):
        self.since = None
        self.last = None

    def update(self, pointing, now_ms, max_gap_ms=MAX_GESTURE_GAP_MS):
        if self.last is not None and now_ms - self.last > max_gap_ms:
            self.since = None
        self.last = now_ms
        if not pointing:
            self.since = None
            return False
        if self.since is None:
            self.since = now_ms
        return now_ms - self.since >= POINTING_HOLD_MS


def draw_hand(bgr, points, label, confidence, score_label="", *, overlay="hand"):
    if overlay == "gesture":
        return
    if overlay == "index":
        x, y = points[INDEX_TIP]
        height, width = bgr.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            tip = (int(x), int(y))
            cv2.circle(bgr, tip, 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(bgr, tip, 6, (75, 255, 190), -1, cv2.LINE_AA)
        return
    points = [(int(x), int(y)) for x, y in points]
    for start, end in HAND_CONNECTIONS:
        cv2.line(bgr, points[start], points[end], (65, 220, 80), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(bgr, point, 3, (30, 170, 255), -1, cv2.LINE_AA)
    score_text = f"{score_label} {confidence:.2f}" if score_label else f"{confidence:.0%}"
    cv2.putText(bgr, f"{label} {score_text}",
                (max(0, points[0][0] - 30), max(20, points[0][1] - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


class RecognitionStrategy(ABC):
    @abstractmethod
    def process(self, bgr, timestamp_ms: int) -> dict:
        """Annotate BGR in place and return JSON-serializable recognition data."""

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""


class MediaPipeHandsStrategy(RecognitionStrategy):
    def __init__(self, settings):
        import mediapipe as mp
        self.mp = mp
        self.overlay = settings.overlay
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=settings.model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process(self, bgr, timestamp_ms):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
        height, width = bgr.shape[:2]
        hands = []
        for landmarks, categories in zip(result.hand_landmarks, result.handedness):
            points = [(p.x * width, p.y * height) for p in landmarks]
            label = categories[0]
            if self.overlay == "hand" or not hands:
                draw_hand(bgr, points, label.category_name, label.score, overlay=self.overlay)
            hands.append({"handedness": label.category_name, "confidence": label.score,
                          "landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in landmarks]})
        return {"hands": hands}

    def close(self):
        self.detector.close()


class PassthroughStrategy(RecognitionStrategy):
    """Camera-only baseline for measuring inference overhead."""
    def process(self, bgr, timestamp_ms):
        return {"hands": []}

    def close(self):
        pass


STRATEGIES = {"mediapipe": MediaPipeHandsStrategy, "passthrough": PassthroughStrategy}


def create_strategy(settings) -> RecognitionStrategy:
    if settings.strategy == "tensorrt":
        from .tensorrt_hands import TensorRTHandsStrategy
        return TensorRTHandsStrategy(settings)
    try:
        factory = STRATEGIES[settings.strategy]
    except KeyError:
        raise ValueError(f"Unknown recognition strategy: {settings.strategy}") from None
    return factory(settings)
