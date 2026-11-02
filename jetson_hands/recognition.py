"""Recognition strategies share BGR input and return portable metadata."""
from abc import ABC, abstractmethod

import cv2

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def draw_hand(bgr, points, label, confidence, score_label=""):
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
            points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
            label = categories[0]
            draw_hand(bgr, points, label.category_name, label.score)
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
