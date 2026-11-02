from collections import deque

import numpy as np

from jetson_hands.tensorrt_hands import HandROI, TensorRTHandsStrategy


class LandmarkEngine:
    def __init__(self, scores):
        self.scores = deque(scores)

    def infer(self, image):
        # Finite, non-degenerate landmarks inside a model crop.
        xyz = np.zeros((21, 3), np.float32)
        xyz[:, :2] = [[80 + i % 5 * 12, 170 - i // 5 * 20] for i in range(21)]
        return {"Identity": xyz, "Identity_1": np.array([self.scores.popleft()]),
                "Identity_2": np.array([0.9])}


def tracker(scores, detection):
    strategy = object.__new__(TensorRTHandsStrategy)
    strategy.hand = LandmarkEngine(scores)
    strategy.roi = HandROI(160, 120, 180, 0)
    strategy.age = 60
    strategy.previous_timestamp = 0
    calls = []

    def detect(image):
        calls.append(True)
        return detection

    strategy._detect = detect
    return strategy, calls


def test_valid_track_is_not_discarded_after_sixty_frames():
    strategy, calls = tracker([0.99], detection=None)
    result = strategy.process(np.zeros((240, 320, 3), np.uint8), 16)
    assert len(result["hands"]) == 1
    assert result["hands"][0]["confidence_kind"] == "hand_presence"
    assert calls == []
    assert result["recognition"]["state"] == "tracking"


def test_lost_crop_recovers_in_same_frame_with_new_palm():
    strategy, calls = tracker([0.1, 0.99], HandROI(160, 120, 180, 0))
    result = strategy.process(np.zeros((240, 320, 3), np.uint8), 16)
    assert len(result["hands"]) == 1
    assert len(calls) == 1
    assert result["recognition"]["recovery_attempted"]
    assert result["recognition"]["state"] == "reacquired"


def test_failed_recovery_does_not_return_stale_hand():
    strategy, calls = tracker([0.1], detection=None)
    result = strategy.process(np.zeros((240, 320, 3), np.uint8), 16)
    assert result["hands"] == []
    assert strategy.roi is None
    assert len(calls) == 1


def test_bad_new_landmarks_do_not_create_unbounded_retry():
    strategy, calls = tracker([0.1, 0.2], HandROI(160, 120, 180, 0))
    result = strategy.process(np.zeros((240, 320, 3), np.uint8), 16)
    assert result["hands"] == []
    assert strategy.roi is None
    assert len(calls) == 1


def test_stale_timestamp_requires_reacquisition():
    strategy, calls = tracker([], detection=None)
    result = strategy.process(np.zeros((240, 320, 3), np.uint8), 1000)
    assert result["hands"] == []
    assert len(calls) == 1
