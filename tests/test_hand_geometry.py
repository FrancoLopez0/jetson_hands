import cv2
import numpy as np
import pytest

from jetson_hands.tensorrt_hands import HandROI, hand_angle, palm_anchors, tracking_roi


@pytest.mark.parametrize("angle", [-2.5, -0.7, 0, 1.1, 2.9])
def test_roi_affine_round_trip_and_orientation(angle):
    roi = HandROI(150, 90, 170, angle)
    axis_up = np.array([np.sin(angle), -np.cos(angle)])
    points = np.array([[roi.cx, roi.cy], np.array([roi.cx, roi.cy]) + axis_up * 40], np.float32)
    transformed = cv2.transform(points[None], roi.matrix())[0]
    np.testing.assert_allclose(transformed[0], [112, 112], atol=1e-4)
    assert transformed[1, 1] < 112
    np.testing.assert_allclose(transformed[1, 0], 112, atol=1e-4)
    restored = cv2.transform(transformed[None], cv2.invertAffineTransform(roi.matrix()))[0]
    np.testing.assert_allclose(restored, points, atol=1e-4)


def test_tracking_roi_contains_all_landmarks_after_rotation():
    points = np.array([[100 + i % 5 * 7, 180 - i // 5 * 16] for i in range(21)], np.float32)
    roi = tracking_roi(points)
    transformed = cv2.transform(points[None], roi.matrix())[0]
    assert np.all(transformed >= 0) and np.all(transformed <= 224)
    assert abs(hand_angle(np.array([0, 1]), np.array([0, 0]))) < 1e-6


def test_ssd_anchor_layout_matches_pinned_palm_model():
    anchors = palm_anchors()
    assert anchors.shape == (2016, 2)
    np.testing.assert_allclose(anchors[:2], [[1 / 48, 1 / 48]] * 2)
    np.testing.assert_allclose(anchors[1152:1158], [[1 / 24, 1 / 24]] * 6)


def test_tracking_roi_does_not_grow_with_distal_fingertip_outliers():
    points = np.array([[100 + i % 5 * 7, 180 - i // 5 * 16] for i in range(21)], np.float32)
    before = tracking_roi(points)
    points[[4, 7, 8, 11, 12, 15, 16, 19, 20]] *= 3
    assert tracking_roi(points) == before
