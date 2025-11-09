from __future__ import annotations

import numpy as np

from app.pose_estimator import classify_pose_from_keypoints


def _build_keypoints(points: dict[int, tuple[float, float, float]]) -> np.ndarray:
    arr = np.zeros((17, 3), dtype=np.float32)
    for idx, (x, y, conf) in points.items():
        arr[idx] = (x, y, conf)
    return arr


def test_classify_pose_standing():
    keypoints = _build_keypoints(
        {
            5: (50, 10, 0.9),
            6: (55, 10, 0.9),
            11: (52, 60, 0.9),
            12: (57, 60, 0.9),
            0: (53, 5, 0.8),
            15: (52, 95, 0.8),
        }
    )
    label, confidence = classify_pose_from_keypoints(
        keypoints,
        min_conf=0.3,
        lying_aspect_ratio=0.7,
        lying_torso_angle_deg=35,
    )
    assert label == "standing"
    assert confidence > 0.5


def test_classify_pose_lying():
    keypoints = _build_keypoints(
        {
            5: (30, 30, 0.9),
            6: (80, 32, 0.9),
            11: (35, 40, 0.9),
            12: (85, 42, 0.9),
            0: (25, 28, 0.8),
            15: (90, 45, 0.8),
        }
    )
    label, confidence = classify_pose_from_keypoints(
        keypoints,
        min_conf=0.3,
        lying_aspect_ratio=0.7,
        lying_torso_angle_deg=35,
    )
    assert label == "lying"
    assert confidence > 0.5


def test_classify_pose_unknown_with_sparse_points():
    keypoints = _build_keypoints({0: (10, 10, 0.2), 1: (12, 11, 0.2)})
    label, confidence = classify_pose_from_keypoints(
        keypoints,
        min_conf=0.3,
        lying_aspect_ratio=0.7,
        lying_torso_angle_deg=35,
    )
    assert label == "unknown"
    assert confidence == 0.0
