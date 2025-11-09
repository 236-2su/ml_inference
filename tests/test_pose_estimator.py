from __future__ import annotations

import numpy as np

from app.pose_estimator import classify_pose_from_keypoints


def _build_keypoints(points: dict[int, tuple[float, float, float]]) -> np.ndarray:
    arr = np.zeros((17, 3), dtype=np.float32)
    for idx, (x, y, conf) in points.items():
        arr[idx] = (x, y, conf)
    return arr


def _classify(keypoints: np.ndarray) -> tuple[str, float]:
    return classify_pose_from_keypoints(
        keypoints,
        min_conf=0.3,
        lying_aspect_ratio=0.65,
        lying_torso_angle_deg=35,
    )


def test_classify_pose_standing():
    keypoints = _build_keypoints(
        {
            5: (50, 5, 0.9),
            6: (80, 8, 0.9),
            11: (55, 120, 0.9),
            12: (78, 122, 0.9),
            13: (57, 170, 0.9),
            14: (76, 172, 0.9),
            15: (58, 210, 0.9),
            16: (74, 212, 0.9),
        }
    )
    label, confidence = _classify(keypoints)
    assert label == "standing"
    assert confidence > 0.5


def test_classify_pose_lying():
    keypoints = _build_keypoints(
        {
            5: (20, 80, 0.9),
            6: (120, 82, 0.9),
            11: (25, 95, 0.9),
            12: (125, 97, 0.9),
            13: (30, 105, 0.9),
            14: (130, 107, 0.9),
        }
    )
    label, confidence = _classify(keypoints)
    assert label == "lying"
    assert confidence > 0.5


def test_classify_pose_crouching():
    keypoints = _build_keypoints(
        {
            5: (30, 10, 0.9),
            6: (110, 12, 0.9),
            11: (42, 150, 0.9),
            12: (98, 152, 0.9),
            13: (46, 190, 0.9),
            14: (94, 192, 0.9),
            15: (48, 215, 0.9),
            16: (92, 217, 0.9),
        }
    )
    label, confidence = _classify(keypoints)
    assert label == "crouching"
    assert confidence > 0.2


def test_classify_pose_sitting():
    keypoints = _build_keypoints(
        {
            5: (40, 15, 0.9),
            6: (100, 18, 0.9),
            11: (50, 80, 0.9),
            12: (92, 82, 0.9),
            13: (55, 115, 0.9),
            14: (88, 117, 0.9),
            15: (60, 145, 0.9),
            16: (84, 147, 0.9),
        }
    )
    label, confidence = _classify(keypoints)
    assert label == "sitting"
    assert confidence > 0.2


def test_classify_pose_unknown_with_sparse_points():
    keypoints = _build_keypoints({0: (10, 10, 0.2), 1: (12, 11, 0.2)})
    label, confidence = _classify(keypoints)
    assert label == "unknown"
    assert confidence == 0.0
