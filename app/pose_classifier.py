"""Pose classification heuristics shared across estimators and tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PoseClassifierConfig",
    "PoseMetrics",
    "classify_pose_from_keypoints",
    "label_pose",
]


@dataclass(frozen=True)
class PoseClassifierConfig:
    """Thresholds that govern how human poses are labeled."""

    min_keypoint_conf: float = 0.3
    lying_aspect_ratio: float = 0.65
    lying_torso_angle_deg: float = 35.0
    crouch_hip_threshold: float = 0.65
    crouch_gap_threshold: float = 0.2


@dataclass(frozen=True)
class PoseMetrics:
    aspect_ratio: float
    torso_angle: Optional[float]
    hip_norm: Optional[float]
    knee_norm: Optional[float]
    ankle_norm: Optional[float]


def classify_pose_from_keypoints(
    keypoints: np.ndarray,
    config: PoseClassifierConfig | None = None,
) -> Tuple[str, float]:
    """Return (label, confidence) given COCO-style keypoints."""
    cfg = config or PoseClassifierConfig()
    metrics = _summarize_keypoints(keypoints, cfg.min_keypoint_conf)
    if metrics is None:
        return "unknown", 0.0

    scores = {
        "lying": _score_lying(metrics, cfg),
        "standing": _score_standing(metrics),
        "crouching": _score_crouching(metrics, cfg),
        "sitting": _score_sitting(metrics),
    }
    if metrics.hip_norm is not None and metrics.hip_norm >= cfg.crouch_hip_threshold:
        scores["crouching"] = min(1.0, scores["crouching"] + 0.15)
        scores["standing"] *= 0.3

    label, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.1:
        return "unknown", 0.0
    return label, float(np.clip(score, 0.0, 1.0))


def label_pose(
    keypoints_xy: np.ndarray,
    keypoint_scores: np.ndarray,
    config: PoseClassifierConfig | None = None,
) -> str:
    """Legacy helper used by quick scripts."""
    if keypoints_xy.shape[0] != keypoint_scores.shape[0]:
        raise ValueError("Keypoint coords and scores must align.")
    combined = np.concatenate(
        (keypoints_xy.astype(np.float32), keypoint_scores.reshape(-1, 1).astype(np.float32)),
        axis=1,
    )
    label, _ = classify_pose_from_keypoints(combined, config=config)
    return label


def _summarize_keypoints(keypoints: np.ndarray, min_conf: float) -> Optional[PoseMetrics]:
    if keypoints.size == 0:
        return None
    conf_mask = keypoints[:, 2] >= min_conf
    if np.count_nonzero(conf_mask) < 4:
        return None
    filtered = keypoints[conf_mask]
    xs = filtered[:, 0]
    ys = filtered[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    if width < 1.0 or height < 1.0:
        return None

    aspect_ratio = height / (width + 1e-6)
    torso_angle = _estimate_torso_angle_deg(keypoints, min_conf)
    hip_norm = _normalized_y(keypoints, [11, 12], ys.min(), height, min_conf)
    knee_norm = _normalized_y(keypoints, [13, 14], ys.min(), height, min_conf)
    ankle_norm = _normalized_y(keypoints, [15, 16], ys.min(), height, min_conf)

    return PoseMetrics(
        aspect_ratio=aspect_ratio,
        torso_angle=torso_angle,
        hip_norm=hip_norm,
        knee_norm=knee_norm,
        ankle_norm=ankle_norm,
    )


def _score_lying(metrics: PoseMetrics, cfg: PoseClassifierConfig) -> float:
    aspect_score = 0.0
    if metrics.aspect_ratio < cfg.lying_aspect_ratio:
        aspect_score = (cfg.lying_aspect_ratio - metrics.aspect_ratio) / cfg.lying_aspect_ratio
    angle_score = 0.0
    if metrics.torso_angle is not None:
        angle_score = max(
            0.0,
            (cfg.lying_torso_angle_deg - metrics.torso_angle) / max(cfg.lying_torso_angle_deg, 1e-6),
        )
    return float(np.clip(max(aspect_score, angle_score), 0.0, 1.0))


def _score_standing(metrics: PoseMetrics) -> float:
    scores: list[float] = []
    aspect_min = 1.35
    aspect_max = 3.0
    if metrics.aspect_ratio > aspect_min:
        denom = max(aspect_max - aspect_min, 1e-6)
        scores.append(np.clip((metrics.aspect_ratio - aspect_min) / denom, 0.0, 1.0))
    if metrics.torso_angle is not None:
        torso_min = 55.0
        scores.append(
            np.clip((metrics.torso_angle - torso_min) / max(90.0 - torso_min, 1e-6), 0.0, 1.0)
        )
    if metrics.hip_norm is not None:
        hip_threshold = 0.75
        scores.append(np.clip((hip_threshold - metrics.hip_norm) / hip_threshold, 0.0, 1.0))
    return float(np.mean(scores)) if scores else 0.0


def _score_crouching(metrics: PoseMetrics, cfg: PoseClassifierConfig) -> float:
    scores: list[float] = []
    if metrics.hip_norm is not None:
        hip_min = 0.6
        scores.append(np.clip((metrics.hip_norm - hip_min) / (1 - hip_min), 0.0, 1.0))
    if metrics.knee_norm is not None and metrics.ankle_norm is not None:
        gap = max(0.0, metrics.ankle_norm - metrics.knee_norm)
        scores.append(np.clip((cfg.crouch_gap_threshold - gap) / cfg.crouch_gap_threshold, 0.0, 1.0))
    aspect_center = 1.1
    scores.append(np.clip(1.0 - abs(metrics.aspect_ratio - aspect_center) / 0.6, 0.0, 1.0))
    if metrics.torso_angle is not None:
        scores.append(np.clip(1.0 - abs(metrics.torso_angle - 55.0) / 40.0, 0.0, 1.0))
    crouch_score = float(np.mean(scores)) if scores else 0.0
    if (
        metrics.hip_norm is not None
        and metrics.knee_norm is not None
        and metrics.ankle_norm is not None
        and metrics.hip_norm >= cfg.crouch_hip_threshold
        and (metrics.ankle_norm - metrics.knee_norm) < cfg.crouch_gap_threshold
    ):
        crouch_score = max(crouch_score + 0.2, 0.8)
    return crouch_score


def _score_sitting(metrics: PoseMetrics) -> float:
    scores: list[float] = []
    scores.append(np.clip(1.0 - abs(metrics.aspect_ratio - 1.0) / 0.7, 0.0, 1.0))
    if metrics.hip_norm is not None:
        scores.append(np.clip(1.0 - abs(metrics.hip_norm - 0.55) / 0.35, 0.0, 1.0))
    if metrics.knee_norm is not None:
        scores.append(np.clip(1.0 - abs(metrics.knee_norm - 0.75) / 0.3, 0.0, 1.0))
    if metrics.ankle_norm is not None and metrics.knee_norm is not None:
        gap = max(0.0, metrics.ankle_norm - metrics.knee_norm)
        scores.append(np.clip(1.0 - abs(gap - 0.25) / 0.25, 0.0, 1.0))
    if metrics.torso_angle is not None:
        scores.append(np.clip((metrics.torso_angle - 35.0) / 30.0, 0.0, 1.0))
    return float(np.mean(scores)) if scores else 0.0


def _normalized_y(
    keypoints: np.ndarray,
    indices: Sequence[int],
    min_y: float,
    height: float,
    min_conf: float,
) -> Optional[float]:
    valid = []
    for idx in indices:
        if idx >= keypoints.shape[0]:
            continue
        kp = keypoints[idx]
        if kp[2] < min_conf:
            continue
        normalized = (kp[1] - min_y) / max(height, 1e-6)
        valid.append(float(np.clip(normalized, 0.0, 1.0)))
    if not valid:
        return None
    return float(np.mean(valid))


def _estimate_torso_angle_deg(keypoints: np.ndarray, min_conf: float) -> Optional[float]:
    """Return the absolute angle (degrees) of the torso line relative to the horizontal axis."""
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if max(needed) >= keypoints.shape[0]:
        return None
    pts = keypoints[needed]
    if (pts[:, 2] < min_conf).any():
        return None

    shoulder_mid = (pts[0][:2] + pts[1][:2]) / 2.0
    hip_mid = (pts[2][:2] + pts[3][:2]) / 2.0
    dx = float(abs(shoulder_mid[0] - hip_mid[0]))
    dy = float(abs(shoulder_mid[1] - hip_mid[1]))
    angle_rad = math.atan2(dy, dx + 1e-6)
    return math.degrees(angle_rad)
