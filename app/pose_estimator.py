"""Placeholder pose estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .detector import Detection
from .stream_listener import Frame


default_keypoints: List[tuple[float, float, float]] = []


@dataclass
class PoseResult:
    label: str
    confidence: float
    keypoints: List[tuple[float, float, float]]


class PoseEstimator:
    """Stub estimator returning unknown pose for all detections."""

    def __init__(self, model_path: str | None = None, conf_threshold: float = 0.25) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def infer(self, frame: Frame, detection: Detection) -> PoseResult:
        return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
