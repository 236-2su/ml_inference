"""RTMPose skeleton returning canned posture results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .detector import Detection
from .stream_listener import Frame


@dataclass
class PoseResult:
    label: str
    confidence: float
    keypoints: List[tuple[float, float, float]]


class PoseEstimator:
    def infer(self, frame: Frame, detection: Detection) -> PoseResult:
        """Return a dummy pose based on the frame index."""
        if frame.index % 60 == 0:
            label = "collapse"
            confidence = 0.9
        else:
            label = "standing"
            confidence = 0.7

        return PoseResult(
            label=label,
            confidence=confidence,
            keypoints=[(0.0, 0.0, 1.0)],  # placeholder result
        )
