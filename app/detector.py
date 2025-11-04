"""YOLO detector stub.

Real implementation should load the configured model and run inference on
incoming frames. For the skeleton we emit mock detections on a cadence so the
rest of the pipeline can be developed independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from .stream_listener import Frame

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h


class Detector:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        log.info("Detector initialized with model %s (placeholder)", model_path)

    def detect(self, frame: Frame) -> List[Detection]:
        """Return mock detections to drive downstream components."""
        if frame.index % 15 == 0:
            return [
                Detection(label="human", confidence=0.82, bbox=(100, 120, 80, 160)),
            ]
        if frame.index % 40 == 0:
            return [
                Detection(label="boar", confidence=0.74, bbox=(220, 180, 140, 120)),
            ]
        return []
