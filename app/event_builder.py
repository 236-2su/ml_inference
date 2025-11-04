"""Helpers to build outbound event payloads."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from .detector import Detection
from .pose_estimator import PoseEstimator, PoseResult
from .tracker import Track


@dataclass
class EventContext:
    stream_id: str
    gpu_enabled: bool


class EventBuilder:
    def __init__(self, pose_estimator: Optional[PoseEstimator] = None) -> None:
        self.pose_estimator = pose_estimator

    def build_human_event(
        self,
        ctx: EventContext,
        track: Track,
        frame_timestamp: datetime,
        pose: PoseResult,
    ) -> Dict[str, object]:
        detection = track.detection
        return {
            "event_id": str(uuid.uuid4()),
            "category": "human",
            "stream_id": ctx.stream_id,
            "track_id": track.track_id,
            "timestamp_utc": frame_timestamp.isoformat(),
            "bbox": detection.bbox,
            "confidence": detection.confidence,
            "pose_label": pose.label,
            "pose_confidence": pose.confidence,
            "keypoints": pose.keypoints,
            "gpu_enabled": ctx.gpu_enabled,
        }

    def build_wildlife_event(
        self,
        ctx: EventContext,
        track: Track,
        frame_timestamp: datetime,
    ) -> Dict[str, object]:
        detection = track.detection
        threat_level = "warning" if detection.label in {"boar"} else "info"
        return {
            "event_id": str(uuid.uuid4()),
            "category": "wildlife",
            "species": detection.label,
            "species_confidence": detection.confidence,
            "threat_level": threat_level,
            "stream_id": ctx.stream_id,
            "track_id": track.track_id,
            "timestamp_utc": frame_timestamp.isoformat(),
            "bbox": detection.bbox,
            "gpu_enabled": ctx.gpu_enabled,
        }
