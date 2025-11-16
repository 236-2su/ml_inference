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
    """Metadata for event serialization."""

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
        inference_latency_ms: int,
        status: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        snapshot_b64: Optional[str] = None,
    ) -> Dict[str, object]:
        detection = track.detection
        event: Dict[str, object] = {
            "event_id": str(uuid.uuid4()),
            "category": "human",
            "stream_id": ctx.stream_id,
            "track_id": track.track_id,
            "timestamp_utc": frame_timestamp.isoformat(),
            "bbox": detection.bbox,
            "confidence": detection.confidence,
            "inference_latency_ms": inference_latency_ms,
            "pose_label": pose.label,
            "pose_confidence": pose.confidence,
        }
        if pose.keypoints:
            event["keypoints"] = [
                [float(x), float(y), float(conf)]
                for x, y, conf in pose.keypoints
            ]
        # status 필드 제거됨 - 백엔드에서 불필요
        if duration_seconds is not None:
            event["duration_seconds"] = duration_seconds
        if snapshot_b64:
            event["image_jpeg_base64"] = snapshot_b64
        return event

    def build_wildlife_event(
        self,
        ctx: EventContext,
        track: Track,
        frame_timestamp: datetime,
        inference_latency_ms: int,
        snapshot_b64: Optional[str] = None,
    ) -> Dict[str, object]:
        detection = track.detection
        event: Dict[str, object] = {
            "event_id": str(uuid.uuid4()),
            "category": "wildlife",
            "stream_id": ctx.stream_id,
            "species": detection.label,
            "species_confidence": detection.confidence,
            "track_id": track.track_id,
            "timestamp_utc": frame_timestamp.isoformat(),
            "bbox": detection.bbox,
            "confidence": detection.confidence,
            "inference_latency_ms": inference_latency_ms,
        }
        if snapshot_b64:
            event["image_jpeg_base64"] = snapshot_b64
        return event
