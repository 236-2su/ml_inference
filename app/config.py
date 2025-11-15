"""Application configuration loaded from environment variables.

The defaults here mirror the placeholders listed in `docs/settings.md`. Swap them
with real values (or rely on dotenv / SSM) when wiring the pipeline for real.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    media_rpi_rtsp_url: str = Field(
        "rtsp://example.local:8554/stream",
        description="Primary RTSP source pulled by mediaMTX.",
    )
    scarecrow_serial_number: str = Field(
        "00000000",
        description="Scarecrow device serial number for HEOBY backend identification (8 digits).",
    )
    fastapi_endpoint: HttpUrl = Field(
        "http://localhost:8000/events",
        description="Endpoint that receives inference events.",
    )
    fastapi_token: Optional[str] = Field(
        "local-dev-token",
        description="Bearer token injected as Authorization header when present.",
    )
    yolo_model_path: str = Field(
        "models/yolov8n-custom.pt",
        description="File system path or URI to the YOLO weights.",
    )
    yolo_conf_threshold: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to keep a detection.",
    )
    yolo_iou_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold used during NMS.",
    )
    yolo_human_model_path: Optional[str] = Field(
        None,
        description="Secondary YOLO weights dedicated to human detection.",
    )
    yolo_human_conf_threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for the human-only model.",
    )
    yolo_human_iou_threshold: float = Field(
        0.45,
        ge=0.0,
        le=1.0,
        description="IoU used by the human-only model.",
    )
    human_skip_conf_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="When a human detection exceeds this confidence, wildlife inference is skipped for that frame.",
    )
    yolo_pose_model_path: str = Field(
        "yolov8x-pose.pt",
        description="Ultralytics pose model used for human posture estimation (RTMPose).",
    )
    yolo_pose_conf_threshold: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Confidence threshold applied to pose predictions.",
    )
    svm_pose_model_path: Optional[str] = Field(
        "models/svm_model_sit_stand_lie.pkl",
        description="MediaPipe + SVM pose classifier model path.",
    )
    use_mediapipe_svm: bool = Field(
        True,
        description="Use MediaPipe + SVM instead of RTMPose for pose estimation.",
    )
    mediapipe_min_detection_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detection confidence for MediaPipe Pose.",
    )
    mediapipe_min_tracking_confidence: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum tracking confidence for MediaPipe Pose.",
    )
    pose_keypoint_conf_threshold: float = Field(
        0.002,
        ge=0.0,
        le=1.0,
        description="Minimum keypoint confidence retained when classifying pose.",
    )
    pose_lying_aspect_ratio: float = Field(
        0.65,
        ge=0.1,
        description="If height/width ratio is below this threshold, treat person as lying.",
    )
    pose_lying_torso_angle_deg: float = Field(
        35.0,
        ge=0.0,
        le=90.0,
        description="If torso angle (deg) is below this threshold, treat person as lying.",
    )
    pose_heatstroke_watch_seconds: int = Field(
        300,
        ge=0,
        description="Duration (seconds) of continuous lying before raising a heatstroke watch.",
    )
    pose_heatstroke_alert_seconds: int = Field(
        900,
        ge=0,
        description="Duration (seconds) of continuous lying before raising a heatstroke alert.",
    )
    pose_state_idle_ttl_seconds: int = Field(
        120,
        ge=5,
        description="How long to keep pose state for tracks that disappeared.",
    )
    pose_state_min_confidence: float = Field(
        0.35,
        ge=0.0,
        le=1.0,
        description="Minimum pose confidence to feed into the state machine.",
    )
    media_output_root: str = Field(
        "./artifacts",
        description="Base directory used for temporary snapshots or clips.",
    )
    include_snapshot: bool = Field(
        False,
        description="When true, include a JPEG snapshot in event payloads.",
    )
    gpu_enabled: bool = Field(
        False,
        description="Feature flag toggled during GPU cutover.",
    )
    default_fps: int = Field(
        12,
        description="Processing frame rate cap for the prototype stage.",
    )
    event_filter_enabled: bool = Field(
        True,
        description="Enable per-track event filtering to suppress duplicate payloads.",
    )
    event_filter_pose_change: bool = Field(
        True,
        description="Emit human events when pose labels change.",
    )
    event_filter_presence_change: bool = Field(
        True,
        description="Emit events when new tracks appear or disappear.",
    )
    event_filter_important_status: bool = Field(
        True,
        description="Always emit events when status is fall/heatstroke.",
    )
    event_filter_position_change: bool = Field(
        False,
        description="Emit events when bounding boxes move more than the configured threshold.",
    )
    event_filter_position_threshold: int = Field(
        120,
        ge=0,
        description="Minimum center-point distance (pixels) treated as a position change.",
    )


# NOTE: functools.lru_cache requires an explicit call on Python 3.7.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
