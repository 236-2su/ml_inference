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
    media_output_root: str = Field(
        "./artifacts",
        description="Base directory used for temporary snapshots or clips.",
    )
    gpu_enabled: bool = Field(
        False,
        description="Feature flag toggled during GPU cutover.",
    )
    default_fps: int = Field(
        12,
        description="Processing frame rate cap for the prototype stage.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
