"""Streaming interface for pulling frames from mediaMTX."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, Iterable, Optional

import numpy as np

try:  # pragma: no cover - runtime dependency
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "opencv-python-headless is required for RTSP ingestion. "
        "Install it via `pip install opencv-python-headless`."
    ) from exc


log = logging.getLogger(__name__)


@dataclass
class Frame:
    index: int
    timestamp: datetime
    image: Optional[np.ndarray] = None


class StreamListener:
    """RTSP listener backed by OpenCV VideoCapture."""

    def __init__(
        self,
        source_url: str,
        fps_limit: float = 12.0,
        reconnect_delay: float = 3.0,
    ) -> None:
        self.source_url = source_url
        fps_value = float(fps_limit)
        if fps_value <= 0:
            raise ValueError("fps_limit must be greater than zero")
        self.fps_limit = fps_value
        self._interval = 1.0 / self.fps_limit
        self._reconnect_delay = max(reconnect_delay, 1.0)
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_index = 0

    def connect(self) -> None:
        """Open the RTSP stream."""
        if self._cap is not None and self._cap.isOpened():
            return
        log.info("Connecting to stream %s", self.source_url)
        # Use RTSP over TCP with larger buffer for better stability over long distances
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|buffer_size;10240000|max_delay;500000000|reorder_queue_size;5000"
        self._cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
        if self._cap.isOpened():
            # Increase OpenCV buffer to handle latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        else:
            raise RuntimeError(f"Unable to open RTSP stream {self.source_url}")

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        success, frame = self._cap.read()
        if not success or frame is None:
            return None
        return frame

    def _reconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
        log.warning("RTSP stream %s disconnected; retrying in %.1fs", self.source_url, self._reconnect_delay)
        time.sleep(self._reconnect_delay)
        self.connect()

    def frames(self) -> Iterable[Frame]:
        """Yield frames at approximately the configured FPS."""
        self.connect()
        last_emit = 0.0
        while True:
            start_time = time.perf_counter()
            frame = self._read_frame()
            if frame is None:
                self._reconnect()
                continue
            now = datetime.now(tz=timezone.utc)
            self._frame_index += 1

            elapsed = start_time - last_emit
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_emit = time.perf_counter()
            yield Frame(index=self._frame_index, timestamp=now, image=frame)

    def once(self, limit: int = 5) -> Generator[Frame, None, None]:
        """Yield a finite number of frames; handy for tests."""
        for idx, frame in enumerate(self.frames()):
            if idx >= limit:
                break
            yield frame
