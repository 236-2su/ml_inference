"""Streaming interface backed by OpenCV VideoCapture."""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, Iterable, Optional, Union

try:
    import cv2
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "opencv-python-headless is required for the StreamListener. "
        "Install it via `pip install opencv-python-headless`."
    ) from exc

import numpy as np


log = logging.getLogger(__name__)


@dataclass
class Frame:
    index: int
    timestamp: datetime
    image: Optional[np.ndarray] = None


class StreamListener:
    """RTSP/Webcam listener with basic reconnection handling."""

    def __init__(
        self,
        source_url: str,
        fps_limit: int = 12,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.source_url = source_url
        self.fps_limit = max(fps_limit, 1)
        self._interval = 1.0 / float(self.fps_limit)
        self._reconnect_delay = max(reconnect_delay, 1.0)
        self._capture: Optional[cv2.VideoCapture] = None
        self._resolved_source: Union[int, str] = self._resolve_source(source_url)
        self._last_frame_ts = 0.0

    def connect(self) -> None:
        """Connect to the source immediately or raise."""
        self._open_capture()

    def frames(self) -> Iterable[Frame]:
        """Yield frames from the configured source with FPS limiting."""
        self.connect()
        frame_counter = itertools.count()
        while True:
            image = self._read_frame_with_retry()
            idx = next(frame_counter)
            yield Frame(
                index=idx,
                timestamp=datetime.now(tz=timezone.utc),
                image=image,
            )

    def once(self, limit: int = 5) -> Generator[Frame, None, None]:
        """Yield a finite number of frames; handy for tests."""
        for frame in itertools.islice(self.frames(), limit):
            yield frame

    def _resolve_source(self, source_url: str) -> Union[int, str]:
        stripped = source_url.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    def _open_capture(self) -> None:
        self._release_capture()
        log.info("Connecting to stream %s", self.source_url)
        capture = cv2.VideoCapture(self._resolved_source)
        if not capture.isOpened():
            capture.release()
            raise ConnectionError(f"Unable to open stream {self.source_url}")
        self._capture = capture
        self._last_frame_ts = 0.0

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read_frame_with_retry(self) -> np.ndarray:
        while True:
            try:
                return self._read_frame()
            except ConnectionError as exc:
                log.warning("Stream connection lost: %s", exc)
            except RuntimeError as exc:
                log.warning("Failed to read frame: %s", exc)
            time.sleep(self._reconnect_delay)
            try:
                self._open_capture()
            except ConnectionError as exc:
                log.error("Reconnect attempt failed: %s", exc)

    def _read_frame(self) -> np.ndarray:
        if self._capture is None or not self._capture.isOpened():
            raise ConnectionError("VideoCapture is not open")
        ok, image = self._capture.read()
        if not ok or image is None:
            self._release_capture()
            raise RuntimeError("Capture returned empty frame")
        self._enforce_fps()
        return image

    def _enforce_fps(self) -> None:
        now = time.perf_counter()
        if self._last_frame_ts:
            sleep_for = self._interval - (now - self._last_frame_ts)
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.perf_counter()
        self._last_frame_ts = now

    def __del__(self) -> None:  # pragma: no cover - cleanup helper
        self._release_capture()

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream listener quick test")
    parser.add_argument("--source", default="0", help="RTSP URL, webcam index (as string), or file path")
    parser.add_argument("--fps", type=int, default=12, help="Frame rate limit")
    parser.add_argument("--limit", type=int, default=5, help="Number of frames to pull")
    parser.add_argument("--reconnect-delay", type=float, default=5.0, help="Seconds to wait before reconnect attempts")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover - manual utility
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    listener = StreamListener(
        source_url=args.source,
        fps_limit=args.fps,
        reconnect_delay=args.reconnect_delay,
    )
    for frame in listener.once(limit=args.limit):
        shape = None if frame.image is None else "x".join(str(dim) for dim in frame.image.shape)
        print(f"Frame {frame.index} at {frame.timestamp.isoformat()} image_shape={shape}")

