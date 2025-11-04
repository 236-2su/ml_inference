"""Streaming interface for pulling frames from mediaMTX (skeleton version)."""

from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generator, Iterable, Optional


log = logging.getLogger(__name__)


@dataclass
class Frame:
    index: int
    timestamp: datetime
    image: Optional[bytes] = None  # placeholder buffer


class StreamListener:
    """Stub RTSP listener that yields synthetic frames for now."""

    def __init__(self, source_url: str, fps_limit: int = 12) -> None:
        self.source_url = source_url
        self.fps_limit = max(fps_limit, 1)
        self._interval = 1.0 / float(self.fps_limit)

    def connect(self) -> None:
        """Simulate the connection step."""
        log.info("Connecting to stream %s (simulated)", self.source_url)

    def frames(self) -> Iterable[Frame]:
        """Yield placeholder frames at the configured frame rate."""
        self.connect()
        for idx in itertools.count():
            yield Frame(index=idx, timestamp=datetime.now(tz=timezone.utc))
            time.sleep(self._interval)

    def once(self, limit: int = 5) -> Generator[Frame, None, None]:
        """Yield a finite number of frames; handy for tests."""
        for frame in itertools.islice(self.frames(), limit):
            yield frame
