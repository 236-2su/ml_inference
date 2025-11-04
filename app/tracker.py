"""Simplified tracker abstraction."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable

from .detector import Detection
from .stream_listener import Frame


@dataclass
class Track:
    track_id: int
    detection: Detection
    first_seen: int
    last_seen: int


class Tracker:
    def __init__(self) -> None:
        self._id_gen = itertools.count(start=1)
        self._tracks: Dict[int, Track] = {}

    def update(self, frame: Frame, detections: Iterable[Detection]) -> Iterable[Track]:
        """Assign simple sequential IDs to detections."""
        updated_tracks: list[Track] = []
        for detection in detections:
            track_id = next(self._id_gen)
            track = Track(
                track_id=track_id,
                detection=detection,
                first_seen=frame.index,
                last_seen=frame.index,
            )
            self._tracks[track_id] = track
            updated_tracks.append(track)
        return updated_tracks
