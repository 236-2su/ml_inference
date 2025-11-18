"""Simplified tracker abstraction."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List

from .detector import Detection
from .stream_listener import Frame


@dataclass
class Track:
    track_id: int
    detection: Detection
    first_seen: int
    last_seen: int
    first_seen_at: datetime
    last_seen_at: datetime
    missed_frames: int = 0


class Tracker:
    """Lightweight IoU-based tracker that keeps IDs stable across frames."""

    def __init__(self, iou_threshold: float = 0.4, max_idle_frames: int = 60) -> None:
        self._id_gen = itertools.count(start=1)
        self._tracks: Dict[int, Track] = {}
        self._iou_threshold = iou_threshold
        self._max_idle_frames = max(1, max_idle_frames)

    def update(self, frame: Frame, detections: Iterable[Detection]) -> List[Track]:
        """Associate detections with existing tracks using greedy IoU matching."""
        detections_list = list(detections)
        updated_tracks: list[Track] = []
        unmatched_tracks = set(self._tracks.keys())

        for detection in detections_list:
            best_track_id = None
            best_iou = 0.0
            for track_id in list(unmatched_tracks):
                track = self._tracks[track_id]
                iou = self._compute_iou(track.detection.bbox, detection.bbox)
                if iou >= self._iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                track = self._tracks[best_track_id]
                track.detection = detection
                track.last_seen = frame.index
                track.last_seen_at = frame.timestamp
                track.missed_frames = 0
                updated_tracks.append(track)
                unmatched_tracks.remove(best_track_id)
            else:
                track_id = next(self._id_gen)
                track = Track(
                    track_id=track_id,
                    detection=detection,
                    first_seen=frame.index,
                    last_seen=frame.index,
                    first_seen_at=frame.timestamp,
                    last_seen_at=frame.timestamp,
                )
                self._tracks[track_id] = track
                updated_tracks.append(track)

        # Age unmatched tracks and drop stale ones.
        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.missed_frames += 1
            track.last_seen_at = frame.timestamp
            if track.missed_frames > self._max_idle_frames:
                del self._tracks[track_id]

        return updated_tracks

    @staticmethod
    def _compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
        """Compute IoU between two (x, y, w, h) boxes."""
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def active_track_ids(self) -> Iterable[int]:
        """Return the set of track IDs that are still being tracked."""
        return tuple(self._tracks.keys())
