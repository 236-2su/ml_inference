"""Filter duplicate per-track events so downstream systems only see changes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class TrackState:
    """Snapshot of the last event that was allowed to pass through."""

    category: str = "unknown"
    pose_label: str = "unknown"
    species: str = "unknown"
    bbox: BBox = (0, 0, 0, 0)
    is_present: bool = False


class EventFilter:
    """Simple in-memory filter that debounces repetitive per-track events."""

    _DEFAULT_IMPORTANT_STATUSES = ("fall_detected", "heatstroke_watch", "heatstroke_alert")

    def __init__(
        self,
        enable_pose_change: bool = True,
        enable_presence_change: bool = True,
        enable_important_status: bool = True,
        enable_position_change: bool = False,
        position_threshold: int = 100,
        important_statuses: Optional[Sequence[str]] = None,
    ):
        self.enable_pose_change = enable_pose_change
        self.enable_presence_change = enable_presence_change
        self.enable_important_status = enable_important_status
        self.enable_position_change = enable_position_change
        self.position_threshold = max(0, position_threshold)
        statuses = important_statuses or self._DEFAULT_IMPORTANT_STATUSES
        self._important_statuses = {status for status in statuses if status}
        self._track_states: Dict[int, TrackState] = {}

    def should_send_event(self, event: dict) -> bool:
        """Return True when the event should be dispatched."""

        track_id = event.get("track_id")
        if track_id is None:
            return True

        status = event.get("status")
        status_triggered = (
            self.enable_important_status and status is not None and status in self._important_statuses
        )

        prev_state = self._track_states.get(track_id)
        category = event.get("category", prev_state.category if prev_state else "unknown")
        pose_label = event.get("pose_label", prev_state.pose_label if prev_state else "unknown")
        species = event.get("species", prev_state.species if prev_state else "unknown")
        bbox = self._normalize_bbox(event.get("bbox"))

        should_send = False
        if prev_state is None:
            should_send = self.enable_presence_change
        else:
            if prev_state.category != category:
                should_send = True
            elif self.enable_pose_change and category == "human" and pose_label != prev_state.pose_label:
                should_send = True
            elif category == "wildlife" and species != prev_state.species:
                should_send = True
            elif self.enable_position_change and self._has_position_changed(prev_state.bbox, bbox):
                should_send = True

        if status_triggered:
            should_send = True

        self._track_states[track_id] = TrackState(
            category=category,
            pose_label=pose_label,
            species=species,
            bbox=bbox,
            is_present=True,
        )
        return should_send

    def mark_track_disappeared(self, track_id: int) -> bool:
        """Remove cached state for a disappeared track and emit presence-change events."""

        if not self.enable_presence_change or track_id not in self._track_states:
            return False
        del self._track_states[track_id]
        return True

    def prune(self, active_track_ids: Iterable[int]) -> None:
        """Drop cached states for tracks no longer monitored."""

        active = set(active_track_ids)
        for track_id in list(self._track_states.keys()):
            if track_id not in active:
                del self._track_states[track_id]

    def get_stats(self) -> dict:
        """Return basic filter stats used for debugging."""

        return {
            "active_tracks": len(self._track_states),
            "tracked_ids": list(self._track_states.keys()),
        }

    @staticmethod
    def _normalize_bbox(raw_bbox: object) -> BBox:
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
            try:
                return (
                    float(raw_bbox[0]),
                    float(raw_bbox[1]),
                    float(raw_bbox[2]),
                    float(raw_bbox[3]),
                )
            except (TypeError, ValueError):
                return (0, 0, 0, 0)
        return (0, 0, 0, 0)

    def _has_position_changed(self, prev_bbox: BBox, current_bbox: BBox) -> bool:
        if prev_bbox == (0, 0, 0, 0) or current_bbox == (0, 0, 0, 0):
            return False

        prev_x = prev_bbox[0] + prev_bbox[2] / 2
        prev_y = prev_bbox[1] + prev_bbox[3] / 2
        curr_x = current_bbox[0] + current_bbox[2] / 2
        curr_y = current_bbox[1] + current_bbox[3] / 2

        distance = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
        return distance > self.position_threshold
