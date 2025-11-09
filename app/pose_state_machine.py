"""State machine that derives human-status events from pose predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Set, Tuple

from .pose_estimator import PoseResult
from .tracker import Track


@dataclass
class PoseStatusUpdate:
    status: Optional[str] = None
    duration_seconds: Optional[int] = None


@dataclass
class _TrackPoseState:
    current_label: str = "unknown"
    previous_label: str = "unknown"
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    lying_since: Optional[datetime] = None
    fall_reported_for: Optional[datetime] = None
    watch_reported_for: Optional[datetime] = None
    alert_reported_for: Optional[datetime] = None


class PoseStateMachine:
    """Derives fall/heatstroke statuses from successive pose labels per track."""

    def __init__(
        self,
        fall_transition_labels: Tuple[str, ...] = ("standing", "sitting", "crouching", "unknown"),
        heatstroke_watch_seconds: int = 300,
        heatstroke_alert_seconds: int = 900,
        idle_ttl_seconds: int = 120,
        min_pose_confidence: float = 0.35,
    ) -> None:
        self._states: Dict[int, _TrackPoseState] = {}
        self._fall_transition_labels = fall_transition_labels
        self._watch_seconds = max(0, heatstroke_watch_seconds)
        self._alert_seconds = max(self._watch_seconds, heatstroke_alert_seconds)
        self._idle_ttl_seconds = max(5, idle_ttl_seconds)
        self._min_pose_confidence = min(max(min_pose_confidence, 0.0), 1.0)

    def update(self, track: Track, pose: PoseResult, frame_timestamp: datetime) -> PoseStatusUpdate:
        """Update pose state for the given track and return any status change."""
        state = self._states.setdefault(track.track_id, _TrackPoseState())
        state.updated_at = frame_timestamp
        status = PoseStatusUpdate()

        effective_label = pose.label if pose.confidence >= self._min_pose_confidence else "unknown"
        if effective_label != state.current_label:
            state.previous_label = state.current_label
            state.current_label = effective_label
            if effective_label == "lying":
                state.lying_since = frame_timestamp
                state.fall_reported_for = None
                state.watch_reported_for = None
                state.alert_reported_for = None
            else:
                state.lying_since = None
                state.fall_reported_for = None
                state.watch_reported_for = None
                state.alert_reported_for = None

        if state.current_label == "lying" and state.lying_since is None:
            state.lying_since = frame_timestamp

        if (
            state.current_label == "lying"
            and state.lying_since
            and state.previous_label in self._fall_transition_labels
            and state.fall_reported_for is None
            and pose.confidence >= self._min_pose_confidence
        ):
            state.fall_reported_for = state.lying_since
            status.status = "fall_detected"
            status.duration_seconds = 0
            return status

        if state.current_label == "lying" and state.lying_since:
            elapsed = int((frame_timestamp - state.lying_since).total_seconds())
            if elapsed >= self._alert_seconds and state.alert_reported_for is None:
                state.alert_reported_for = state.lying_since
                status.status = "heatstroke_alert"
                status.duration_seconds = elapsed
                return status
            if elapsed >= self._watch_seconds and state.watch_reported_for is None:
                state.watch_reported_for = state.lying_since
                status.status = "heatstroke_watch"
                status.duration_seconds = elapsed
                return status

        return status

    def prune(self, active_track_ids: Iterable[int], now: datetime) -> None:
        """Drop pose state for tracks that have been idle longer than the TTL."""
        active: Set[int] = set(active_track_ids)
        for track_id, state in list(self._states.items()):
            if track_id in active:
                continue
            idle_seconds = (now - state.updated_at).total_seconds()
            if idle_seconds > self._idle_ttl_seconds:
                del self._states[track_id]
