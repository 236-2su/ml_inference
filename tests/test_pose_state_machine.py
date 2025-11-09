from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.detector import Detection
from app.pose_estimator import PoseResult
from app.pose_state_machine import PoseStateMachine
from app.tracker import Track


def _make_track(track_id: int = 1) -> Track:
    now = datetime.now(tz=timezone.utc)
    detection = Detection(label="human", confidence=0.9, bbox=(0, 0, 10, 10))
    return Track(
        track_id=track_id,
        detection=detection,
        first_seen=0,
        last_seen=0,
        first_seen_at=now,
        last_seen_at=now,
    )


def test_fall_detected_on_transition():
    machine = PoseStateMachine(heatstroke_watch_seconds=30, heatstroke_alert_seconds=60)
    track = _make_track()
    now = datetime.now(tz=timezone.utc)
    machine.update(track, PoseResult("standing", 0.9, []), now)
    status = machine.update(track, PoseResult("lying", 0.9, []), now + timedelta(seconds=1))
    assert status.status == "fall_detected"
    assert status.duration_seconds == 0


def test_heatstroke_watch_and_alert():
    machine = PoseStateMachine(heatstroke_watch_seconds=5, heatstroke_alert_seconds=10)
    track = _make_track()
    now = datetime.now(tz=timezone.utc)
    machine.update(track, PoseResult("standing", 0.9, []), now)
    machine.update(track, PoseResult("lying", 0.9, []), now + timedelta(seconds=1))

    watch = machine.update(track, PoseResult("lying", 0.9, []), now + timedelta(seconds=6))
    assert watch.status == "heatstroke_watch"
    assert watch.duration_seconds >= 5

    alert = machine.update(track, PoseResult("lying", 0.9, []), now + timedelta(seconds=11))
    assert alert.status == "heatstroke_alert"
    assert alert.duration_seconds >= 10


def test_prune_expires_inactive_tracks():
    machine = PoseStateMachine(idle_ttl_seconds=5)
    track = _make_track()
    now = datetime.now(tz=timezone.utc)
    machine.update(track, PoseResult("standing", 0.9, []), now)
    machine.prune(active_track_ids=(), now=now + timedelta(seconds=10))
    status = machine.update(track, PoseResult("lying", 0.9, []), now + timedelta(seconds=11))
    assert status.status == "fall_detected"
