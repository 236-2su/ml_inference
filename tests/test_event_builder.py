from datetime import datetime, timezone

from ml_inference.app.event_builder import EventBuilder, EventContext
from ml_inference.app.detector import Detection
from ml_inference.app.pose_estimator import PoseEstimator, PoseResult
from ml_inference.app.stream_listener import Frame
from ml_inference.app.tracker import Track


def test_build_human_event():
    builder = EventBuilder(pose_estimator=PoseEstimator())
    ctx = EventContext(stream_id="stream-1", gpu_enabled=False)
    detection = Detection(label="human", confidence=0.9, bbox=(1, 2, 3, 4))
    track = Track(track_id=42, detection=detection, first_seen=0, last_seen=0)
    pose = PoseResult(label="standing", confidence=0.8, keypoints=[(0.0, 0.0, 1.0)])
    timestamp = datetime.now(tz=timezone.utc)

    event = builder.build_human_event(ctx, track, timestamp, pose)

    assert event["category"] == "human"
    assert event["pose_label"] == "standing"
    assert event["bbox"] == (1, 2, 3, 4)


def test_build_wildlife_event():
    builder = EventBuilder()
    ctx = EventContext(stream_id="stream-1", gpu_enabled=True)
    detection = Detection(label="boar", confidence=0.7, bbox=(10, 20, 30, 40))
    track = Track(track_id=99, detection=detection, first_seen=0, last_seen=0)
    timestamp = datetime.now(tz=timezone.utc)

    event = builder.build_wildlife_event(ctx, track, timestamp)

    assert event["category"] == "wildlife"
    assert event["threat_level"] == "warning"
    assert event["gpu_enabled"] is True
