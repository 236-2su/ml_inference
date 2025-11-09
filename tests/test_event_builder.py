from datetime import datetime, timezone

from app.detector import Detection
from app.event_builder import EventBuilder, EventContext
from app.pose_estimator import PoseEstimator, PoseResult
from app.tracker import Track


def test_build_human_event():
    builder = EventBuilder(pose_estimator=PoseEstimator())
    ctx = EventContext(stream_id="stream-1", gpu_enabled=False)
    detection = Detection(label="human", confidence=0.9, bbox=(1, 2, 3, 4))
    track = Track(track_id=42, detection=detection, first_seen=0, last_seen=0)
    pose = PoseResult(label="standing", confidence=0.8, keypoints=[(0.0, 0.0, 1.0)])
    timestamp = datetime.now(tz=timezone.utc)

    event = builder.build_human_event(
        ctx=ctx,
        track=track,
        frame_timestamp=timestamp,
        pose=pose,
        inference_latency_ms=123,
        status="fall_detected",
        duration_seconds=17,
        snapshot_b64="snapshot",
    )

    assert event["category"] == "human"
    assert event["pose_label"] == "standing"
    assert event["inference_latency_ms"] == 123
    assert event["status"] == "fall_detected"
    assert event["duration_seconds"] == 17
    assert event["image_jpeg_base64"] == "snapshot"
    assert "stream_id" not in event


def test_build_wildlife_event():
    builder = EventBuilder()
    ctx = EventContext(stream_id="stream-1", gpu_enabled=True)
    detection = Detection(label="boar", confidence=0.7, bbox=(10, 20, 30, 40))
    track = Track(track_id=99, detection=detection, first_seen=0, last_seen=0)
    timestamp = datetime.now(tz=timezone.utc)

    event = builder.build_wildlife_event(
        ctx=ctx,
        track=track,
        frame_timestamp=timestamp,
        inference_latency_ms=77,
        snapshot_b64=None,
    )

    assert event["category"] == "wildlife"
    assert event["species"] == "boar"
    assert event["species_confidence"] == 0.7
    assert event["inference_latency_ms"] == 77
    assert "gpu_enabled" not in event
