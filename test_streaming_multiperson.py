"""Test script for multi-person streaming pose inference using MediaPipe + SVM."""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import Settings
from app.detector import Detector
from app.pose_estimator_mediapipe import MediaPipePoseEstimator
from app.stream_listener import StreamListener
from app.tracker import Tracker

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_video_multiperson(video_path: str, limit_frames: int = 100):
    """Test multi-person pose detection on a video file.

    Args:
        video_path: Path to video file
        limit_frames: Number of frames to process
    """
    log.info("="*80)
    log.info("Testing Multi-Person Streaming Pose Inference")
    log.info("Video: %s", video_path)
    log.info("="*80)

    # Initialize components
    settings = Settings()

    # Human detector (YOLO)
    log.info("Loading YOLO human detector...")
    human_detector = Detector(
        model_path=settings.yolo_human_model_path,
        conf_threshold=settings.yolo_human_conf_threshold,
        iou_threshold=settings.yolo_human_iou_threshold,
        allowed_labels={"human", "person"},
    )

    # Pose estimator (MediaPipe + SVM)
    log.info("Loading MediaPipe + SVM pose estimator...")
    pose_estimator = MediaPipePoseEstimator(
        svm_model_path=settings.svm_pose_model_path,
        conf_threshold=settings.yolo_pose_conf_threshold,
        min_detection_confidence=settings.mediapipe_min_detection_confidence,
        min_tracking_confidence=settings.mediapipe_min_tracking_confidence,
    )

    # Tracker
    tracker = Tracker()

    # Stream listener
    log.info("Opening video stream...")
    listener = StreamListener(video_path, fps_limit=12)

    log.info("\nProcessing frames...")
    log.info("-"*80)

    frame_count = 0
    total_detections = 0
    pose_counts = {"sitting": 0, "standing": 0, "lying": 0, "unknown": 0}
    track_ids_seen = set()

    try:
        for frame in listener.once(limit=limit_frames):
            frame_count += 1

            # Detect humans
            detections = human_detector.detect(frame)

            if detections:
                total_detections += len(detections)

                # Update tracker
                tracks = tracker.update(frame, detections)

                # Process each tracked person
                for track in tracks:
                    track_ids_seen.add(track.track_id)

                    # Estimate pose
                    pose_result = pose_estimator.infer(frame, track.detection)

                    # Count poses
                    pose_counts[pose_result.label] = pose_counts.get(pose_result.label, 0) + 1

                    # Log every 10th frame or when multiple people detected
                    if frame_count % 10 == 0 or len(tracks) > 1:
                        log.info(
                            "Frame %3d | People: %d | Track %d: %s (conf=%.2f)",
                            frame_count,
                            len(tracks),
                            track.track_id,
                            pose_result.label.upper(),
                            pose_result.confidence
                        )

            elif frame_count % 20 == 0:
                log.info("Frame %3d | No people detected", frame_count)

    except KeyboardInterrupt:
        log.info("\nTest interrupted by user")

    # Print summary
    log.info("="*80)
    log.info("Test Complete - Summary")
    log.info("="*80)
    log.info("Total frames processed: %d", frame_count)
    log.info("Total person detections: %d", total_detections)
    log.info("Unique tracks (people): %d", len(track_ids_seen))
    log.info("\nPose Distribution:")
    for pose, count in sorted(pose_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            percentage = (count / sum(pose_counts.values())) * 100 if sum(pose_counts.values()) > 0 else 0
            log.info("  %-10s: %4d (%.1f%%)", pose.upper(), count, percentage)
    log.info("="*80)


if __name__ == "__main__":
    # Test videos from parent directory
    test_videos = [
        "../test.mp4",
        "../test1.mp4",
        "../test2.mp4",
        "../test3.mp4",
        "../test4.mp4",
        "../test5.mp4",
        "../test6.mp4",
    ]

    # Test first video that exists
    for video in test_videos:
        video_path = Path(__file__).parent / video
        if video_path.exists():
            test_video_multiperson(str(video_path), limit_frames=150)
            break
    else:
        log.error("No test videos found. Please ensure test videos are in parent directory.")
        sys.exit(1)
