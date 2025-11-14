"""Test to show actual JSON output format from the system."""

import sys
import pickle
import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed.")
    sys.exit(1)

# Load SVM model
print("Loading SVM model...")
with open('models/svm_model_sit_stand_lie.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Initialize MediaPipe
print("Initializing MediaPipe Pose...\n")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Label mapping
LABELS = {0: 'sitting', 1: 'standing', 2: 'lying'}

def extract_features(image):
    """Extract 66 MediaPipe features from image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None, []

    xs = [lm.x for lm in results.pose_landmarks.landmark]
    ys = [lm.y for lm in results.pose_landmarks.landmark]

    # Get keypoints with visibility
    keypoints = [[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark]

    return xs + ys, keypoints

def predict_pose(features):
    """Predict pose using SVM."""
    if features is None or len(features) != 66:
        return None, 0.0
    prediction = svm_model.predict(np.array(features).reshape(1, -1))[0]

    # Get confidence from decision function
    decision = svm_model.decision_function(np.array(features).reshape(1, -1))
    if len(decision.shape) > 1:
        confidence = float(np.max(np.abs(decision)))
    else:
        confidence = float(np.abs(decision[0]))
    confidence = min(1.0, confidence / 3.0)

    return LABELS[prediction], confidence

def create_event_json(track_id, bbox, pose_label, pose_confidence, keypoints, frame_idx):
    """Create event JSON matching the event_builder format."""
    import uuid

    event = {
        "event_id": str(uuid.uuid4()),
        "category": "human",
        "track_id": track_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "bbox": bbox,
        "confidence": 0.85,  # YOLO detection confidence (example)
        "inference_latency_ms": 15,  # example latency
        "pose_label": pose_label,
        "pose_confidence": pose_confidence,
        "keypoints": keypoints[:5],  # First 5 keypoints for example (nose, eyes, ears)
        "frame_index": frame_idx
    }

    return event

# Test on a video
video_path = '../test1.mp4'  # SITTING video
if not Path(video_path).exists():
    video_path = '../test.mp4'

print("="*80)
print("JSON OUTPUT FORMAT TEST")
print("="*80)
print(f"\nTesting video: {video_path}\n")

cap = cv2.VideoCapture(video_path)
events = []

# Process first 30 frames
for frame_idx in range(30):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % 10 == 0:  # Every 10th frame
        h, w = frame.shape[:2]

        # Extract features
        features, keypoints = extract_features(frame)
        pose_label, pose_confidence = predict_pose(features)

        if pose_label:
            # Simulate bounding box (in real system, YOLO provides this)
            bbox = [int(w*0.3), int(h*0.1), int(w*0.4), int(h*0.8)]

            # Create event
            event = create_event_json(
                track_id=1,
                bbox=bbox,
                pose_label=pose_label,
                pose_confidence=pose_confidence,
                keypoints=keypoints,
                frame_idx=frame_idx
            )
            events.append(event)

            print(f"Frame {frame_idx}: {pose_label.upper()} (confidence: {pose_confidence:.3f})")

cap.release()

# Show JSON examples
print("\n" + "="*80)
print("EXAMPLE JSON OUTPUT (Single Event)")
print("="*80)
if events:
    print(json.dumps(events[0], indent=2))

print("\n" + "="*80)
print("EXAMPLE JSON OUTPUT (Batch of Events)")
print("="*80)
print(json.dumps(events, indent=2))

print("\n" + "="*80)
print("JSON SCHEMA EXPLANATION")
print("="*80)
print("""
Human Event Fields:
{
  "event_id": "unique UUID for this event",
  "category": "human" (or "wildlife" for animals),
  "track_id": 1 (unique ID for this person),
  "timestamp_utc": "2024-11-14T03:30:00.123456",
  "bbox": [x, y, width, height] (bounding box coordinates),
  "confidence": 0.85 (YOLO detection confidence 0-1),
  "inference_latency_ms": 15 (processing time in milliseconds),
  "pose_label": "sitting|standing|lying" (your SVM prediction),
  "pose_confidence": 0.92 (SVM confidence 0-1),
  "keypoints": [[x, y, visibility], ...] (33 MediaPipe landmarks),
  "frame_index": 10 (frame number)
}

Optional Fields (when status changes):
{
  "status": "fall_detected|heatstroke_watch|heatstroke_alert",
  "duration_seconds": 300 (how long in this pose),
  "image_jpeg_base64": "base64 encoded snapshot (if enabled)"
}
""")

print("\n" + "="*80)
print("REAL-TIME EVENT DELIVERY")
print("="*80)
print(f"""
In production, these events are sent to your FastAPI endpoint:
  POST {events[0].get('timestamp_utc', 'https://k13e106.p.ssafy.io/dev/api/events')}

Headers:
  Authorization: Bearer <your-token>
  Content-Type: application/json

Body:
  [list of event objects like shown above]

The system batches events from all detected people and sends them
together for efficient processing.
""")

print("="*80)
print(f"Total events generated: {len(events)}")
print("="*80)
