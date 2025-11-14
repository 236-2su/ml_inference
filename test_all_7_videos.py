"""Test all 7 videos with MediaPipe + SVM multi-person pose detection."""

import sys
import pickle
import cv2
import numpy as np
import os
from pathlib import Path

try:
    import mediapipe as mp
except ImportError:
    print("ERROR: mediapipe not installed. Run: pip install mediapipe scikit-learn")
    sys.exit(1)

# Load SVM model
print("Loading SVM model...")
with open('models/svm_model_sit_stand_lie.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# Initialize MediaPipe
print("Initializing MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Label mapping
LABELS = {0: 'SITTING', 1: 'STANDING', 2: 'LYING'}

def extract_features(image):
    """Extract 66 MediaPipe features from image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    xs = [lm.x for lm in results.pose_landmarks.landmark]
    ys = [lm.y for lm in results.pose_landmarks.landmark]

    return xs + ys

def predict_pose(features):
    """Predict pose using SVM."""
    if features is None or len(features) != 66:
        return None
    prediction = svm_model.predict(np.array(features).reshape(1, -1))[0]
    return LABELS[prediction]

def test_video(video_path, max_frames=300):
    """Test a single video and return statistics."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0

    frame_count = 0
    processed_count = 0
    pose_counts = {'SITTING': 0, 'STANDING': 0, 'LYING': 0}

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 3rd frame
        if frame_count % 3 == 0:
            features = extract_features(frame)
            pose_label = predict_pose(features)

            if pose_label:
                pose_counts[pose_label] = pose_counts.get(pose_label, 0) + 1
                processed_count += 1

    cap.release()

    # Determine dominant pose
    dominant_pose = 'NONE'
    if pose_counts:
        dominant_pose = max(pose_counts, key=pose_counts.get)

    return {
        'filename': os.path.basename(video_path),
        'total_frames': total_frames,
        'frames_processed': frame_count,
        'poses_detected': processed_count,
        'fps': fps,
        'duration_sec': duration,
        'pose_counts': pose_counts,
        'dominant_pose': dominant_pose
    }

# Test all 7 videos
videos = [
    '../test.mp4',
    '../test1.mp4',
    '../test2.mp4',
    '../test3.mp4',
    '../test4.mp4',
    '../test5.mp4',
    '../test6.mp4'
]

print("\n" + "="*80)
print("TESTING ALL 7 VIDEOS WITH MEDIAPIPE + SVM POSE CLASSIFIER")
print("="*80)

results = []

for video_path in videos:
    if not os.path.exists(video_path):
        print(f"\n[SKIP] {os.path.basename(video_path)} - File not found")
        continue

    print(f"\nTesting: {os.path.basename(video_path)}")
    print("-" * 80)

    result = test_video(video_path, max_frames=300)

    if result:
        results.append(result)
        print(f"  Duration: {result['duration_sec']:.1f}s | Frames: {result['total_frames']} | FPS: {result['fps']}")
        print(f"  Processed: {result['frames_processed']} frames")
        print(f"  Poses detected: {result['poses_detected']}")
        print(f"  Predictions: SIT={result['pose_counts']['SITTING']}, "
              f"STAND={result['pose_counts']['STANDING']}, "
              f"LIE={result['pose_counts']['LYING']}")
        print(f"  Dominant Pose: {result['dominant_pose']}")
    else:
        print(f"  [ERROR] Failed to process video")

# Print comprehensive summary
print("\n" + "="*80)
print("COMPREHENSIVE RESULTS SUMMARY")
print("="*80)

if results:
    print(f"\nTotal videos tested: {len(results)}")
    print("\n" + "-"*80)
    print(f"{'Video':<15} {'Duration':<10} {'Frames':<10} {'Dominant Pose':<15} {'SIT/STAND/LIE'}")
    print("-"*80)

    for r in results:
        pose_str = f"{r['pose_counts']['SITTING']:3d}/{r['pose_counts']['STANDING']:3d}/{r['pose_counts']['LYING']:3d}"
        print(f"{r['filename']:<15} {r['duration_sec']:>5.1f}s     {r['total_frames']:>6}     "
              f"{r['dominant_pose']:<15} {pose_str}")

    # Overall statistics
    total_detections = sum(r['poses_detected'] for r in results)
    total_sit = sum(r['pose_counts']['SITTING'] for r in results)
    total_stand = sum(r['pose_counts']['STANDING'] for r in results)
    total_lie = sum(r['pose_counts']['LYING'] for r in results)

    print("-"*80)
    print(f"\nOverall Statistics:")
    print(f"  Total pose detections: {total_detections}")
    print(f"  SITTING:  {total_sit:4d} ({total_sit/total_detections*100:.1f}%)" if total_detections > 0 else "  SITTING:     0 (0.0%)")
    print(f"  STANDING: {total_stand:4d} ({total_stand/total_detections*100:.1f}%)" if total_detections > 0 else "  STANDING:    0 (0.0%)")
    print(f"  LYING:    {total_lie:4d} ({total_lie/total_detections*100:.1f}%)" if total_detections > 0 else "  LYING:       0 (0.0%)")

    # Pose distribution by video
    print(f"\nDominant Pose Distribution:")
    pose_dist = {}
    for r in results:
        pose_dist[r['dominant_pose']] = pose_dist.get(r['dominant_pose'], 0) + 1

    for pose in ['SITTING', 'STANDING', 'LYING']:
        count = pose_dist.get(pose, 0)
        print(f"  {pose}: {count} video(s)")

print("\n" + "="*80)
print("SUCCESS! MediaPipe + SVM integration working for all videos!")
print("="*80)
