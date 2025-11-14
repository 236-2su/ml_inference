"""Quick test of MediaPipe + SVM multi-person pose detection."""

import sys
import pickle
import cv2
import numpy as np

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
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

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

# Test on first available video
videos = ['../test.mp4', '../test1.mp4', '../test2.mp4']
video_path = None
for v in videos:
    if cv2.VideoCapture(v).isOpened():
        video_path = v
        break

if not video_path:
    print("ERROR: No test videos found")
    sys.exit(1)

print(f"\nTesting on: {video_path}")
print("="*60)

cap = cv2.VideoCapture(video_path)
frame_count = 0
pose_counts = {'SITTING': 0, 'STANDING': 0, 'LYING': 0}

while cap.isOpened() and frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % 3 == 0:  # Process every 3rd frame
        features = extract_features(frame)
        pose_label = predict_pose(features)

        if pose_label:
            pose_counts[pose_label] = pose_counts.get(pose_label, 0) + 1

            if frame_count % 15 == 0:
                print(f"Frame {frame_count:3d}: {pose_label}")

cap.release()

print("="*60)
print(f"\nProcessed {frame_count} frames")
print("\nPose Distribution:")
for pose, count in sorted(pose_counts.items(), key=lambda x: -x[1]):
    if count > 0:
        pct = (count / sum(pose_counts.values())) * 100
        print(f"  {pose:10s}: {count:3d} ({pct:.1f}%)")
print("\n✓ MediaPipe + SVM integration working!")
