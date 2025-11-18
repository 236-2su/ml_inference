"""MediaPipe + SVM based pose estimator for multi-person streaming inference."""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .detector import Detection
from .stream_listener import Frame

try:  # pragma: no cover - runtime dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - runtime dependency
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover
    mp = None  # type: ignore

log = logging.getLogger(__name__)

# Pose labels mapping from SVM output
LABEL_MAP = {0: "sitting", 1: "standing", 2: "lying"}
default_keypoints: List[tuple[float, float, float]] = []


@dataclass
class PoseResult:
    """Result from pose estimation matching the existing interface."""
    label: str
    confidence: float
    keypoints: List[tuple[float, float, float]]


class MediaPipePoseEstimator:
    """MediaPipe-backed estimator with trained SVM classifier for pose labeling."""

    def __init__(
        self,
        svm_model_path: str | None = None,
        conf_threshold: float = 0.35,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.svm_model_path = svm_model_path
        self.conf_threshold = conf_threshold
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._svm_model = None
        self._mp_pose = None
        self._pose = None
        self._load_model()

    def infer(self, frame: Frame, detection: Detection) -> PoseResult:
        """Infer pose for a single person detection.

        Args:
            frame: Video frame containing the person
            detection: Person detection with bounding box

        Returns:
            PoseResult with label, confidence, and keypoints
        """
        inference_start = time.perf_counter()

        if detection.confidence < self.conf_threshold:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        if not self._svm_model or not self._pose:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        image = self._frame_to_ndarray(frame)
        if image is None or cv2 is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        # Extract person ROI from bounding box
        crop_start = time.perf_counter()
        cropped_person = self._extract_person_roi(image, detection.bbox)
        if cropped_person is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        crop_time_ms = (time.perf_counter() - crop_start) * 1000

        # Extract MediaPipe landmarks
        mediapipe_start = time.perf_counter()
        features, keypoints_list = self._extract_mediapipe_features(cropped_person)
        mediapipe_time_ms = (time.perf_counter() - mediapipe_start) * 1000

        if features is None or len(features) != 66:
            log.debug("MediaPipe pose estimation took %.2f ms (no landmarks)", mediapipe_time_ms)
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        # Predict using SVM
        try:
            svm_start = time.perf_counter()
            prediction = self._svm_model.predict(np.array(features).reshape(1, -1))[0]
            # Get decision function scores for confidence
            decision_scores = self._svm_model.decision_function(np.array(features).reshape(1, -1))
            svm_time_ms = (time.perf_counter() - svm_start) * 1000

            # Convert decision scores to confidence (using softmax-like normalization)
            if len(decision_scores.shape) > 1:
                # Multi-class: get max score
                confidence = float(np.max(np.abs(decision_scores)))
            else:
                confidence = float(np.abs(decision_scores[0]))

            # Normalize confidence to 0-1 range (rough approximation)
            confidence = min(1.0, confidence / 3.0)  # Scale down typical SVM scores
            confidence = max(0.0, confidence)

            label = LABEL_MAP.get(int(prediction), "unknown")

            total_time_ms = (time.perf_counter() - inference_start) * 1000
            log.info(
                "Pose estimation complete: crop=%.2f ms, mediapipe=%.2f ms, svm=%.2f ms, total=%.2f ms, label=%s, confidence=%.3f",
                crop_time_ms,
                mediapipe_time_ms,
                svm_time_ms,
                total_time_ms,
                label,
                confidence
            )

            return PoseResult(label=label, confidence=confidence, keypoints=keypoints_list)

        except Exception as e:
            log.warning("SVM prediction failed: %s", e)
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

    def _load_model(self) -> None:
        """Load SVM model and initialize MediaPipe Pose."""
        # Load SVM model
        if not self.svm_model_path:
            log.info("Pose estimator disabled (SVM model path missing).")
            return

        try:
            with open(self.svm_model_path, 'rb') as f:
                self._svm_model = pickle.load(f)
            log.info("Loaded SVM model from %s", self.svm_model_path)
        except FileNotFoundError:
            log.warning("SVM model %s not found; pose estimator disabled.", self.svm_model_path)
            return
        except Exception as e:
            log.exception("Failed to load SVM model %s: %s", self.svm_model_path, e)
            return

        # Initialize MediaPipe Pose
        if mp is None:
            log.warning("mediapipe not installed; pose estimator disabled.")
            return

        try:
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                model_complexity=1  # 0=lite, 1=full, 2=heavy
            )
            log.info("Initialized MediaPipe Pose estimator")
        except Exception as e:
            log.exception("Failed to initialize MediaPipe Pose: %s", e)

    def _frame_to_ndarray(self, frame: Frame) -> Optional[np.ndarray]:
        """Convert Frame to numpy array."""
        if frame.image is None:
            return None
        if isinstance(frame.image, np.ndarray):
            return frame.image
        if isinstance(frame.image, bytes):
            if cv2 is None:
                return None
            np_buffer = np.frombuffer(frame.image, dtype=np.uint8)
            decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
            return decoded
        return None

    def _extract_person_roi(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """Extract person region from image using bounding box.

        Args:
            image: Full frame image
            bbox: (x, y, width, height) bounding box

        Returns:
            Cropped image of the person or None if invalid
        """
        h, w = image.shape[:2]
        x, y, bw, bh = bbox

        # Add small padding
        pad_ratio = 0.05
        pad_w = int(bw * pad_ratio)
        pad_h = int(bh * pad_ratio)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + bw + pad_w)
        y2 = min(h, y + bh + pad_h)

        if x2 <= x1 or y2 <= y1:
            return None

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        return cropped

    def _extract_mediapipe_features(
        self, image: np.ndarray
    ) -> tuple[Optional[List[float]], List[tuple[float, float, float]]]:
        """Extract 66 features (33 landmarks × 2 coords) using MediaPipe.

        Args:
            image: Cropped person image

        Returns:
            Tuple of (features_list, keypoints_list)
            - features_list: 66 floats [x0, x1, ..., x32, y0, y1, ..., y32]
            - keypoints_list: List of (x, y, confidence) tuples for compatibility
        """
        if self._pose is None:
            return None, []

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = self._pose.process(image_rgb)

        if not results.pose_landmarks:
            return None, []

        # Extract x and y coordinates
        xs = []
        ys = []
        keypoints_list = []

        for landmark in results.pose_landmarks.landmark:
            xs.append(landmark.x)
            ys.append(landmark.y)
            # Store as (x, y, visibility) for keypoints output
            keypoints_list.append((
                float(landmark.x),
                float(landmark.y),
                float(landmark.visibility)
            ))

        # Concatenate as required by SVM: [x0, x1, ..., x32, y0, y1, ..., y32]
        features = xs + ys

        if len(features) != 66:
            log.warning("Expected 66 features, got %d", len(features))
            return None, []

        return features, keypoints_list

    def __del__(self):
        """Cleanup MediaPipe Pose instance."""
        if self._pose:
            try:
                self._pose.close()
            except Exception:
                pass
