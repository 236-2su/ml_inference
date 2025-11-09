"""RTMpose-based pose estimator used to classify standing vs. lying humans."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .detector import Detection
from .stream_listener import Frame

try:  # pragma: no cover - runtime dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - runtime dependency
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore

log = logging.getLogger(__name__)

default_keypoints: List[tuple[float, float, float]] = []


@dataclass
class PoseResult:
    label: str
    confidence: float
    keypoints: List[tuple[float, float, float]]


def classify_pose_from_keypoints(
    keypoints: np.ndarray,
    min_conf: float,
    lying_aspect_ratio: float,
    lying_torso_angle_deg: float,
) -> Tuple[str, float]:
    """Return (label, confidence) given COCO-style keypoints."""
    if keypoints.size == 0:
        return "unknown", 0.0

    conf_mask = keypoints[:, 2] >= min_conf
    if np.count_nonzero(conf_mask) < 4:
        return "unknown", 0.0

    filtered = keypoints[conf_mask]
    xs = filtered[:, 0]
    ys = filtered[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    if width <= 1.0 or height <= 1.0:
        return "unknown", 0.0

    aspect_ratio = height / (width + 1e-6)
    lying_scores: List[float] = []
    standing_scores: List[float] = []

    if aspect_ratio < lying_aspect_ratio:
        score = np.clip((lying_aspect_ratio - aspect_ratio) / lying_aspect_ratio, 0.0, 1.0)
        lying_scores.append(float(score))
    else:
        score = np.clip((aspect_ratio - lying_aspect_ratio) / max(1e-6, 1.5 - lying_aspect_ratio), 0.0, 1.0)
        standing_scores.append(float(score))

    torso_angle = _estimate_torso_angle_deg(keypoints, min_conf)
    if torso_angle is not None:
        if torso_angle < lying_torso_angle_deg:
            score = np.clip(
                (lying_torso_angle_deg - torso_angle) / max(lying_torso_angle_deg, 1e-6),
                0.0,
                1.0,
            )
            lying_scores.append(float(score))
        else:
            score = np.clip(
                (torso_angle - lying_torso_angle_deg) / max(90.0 - lying_torso_angle_deg, 1e-6),
                0.0,
                1.0,
            )
            standing_scores.append(float(score))

    lying_conf = float(np.mean(lying_scores)) if lying_scores else 0.0
    standing_conf = float(np.mean(standing_scores)) if standing_scores else 0.0
    top_conf = max(lying_conf, standing_conf)
    if top_conf < 0.15:
        return "unknown", 0.0

    if lying_conf > standing_conf:
        return "lying", lying_conf
    return "standing", standing_conf


def _estimate_torso_angle_deg(keypoints: np.ndarray, min_conf: float) -> Optional[float]:
    """Return the absolute angle (degrees) of the torso line relative to the horizontal axis."""
    # COCO indices
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    if max(needed) >= keypoints.shape[0]:
        return None
    pts = keypoints[needed]
    if (pts[:, 2] < min_conf).any():
        return None

    shoulder_mid = (pts[0][:2] + pts[1][:2]) / 2.0
    hip_mid = (pts[2][:2] + pts[3][:2]) / 2.0
    dx = float(abs(shoulder_mid[0] - hip_mid[0]))
    dy = float(abs(shoulder_mid[1] - hip_mid[1]))
    angle_rad = math.atan2(dy, dx + 1e-6)
    return math.degrees(angle_rad)


def label_pose(
    keypoints_xy: np.ndarray,
    keypoint_scores: np.ndarray,
    min_conf: float = 0.3,
    lying_aspect_ratio: float = 0.65,
    lying_torso_angle_deg: float = 35,
) -> str:
    """Utility used by legacy pose_test.py script."""
    if keypoints_xy.shape[0] != keypoint_scores.shape[0]:
        raise ValueError("Keypoint coords and scores must align.")
    combined = np.concatenate(
        (keypoints_xy.astype(np.float32), keypoint_scores.reshape(-1, 1).astype(np.float32)),
        axis=1,
    )
    label, _ = classify_pose_from_keypoints(combined, min_conf, lying_aspect_ratio, lying_torso_angle_deg)
    return label


class PoseEstimator:
    """RTMpose-backed estimator that annotates human detections with pose labels."""

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float = 0.35,
        keypoint_conf_threshold: float = 0.3,
        lying_aspect_ratio: float = 0.65,
        lying_torso_angle_deg: float = 35.0,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.lying_aspect_ratio = lying_aspect_ratio
        self.lying_torso_angle_deg = lying_torso_angle_deg
        self.model = self._load_model()

    def infer(self, frame: Frame, detection: Detection) -> PoseResult:
        if not self.model:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        image = self._frame_to_ndarray(frame)
        if image is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        x, y, w, h = detection.bbox
        x1 = max(int(x), 0)
        y1 = max(int(y), 0)
        x2 = min(int(x + w), image.shape[1])
        y2 = min(int(y + h), image.shape[0])
        if x2 <= x1 or y2 <= y1:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        crop = image[y1:y2, x1:x2]
        results = self.model.predict(source=crop, conf=self.conf_threshold, verbose=False)
        if not results:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        keypoints = self._select_keypoints(results[0])
        if keypoints is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        # Convert to absolute coordinates in the original frame.
        keypoints[:, 0] += x1
        keypoints[:, 1] += y1

        label, score = classify_pose_from_keypoints(
            keypoints,
            min_conf=self.keypoint_conf_threshold,
            lying_aspect_ratio=self.lying_aspect_ratio,
            lying_torso_angle_deg=self.lying_torso_angle_deg,
        )
        keypoints_list = [(float(k[0]), float(k[1]), float(k[2])) for k in keypoints]
        return PoseResult(label=label, confidence=score, keypoints=keypoints_list)

    def _load_model(self):
        if not self.model_path:
            log.info("Pose estimator disabled (model path missing).")
            return None
        if YOLO is None:
            log.warning("ultralytics not installed; pose estimator disabled.")
            return None
        try:
            model = YOLO(self.model_path)
            log.info("Loaded pose model %s", self.model_path)
            return model
        except FileNotFoundError:
            log.warning("Pose model %s not found; pose estimator disabled.", self.model_path)
            return None
        except Exception:  # pragma: no cover
            log.exception("Failed to load pose model %s", self.model_path)
            return None

    def _frame_to_ndarray(self, frame: Frame) -> Optional[np.ndarray]:
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

    def _select_keypoints(self, result) -> Optional[np.ndarray]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.data is None or keypoints.data.shape[0] == 0:
            return None
        data = keypoints.data.cpu().numpy()
        # Pick the person with the highest mean confidence.
        best_idx = int(np.argmax(data[:, :, 2].mean(axis=1)))
        return data[best_idx]
