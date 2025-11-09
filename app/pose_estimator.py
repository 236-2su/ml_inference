"""RTMPose ONNX-based pose estimator used to classify detailed human postures."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .detector import Detection
from .stream_listener import Frame

try:  # pragma: no cover - runtime dependency
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

try:  # pragma: no cover - runtime dependency
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None  # type: ignore

log = logging.getLogger(__name__)

INPUT_SIZE = (256, 192)  # (height, width)
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT_RATIO = 2.0
PAD_RATIO = 0.1

default_keypoints: List[tuple[float, float, float]] = []


@dataclass
class PoseResult:
    label: str
    confidence: float
    keypoints: List[tuple[float, float, float]]


@dataclass
class _PoseMetrics:
    aspect_ratio: float
    torso_angle: Optional[float]
    hip_norm: Optional[float]
    knee_norm: Optional[float]
    ankle_norm: Optional[float]


def classify_pose_from_keypoints(
    keypoints: np.ndarray,
    min_conf: float,
    lying_aspect_ratio: float,
    lying_torso_angle_deg: float,
) -> Tuple[str, float]:
    """Return (label, confidence) given COCO-style keypoints."""
    metrics = _summarize_keypoints(keypoints, min_conf)
    if metrics is None:
        return "unknown", 0.0

    scores = {
        "lying": _score_lying(metrics, lying_aspect_ratio, lying_torso_angle_deg),
        "standing": _score_standing(metrics),
        "crouching": _score_crouching(metrics),
        "sitting": _score_sitting(metrics),
    }
    if metrics.hip_norm is not None and metrics.hip_norm >= 0.7:
        scores["crouching"] = min(1.0, scores["crouching"] + 0.15)
        scores["sitting"] *= 0.7
    label, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.1:
        return "unknown", 0.0
    return label, float(np.clip(score, 0.0, 1.0))


def _summarize_keypoints(keypoints: np.ndarray, min_conf: float) -> Optional[_PoseMetrics]:
    if keypoints.size == 0:
        return None
    conf_mask = keypoints[:, 2] >= min_conf
    if np.count_nonzero(conf_mask) < 4:
        return None
    filtered = keypoints[conf_mask]
    xs = filtered[:, 0]
    ys = filtered[:, 1]
    width = float(xs.max() - xs.min())
    height = float(ys.max() - ys.min())
    if width < 1.0 or height < 1.0:
        return None

    aspect_ratio = height / (width + 1e-6)
    torso_angle = _estimate_torso_angle_deg(keypoints, min_conf)
    hip_norm = _normalized_y(keypoints, [11, 12], ys.min(), height, min_conf)
    knee_norm = _normalized_y(keypoints, [13, 14], ys.min(), height, min_conf)
    ankle_norm = _normalized_y(keypoints, [15, 16], ys.min(), height, min_conf)

    return _PoseMetrics(
        aspect_ratio=aspect_ratio,
        torso_angle=torso_angle,
        hip_norm=hip_norm,
        knee_norm=knee_norm,
        ankle_norm=ankle_norm,
    )


def _score_lying(metrics: _PoseMetrics, aspect_threshold: float, torso_threshold: float) -> float:
    aspect_score = 0.0
    if metrics.aspect_ratio < aspect_threshold:
        aspect_score = (aspect_threshold - metrics.aspect_ratio) / aspect_threshold
    angle_score = 0.0
    if metrics.torso_angle is not None:
        angle_score = max(
            0.0,
            (torso_threshold - metrics.torso_angle) / max(torso_threshold, 1e-6),
        )
    return float(np.clip(max(aspect_score, angle_score), 0.0, 1.0))


def _score_standing(metrics: _PoseMetrics) -> float:
    scores: list[float] = []
    aspect_min = 1.35
    aspect_max = 3.0
    if metrics.aspect_ratio > aspect_min:
        denom = max(aspect_max - aspect_min, 1e-6)
        scores.append(np.clip((metrics.aspect_ratio - aspect_min) / denom, 0.0, 1.0))
    if metrics.torso_angle is not None:
        torso_min = 55.0
        scores.append(
            np.clip((metrics.torso_angle - torso_min) / max(90.0 - torso_min, 1e-6), 0.0, 1.0)
        )
    if metrics.hip_norm is not None:
        hip_threshold = 0.75
        scores.append(np.clip((hip_threshold - metrics.hip_norm) / hip_threshold, 0.0, 1.0))
    standing_score = float(np.mean(scores)) if scores else 0.0
    if metrics.hip_norm is not None and metrics.hip_norm > 0.7:
        standing_score = min(standing_score * 0.25, 0.3)
    return standing_score


def _score_crouching(metrics: _PoseMetrics) -> float:
    scores: list[float] = []
    if metrics.hip_norm is not None:
        hip_min = 0.6
        scores.append(np.clip((metrics.hip_norm - hip_min) / (1 - hip_min), 0.0, 1.0))
    if metrics.knee_norm is not None and metrics.ankle_norm is not None:
        gap = max(0.0, metrics.ankle_norm - metrics.knee_norm)
        scores.append(np.clip((0.22 - gap) / 0.22, 0.0, 1.0))
    aspect_center = 1.1
    scores.append(np.clip(1.0 - abs(metrics.aspect_ratio - aspect_center) / 0.6, 0.0, 1.0))
    if metrics.torso_angle is not None:
        scores.append(np.clip(1.0 - abs(metrics.torso_angle - 55.0) / 40.0, 0.0, 1.0))
    crouch_score = float(np.mean(scores)) if scores else 0.0
    if (
        metrics.hip_norm is not None
        and metrics.knee_norm is not None
        and metrics.ankle_norm is not None
    ):
        gap = metrics.ankle_norm - metrics.knee_norm
        if metrics.hip_norm >= 0.65 and gap < 0.2:
            crouch_score = max(crouch_score + 0.2, 0.8)
    return crouch_score


def _score_sitting(metrics: _PoseMetrics) -> float:
    scores: list[float] = []
    scores.append(np.clip(1.0 - abs(metrics.aspect_ratio - 1.0) / 0.7, 0.0, 1.0))
    if metrics.hip_norm is not None:
        scores.append(np.clip(1.0 - abs(metrics.hip_norm - 0.55) / 0.35, 0.0, 1.0))
    if metrics.knee_norm is not None:
        scores.append(np.clip(1.0 - abs(metrics.knee_norm - 0.75) / 0.3, 0.0, 1.0))
    if metrics.ankle_norm is not None and metrics.knee_norm is not None:
        gap = max(0.0, metrics.ankle_norm - metrics.knee_norm)
        scores.append(np.clip(1.0 - abs(gap - 0.25) / 0.25, 0.0, 1.0))
    if metrics.torso_angle is not None:
        scores.append(np.clip((metrics.torso_angle - 35.0) / 30.0, 0.0, 1.0))
    return float(np.mean(scores)) if scores else 0.0


def _normalized_y(
    keypoints: np.ndarray,
    indices: Sequence[int],
    min_y: float,
    height: float,
    min_conf: float,
) -> Optional[float]:
    valid = []
    for idx in indices:
        if idx >= keypoints.shape[0]:
            continue
        kp = keypoints[idx]
        if kp[2] < min_conf:
            continue
        normalized = (kp[1] - min_y) / max(height, 1e-6)
        valid.append(float(np.clip(normalized, 0.0, 1.0)))
    if not valid:
        return None
    return float(np.mean(valid))


def _estimate_torso_angle_deg(keypoints: np.ndarray, min_conf: float) -> Optional[float]:
    """Return the absolute angle (degrees) of the torso line relative to the horizontal axis."""
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
    """RTMPose-backed estimator that annotates human detections with pose labels."""

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
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._load_model()

    def infer(self, frame: Frame, detection: Detection) -> PoseResult:
        if detection.confidence < self.conf_threshold:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        if not self._session or not self._input_name:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        image = self._frame_to_ndarray(frame)
        if image is None or cv2 is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        prepared = self._prepare_input(image, detection.bbox)
        if prepared is None:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)
        tensor, meta = prepared
        outputs = self._session.run(None, {self._input_name: tensor})  # type: ignore[arg-type]
        keypoints = self._decode_simcc(outputs, meta)
        if keypoints.size == 0:
            return PoseResult(label="unknown", confidence=0.0, keypoints=default_keypoints)

        label, score = classify_pose_from_keypoints(
            keypoints,
            min_conf=self.keypoint_conf_threshold,
            lying_aspect_ratio=self.lying_aspect_ratio,
            lying_torso_angle_deg=self.lying_torso_angle_deg,
        )
        keypoints_list = [(float(k[0]), float(k[1]), float(k[2])) for k in keypoints]
        return PoseResult(label=label, confidence=score, keypoints=keypoints_list)

    def _load_model(self) -> None:
        if not self.model_path:
            log.info("Pose estimator disabled (model path missing).")
            return
        if ort is None:
            log.warning("onnxruntime not installed; pose estimator disabled.")
            return
        try:
            session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
            self._session = session
            self._input_name = session.get_inputs()[0].name
            log.info("Loaded RTMPose model %s", self.model_path)
        except FileNotFoundError:
            log.warning("Pose model %s not found; pose estimator disabled.", self.model_path)
        except Exception:  # pragma: no cover - diagnostics only
            log.exception("Failed to load pose model %s", self.model_path)

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

    def _prepare_input(
        self, image: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> Optional[tuple[np.ndarray, dict]]:
        h, w = image.shape[:2]
        x, y, bw, bh = bbox
        pad_w = int(bw * PAD_RATIO)
        pad_h = int(bh * PAD_RATIO)
        x1 = int(np.clip(x - pad_w, 0, w - 1))
        y1 = int(np.clip(y - pad_h, 0, h - 1))
        x2 = int(np.clip(x + bw + pad_w, 0, w - 1))
        y2 = int(np.clip(y + bh + pad_h, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        resized = cv2.resize(crop, (INPUT_SIZE[1], INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32)
        tensor = (tensor - MEAN) / STD
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        meta = {
            "bbox": (x1, y1, x2, y2),
            "scale_x": crop.shape[1] / INPUT_SIZE[1],
            "scale_y": crop.shape[0] / INPUT_SIZE[0],
        }
        return tensor, meta

    def _decode_simcc(self, outputs: Sequence[np.ndarray], meta: dict) -> np.ndarray:
        pred_x, pred_y = outputs
        pred_x = np.squeeze(pred_x, axis=0)
        pred_y = np.squeeze(pred_y, axis=0)

        x_probs = self._softmax(pred_x, axis=-1)
        y_probs = self._softmax(pred_y, axis=-1)

        x_idx = np.argmax(x_probs, axis=-1)
        y_idx = np.argmax(y_probs, axis=-1)

        x_conf = np.take_along_axis(x_probs, x_idx[..., None], axis=-1).squeeze(-1)
        y_conf = np.take_along_axis(y_probs, y_idx[..., None], axis=-1).squeeze(-1)
        conf = (x_conf + y_conf) / 2.0

        coords = np.stack(
            [
                x_idx.astype(np.float32) / SIMCC_SPLIT_RATIO,
                y_idx.astype(np.float32) / SIMCC_SPLIT_RATIO,
                conf.astype(np.float32),
            ],
            axis=-1,
        )
        coords[:, 0] = coords[:, 0] * meta["scale_x"] + meta["bbox"][0]
        coords[:, 1] = coords[:, 1] * meta["scale_y"] + meta["bbox"][1]
        return coords

    @staticmethod
    def _softmax(arr: np.ndarray, axis: int = -1) -> np.ndarray:
        arr = arr - np.max(arr, axis=axis, keepdims=True)
        exp = np.exp(arr)
        return exp / np.sum(exp, axis=axis, keepdims=True)
